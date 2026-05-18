import contextlib
import logging
import math
from collections.abc import Callable
from enum import Enum
from typing import NamedTuple

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from dec_torch.utils.data import extract_batch_pairs

logger = logging.getLogger(__name__)


class HistoryTracker:
    """Performance log designed for efficient storage and access to records.

    This class provides an efficient way to track and store training/validation
    metrics during model training. It uses enums internally for fast lookup
    and provides convenient conversion to pandas DataFrame for analysis.

    Example:
        >>> tracker = HistoryTracker(phases=['training', 'validation'],
        ...                          metrics=['loss', 'accuracy'])
        >>> tracker.add_record(epoch=1, phase='training', metric='loss', score=0.45)
        >>> tracker.add_record(epoch=1, phase='validation', metric='loss', score=0.42)
        >>> df = tracker.history
        >>> print(df.head())
           epoch     phase  metric  score
        0      1  training    loss   0.45
        1      1 validation    loss   0.42
    """

    def __init__(self, phases: list[str], metrics: list[str]) -> None:
        """Initialize HistoryTracker with predetermined phases and metrics.

        Args:
            phases: List of model phases, e.g., 'training', 'validation'.
            metrics: Model performance metrics, e.g., 'loss', 'accuracy'.
        """
        Phase = Enum("Phase", phases)
        Metric = Enum("Metric", metrics)
        RecordKey = NamedTuple("RecordKey", epoch=int, phase=Enum, metric=Enum)

        self.Phase = Phase
        self.Metric = Metric
        self.RecordKey = RecordKey

        self._history: dict[RecordKey, float] = {}

    def add_record(self, epoch: int, phase: str, metric: str, score: float) -> None:
        """Record a score in history log.

        Args:
            epoch: The epoch number.
            phase: The phase name ('training' or 'validation').
            metric: The metric name ('loss', 'accuracy', etc.).
            score: The metric value to record.
        """
        self._history[
            self.RecordKey(
                epoch=int(epoch), phase=self.Phase[phase], metric=self.Metric[metric]
            )
        ] = float(score)

    def get_record(self, epoch: int, phase: str, metric: str) -> float:
        """Get a record from history log.

        Args:
            epoch: The epoch number.
            phase: The phase name ('training' or 'validation').
            metric: The metric name ('loss', 'accuracy', etc.).

        Returns:
            The recorded score for the specified epoch, phase, and metric.
        """
        return self._history[
            self.RecordKey(
                epoch=int(epoch), phase=self.Phase[phase], metric=self.Metric[metric]
            )
        ]

    @property
    def history(self) -> pd.DataFrame:
        """Access all history at once as a DataFrame.

        Converts the internal efficient storage format to a pandas DataFrame
        for analysis and visualization. This operation involves data copying
        and may be inefficient for very large training histories.

        Returns:
            DataFrame with columns ['epoch', 'phase', 'metric', 'score'].

        Example:
            >>> df = tracker.history
            >>> print(df[df['metric'] == 'loss'].head())
        """
        rows = [
            {
                "epoch": key.epoch,
                "phase": key.phase.name,
                "metric": key.metric.name,
                "score": value,
            }
            for key, value in self._history.items()
        ]

        df = pd.DataFrame(rows)
        df["phase"] = df["phase"].astype("category")
        df["metric"] = df["metric"].astype("category")
        return df.sort_values(by=["epoch", "phase", "metric"])

    def __str__(self) -> str:
        """String representation of the history tracker."""
        return str(self.history)


def run_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    metrics: dict[str, Callable],
    optimizer: torch.optim.Optimizer,  # Not used if train=False
    train: bool,
    device: str | torch.device | None = None,
    transform: Callable | None = None,
    derive_loss_target_fn: Callable | None = None,
    return_label: bool = False,
) -> tuple[dict[str, float], torch.Tensor | None]:
    """Run one epoch loop of an autoencoder or DEC model.

    This function executes a single training or validation epoch, computing
    metrics and optionally updating model parameters. It handles both standard
    autoencoder training and DEC-specific training with custom target derivation.

    Args:
        model: The model to train or evaluate (autoencoder or DEC).
        data_loader: DataLoader providing batches of data.
        metrics: Dictionary of metric functions.
            Must include 'loss' metric. Functions should return cumulative scores
            (not averaged over batch), except for loss which should be per-batch.
        optimizer: Optimizer for updating parameters.
            Only used when train=True.
        train: Whether to train (True) or evaluate (False).
        device: Device to move data to.
        transform: Optional transform to apply to inputs
            (e.g., for layer-wise training of StackedAutoEncoder).
        derive_loss_target_fn: Function to derive loss
            targets from model outputs (used in DEC training).
        return_label: Whether to return predicted labels/assignments.

    Returns:
        A tuple containing:
            - Dictionary of metric names to averaged scores.
            - Optional tensor of predicted labels (if return_label=True).

    Notes:
        The DataLoader is expected to return either:
        - A Tensor (just the input) for autoencoder training
        - A tuple of (input, target) for supervised or custom training
        - A tuple of (input, target) tensors

        The transform parameter is used during layer-wise training of
        dec_torch.autoencoder.StackedAutoEncoder to transform inputs
        using previously trained encoder layers.

        The derive_loss_target_fn parameter is used in DEC training to
        compute the target distribution P from soft assignment Q predictions.

        When return_label=True is used for DEC training, the DataLoader
        should be initialized with shuffle=False to properly track cluster
        reassignments.

    Example:
        >>> metrics = {'loss': nn.MSELoss()}
        >>> scores, _ = run_one_epoch(model, loader, metrics, optimizer, train=True)
        >>> print(f"Average loss: {scores['loss']:.4f}")
    """
    if "loss" not in metrics:
        msg = "loss not found in metrics"
        raise AssertionError(msg)

    if train:
        model.train()
        context = contextlib.nullcontext()
    else:
        model.eval()
        context = torch.no_grad()

    scores = dict.fromkeys(metrics, 0.0)
    n_samples = 0
    label_list = [] if return_label else None

    for batch in data_loader:
        batch_input, batch_target = extract_batch_pairs(batch, device, transform)

        batch_output = None
        with context:
            batch_output = model(batch_input)

        if derive_loss_target_fn is None:
            loss = metrics["loss"](batch_output, batch_target)
        else:
            with torch.no_grad():
                local_target = derive_loss_target_fn(batch_output)
            loss = metrics["loss"](batch_output, local_target)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if return_label and label_list is not None:
            label_list.append(torch.argmax(batch_output, dim=1))

        # Storing loss, and other optional metric scores.
        for metric, score_fn in metrics.items():
            if metric == "loss":
                scores[metric] += loss.item() * batch_input.size(0)
            else:
                scores[metric] += score_fn(batch_output, batch_target)
        n_samples += batch_input.size(0)

    # Averaging metric scores.
    for metric in scores:
        scores[metric] /= n_samples

    labels = torch.cat(label_list) if return_label else None
    return scores, labels


def train_ae_model(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.modules.loss._Loss,
    val_loader: DataLoader | None = None,
    n_epoch: int = 100,
    transform: Callable | None = None,
    device: str | torch.device | None = None,
    verbose: bool = True,
    max_verbose: int = 20,
) -> pd.DataFrame:
    """Train an autoencoder with specified hyperparameters.

    This function provides a complete training loop for autoencoders (both
    standard and stacked). It handles training and validation, tracks history,
    and provides verbose logging.

    Args:
        model: AutoEncoder or StackedAutoEncoder module.
        train_loader: DataLoader of training set.
        optimizer: Optimizer tied to model.parameters().
        loss_fn: Loss function for reconstruction error.
        val_loader: DataLoader of validation set.
            If None, skips validation. Defaults to None.
        n_epoch: Total number of epochs during training. Defaults to 100.
        transform: Apply transformation on the loaded batch
            (used for layer-wise training of StackedAutoEncoder).
        device: Tensor computation device to load data.
            If None, uses the device of the model. Defaults to None.
        verbose: Specifies if training performance is logged.
            Defaults to True.
        max_verbose: Maximum (or +1) number of status lines to log.
            Defaults to 20.

    Returns:
        Training and validation loss history as a DataFrame
            with columns ['epoch', 'phase', 'metric', 'score'].

    Notes:
        The DataLoader is expected to return either:
        - A Tensor (just the input) for autoencoder training
        - A tuple of (input, target) for custom training scenarios

        The transform parameter is specifically used during layer-wise
        training of StackedAutoEncoder to transform inputs using previously
        trained encoder layers.

    Example:
        >>> from dec_torch.autoencoder import AutoEncoder, AutoEncoderConfig
        >>> from torch.utils.data import DataLoader, TensorDataset
        >>> from torch import optim, nn
        >>>
        >>> # Setup model and data
        >>> config = AutoEncoderConfig.build(input_dim=784, latent_dim=128)
        >>> model = AutoEncoder(config)
        >>> dataset = TensorDataset(torch.randn(1000, 784))
        >>> train_loader = DataLoader(dataset, batch_size=32)
        >>>
        >>> # Training setup
        >>> optimizer = optim.Adam(model.parameters(), lr=0.001)
        >>> loss_fn = nn.MSELoss()
        >>>
        >>> # Train model
        >>> history = train_ae_model(
        ...     model, train_loader, optimizer, loss_fn,
        ...     n_epoch=50, verbose=True
        ... )
        >>> print(f"Final training loss: {history.iloc[-1]['score']:.4f}")
    """
    phases = [("training", train_loader, True)]
    if val_loader is not None:
        phases.append(("validation", val_loader, False))
    metrics = {"loss": loss_fn}
    tracker = HistoryTracker(
        phases=[p[0] for p in phases], metrics=list(metrics.keys())
    )
    verbose_steps = _verbosity_steps(n_epoch, max_verbose) if verbose else ()

    for epoch_i in range(n_epoch):
        for phase, loader, train_mode in phases:
            scores, _ = run_one_epoch(
                model,
                loader,
                metrics,
                optimizer,
                train=train_mode,
                device=device,
                transform=transform,
            )
            for metric, score in scores.items():
                tracker.add_record(epoch_i + 1, phase, metric, score)

        if epoch_i in verbose_steps:
            train_loss = tracker.get_record(epoch_i + 1, "training", "loss")
            msg = (
                f"[Epoch: {epoch_i + 1:4d}] | "
                f"Train. loss: {train_loss:.4f} | "
            )
            if val_loader is not None:
                val_loss = tracker.get_record(epoch_i + 1, "validation", "loss")
                msg += f"Val. loss: {val_loss:.4f} |"
            logger.info(msg)

    return tracker.history


def train_dec_model(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.modules.loss._Loss,
    val_loader: DataLoader | None = None,
    tolerance: float = 0.01,
    derive_loss_target_fn: Callable | None = None,
    device: str | torch.device | None = None,
    verbose: bool = True,
    max_verbose: int = 10000,
    max_epoch: int = 10000,
) -> pd.DataFrame:
    """Train a DEC model with specified hyperparameters.

    This function implements the complete DEC training loop with automatic
    stopping based on cluster assignment stability. It tracks the fraction
    of samples that change cluster assignments between epochs and stops when
    this fraction falls below the tolerance threshold.

    Args:
        model: A DEC model instance.
        train_loader: DataLoader of training set.
            Should be initialized with shuffle=False for proper tracking.
        optimizer: Optimizer for model parameters.
        loss_fn: Loss function (typically KLDivLoss).
        val_loader: DataLoader of validation set.
            If None, skips validation. Defaults to None.
        tolerance: Maximum cluster reassignment threshold to stop training.
            Training stops when reassignment fraction < tolerance. Defaults to 0.01.
        derive_loss_target_fn: Function to derive loss targets
            from model outputs (should be DEC.target_distribution).
        device: Tensor computation device.
            If None, uses the model's device. Defaults to None.
        verbose: Whether to log training progress. Defaults to True.
        max_verbose: Maximum number of status lines to log. Defaults to 10000.
        max_epoch: Maximum number of epochs before forced stopping.
            Defaults to 10000.

    Returns:
        Training and validation loss history as a DataFrame.

    Training Process:
        1. For each epoch, compute soft assignments for all training samples
        2. Calculate target distribution from soft assignments
        3. Compute KL divergence loss between Q and P
        4. Track cluster reassignments compared to previous epoch
        5. Stop when reassignments < tolerance or max_epoch reached

    Notes:
        The DataLoader is expected to return either:
        - A Tensor (just the input) for standard DEC training
        - A tuple of (input, target) for custom scenarios

        The training DataLoader must be initialized with shuffle=False
        to properly track cluster reassignments across epochs.

        The derive_loss_target_fn should be set to DEC.target_distribution
        for standard DEC training.

    Example:
        >>> from dec_torch.dec import DEC, KLDivLoss
        >>> from torch import optim
        >>>
        >>> # Setup
        >>> dec_model = DEC(encoder, centroids, alpha=1.0)
        >>> loss_fn = KLDivLoss()
        >>> optimizer = optim.SGD(dec_model.parameters(), lr=0.001, momentum=0.9)
        >>>
        >>> # Train with default tolerance (1%)
        >>> history = train_dec_model(
        ...     dec_model, train_loader, optimizer, loss_fn,
        ...     derive_loss_target_fn=DEC.target_distribution,
        ...     verbose=True
        ... )
        >>>
        >>> # Train with stricter tolerance
        >>> history = train_dec_model(
        ...     dec_model, train_loader, optimizer, loss_fn,
        ...     tolerance=0.001,  # Stop when < 0.01%% reassignments
        ...     derive_loss_target_fn=DEC.target_distribution,
        ...     verbose=True
        ... )

    Note:
        Training automatically stops when the cluster assignments stabilize,
        indicating convergence. The final reassignment fraction is logged.
    """
    phases = [("training", train_loader, True)]
    if val_loader is not None:
        phases.append(("validation", val_loader, False))
    metrics = {"loss": loss_fn}
    tracker = HistoryTracker(phases=[p[0] for p in phases], metrics=["loss"])
    verbose_steps = _verbosity_steps(max_epoch, max_verbose) if verbose else ()
    previous_labels = None
    current_labels = None
    reassignment_fraction = 1.0

    for epoch_i in range(max_epoch):
        for phase, loader, train_mode in phases:
            scores, labels = run_one_epoch(
                model,
                loader,
                metrics,
                optimizer,
                train=train_mode,
                device=device,
                derive_loss_target_fn=derive_loss_target_fn,
                return_label=(phase == "training"),
            )

            if labels is not None:
                current_labels = labels

            for metric, score in scores.items():
                tracker.add_record(epoch_i + 1, phase, metric, score)

        if previous_labels is not None:
            reassignments = sum(previous_labels != current_labels)
            reassignment_fraction = reassignments / len(previous_labels)
        previous_labels = current_labels

        if epoch_i in verbose_steps or reassignment_fraction < tolerance:
            train_loss = tracker.get_record(epoch_i + 1, "training", "loss")
            msg = (
                f"[Epoch: {epoch_i + 1:4d}] | "
                f"Train. loss: {train_loss:.4f} | "
                f"Reassignment: {reassignment_fraction:7.2%} | "
            )
            if val_loader is not None:
                val_loss = tracker.get_record(epoch_i + 1, "validation", "loss")
                msg += f"Val. loss: {val_loss:.4f} |"
            logger.info(msg)

        if reassignment_fraction < tolerance:
            break

    return tracker.history


def _verbosity_steps(n_epoch: int, max_verbose: int) -> set[int]:
    """Determine which epochs should be reported during training.

    Args:
        n_epoch: total number of epochs in training.
        max_verbose: maximum (potentially one extra) number of logs to print.
    """
    verbose_steps = set(range(0, n_epoch, math.ceil(n_epoch / max_verbose)))
    verbose_steps.add(n_epoch - 1)

    return verbose_steps
