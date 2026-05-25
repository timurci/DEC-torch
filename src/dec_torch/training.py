import contextlib
import logging
import math
from collections.abc import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from dec_torch.utils.data import extract_batch_pairs

logger = logging.getLogger(__name__)


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
    trackers: list | None = None,
    phase_prefix: str | None = None,
) -> None:
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
        trackers: List of experiment trackers to log metrics to.
            If None, no tracking is performed. Defaults to None.
        phase_prefix: Optional prefix to prepend to phase names when logging.
            For example, "layer_0" produces phases "layer_0_train" and
            "layer_0_val". Defaults to None.

    Notes:
        The DataLoader is expected to return either:
        - A Tensor (just the input) for autoencoder training
        - A tuple of (input, target) for custom training scenarios

        The transform parameter is specifically used during layer-wise
        training of StackedAutoEncoder to transform inputs using previously
        trained encoder layers.

    Example:
        >>> from dec_torch.autoencoder import AutoEncoder, AutoEncoderConfig
        >>> from dec_torch.trackers import HistoryTracker
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
        >>> tracker = HistoryTracker()
        >>> train_ae_model(
        ...     model, train_loader, optimizer, loss_fn,
        ...     n_epoch=50, verbose=True, trackers=[tracker]
        ... )
        >>> print(f"Final training loss: {tracker.history.iloc[-1]['score']:.4f}")
    """
    from dec_torch.trackers import Phase

    phases = [(Phase.TRAIN, train_loader, True)]
    if val_loader is not None:
        phases.append((Phase.VAL, val_loader, False))
    metrics = {"loss": loss_fn}

    experiment_trackers = trackers or []
    if experiment_trackers:
        for tracker in experiment_trackers:
            tracker.log_params(
                {
                    "n_epoch": n_epoch,
                    "has_validation": val_loader is not None,
                }
            )

    verbose_steps = _verbosity_steps(n_epoch, max_verbose) if verbose else ()

    for epoch_i in range(n_epoch):
        epoch_train_scores: dict[str, float] = {}
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
            if experiment_trackers:
                log_phase = (
                    f"{phase_prefix}_{phase}"
                    if phase_prefix is not None
                    else str(phase)
                )
                for tracker in experiment_trackers:
                    tracker.log_metrics(
                        phase=log_phase, step=epoch_i + 1, metrics=scores
                    )

            if phase == Phase.TRAIN:
                epoch_train_scores = scores

        if epoch_i in verbose_steps:
            msg = (
                f"[Epoch: {epoch_i + 1:4d}] | "
                f"Train. loss: {epoch_train_scores['loss']:.4f} | "
            )
            if val_loader is not None:
                msg += f"Val. loss: {scores['loss']:.4f} |"
            logger.info(msg)


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
    trackers: list | None = None,
    phase_prefix: str | None = None,
) -> None:
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
        trackers: List of experiment trackers to log metrics to.
            If None, no tracking is performed. Defaults to None.
        phase_prefix: Optional prefix to prepend to phase names when logging.
            For example, "layer_0" produces phases "layer_0_train" and
            "layer_0_val". Defaults to None.

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
        >>> from dec_torch.dec import DEC
        >>> from dec_torch.loss import KLDivLoss
        >>> from dec_torch.trackers import HistoryTracker
        >>> from torch import optim
        >>>
        >>> # Setup
        >>> dec_model = DEC(encoder, centroids, alpha=1.0)
        >>> loss_fn = KLDivLoss()
        >>> optimizer = optim.SGD(dec_model.parameters(), lr=0.001, momentum=0.9)
        >>>
        >>> # Train with default tolerance (1%)
        >>> tracker = HistoryTracker()
        >>> train_dec_model(
        ...     dec_model, train_loader, optimizer, loss_fn,
        ...     derive_loss_target_fn=DEC.target_distribution,
        ...     verbose=True, trackers=[tracker]
        ... )
        >>>
        >>> # Train with stricter tolerance
        >>> tracker = HistoryTracker()
        >>> train_dec_model(
        ...     dec_model, train_loader, optimizer, loss_fn,
        ...     tolerance=0.001,
        ...     derive_loss_target_fn=DEC.target_distribution,
        ...     verbose=True, trackers=[tracker]
        ... )

    Note:
        Training automatically stops when the cluster assignments stabilize,
        indicating convergence. The final reassignment fraction is logged.
    """
    from dec_torch.trackers import Phase

    phases = [(Phase.TRAIN, train_loader, True)]
    if val_loader is not None:
        phases.append((Phase.VAL, val_loader, False))
    metrics = {"loss": loss_fn}

    experiment_trackers = trackers or []
    if experiment_trackers:
        for tracker in experiment_trackers:
            tracker.log_params(
                {
                    "max_epoch": max_epoch,
                    "tolerance": tolerance,
                    "has_validation": val_loader is not None,
                }
            )

    verbose_steps = _verbosity_steps(max_epoch, max_verbose) if verbose else ()
    previous_labels = None
    current_labels = None
    reassignment_fraction = 1.0

    for epoch_i in range(max_epoch):
        epoch_train_scores: dict[str, float] = {}
        for phase, loader, train_mode in phases:
            scores, labels = run_one_epoch(
                model,
                loader,
                metrics,
                optimizer,
                train=train_mode,
                device=device,
                derive_loss_target_fn=derive_loss_target_fn,
                return_label=(phase == Phase.TRAIN),
            )

            if labels is not None:
                current_labels = labels

            if experiment_trackers:
                log_phase = (
                    f"{phase_prefix}_{phase}"
                    if phase_prefix is not None
                    else str(phase)
                )
                for tracker in experiment_trackers:
                    tracker.log_metrics(
                        phase=log_phase, step=epoch_i + 1, metrics=scores
                    )

            if phase == Phase.TRAIN:
                epoch_train_scores = scores

        if previous_labels is not None:
            reassignments = sum(previous_labels != current_labels)
            reassignment_fraction = reassignments / len(previous_labels)
        previous_labels = current_labels

        if epoch_i in verbose_steps or reassignment_fraction < tolerance:
            msg = (
                f"[Epoch: {epoch_i + 1:4d}] | "
                f"Train. loss: {epoch_train_scores['loss']:.4f} | "
                f"Reassignment: {reassignment_fraction:7.2%} | "
            )
            if val_loader is not None:
                msg += f"Val. loss: {scores['loss']:.4f} |"
            logger.info(msg)

        if reassignment_fraction < tolerance:
            break


def _verbosity_steps(n_epoch: int, max_verbose: int) -> set[int]:
    """Determine which epochs should be reported during training.

    Args:
        n_epoch: total number of epochs in training.
        max_verbose: maximum (potentially one extra) number of logs to print.
    """
    verbose_steps = set(range(0, n_epoch, math.ceil(n_epoch / max_verbose)))
    verbose_steps.add(n_epoch - 1)

    return verbose_steps
