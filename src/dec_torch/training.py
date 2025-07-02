import torch
from torch import nn
from torch.utils.data import DataLoader

from dec_torch.utils.data import extract_batch_pairs

import pandas as pd
import math

from typing import Optional, NamedTuple
from collections.abc import Callable
from enum import Enum

import contextlib
import logging


logger = logging.getLogger(__name__)


class HistoryTracker:
    """Performance log designed for efficient storage and access to records."""
    def __init__(self, phases: list[str], metrics: list[str]):
        """Initialize HistoryTracker with predetermined phases and metrics.

        Arguments:
        phases: List of model phases, e.g 'training', 'validation'.
        metrics: Model performance metrics, e.g 'loss', 'accuracy'.
        """
        Phase = Enum("Phase", phases)
        Metric = Enum("Metric", metrics)
        RecordKey = NamedTuple("RecordKey", epoch=int, phase=Enum, metric=Enum)

        self.Phase = Phase
        self.Metric = Metric
        self.RecordKey = RecordKey

        self._history: dict[RecordKey, float] = {}

    def add_record(self, epoch: int, phase: str, metric: str, score: float):
        """Record a score in history log."""
        self._history[self.RecordKey(
            epoch=int(epoch),
            phase=self.Phase[phase],
            metric=self.Metric[metric]
        )] = float(score)

    def get_record(self, epoch: int, phase: str, metric: str) -> float:
        """Get a record from history log."""
        return self._history[self.RecordKey(
            epoch=int(epoch),
            phase=self.Phase[phase],
            metric=self.Metric[metric]
        )]

    @property
    def history(self) -> pd.DataFrame:
        """Access all history at once as a dataframe (inefficient)."""
        rows = [{"epoch": key.epoch,
                 "phase": key.phase.name,
                 "metric": key.metric.name,
                 "score": value} for key, value in self._history.items()]

        df = pd.DataFrame(rows)
        df["phase"] = df["phase"].astype("category")
        df["metric"] = df["metric"].astype("category")
        return df.sort_values(by=["epoch", "phase", "metric"])

    def __str__(self):
        return str(self.history)


def run_one_epoch(
        model: nn.Module,
        data_loader: DataLoader,
        metrics: dict[str, Callable],
        optimizer: torch.optim.Optimizer,  # Not used if train=False
        train: bool,
        device: Optional[str | torch.device] = None,
        transform: Optional[Callable] = None,
        derive_loss_target_fn: Optional[Callable] = None,
        return_label: bool = False
) -> tuple[dict[str, float], Optional[torch.Tensor]]:
    """Run one epoch loop of an autoencoder or DEC model.

    Arguments:
    metrics: Metrics with "fn(output, target)" functions. Must include 'loss'.
    train: Whether the model should be set to train, otherwise eval mode.
    transform: Apply transformation on the loaded batch input.
    derive_loss_target_fn: Derives loss function target from model output.
    return_labels: Returns all predicted labels.

    Returns:
    Tuple of `{metric: score}` dictionary and optional label tensor.

    Note:
    Return value of the functions in `metrics` should be cumulative (i.e must
    not return average score of the batch), except for the loss function.

    Note:
    `data_loader` is expected to return either a Tensor (i.e just the input) to
    train an autoencoder, or a tuple consisting of input and target of type
    `tuple[Tensor, Tensor]`.

    Note:
    `transform` is added to transform the input using sequential encoders
    during layer-wise training phase of a `StackedAutoEncoder`.

    Note:
    `derive_loss_target_fn` is added to compute the target distribution (p)
    from soft assignment (q) predictions of a DEC model.

    Note:
    `return_label` is added to keep track of cluster assignments in each epoch
    while training a DEC model. Therefore, data_loader should be initialized
    with `shuffle=False`.
    """
    assert "loss" in metrics

    if train:
        model.train()
        context = contextlib.nullcontext()
    else:
        model.eval()
        context = torch.no_grad()

    scores = {k: 0.0 for k in metrics}
    n_samples = 0
    label_list = [] if return_label else None

    for batch in data_loader:
        batch_input, batch_target = extract_batch_pairs(
            batch,
            device,
            transform
        )

        batch_output = None
        with context:
            batch_output = model(batch_input)

        if derive_loss_target_fn is None:
            loss = metrics["loss"](batch_output,
                                   batch_target)
        else:
            loss = metrics["loss"](batch_output,
                                   derive_loss_target_fn(batch_output))

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
        val_loader: Optional[DataLoader] = None,
        n_epoch: int = 100,
        transform: Optional[Callable] = None,
        device: Optional[str | torch.device] = None,
        verbose: bool = True,
        max_verbose: int = 20
) -> pd.DataFrame:
    """Train an autoencoder module with specified hyperparameters.

    Arguments:
    model: `AutoEncoder` or `StackedAutoEncoder` module.
    train_loader: DataLoader of training set.
    optimizer: Optimizer tied to `model.parameters()`.
    val_loader: DataLoader of validation set.
    device: Tensor computation device to load the data.
    n_epoch: Total number of epochs during training.
    verbose: Specifies if training performance reported via logging module.
    max_verbose: Maximum (or +1) number of lines to log.
    transform: Apply transformation on the loaded batch.
    loss_target_derive_fn: Derives loss function target from model output.

    Returns:
    Training and validation loss history.

    Note:
    `DataLoader` is expected to return either a Tensor (i.e just the input) to
    train an autoencoder, or a tuple consisting of input and target of type
    `tuple[Tensor, Tensor]`.

    Note:
    `transform` is added to transform the input using sequential encoders
    during layer-wise training phase of a `StackedAutoEncoder`.
    """
    phases = [("training", train_loader, True)]
    if val_loader is not None:
        phases.append(("validation", val_loader, False))
    metrics = {"loss": loss_fn}
    tracker = HistoryTracker(phases=[p[0] for p in phases],
                             metrics=[k for k in metrics])
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
                transform=transform
            )
            for metric, score in scores.items():
                tracker.add_record(epoch_i + 1, phase, metric, score)

        if epoch_i in verbose_steps:
            train_loss = tracker.get_record(epoch_i + 1, "training", "loss")
            val_loss = tracker.get_record(epoch_i + 1, "validation", "loss")
            msg = (f"[Epoch: {epoch_i + 1:4d}] | "
                   f"Train. loss: {train_loss:.4f} | "
                   f"Val. loss: {val_loss:.4f} |")
            logger.info(msg)

    return tracker.history


def train_dec_model(
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.modules.loss._Loss,
        val_loader: Optional[DataLoader] = None,
        tolerance: float = .01,
        derive_loss_target_fn: Optional[Callable] = None,
        device: Optional[str | torch.device] = None,
        verbose: bool = True,
        max_verbose: int = 10000,
        max_epoch: int = 10000
):
    """Train a DEC model with specified hyperparameters.

    Arguments:
    model: A `DEC` model.
    train_loader: DataLoader of training set (with `shuffle=False`).
    optimizer: Optimizer tied to `model.parameters()`.
    val_loader: DataLoader of validation set.
    device: Tensor computation device to load the data.
    n_epoch: Total number of epochs during training.
    verbose: Specifies if training performance reported via logging module.
    max_verbose: Maximum (or +1) number of lines to log.
    tolerance: Maximum cluster reassignment threshold to stop training.
    loss_target_derive_fn: Derives loss function target from model output.

    Returns:
    Training and validation loss history.

    Note:
    `DataLoader` is expected to return either a Tensor (i.e just the input) to
    train an autoencoder, or a tuple consisting of input and target of type
    `tuple[Tensor, Tensor]`.
    """
    phases = [("training", train_loader, True)]
    if val_loader is not None:
        phases.append(("validation", val_loader, False))
    metrics = {"loss": loss_fn}
    tracker = HistoryTracker(phases=[p[0] for p in phases],
                             metrics=["loss"])
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
                return_label=True if phase == "training" else False
            )

            if labels is not None:
                current_labels = labels

            for metric, score in scores.items():
                tracker.add_record(epoch_i + 1, phase, metric, score)

        if previous_labels is None:
            previous_labels = current_labels
        else:
            reassignments = sum(previous_labels != current_labels)
            reassignment_fraction = reassignments / len(previous_labels)

        if epoch_i in verbose_steps or reassignment_fraction < tolerance:
            train_loss = tracker.get_record(epoch_i + 1, "training", "loss")
            val_loss = tracker.get_record(epoch_i + 1, "validation", "loss")
            msg = (f"[Epoch: {epoch_i + 1:4d}] | "
                   f"Train. loss: {train_loss:.4f} | "
                   f"Reassignment: {reassignment_fraction:6.2%} | "
                   f"Val. loss: {val_loss:.4f} |")
            logger.info(msg)

        if reassignment_fraction < tolerance:
            break

    return tracker.history


def _verbosity_steps(n_epoch: int, max_verbose: int) -> set[int]:
    """Determine which epochs should be reported during training.

    Arguments:
    n_epoch: total number of epochs in training.
    max_verbose: maximum (potentially one extra) number of logs to print.
    """
    verbose_steps = set(range(0, n_epoch, math.ceil(n_epoch / max_verbose)))
    verbose_steps.add(n_epoch - 1)

    return verbose_steps
