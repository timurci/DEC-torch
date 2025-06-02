import torch
from torch import nn
from torch.utils.data import DataLoader

import pandas as pd
import math

from typing import Optional, NamedTuple, Union
from collections.abc import Callable, Sequence
from enum import Enum

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


def _train_pass(
        net: nn.Module,
        input: torch.Tensor,
        target: Union[torch.Tensor, Callable],
        loss_fn: nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer
):
    net.train()
    optimizer.zero_grad()

    output = net(input)
    if isinstance(target, Callable):
        target = target(output)
    loss = loss_fn(output, target)

    loss.backward()
    optimizer.step()

    return loss.item()


def _test_pass(
        net: nn.Module,
        input: torch.Tensor,
        target: Union[torch.Tensor, Callable],
        loss_fn: nn.modules.loss._Loss,
        optimizer: Optional[torch.optim.Optimizer] = None  # not used
):
    net.eval()

    with torch.no_grad():
        output = net(input)
        if isinstance(target, Callable):
            target = target(output)
        loss = loss_fn(output, target)

        return loss.item()


def _extract_batch_pairs(
        batch: Union[
            torch.Tensor,
            tuple[torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
            Sequence,
        ],
        device: Optional[str | torch.device] = None,
        transform: Optional[Callable] = None,
        target_function: Optional[Callable] = None,
) -> tuple[torch.Tensor, torch.Tensor | Callable]:
    """Extract input-target pairs from a batch.

    Arguments:
    batch: Input tensor or a sequence of (input, target).
    device: Tensor computation device.
    transform: Transforms the input. Applied after loading input to the device.
    target_function: Function to compute target values from outputs.
    """
    if isinstance(batch, Sequence) and len(batch) > 1:
        batch_input, batch_target = batch[0], batch[1]
    else:
        batch_input = batch[0] if isinstance(batch, Sequence) else batch
        batch_target = None

    if device:
        batch_input = batch_input.to(device)
        if isinstance(batch_target, torch.Tensor):
            batch_target = batch_target.to(device)
    if transform:
        batch_input = transform(batch_input)

    if batch_target is None:
        if target_function:
            batch_target = target_function
        else:
            batch_target = batch_input

    return batch_input, batch_target


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.modules.loss._Loss,
        val_loader: Optional[DataLoader] = None,
        device: Optional[str | torch.device] = None,
        n_epoch: int = 100,
        verbose: bool = True,
        max_verbose: int = 20,
        transform: Optional[Callable] = None,
        target_function: Optional[Callable] = None
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
    target_function: Function to apply on output to compute target values.

    Returns:
    pandas.DataFrame: Training and validation loss history.

    Note:
    `DataLoader` is expected to return either a Tensor (just the input) to
    train an autoencoder, or a tuple consisting of input and target of type
    `tuple[Tensor, Tensor]`.

    Note:
    `transform` is added to transform the input using sequential encoders
    during layer-wise training phase of a `StackedAutoEncoder`.

    Note:
    `target_function` is added to compute target_distribution from the output
    of a `DEC` model.
    """
    phases = [("training", train_loader, _train_pass)]

    if val_loader is not None:
        phases.append(("validation", val_loader, _test_pass))

    tracker = HistoryTracker(phases=[p[0] for p in phases],
                             metrics=["loss"])

    verbose_steps = _verbosity_steps(n_epoch, max_verbose) if verbose else ()

    for epoch_i in range(n_epoch):
        for phase, loader, phase_pass in phases:
            sum_sq_error = 0.
            n_samples = 0
            for batch in loader:
                batch_input, batch_target = _extract_batch_pairs(
                    batch,
                    device,
                    transform,
                    target_function
                )
                batch_mse = phase_pass(model,
                                       batch_input,
                                       batch_target,
                                       loss_fn, optimizer)

                batch_size = batch_input.size(0)
                n_samples += batch_size
                sum_sq_error += batch_mse * batch_size

            overall_mse = sum_sq_error / n_samples
            tracker.add_record(epoch_i + 1, phase, "loss", overall_mse)

        if epoch_i in verbose_steps:
            train_loss = tracker.get_record(epoch_i + 1, "training", "loss")
            val_loss = tracker.get_record(epoch_i + 1, "validation", "loss")
            msg = (f"[Epoch: {epoch_i + 1:4d}] | "
                   f"Train. loss: {train_loss:.4f} | "
                   f"Val. loss: {val_loss:.4f} |")
            logger.info(msg)

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
