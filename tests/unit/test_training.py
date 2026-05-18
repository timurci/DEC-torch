"""Regression tests for training loops."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dec_torch.training import run_one_epoch


def test_run_one_epoch_raises_when_loss_metric_missing() -> None:
    """Missing loss metrics fail immediately with a raised assertion."""
    model = nn.Linear(2, 1)
    data_loader = DataLoader(
        TensorDataset(torch.ones(2, 2), torch.ones(2, 1)), batch_size=1
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(AssertionError, match="loss not found in metrics"):
        run_one_epoch(
            model=model,
            data_loader=data_loader,
            metrics={},
            optimizer=optimizer,
            train=True,
        )
