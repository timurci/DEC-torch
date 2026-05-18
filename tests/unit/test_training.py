"""Regression tests for training loops."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dec_torch.dec.dec import DEC, KLDivLoss
from dec_torch.training import run_one_epoch, train_ae_model, train_dec_model


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


def test_train_ae_model_logs_without_validation_loader() -> None:
    """Verbose autoencoder training does not require validation records."""
    model = nn.Linear(2, 2)
    data_loader = DataLoader(TensorDataset(torch.ones(2, 2)), batch_size=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    history = train_ae_model(
        model=model,
        train_loader=data_loader,
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
        n_epoch=1,
        verbose=True,
    )

    assert history["phase"].tolist() == ["training"]  # noqa: S101


def test_train_dec_model_logs_without_validation_loader() -> None:
    """Verbose DEC training does not require validation records."""
    model = DEC(encoder=nn.Identity(), centroids=torch.tensor([[0.0], [1.0]]))
    data_loader = DataLoader(
        TensorDataset(torch.tensor([[0.0], [0.1], [1.0], [1.1]])), batch_size=2
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    history = train_dec_model(
        model=model,
        train_loader=data_loader,
        optimizer=optimizer,
        loss_fn=KLDivLoss(),
        derive_loss_target_fn=DEC.target_distribution,
        verbose=True,
        max_epoch=1,
    )

    assert history["phase"].tolist() == ["training"]  # noqa: S101
