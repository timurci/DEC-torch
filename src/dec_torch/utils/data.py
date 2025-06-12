import torch
from torch import nn
from torch.utils.data import DataLoader

from typing import Optional, Union
from collections.abc import Callable, Sequence


def extract_batch_pairs(
        batch: Union[
            torch.Tensor,
            tuple[torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
            Sequence[torch.Tensor],
        ],
        device: Optional[str | torch.device] = None,
        transform: Optional[Callable] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract input-target pairs from a batch.

    Arguments:
    batch: Input tensor or a `Sequence` of (input, target) pair.
    device: Device to move all tensors to. Applied before transformation.
    transform: Apply transform (e.g., embedding model) to input.

    Returns:
    Input-label pair or input-input pair.

    Note:
    If the batch has only input, target will be set to input (self-supervised).
    """
    if isinstance(batch, Sequence) and len(batch) > 1:
        batch_input, batch_target = batch[0], batch[1]
    else:
        batch_input = batch[0] if isinstance(batch, Sequence) else batch
        batch_target = None

    if device is not None:
        batch_input = batch_input.to(device)
        if isinstance(batch_target, torch.Tensor):
            batch_target = batch_target.to(device)

    if transform is not None:
        if isinstance(transform, nn.Module):
            restore_training = transform.training
            transform.eval()
            with torch.no_grad():
                batch_input = transform(batch_input)
            if restore_training:
                transform.train()
        else:
            batch_input = transform(batch_input)

    if batch_target is None:
        batch_target = batch_input

    return batch_input, batch_target


def extract_all_data(
        data_loader: DataLoader,
        device: Optional[str | torch.device] = None,
        transform: Optional[Callable] = None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Load all data from `DataLoader`, optionally transform inputs.

    Arguments:
    data_loader: Expected to load input or input-label pairs in each batch.
    device: Device to move all tensors to. Applied before transformation.
    transform: Apply transform (e.g., embedding model) to input.

    Returns:
    Concatenated input-label pair. If no label, then returns `None` for label.
    """
    inputs_list = []
    targets_list = []

    for batch in data_loader:
        inputs, targets = extract_batch_pairs(batch, device, transform)

        inputs_list.append(inputs)
        if targets is not inputs:
            targets_list.append(targets)

    inputs = torch.cat(inputs_list, dim=0)
    targets = torch.cat(targets_list, dim=0) if len(targets_list) > 0 else None

    return inputs, targets
