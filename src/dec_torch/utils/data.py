from collections.abc import Callable, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader


def extract_batch_pairs(
    batch: torch.Tensor
    | tuple[torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor]
    | Sequence[torch.Tensor],
    device: str | torch.device | None = None,
    transform: Callable | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract input-target pairs from a batch.

    This utility function handles various batch formats that DataLoaders might
    return and prepares them for model training. It supports:
    - Single tensor batches (for autoencoder self-supervised learning)
    - Tuple of (input, target) pairs
    - Multi-element sequences

    Optionally, it can move tensors to a specified device and apply
    transformations (e.g., embedding models) to the input.

    Args:
        batch: Batch data from a DataLoader. Can be:
            - A single tensor (treated as both input and target)
            - A tuple of (input, target)
            - A sequence with at least one tensor
        device: Device to move all tensors to. Applied before transformation.
        transform: Optional transform to apply to input, such as an embedding model.

    Returns:
        A tuple of (inputs, targets). If the batch only contains inputs, returns
        (inputs, inputs) for self-supervised learning scenarios.

    Notes:
        The batch is expected to come from a PyTorch DataLoader and may have
        different formats depending on the dataset.

        If the batch has only input (common for autoencoders), the target is
        set to the input for self-supervised reconstruction.

        When a transform is provided and is a nn.Module, it is temporarily
        set to eval mode (WARN: side effect) and runs without gradient tracking.

    Example:
        >>> # Single tensor batch (autoencoder)
        >>> batch = torch.randn(32, 784)
        >>> inputs, targets = extract_batch_pairs(batch)
        >>> print(inputs is targets)  # True for self-supervised
        True
        >>>
        >>> # Tuple batch (input, target)
        >>> batch = (torch.randn(32, 784), torch.randn(32, 10))
        >>> inputs, targets = extract_batch_pairs(batch)
        >>> print(inputs.shape, targets.shape)
        torch.Size([32, 784]) torch.Size([32, 10])
        >>>
        >>> # With transform
        >>> encoder = Coder(config)
        >>> inputs, targets = extract_batch_pairs(batch, transform=encoder)
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
    device: str | torch.device | None = None,
    transform: Callable | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Load all data from a DataLoader with optional transformations.

    This function iterates through an entire DataLoader and concatenates all
    batches into single tensors. It's useful for computing embeddings on the
    full dataset or for visualization purposes.

    Args:
        data_loader: DataLoader to extract all data from.
        device: Device to move tensors to. Applied before transformation.
        transform: Optional transform to apply to inputs, such as an encoder model.

    Returns:
        A tuple containing:
            - Concatenated inputs tensor
            - Concatenated targets tensor, or None if no targets are present

    Notes:
        This function loads the entire dataset into memory. Use with caution
        for very large datasets.

        If the DataLoader returns only inputs (no targets), the second return
        value will be None.

        When a transform is provided, it is applied to each batch's inputs
        before concatenation.

    Example:
        >>> from torch.utils.data import DataLoader, TensorDataset
        >>> from dec_torch.autoencoder import Coder, CoderConfig
        >>>
        >>> # Create dataset and loader
        >>> data = torch.randn(1000, 784)
        >>> dataset = TensorDataset(data)
        >>> loader = DataLoader(dataset, batch_size=100)
        >>>
        >>> # Extract all raw data
        >>> all_data, _ = extract_all_data(loader)
        >>> print(all_data.shape)
        torch.Size([1000, 784])
        >>>
        >>> # Extract embeddings
        >>> config = CoderConfig(input_dim=784, output_dim=128)
        >>> encoder = Coder(config)
        >>> embeddings, _ = extract_all_data(loader, transform=encoder)
        >>> print(embeddings.shape)
        torch.Size([1000, 128])
        >>>
        >>> # Use embeddings for k-means initialization
        >>> from dec_torch.dec import init_clusters
        >>> centroids = init_clusters(embeddings.detach().numpy(), n_clusters=10)
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
