"""Core Deep Embedded Clustering (DEC) module.

This module provides the main DEC model implementation along with
cluster initialization utilities and custom loss functions.

Main Components:
    - DEC: Deep Embedded Clustering model
    - KLDivLoss: Custom KL divergence loss for DEC training
    - init_clusters: Initialize centroids using k-means
    - init_clusters_random: Random centroid initialization
    - init_clusters_trials: Multiple k-means trials for better initialization

Example:
    >>> from dec_torch.dec import DEC, init_clusters, KLDivLoss
    >>> import torch
    >>>
    >>> # Initialize centroids with k-means
    >>> centroids = init_clusters(embeddings.numpy(), n_clusters=10)
    >>>
    >>> # Create DEC model
    >>> dec_model = DEC(encoder=encoder, centroids=centroids, alpha=1.0)
    >>>
    >>> # Define loss and optimizer
    >>> loss_fn = KLDivLoss()
    >>> optimizer = torch.optim.SGD(dec_model.parameters(), lr=0.001)
    >>>
    >>> # Train the model
    >>> history = dec_model.fit(train_loader, optimizer, loss_fn)
"""

from . import io
from .dec import (
    DEC,
    KLDivLoss,
    init_clusters,
    init_clusters_random,
    init_clusters_trials,
)

__all__ = [
    "DEC",
    "KLDivLoss",
    "init_clusters",
    "init_clusters_random",
    "init_clusters_trials",
    "io",
]
