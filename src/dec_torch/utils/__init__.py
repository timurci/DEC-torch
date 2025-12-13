"""Utilities for DEC-torch.

This module provides helper functions for data handling and visualization
when working with DEC models.

Submodules:
    - data: Data loading and batch processing utilities
    - visualization: Visualization tools for training history and cluster analysis

Common Usage:
    >>> from dec_torch.utils.data import extract_all_data
    >>> from dec_torch.utils.visualization import loss_plot, cluster_plot
    >>>
    >>> # Extract all embeddings from a DataLoader
    >>> embeddings, _ = extract_all_data(data_loader, transform=encoder)
    >>>
    >>> # Plot training history
    >>> loss_plot(history)
    >>>
    >>> # Visualize clusters
    >>> cluster_plot(embeddings, labels=cluster_labels, centroids=centroids)
"""

from . import data, visualization

__all__ = [
    "data",
    "visualization",
]
