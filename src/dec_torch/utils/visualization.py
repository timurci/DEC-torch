"""Visualization utilities for DEC models and training.

This module provides plotting functions for analyzing training history and
visualizing cluster assignments in the latent space. It supports various
dimensionality reduction techniques for visualizing high-dimensional embeddings.

Main Functions:
    loss_plot: Plot training/validation loss curves
    cluster_plot: Visualize clusters in 2D projections

Dimensionality Reduction Methods:
    UMAP (default): Uniform Manifold Approximation and Projection
    t-SNE: t-Distributed Stochastic Neighbor Embedding
    PCA: Principal Component Analysis

Example: Loss Plot:
    >>> from dec_torch.utils.visualization import loss_plot
    >>> import matplotlib.pyplot as plt
    >>>
    >>> # After training, plot history
    >>> loss_plot(history)
    >>> plt.show()

Example: Cluster Visualization:
    >>> from dec_torch.utils.visualization import cluster_plot
    >>> import matplotlib.pyplot as plt
    >>> from dec_torch.utils.data import extract_all_data
    >>>
    >>> # Extract embeddings and get cluster assignments
    >>> embeddings, _ = extract_all_data(loader, transform=encoder)
    >>> assignments = dec_model(embeddings).argmax(dim=1).numpy()
    >>> centroids = dec_model.centroids.detach().numpy()
    >>>
    >>> # Plot clusters
    >>> cluster_plot(embeddings.numpy(), labels=assignments, centroids=centroids)
    >>> plt.show()
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy import typing as npt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


def loss_plot(history: pd.DataFrame, ax: Axes | None = None, **sns_kwargs) -> Axes:
    """Plot training-validation loss history.

    This function creates a line plot of training and validation loss over epochs.
    It's designed to work directly with the history DataFrame returned by
    training functions.

    Args:
        history: Output of dec_torch.training.train_ae_model or
            dec_torch.training.train_dec_model.
        ax: If provided, the plot will be drawn on this axes. If None, creates
            a new figure.
        **sns_kwargs: Additional arguments passed to seaborn.lineplot.

    Returns:
        The axes object containing the plot.

    Example:
        >>> from dec_torch.utils.visualization import loss_plot
        >>> import matplotlib.pyplot as plt
        >>>
        >>> # After training
        >>> loss_plot(history)
        >>> plt.title("DEC Training Loss")
        >>> plt.show()
        >>>
        >>> # Customize appearance
        >>> loss_plot(history, palette="viridis", linewidth=2.5)
        >>> plt.show()
    """
    if ax is None:
        _, ax = plt.subplots()

    loss_history = history[history["metric"] == "loss"]
    assert isinstance(loss_history, pd.DataFrame)

    sns.lineplot(
        data=loss_history, x="epoch", y="score", ax=ax, hue="phase", **sns_kwargs
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()

    return ax


def _create_2d_embedding_model(
    method: Literal["umap", "tsne", "pca"], **model_options
) -> UMAP | TSNE | PCA:
    match method:
        case "umap":
            return UMAP(n_components=2, **model_options)
        case "tsne":
            return TSNE(n_components=2, **model_options)
        case "pca":
            return PCA(n_components=2, **model_options)
        case _:
            raise AssertionError("unsupported reduction method")


def _transform_2d_embeddings(
    model: UMAP | TSNE | PCA, data: npt.NDArray, centroids: npt.NDArray | None
) -> tuple[npt.NDArray, npt.NDArray | None]:
    centroids_2D = None

    # Ideally the model should only train on embeddings. However,
    # t-SNE does not allow to use transform after fitting the model.
    if isinstance(model, TSNE) and centroids is not None:
        combined = np.vstack([data, centroids])
        combined_2D = model.fit_transform(combined)
        assert isinstance(combined_2D, np.ndarray)
        embeddings_2D = combined_2D[: len(data)]
        centroids_2D = combined_2D[len(data) :]
    else:
        embeddings_2D = model.fit_transform(data)
        if centroids is not None:
            centroids_2D = model.transform(centroids)  # pyright: ignore

    return embeddings_2D, centroids_2D


def cluster_plot(
    embeddings: np.ndarray,
    labels: Sequence | dict[str, Sequence] | None = None,
    centroids: np.ndarray | None = None,
    reduction: Literal["umap", "tsne", "pca"] | None = "umap",
    reduction_options: dict = {},
    ax: Axes | npt.NDArray[np.object_] | None = None,
    centroids_options: dict = {},
    **sns_kwargs,
) -> Axes | npt.NDArray[np.object_]:
    """Plot high-dimensional embeddings in a 2D scatterplot.

    This function visualizes clusters by reducing high-dimensional embeddings
    to 2D using dimensionality reduction techniques (UMAP, t-SNE, or PCA).
    It can plot cluster assignments, ground truth labels, and centroids.

    Args:
        embeddings: High-dimensional embeddings of shape (n_samples, n_dimensions).
        labels: Cluster assignments or labels for coloring points. Can be:
            - A single sequence/array of labels
            - A dictionary mapping label set names to label sequences
        centroids: Cluster centroids to plot as stars, shape (n_clusters, n_dimensions).
        reduction: Dimensionality reduction method. Defaults to "umap".
        reduction_options: Additional arguments passed to the reduction class constructor.
        ax: Matplotlib axes to plot on. If None, creates new figure. For multiple
            label sets, provide array of axes.
        centroids_options: Additional arguments for centroid scatter plots.
        **sns_kwargs: Additional arguments passed to seaborn.scatterplot.

    Returns:
        The matplotlib axes containing the plot(s).

    Dimensionality Reduction Methods:
        umap: Uniform Manifold Approximation and Projection. Good for preserving
            global structure. Fast for large datasets.
        tsne: t-Distributed Stochastic Neighbor Embedding. Excellent for visualizing
            local structure and clusters. Computationally intensive.
        pca: Principal Component Analysis. Linear method, very fast, good for
            initial exploration.

    Notes:
        If embeddings are already 2-dimensional and no reduction is specified,
        they are plotted directly. For higher dimensions, a reduction method
        is required.

        When labels is a dictionary with multiple label sets, the function
        creates multiple subplots, one for each label set.

        Centroids are plotted as star markers on top of the scatter plot.

    Example:
        >>> from dec_torch.utils.visualization import cluster_plot
        >>> from dec_torch.utils.data import extract_all_data
        >>>
        >>> # Get embeddings and cluster assignments
        >>> embeddings, _ = extract_all_data(loader, transform=encoder)
        >>> assignments = dec_model(embeddings).argmax(dim=1).numpy()
        >>> centroids = dec_model.centroids.detach().numpy()
        >>>
        >>> # Basic cluster plot
        >>> cluster_plot(embeddings.numpy(), labels=assignments, centroids=centroids)
        >>> plt.show()
        >>>
        >>> # Compare with ground truth
        >>> cluster_plot(
        ...     embeddings.numpy(),
        ...     labels={'DEC': assignments, 'Ground Truth': true_labels},
        ...     centroids=centroids,
        ...     reduction='tsne',
        ...     reduction_options={'perplexity': 30}
        ... )
        >>> plt.show()
        >>>
        >>> # Custom styling
        >>> cluster_plot(
        ...     embeddings.numpy(),
        ...     labels=assignments,
        ...     centroids=centroids,
        ...     palette='Set2',
        ...     s=50,  # point size
        ...     centroids_options={'s': 200, 'marker': '*', 'color': 'red'}
        ... )
        >>> plt.show()
    """
    # Standardize `labels` type to dict[str, Optional[Sequence]]
    if isinstance(labels, dict):
        label_map = labels
    else:
        label_map: dict[str, Sequence | None] = {"": labels}

    # Set up ax iterator
    if ax is None:
        _, ax = plt.subplots(ncols=len(label_map))
        assert ax is not None

    ax_itr = ax.flat if isinstance(ax, np.ndarray) else [ax]
    assert (
        len(ax_itr) >= len(label_map),
        "there are more labels than provided subplots",
    )

    # Create 2D embeddings of the input matrices
    if reduction:
        reducer = _create_2d_embedding_model(reduction, **reduction_options)
        embeddings_2D, centroids_2D = _transform_2d_embeddings(
            reducer, embeddings, centroids
        )
    elif embeddings.shape[1] > 2 or (centroids is not None and centroids.shape[1] > 2):
        raise AssertionError(
            "cannot plot high-dimensional embeddings without 2D mapping"
        )
    else:
        embeddings_2D = embeddings
        centroids_2D = centroids
    assert isinstance(embeddings_2D, np.ndarray)

    # Plot 2D embeddings
    for axis, (label_title, label_values) in zip(ax_itr, label_map.items()):
        assert isinstance(axis, Axes)

        # Plot all data points in subplot
        sns.scatterplot(
            x=embeddings_2D[:, 0],
            y=embeddings_2D[:, 1],
            hue=label_values,
            ax=axis,
            **sns_kwargs,
        )

        # Plot centroids in subplot
        if centroids_2D is not None:
            assert isinstance(centroids_2D, np.ndarray)
            sns.scatterplot(
                x=centroids_2D[:, 0],
                y=centroids_2D[:, 1],
                label="Centroids",
                ax=axis,
                **centroids_options,
            )

        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_title(f"{label_title} (Projection: {str(reduction).upper()})")

        if label_values is not None:
            axis.legend()

    return ax
