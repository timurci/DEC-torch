from typing import Optional, Literal
from collections.abc import Sequence

import pandas as pd
import numpy as np
from numpy import typing as npt

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns

from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def loss_plot(
        history: pd.DataFrame,
        ax: Optional[Axes] = None,
        **sns_kwargs
) -> Axes:
    """Plot training-validation loss history.

    Arguments:
    history: Output of `training.train_model()`.
    ax: If provided, the plot will be populated into this subplot instead.
    **sns_kwargs: Additional arguments for `searborn.lineplot()` function.
    """
    if ax is None:
        _, ax = plt.subplots()

    loss_history = history[history["metric"] == "loss"]
    assert isinstance(loss_history, pd.DataFrame)

    sns.lineplot(data=loss_history,
                 x="epoch",
                 y="score",
                 ax=ax,
                 hue="phase",
                 **sns_kwargs)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()

    return ax


def _create_2d_embedding_model(
    method: Literal["umap", "tsne", "pca"],
    **model_options
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
    model: UMAP | TSNE | PCA,
    data: npt.NDArray,
    centroids: Optional[npt.NDArray]
) -> tuple[npt.NDArray, Optional[npt.NDArray]]:
    centroids_2D = None

    # Ideally the model should only train on embeddings. However,
    # t-SNE does not allow to use transform after fitting the model.
    if isinstance(model, TSNE) and centroids is not None:
        combined = np.vstack([data, centroids])
        combined_2D = model.fit_transform(combined)
        assert isinstance(combined_2D, np.ndarray)
        embeddings_2D = combined_2D[:len(data)]
        centroids_2D = combined_2D[len(data):]
    else:
        embeddings_2D = model.fit_transform(data)
        if centroids is not None:
            centroids_2D = model.transform(centroids)  # pyright: ignore

    return embeddings_2D, centroids_2D


def cluster_plot(
        embeddings: np.ndarray,
        labels: Optional[Sequence | dict[str, Sequence]] = None,
        centroids: Optional[np.ndarray] = None,
        reduction: Optional[Literal["umap", "tsne", "pca"]] = "umap",
        reduction_options: dict = {},
        ax: Optional[Axes | npt.NDArray[np.object_]] = None,
        centroids_options: dict = {},
        **sns_kwargs,
) -> Axes | npt.NDArray[np.object_]:
    """Plot high-dimensional embeddings in a 2D scatterplot.

    Arguments:
    embeddings: Embedded representation of the data. (n_sample, n_dim)
    labels: Label set(s) for each sample in the data.
    centroids: Plot centroids on top of clusters. (n_centroid, n_dim).
    reduction: Specifies embedding technique to reduce input to 2D.
    reduction_options: Passed to embedding class constructor.
    ax: The plot will be populated into the provided subplot(s).
    centroids_options: Additional scatterplot arguments for centroids.
    **sns_kwargs: Additional arguments for `seaborn.scatterplot()`.
    """
    # Standardize `labels` type to dict[str, Optional[Sequence]]
    if isinstance(labels, dict):
        label_map = labels
    else:
        label_map: dict[str, Optional[Sequence]] = {"": labels}

    # Set up ax iterator
    if ax is None:
        _, ax = plt.subplots(ncols=len(label_map))
        assert(ax is not None)

    ax_itr = ax.flat if isinstance(ax, np.ndarray) else [ax]
    assert(len(ax_itr) >= len(label_map), "there are more labels than provided subplots")

    # Create 2D embeddings of the input matrices
    if reduction:
        reducer = _create_2d_embedding_model(reduction, **reduction_options)
        embeddings_2D, centroids_2D = _transform_2d_embeddings(
            reducer,
            embeddings,
            centroids
        )
    elif embeddings.shape[1] > 2 \
            or (centroids is not None and centroids.shape[1] > 2):
        raise AssertionError("cannot plot high-dimensional embeddings without 2D mapping")
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
            **sns_kwargs
        )

        # Plot centroids in subplot
        if centroids_2D is not None:
            assert isinstance(centroids_2D, np.ndarray)
            sns.scatterplot(
                x=centroids_2D[:, 0],
                y=centroids_2D[:, 1],
                label="Centroids",
                ax=axis,
                **centroids_options
            )

        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_title(f"{label_title} (Projection: {str(reduction).upper()})")

        if label_values is not None:
            axis.legend()

    return ax
