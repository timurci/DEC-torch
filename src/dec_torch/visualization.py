from typing import Optional, Literal
from collections.abc import Sequence

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns

import umap
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


def cluster_plot(
        embeddings: np.ndarray,
        labels: Optional[Sequence] = None,
        centroids: Optional[np.ndarray] = None,
        reduction: Optional[Literal["umap", "tsne", "pca"]] = "umap",
        reduction_options: dict = {},
        ax: Optional[Axes] = None,
        centroids_options: dict = {},
        **sns_kwargs,
) -> Axes:
    """Plot high-dimensional embeddings in a 2D scatterplot.

    Arguments:
    embeddings: Embedded representation of the data. (n_sample, n_dim)
    labels: Label of each sample in the data.
    centroids: Plot centroids on top of clusters. (n_centroid, n_dim).
    reduction: Specifies embedding technique to reduce input to 2D.
    reduction_options: Passed to embedding class constructor.
    ax: If provided, the plot will be populated into this subplot instead.
    centroids_options: Additional scatterplot arguments for centroids.
    **sns_kwargs: Additional arguments for `seaborn.scatterplot()`.
    """
    reducer = None

    if reduction:
        assert reduction in ("umap", "tsne", "pca"), "unsupported reduction"

        if reduction == "umap":
            reducer = umap.UMAP(n_components=2, **reduction_options)
        elif reduction == "tsne":
            reducer = TSNE(n_components=2, **reduction_options)
        elif reduction == "pca":
            reducer = PCA(n_components=2, **reduction_options)
    elif embeddings.shape[1] != 2:
        raise AssertionError("cannot plot non-2D embeddings without reduction")

    embeddings_2D = embeddings
    centroids_2D = centroids

    if reducer:
        # Ideally the reducer should first fit data, then transform centroids.
        # However, t-SNE does not allow to use transform, hence the if/else.
        if reduction == "tsne" and centroids is not None:
            combined = np.vstack([embeddings, centroids])
            combined_2D = reducer.fit_transform(combined)
            assert isinstance(combined_2D, np.ndarray)
            embeddings_2D = combined_2D[:len(embeddings)]
            centroids_2D = combined_2D[len(embeddings):]
        else:
            embeddings_2D = reducer.fit_transform(embeddings)
            if centroids is not None:
                centroids_2D = reducer.transform(centroids)  # pyright: ignore

    assert isinstance(embeddings_2D, np.ndarray)
    if ax is None:
        _, ax = plt.subplots()

    sns.scatterplot(
        x=embeddings_2D[:, 0],
        y=embeddings_2D[:, 1],
        hue=labels,
        ax=ax,
        **sns_kwargs
    )

    if centroids_2D is not None:
        assert isinstance(centroids_2D, np.ndarray)
        sns.scatterplot(
            x=centroids_2D[:, 0],
            y=centroids_2D[:, 1],
            label="Centroids",
            ax=ax,
            **centroids_options
        )

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend()

    return ax
