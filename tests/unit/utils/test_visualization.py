"""Regression tests for visualization utilities."""

import matplotlib
import numpy as np
import pytest
from matplotlib import pyplot as plt

from dec_torch.utils.visualization import cluster_plot

matplotlib.use("Agg")


def test_cluster_plot_raises_when_labels_exceed_axes() -> None:
    """Multiple label sets require enough provided subplots."""
    embeddings = np.array([[0.0, 0.0], [1.0, 1.0]])
    labels = {"first": [0, 1], "second": [1, 0]}
    _, ax = plt.subplots()

    with pytest.raises(ValueError, match="there are more labels"):
        cluster_plot(embeddings, labels=labels, reduction=None, ax=ax)
