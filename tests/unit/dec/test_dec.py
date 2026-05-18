"""Regression tests for DEC clustering helpers."""

import torch

from dec_torch.dec.dec import init_clusters_trials


def test_init_clusters_trials_accepts_tensor_embeddings() -> None:
    """Tensor embeddings remain valid for k-means trial initialization."""
    embeddings = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
        ]
    )

    centroids_list, scores = init_clusters_trials(
        embeddings, n_clusters=2, n_trials=1
    )

    assert len(centroids_list) == 1  # noqa: S101
    assert centroids_list[0].shape == (2, 2)  # noqa: S101
    assert list(scores.columns) == [  # noqa: S101
        "SIL",
        "CH",
        "SIL-rank",
        "CH-rank",
        "combined-rank",
    ]


def test_init_clusters_trials_preserves_tensor_dtype() -> None:
    """Tensor embeddings are converted for sklearn and restored for DEC use."""
    embeddings = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
        ],
        dtype=torch.float64,
    )

    centroids_list, _ = init_clusters_trials(embeddings, n_clusters=2, n_trials=1)

    assert centroids_list[0].dtype == torch.float64  # noqa: S101
