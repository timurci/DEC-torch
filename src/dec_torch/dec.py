import torch
from torch import nn

from torch.utils.data import DataLoader

from typing import Optional


def init_clusters_random(
        n_clusters: int,
        latent_dim: int,
        mean: float = 0.0,
        std: float = 1.0,
        device: Optional[str] = None,
):
    """Initialize cluster by sampling from normal distribution.

    This is not a recommended method to initialize the cluster centers,
    since it is unaware of the data distribution in latent space.
    """
    clusters = torch.normal(mean, std, size=(n_clusters, latent_dim))
    if device:
        clusters = clusters.to(device)
    return clusters


def init_clusters(
        n_clusters: int,
        data_loader: DataLoader,
        encoder: nn.Module,
        method: str = "k-means",
        device: Optional[str] = None,
):
    """Initialize clusters by clustering embeddings of all data at once.

    Arguments:
    n_clusters: Number of clusters to initialize.
    data_loader: DataLoader where all data is extracted at once.
    encoder: Encoder module to get the latent representation of inputs.
    method: Specifies the clustering method. Options: k-means, hierarchical.
    device: Specifies which computation divce to load the data on.

    Hint: It is possible to use a subset of the dataset to initialize
    the cluster centers using `torch.utils.data.Subset` class.
    """
    raise NotImplementedError()


class DEC(nn.Module):
    """Deep Embedded Clustering Module."""
    def __init__(
            self,
            encoder: nn.Module,
            centroids: torch.Tensor,
            alpha: float = 1.0,
    ):
        """Initialize a DEC module.

        Arguments:
        encoder: A pre-trained encoder module (trained without deep copy).
        centroids: Initial cluster centroids.
        alpha: Degrees of freedom of Student's t-distribution.

        See `init_clusters()` and `init_clusters_random()` utility functions
        to initialize centroids.
        """
        super().__init__()

        self.encoder = encoder
        self.centroids = nn.Parameter(centroids)
        self.alpha = alpha

    def forward(self, x):
        z = self.encoder(x)
        q = self.soft_assignment(z,
                                 self.centroids,
                                 self.alpha)
        return q

    @staticmethod
    def soft_assignment(
            z: torch.Tensor,
            centroids: torch.Tensor,
            alpha: float
    ) -> torch.Tensor:
        """Compute soft assignment a sample to each cluster.

        Arguments:
        z: Latent representation (batch_size, latent_dim) of the input.
        centroids: Centroids of clusters (n_cluster, latent_dim).
        alpha: Degrees of freedom of Student's t-distribution.

        Returns:
        torch.Tensor: Soft assignment distribution (batch_size, n_cluster).

        """
        z = z.unsqueeze(1)
        centroids = centroids.unsqueeze(0)

        norm = torch.sum((z - centroids) ** 2, dim=-1)
        power = - (alpha + 1) / 2
        similarity = (1 + norm / alpha) ** power

        return (similarity / similarity.sum(dim=1, keepdim=True))

    @staticmethod
    def target_distribution(
            q: torch.Tensor,
    ) -> torch.Tensor:
        """Compute hard assignment distribution from soft cluster assignments.

        Arguments:
        q: Soft assignment of a sample to each cluster (batch_size, n_cluster).

        Returns:
        torch.Tensor: Hard assignment distribution (batch_size, n_cluster).
        """
        soft_cluster_freq = q.sum(dim=0, keepdim=True)
        q2_norm = (q ** 2) / soft_cluster_freq

        return q2_norm / q2_norm.sum(dim=1, keepdim=True)
