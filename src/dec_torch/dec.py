import torch
from torch import nn

from torch.utils.data import DataLoader

import pandas as pd
from sklearn.cluster import KMeans

from training import train_model


def init_clusters_random(
        n_clusters: int,
        latent_dim: int,
        mean: float = 0.0,
        std: float = 1.0,
) -> torch.Tensor:
    """Initialize cluster by sampling from normal distribution.

    Returns:
    torch.Tensor: Cluster centroids (n_cluster, latent_dim).

    Notes:
    This is not a recommended method to initialize the cluster centers,
    since it is unaware of the data distribution in latent space.
    """
    clusters = torch.normal(mean, std, size=(n_clusters, latent_dim))
    return clusters


def init_clusters(
        n_clusters: int,
        data_loader: DataLoader,
        encoder: nn.Module,
        device: str,
) -> torch.Tensor:
    """Initialize centroids via k-means algorithm using all data at once.

    Arguments:
    n_clusters: Number of clusters to initialize.
    data_loader: DataLoader where all data is extracted at once.
    encoder: Encoder module to get the latent representation of inputs.
    device: Encoder-compatible tensor computation device to load the data.

    Returns:
    torch.Tensor: Cluster centroids (n_cluster, latent_dim).

    Notes:
    It is possible to use a subset of the dataset to initialize
    the cluster centers using `torch.utils.data.Subset` class.
    """
    data = torch.Tensor([])
    for batch in data_loader:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]  # assuming the first element is input
        data = torch.cat((data, batch))
    data = data.to(device)

    encoder.eval()
    with torch.no_grad():
        z = encoder(data).detach().cpu().numpy()
        kmeans = KMeans(n_clusters)
        kmeans.fit(z)
        centroids = torch.Tensor(kmeans.cluster_centers_)
        return centroids


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

    def fit(
            self,
            train_loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            loss_fn: nn.modules.loss._Loss,
            **kwargs
    ) -> pd.DataFrame:
        """Train the DEC model to minimize clustering loss.

        Notes:
        See `.training.train_model()` for the details of the parameters.
        """
        device = next(self.parameters()).device
        history = train_model(self,
                              train_loader,
                              optimizer,
                              loss_fn,
                              **kwargs,
                              device=device)
        return history

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
