import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from torch import nn
from torch.utils.data import DataLoader

from dec_torch.training import train_dec_model


class KLDivLoss(nn.Module):
    """Custom KL Divergence Loss module for DEC training.

    This loss function computes the KL divergence between the soft assignment
    distribution (q) and the target distribution (p), which is used to train the
    DEC model. The module uses mathematically accurate reduction by default as
    suggested in PyTorch Documentation.

    The module automatically applies log transformation to Q as required by
    nn.functional.kl_div and adds a small epsilon for numerical stability.

    Args:
        reduction: Reduction method for the loss. Defaults to "batchmean" for
            correct KL divergence calculation.
        eps: Small constant added to Q before log transformation for numerical
            stability. Defaults to 1e-10.

    Attributes:
        reduction: The reduction method being used.
        eps: The epsilon value for numerical stability.

    Example:
        >>> loss_fn = KLDivLoss(reduction="batchmean", eps=1e-10)
        >>> q = torch.tensor([[0.1, 0.7, 0.2], [0.3, 0.3, 0.4]])
        >>> p = torch.tensor([[0.05, 0.85, 0.10], [0.4, 0.3, 0.3]])
        >>> loss = loss_fn(q, p)
        >>> print(loss.item())
        0.123...

    Note:
        See PyTorch documentation for nn.functional.kl_div for details about
        the reduction methods and the mathematical formulation.
    """

    def __init__(self, reduction: str = "batchmean", eps: float = 1e-10) -> None:
        """Initialize the KL Divergence loss function.

        Args:
            reduction: Reduction method for the loss. Defaults to "batchmean".
            eps: Epsilon for numerical stability. Defaults to 1e-10.
        """
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence between Q and P distributions.

        Args:
            q (torch.Tensor): Soft assignment distribution (batch_size, n_clusters).
            p (torch.Tensor): Target distribution (batch_size, n_clusters).

        Returns:
            torch.Tensor: KL divergence loss value.

        Note:
            Q is automatically transformed to log space and epsilon is added for
            numerical stability before computing the divergence.
        """
        q = (q + self.eps).log()  # kl_div requires Q to be in log space.
        return nn.functional.kl_div(q, p, reduction=self.reduction)


def init_clusters_random(
    n_clusters: int,
    latent_dim: int,
    mean: float = 0.0,
    std: float = 1.0,
) -> torch.Tensor:
    """Initialize cluster centroids by sampling from a normal distribution.

    This method generates random centroids without considering the actual data
    distribution in the latent space. While simple, this approach is generally
    not recommended as it may lead to poor clustering performance.

    Args:
        n_clusters: Number of clusters to initialize.
        latent_dim: Dimensionality of the latent space.
        mean: Mean of the normal distribution. Defaults to 0.0.
        std: Standard deviation of the normal distribution. Defaults to 1.0.

    Returns:
        Cluster centroids of shape (n_clusters, latent_dim).

    Note:
        This is not a recommended method to initialize cluster centers, since it
        is unaware of the data distribution in latent space. Consider using
        init_clusters() or init_clusters_trials() instead.

    Example:
        >>> centroids = init_clusters_random(n_clusters=10, latent_dim=128)
        >>> print(centroids.shape)
        torch.Size([10, 128])
    """
    return torch.normal(mean, std, size=(n_clusters, latent_dim))


def init_clusters(
    embeddings: np.ndarray | torch.Tensor,
    n_clusters: int,
) -> torch.Tensor:
    """Initialize cluster centroids via k-means algorithm on the embeddings.

    This method uses scikit-learn's KMeans algorithm to find cluster centers
    in the latent space. This is the recommended approach for initializing
    DEC models as it accounts for the actual data distribution.

    Args:
        embeddings: Embedded representation of the whole dataset with shape
            (n_samples, latent_dim).
        n_clusters: Number of clusters to initialize.

    Returns:
        Cluster centroids of shape (n_clusters, latent_dim).

    See Also:
        `init_clusters_trials()`: Run multiple k-means for better initialization.
        `dec_torch.utils.data.extract_all_data()`: Utility to compute embeddings.

    Note:
        It is possible to use a subset of the dataset to initialize the cluster
        centers using `torch.utils.data.Subset` class in large datasets.

    Example:
        >>> embeddings = encoder(training_data).detach().cpu()
        >>> centroids = init_clusters(embeddings, n_clusters=10)
        >>> print(centroids.shape)
        torch.Size([10, 128])
    """
    embeddings_array, embeddings_device, embeddings_dtype = _as_kmeans_input(
        embeddings
    )
    kmeans = KMeans(n_clusters)
    kmeans.fit(embeddings_array)
    return torch.from_numpy(kmeans.cluster_centers_).to(
        device=embeddings_device, dtype=embeddings_dtype
    )


def init_clusters_trials(
    embeddings: np.ndarray | torch.Tensor, n_clusters: int, n_trials: int = 20
) -> tuple[list[torch.Tensor], pd.DataFrame]:
    """Run k-means initialization multiple times and return the best candidates.

    This method runs k-means clustering multiple times with different random
    initializations and evaluates each result using clustering quality metrics
    (Silhouette score and Calinski-Harabasz score). The centroids are ranked
    and returned along with their quality scores.

    Args:
        embeddings: Embedded representation of the dataset with shape
            (n_samples, latent_dim).
        n_clusters: Number of clusters to initialize.
        n_trials: Number of k-means runs with different initializations.
            Defaults to 20.

    Returns:
        A tuple containing:
            - List of centroid arrays (one for each trial)
            - DataFrame with quality metrics and rankings for each trial

    The DataFrame includes:
        - SIL: Silhouette score (higher is better)
        - CH: Calinski-Harabasz score (higher is better)
        - SIL-rank: Rank by Silhouette score
        - CH-rank: Rank by Calinski-Harabasz score
        - combined-rank: Sum of both ranks (lower is better)

    Example:
        >>> centroids_list, scores = init_clusters_trials(
        ... embeddings,
        ... n_clusters=10,
        ... n_trials=5)
        >>> print(scores.head())
               SIL       CH  SIL-rank  CH-rank  combined-rank
        run-id
        0     0.45    120.5       1.0      2.0            3.0
        1     0.42    135.2       2.0      1.0            3.0
        >>> # Select the best centroids
        >>> best_centroids = centroids_list[scores.iloc[0].name]

    Note:
        The combined-rank column can be used to select the best initialization.
        Lower combined-rank values indicate better overall clustering quality.
    """
    centroids_list = []
    trials = {"SIL": [], "CH": []}
    embeddings_array, embeddings_device, embeddings_dtype = _as_kmeans_input(
        embeddings
    )
    embeddings_tensor = torch.as_tensor(
        embeddings_array, device=embeddings_device, dtype=embeddings_dtype
    )

    for _ in range(n_trials):
        centroids = init_clusters(embeddings_tensor, n_clusters)
        soft_assignments = DEC.soft_assignment(
            embeddings_tensor, centroids, alpha=1
        )
        labels_pred = torch.argmax(soft_assignments, dim=1).detach().cpu().numpy()

        centroids_list.append(centroids)
        trials["SIL"].append(silhouette_score(embeddings_array, labels_pred))
        trials["CH"].append(calinski_harabasz_score(embeddings_array, labels_pred))

    trials = pd.DataFrame(trials)
    trials["run-id"] = list(range(len(centroids_list)))
    trials = trials.set_index("run-id")

    trials["SIL-rank"] = trials["SIL"].rank(method="min", ascending=False)
    trials["CH-rank"] = trials["CH"].rank(method="min", ascending=False)

    trials["combined-rank"] = trials["SIL-rank"] + trials["CH-rank"]
    trials = trials.sort_values(by="combined-rank", ascending=True)

    return centroids_list, trials


def _as_kmeans_input(
    embeddings: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, torch.device | None, torch.dtype | None]:
    """Return sklearn-compatible embeddings and original tensor metadata."""
    if isinstance(embeddings, torch.Tensor):
        return (
            embeddings.detach().cpu().numpy(),
            embeddings.device,
            embeddings.dtype,
        )
    return embeddings, None, None


class DEC(nn.Module):
    """Deep Embedded Clustering Module.

    This module implements the Deep Embedded Clustering (DEC) algorithm for
    unsupervised clustering in the latent space of a pre-trained encoder.

    DEC optimizes a clustering objective by iteratively:
    1. Computing soft assignments between data points and cluster centroids
    2. Refining cluster centroids based on high-confidence assignments
    3. Minimizing KL divergence between soft assignments and target distribution

    The Student's t-distribution (with alpha degrees of freedom) is used as
    the kernel to measure similarity between data points and centroids in the
    latent space.

    Args:
        encoder: A pre-trained encoder module (without deep copy). This should
            already be trained to produce meaningful latent representations.
        centroids: Initial cluster centroids of shape (n_clusters, latent_dim).
        alpha: Degrees of freedom of Student's t-distribution. Controls the
            heaviness of the tails. Defaults to 1.0.

    Attributes:
        encoder: The encoder network.
        centroids: Learnable cluster centroids.
        alpha: Degrees of freedom parameter.

    Note:
        The encoder is expected to produce latent representations suitable for
        clustering. Pre-training with an autoencoder is a typical approach.

        See init_clusters() and init_clusters_random() for centroid initialization
        methods.

    References:
        Xie, J., Girshick, R., & Farhadi, A. (2016).
        Unsupervised Deep Embedding for Clustering Analysis.
        (arXiv:1511.06335)

    Example:
        >>> from dec_torch.dec import DEC, init_clusters, KLDivLoss
        >>> import torch.optim as optim
        >>>
        >>> # Initialize centroids in latent space
        >>> embeddings = encoder(training_data)
        >>> centroids = init_clusters(embeddings.detach().cpu(), n_clusters=10)
        >>>
        >>> # Create DEC model
        >>> dec_model = DEC(encoder=encoder, centroids=centroids, alpha=1.0)
        >>>
        >>> # Training setup
        >>> loss_fn = dec.KLDivLoss()
        >>> optimizer = optim.SGD(dec_model.parameters(), lr=0.001)
        >>>
        >>> # Train the model
        >>> history = dec_model.fit(train_loader, optimizer, loss_fn, n_epoch=100)
    """

    def __init__(
        self,
        encoder: nn.Module,
        centroids: torch.Tensor,
        alpha: float = 1.0,
    ) -> None:
        """Initialize a DEC module.

        Args:
            encoder: A pre-trained encoder module (trained without deep copy).
            centroids: Initial cluster centroids.
            alpha: Degrees of freedom of Student's t-distribution.

        Note:
            See init_clusters() and init_clusters_random() for centroid
            initialization methods.
        """
        super().__init__()

        self.encoder = encoder
        self.centroids = nn.Parameter(centroids)
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through DEC - compute soft cluster assignments.

        Args:
            x: Input data of shape (batch_size, input_dim).

        Returns:
            Soft assignment probabilities of shape (batch_size, n_clusters).

        Example:
            >>> dec_model = DEC(encoder, centroids, alpha=1.0)
            >>> data = torch.randn(64, 784)
            >>> soft_assignments = dec_model(data)
            >>> print(soft_assignments.shape)
            torch.Size([64, 10])
            >>> # Get hard assignments (cluster predictions)
            >>> cluster_ids = torch.argmax(soft_assignments, dim=1)
            >>> print(cluster_ids.shape)
            torch.Size([64])
        """
        z = self.encoder(x)
        return self.soft_assignment(z, self.centroids, self.alpha)  # q value

    def fit(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.modules.loss._Loss,
        tolerance: float = 0.01,
        **kwargs,
    ) -> pd.DataFrame:
        """Train the DEC model to minimize clustering loss.

        This method trains the DEC model by optimizing the KL divergence between
        soft assignments and target distributions. Training automatically stops
        when the fraction of cluster reassignments falls below the tolerance threshold.

        Args:
            train_loader: DataLoader for training data. Should have shuffle=False to
                track cluster assignments.
            optimizer: Optimizer for model parameters (encoder and centroids).
            loss_fn: Loss function (typically KLDivLoss).
            tolerance: Cluster reassignment percentage threshold to stop training.
                Training stops when reassignments < tolerance. Defaults to 0.01 (1%).
            **kwargs: Additional arguments passed to train_dec_model().

        Returns:
            Training history with loss values.

        See Also:
            train_dec_model: Detailed parameter documentation.
            target_distribution: How target distributions are computed.

        Example:
            >>> from dec_torch.dec import KLDivLoss
            >>> from torch import optim
            >>>
            >>> loss_fn = KLDivLoss()
            >>> optimizer = optim.SGD(dec_model.parameters(), lr=0.001, momentum=0.9)
            >>>
            >>> # Train with default tolerance
            >>> history = dec_model.fit(train_loader, optimizer, loss_fn)
            >>>
            >>> # Train with custom tolerance
            >>> history = dec_model.fit(train_loader, optimizer, loss_fn)

        Note:
            The training DataLoader should be initialized with shuffle=False to
            properly track cluster reassignments across epochs.
        """
        device = next(self.parameters()).device
        if "device" not in kwargs:
            kwargs["device"] = device
        return train_dec_model(
            self,
            train_loader,
            optimizer,
            loss_fn,
            **kwargs,
            tolerance=tolerance,
            derive_loss_target_fn=self.target_distribution,
        )

    @staticmethod
    def soft_assignment(
        z: torch.Tensor, centroids: torch.Tensor, alpha: float
    ) -> torch.Tensor:
        """Compute soft assignment of samples to clusters.

        This method computes the probability that each data point belongs to each
        cluster based on the Student's t-distribution kernel.

        Args:
            z: Latent representation of shape (batch_size, latent_dim).
            centroids: Cluster centroids of shape (n_clusters, latent_dim).
            alpha: Degrees of freedom of Student's t-distribution.

        Returns:
            Soft assignment probabilities of shape (batch_size, n_clusters).
            Each row sums to 1.

        Example:
            >>> z = torch.randn(100, 128)  # 100 samples in 128-dim latent space
            >>> centroids = torch.randn(10, 128)  # 10 clusters
            >>> soft_assignments = DEC.soft_assignment(z, centroids, alpha=1.0)
            >>> print(soft_assignments.shape)
            torch.Size([100, 10])
            >>> print(torch.allclose(soft_assignments.sum(dim=1), torch.ones(100)))
            True
        """
        z = z.unsqueeze(1)
        centroids = centroids.unsqueeze(0)

        norm = torch.sum((z - centroids) ** 2, dim=-1)
        power = -(alpha + 1) / 2
        similarity = (1 + norm / alpha) ** power

        return similarity / similarity.sum(dim=1, keepdim=True)

    @staticmethod
    def target_distribution(
        q: torch.Tensor,
    ) -> torch.Tensor:
        """Compute target distribution from soft cluster assignments.

        This method computes the target distribution P used in the DEC objective.
        The target distribution emphasizes high-confidence assignments by squaring
        and normalizing the soft assignments. This helps to sharpen the cluster
        assignments during training.

        Args:
            q: Soft assignments of shape (batch_size, n_clusters).

        Returns:
            Target distribution P of shape (batch_size, n_clusters).

        The computation follows:
        p_ij = (q_ij^2 / Σ_i q_ij) / Σ_j'(q_ij'^2 / Σ_i q_ij')

        Example:
            >>> q = torch.tensor([[0.1, 0.7, 0.2], [0.3, 0.3, 0.4]])
            >>> p = DEC.target_distribution(q)
            >>> print(p.shape)
            torch.Size([2, 3])
            >>> print(torch.allclose(p.sum(dim=1), torch.ones(2)))
            True

        Note:
            This is a static method and can be called without instantiating the class.
            The target distribution P is used as the "ground truth" in the KL
            divergence loss during DEC training.
        """
        soft_cluster_freq = q.sum(dim=0, keepdim=True)
        q2_norm = (q**2) / soft_cluster_freq

        return q2_norm / q2_norm.sum(dim=1, keepdim=True)
