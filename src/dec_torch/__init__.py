"""DEC-torch: Deep Embedded Clustering in PyTorch.

A modular PyTorch toolkit for Deep Embedded Clustering (DEC), an unsupervised
clustering method via deep representation learning.

Main Features:
    - Plug-and-play with any encoder: Use custom models or built-in autoencoders
    - Multiple centroid initializations: Compare k-means trials or use random
      centroid initialization
    - Seamless model I/O: Save/load utilities for both DEC and built-in autoencoders
    - Tracking & visualization: History objects record losses/metrics; integrated
      cluster visualization
    - Extensible design: All components are modular and can be swapped, extended,
      or customized

Package Structure:
    - dec_torch.autoencoder: Autoencoder implementations (Basic, Stacked)
    - dec_torch.dec: Core DEC model and clustering utilities
    - dec_torch.loss: Custom loss functions (KLDivLoss)
    - dec_torch.training: Training loops for autoencoders and DEC models
    - dec_torch.trackers: Experiment tracking protocols and implementations
    - dec_torch.utils: Data handling and visualization utilities

Examples:
    Create and train a DEC model:

        >>> from dec_torch import DEC, init_clusters
        >>> from dec_torch.autoencoder import AutoEncoder, AutoEncoderConfig
        >>> import torch
        >>>
        >>> # Create and train an autoencoder
        >>> config = AutoEncoderConfig.build(
        ...     input_dim=784,
        ...     latent_dim=128,
        ...     hidden_dims=[500, 500, 2000]
        ... )
        >>> encoder = AutoEncoder(config).encoder
        >>>
        >>> # Initialize cluster centroids
        >>> embeddings = encoder(training_data)
        >>> centroids = init_clusters(embeddings.detach().numpy(), n_clusters=10)
        >>>
    >>> # Create and train DEC model
    >>> dec_model = DEC(encoder=encoder, centroids=centroids)
    >>> dec_model.fit(train_loader, optimizer, loss_fn)

Note:
    In the original DEC study [1]_, "DEC" refers to the complete workflow combining
    representation learning *and* clustering. In this package, DEC refers only to the
    clustering model, under the assumption that suitable representation learning has
    already been performed in the encoder.

References:
    [1] Xie, J., Girshick, R., & Farhadi, A. (2016).
        Unsupervised deep embedding for clustering analysis.
        (arXiv:1511.06335)
"""

from . import autoencoder, dec, loss, training, trackers, utils

__all__ = [
    "autoencoder",
    "dec",
    "loss",
    "training",
    "trackers",
    "utils",
]
