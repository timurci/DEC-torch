"""Input/Output utilities for DEC models.

This module provides saving and loading functionality for DEC models (instances
of the DEC class) with different encoder types. The save functions store both
the encoder weights and cluster centroids, while the load functions reconstruct
the complete DEC model.

Supported Encoder Types:
    1. Coder encoders: Single or sequential Coder modules from the autoencoder
       module. Use save() and load().

    2. Generic encoders: Custom PyTorch models or third-party encoders.
       Use save_generic() and load_generic().

Main Functions:
    - save: Save a trained DEC model (encoder + centroids)
    - load: Load a saved DEC model (encoder + centroids)
    - save_generic: Save a DEC model with any encoder type
    - load_generic: Load a DEC model with any encoder type

Example: Saving and loading a DEC model with Coder encoder:
    >>> from dec_torch.dec import DEC, init_clusters
    >>> from dec_torch.autoencoder import Coder, CoderConfig
    >>> from dec_torch.dec.io import save, load
    >>>
    >>> # Create and train a DEC model
    >>> encoder = Coder(CoderConfig(input_dim=784, output_dim=128))
    >>> centroids = init_clusters(embeddings.numpy(), n_clusters=10)
    >>> dec_model = DEC(encoder=encoder, centroids=centroids)
    >>> # ... train the DEC model ...
    >>>
    >>> # Save the trained DEC model
    >>> save(dec_model, "encoder_weights.pth", "centroids.pth")
    >>>
    >>> # Load the DEC model later
    >>> loaded_dec_model = load("encoder_weights.pth", "centroids.pth")

Example: Saving and loading a DEC model with generic encoder:
    >>> from dec_torch.dec.io import save_generic, load_generic
    >>>
    >>> # Save the DEC model
    >>> save_generic(dec_model, "encoder_weights.pth", "centroids.pth")
    >>>
    >>> # Load the DEC model (must provide initialized encoder)
    >>> custom_encoder = MyCustomEncoder()  # Same architecture as saved model
    >>> loaded_dec_model = load_generic(
    ...     "encoder_weights.pth",
    ...     "centroids.pth",
    ...     custom_encoder
    ... )

Note:
    These functions specifically save and load complete DEC models (encoder
    weights + cluster centroids), not standalone encoders, decoders, or
    autoencoders. When loading models, additional keyword arguments are
    passed to torch.load() and torch.save(), such as map_location for
    device mapping.
"""

import torch
from torch import nn

from dec_torch.autoencoder import Coder, CoderConfig

from .dec import DEC


class NotBuiltinEncoderError(Exception):
    """The encoder is not derived from a supported builtin type."""

    def __init__(self) -> None:
        """Construct error message."""
        super().__init__(
            f"Encoder type is not derived from type {Coder.__name__}. "
            f"Either a direct {Coder.__name__} object should be provided "
            f"or it should be wrapped inside a {nn.Sequential.__name__}."
        )


def save(dec_model: DEC, encoder_path: str, centroids_path: str, **kwargs) -> None:
    """Save a DEC model with a Coder or sequential Coder encoder.

    This function saves a DEC model where the encoder is either a single
    Coder or a nn.Sequential of Coder modules (e.g., from
    StackedAutoEncoder). The encoder configuration and weights
    are saved along with the cluster centroids.

    Args:
        dec_model: DEC instance with Coder or sequential Coder encoder.
        encoder_path: Path to save encoder weights and configuration.
        centroids_path: Path to save cluster centroids.
        **kwargs: Additional arguments passed to torch.save.

    Raises:
        AssertionError: If encoder is not a Coder or sequential Coder.

    See Also:
        load: Load a DEC model saved with this function.
        save_generic: Save DEC model with any encoder type.

    Example:
        >>> from dec_torch.dec import DEC, init_clusters
        >>> from dec_torch.autoencoder import Coder, CoderConfig
        >>>
        >>> encoder = Coder(CoderConfig(input_dim=784, output_dim=128))
        >>> centroids = init_clusters(embeddings.numpy(), n_clusters=10)
        >>> dec_model = DEC(encoder=encoder, centroids=centroids)
        >>>
        >>> # Save for later use
        >>> save(dec_model, "encoder.pth", "centroids.pth")
        >>> # Save with device mapping
        >>> save(dec_model, "encoder.pth", "centroids.pth", map_location="cpu")
    """
    encoder = dec_model.encoder
    # If encoder is nn.Sequential of Coders, save configs and state_dicts
    if isinstance(encoder, nn.Sequential):
        _save_sequential_encoder(encoder, encoder_path, **kwargs)
    elif isinstance(encoder, Coder):
        encoder.save(encoder_path, **kwargs)
    else:
        raise NotBuiltinEncoderError

    torch.save(dec_model.centroids.data.cpu(), centroids_path, **kwargs)


def load(
    encoder_path: str,
    centroids_path: str,
    alpha: float = 1.0,
    sequential_encoder: bool = False,
    **kwargs,
) -> DEC:
    """Load a DEC model with a Coder or sequential Coder encoder.

    Loads a DEC model that was saved with `dec.io.save()`. Reconstructs the
    encoder architecture from saved configuration and loads the weights
    and centroids.

    Args:
        encoder_path: Path to saved encoder.
        centroids_path: Path to saved centroids.
        alpha: Degrees of freedom of Student's t-distribution.
            Should match the value used during training. Defaults to 1.0.
        sequential_encoder: True if the encoder was saved as a sequential
            stack of Coder modules (i.e., from StackedAutoEncoder).
        **kwargs: Additional arguments passed to torch.load.

    Returns:
        Loaded DEC model with trained encoder and centroids.

    See Also:
        save: Save a DEC model with Coder encoder.
        load_generic: Load DEC model with custom encoder.

    Example:
        >>> # Load standard encoder
        >>> dec_model = load("encoder.pth", "centroids.pth")
        >>>
        >>> # Load sequential encoder (from StackedAutoEncoder)
        >>> dec_model = load(
        ...     "encoder.pth", "centroids.pth",
        ...     sequential_encoder=True
        ... )
        >>>
        >>> # Load with device mapping
        >>> dec_model = load(
        ...     "encoder.pth", "centroids.pth",
        ...     map_location="cpu"
        ... )
    """
    if sequential_encoder:
        encoder_instance = _load_sequential_encoder(encoder_path, **kwargs)
    else:
        encoder_instance = Coder.load(encoder_path)
    centroids = torch.load(centroids_path, **kwargs)

    return DEC(encoder=encoder_instance, centroids=centroids, alpha=alpha)


def save_generic(
    dec_model: DEC, encoder_path: str, centroids_path: str, **kwargs
) -> None:
    """Save a trained DEC model using any encoder architecture.

    This function saves a trained DEC model with a custom or third-party encoder
    that doesn't use the built-in Coder class. Only the encoder's state_dict
    and the cluster centroids are saved (configuration is not saved).

    Args:
        dec_model: Trained DEC instance with any PyTorch encoder.
        encoder_path: Path to save encoder state dictionary.
        centroids_path: Path to save cluster centroids.
        **kwargs: Additional arguments passed to torch.save().

    Note:
        Unlike save(), this function does not save encoder configuration.
        When loading, you must provide an initialized encoder instance with the
        same architecture.

    See Also:
        load_generic: Load DEC model saved with this function.
        save: Save DEC model with Coder-based encoder.

    Example:
        >>> from dec_torch.dec import DEC
        >>> import torchvision.models as models
        >>>
        >>> # Use a pre-trained ResNet as encoder
        >>> resnet = models.resnet18(pretrained=True)
        >>> centroids = torch.randn(10, 512)  # 10 clusters, 512-dim features
        >>> dec_model = DEC(encoder=resnet, centroids=centroids)
        >>> # ... training code ...
        >>>
        >>> # Save DEC model for later
        >>> save_generic(dec_model, "resnet_encoder.pth", "centroids.pth")
    """
    torch.save(dec_model.encoder.state_dict(), encoder_path, **kwargs)
    torch.save(dec_model.centroids.data.cpu(), centroids_path, **kwargs)


def load_generic(
    encoder_path: str,
    centroids_path: str,
    encoder_instance: nn.Module,
    alpha: float = 1.0,
    **kwargs,
) -> DEC:
    """Load a saved DEC model using any encoder.

    Loads a DEC model that was saved with save_generic(). Requires an
    initialized encoder instance with the same architecture as the saved model.

    Args:
        encoder_path: Path to saved encoder state dictionary.
        centroids_path: Path to saved cluster centroids.
        encoder_instance: Initialized encoder to load weights into. Must have the
            same architecture as the saved encoder.
        alpha: Degrees of freedom of Student's t-distribution. Defaults to 1.0.
        **kwargs: Additional arguments passed to torch.load().

    Returns:
        Loaded DEC model with weights restored to the encoder instance and
        centroids loaded from file.

    See Also:
        save_generic: Save DEC model with custom encoder.
        load: Load DEC model with Coder-based encoder.

    Example:
        >>> from dec_torch.dec import DEC
        >>> import torchvision.models as models
        >>>
        >>> # Initialize encoder with same architecture
        >>> resnet = models.resnet18()
        >>>
        >>> # Load saved weights and centroids
        >>> dec_model = load_generic(
        ...     "resnet_encoder.pth",
        ...     "centroids.pth",
        ...     resnet,
        ...     alpha=1.0
        ... )
        >>>
        >>> # Load with device mapping
        >>> dec_model = load_generic(
        ...     "resnet_encoder.pth",
        ...     "centroids.pth",
        ...     resnet,
        ...     map_location="cpu"
        ... )
    """
    state_dict = torch.load(encoder_path, **kwargs)
    encoder_instance.load_state_dict(state_dict)

    centroids = torch.load(centroids_path, **kwargs)

    return DEC(encoder=encoder_instance, centroids=centroids, alpha=alpha)


def _save_sequential_encoder(
    encoders: nn.Sequential, encoder_path: str, **kwargs
) -> None:
    """Save sequential Coder encoder to path."""
    configs = []
    state_dicts = []

    for encoder in encoders:
        if not isinstance(encoder, Coder):
            raise NotBuiltinEncoderError
        configs.append(encoder.config.to_dict())
        state_dicts.append(encoder.state_dict())

    torch.save(
        {"type": "sequential", "configs": configs, "state_dicts": state_dicts},
        encoder_path,
        **kwargs,
    )


def _load_sequential_encoder(encoder_path: str, **kwargs) -> nn.Sequential:
    """Load sequential Coder encoder from path."""
    enc = torch.load(encoder_path, **kwargs)
    modules = []

    for config, state_dict in zip(enc["configs"], enc["state_dicts"], strict=True):
        coder = Coder(CoderConfig.from_dict(config))
        coder.load_state_dict(state_dict)
        modules.append(coder)

    return nn.Sequential(*modules)
