"""Handle I/O of DEC models using an internal or generic encoder class."""

import torch
from torch import nn

from .dec import DEC
from dec_torch.autoencoder import Coder, CoderConfig


def save(
        dec_model: DEC,
        encoder_path: str,
        centroids_path: str,
        **kwargs
):
    """Save a DEC model with a `Coder` or sequential `Coder` encoder.

    Arguments:
    dec_model: DEC instance with `Coder` or sequential `Coder` encoder.
    encoder_path: Path to save encoder.
    centroids_path: Path to save centroids.
    **kwargs: Additional arguments for `torch.save()`.
    """
    encoder = dec_model.encoder
    # If encoder is nn.Sequential of Coders, save configs and state_dicts
    if isinstance(encoder, nn.Sequential):
        _save_sequential_encoder(encoder, encoder_path, **kwargs)
    elif isinstance(encoder, Coder):
        encoder.save(encoder_path, **kwargs)
    else:
        raise AssertionError("encoder is not a Coder or a sequential Coder")

    torch.save(dec_model.centroids.data.cpu(), centroids_path, **kwargs)


def load(
        encoder_path: str,
        centroids_path: str,
        alpha: float = 1.0,
        sequential_encoder: bool = False,
        **kwargs
) -> DEC:
    """Load a DEC model with a `Coder` or sequential `Coder` encoder.

    Arguments:
    encoder_path: Path to saved encoder.
    centroids_path: Path to saved centroids.
    **kwargs: Additional arguments for `torch.load()`.
    """
    if sequential_encoder:
        encoder_instance = _load_sequential_encoder(encoder_path, **kwargs)
    else:
        encoder_instance = Coder.load(encoder_path)
    centroids = torch.load(centroids_path, **kwargs)

    return DEC(encoder=encoder_instance, centroids=centroids, alpha=alpha)


def save_generic(dec_model: DEC,
                 encoder_path: str,
                 centroids_path: str,
                 **kwargs):
    """Save a DEC model using any encoder model.

    Arguments:
    dec_model: DEC instance with generic encoder.
    encoder_path: Path to save encoder.
    centroids_path: Path to save centroids.
    **kwargs: Additional arguments for `torch.save()`.
    """
    torch.save(dec_model.encoder.state_dict(), encoder_path, **kwargs)
    torch.save(dec_model.centroids.data.cpu(), centroids_path, **kwargs)


def load_generic(encoder_path: str,
                 centroids_path: str,
                 encoder_instance: nn.Module,
                 alpha=1.0,
                 **kwargs) -> DEC:
    """Load a DEC model using any encoder model.

    Arguments:
    encoder_path: Path to saved encoder.
    centroids_path: Path to saved centroids.
    encoder_instance: Initialized encoder to load weights into.
    **kwargs: Additional arguments for `torch.load()`.
    """
    state_dict = torch.load(encoder_path, **kwargs)
    encoder_instance.load_state_dict(state_dict)

    centroids = torch.load(centroids_path, **kwargs)

    return DEC(encoder=encoder_instance, centroids=centroids, alpha=alpha)


def _save_sequential_encoder(
    encoders: nn.Sequential,
    encoder_path: str,
    **kwargs
):
    """Save sequential `Coder` encoder to path."""
    configs = []
    state_dicts = []

    for encoder in encoders:
        assert isinstance(encoder, Coder)
        configs.append(encoder.config.to_dict())
        state_dicts.append(encoder.state_dict())

    torch.save({
        "type": "sequential",
        "configs": configs,
        "state_dicts": state_dicts
    }, encoder_path, **kwargs)


def _load_sequential_encoder(
    encoder_path: str,
    **kwargs
) -> nn.Sequential:
    """Load sequential `Coder` encoder from path."""
    enc = torch.load(encoder_path, **kwargs)
    modules = []

    for config, state_dict in zip(enc["configs"], enc["state_dicts"]):
        coder = Coder(CoderConfig.from_dict(config))
        coder.load_state_dict(state_dict)
        modules.append(coder)

    sequential_encoder = nn.Sequential(*modules)
    return sequential_encoder
