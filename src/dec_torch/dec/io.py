"""Handle I/O of DEC models using an internal or generic encoder class."""

import torch
from torch import nn

from .dec import DEC
from dec_torch.autoencoder import Coder


def save(
        dec_model: DEC,
        encoder_path: str,
        centroids_path: str,
        **kwargs
):
    """Save a DEC model with a `Coder` class encoder.

    Arguments:
    dec_model: DEC instance with `Code` class encoder.
    encoder_path: Path to save encoder.
    centroids_path: Path to save centroids.
    **kwargs: Additional arguments for `torch.save()`.
    """
    encoder = dec_model.encoder
    assert isinstance(encoder, Coder), "the encoder should be a Coder type"
    encoder.save(encoder_path)

    torch.save(dec_model.centroids.data.cpu(), centroids_path, **kwargs)


def load(
        encoder_path: str,
        centroids_path: str,
        alpha: float = 1.0,
        **kwargs
) -> DEC:
    """Load a DEC model with a `Coder` class encoder.

    Arguments:
    encoder_path: Path to saved encoder.
    centroids_path: Path to saved centroids.
    **kwargs: Additional arguments for `torch.load()`.
    """
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
