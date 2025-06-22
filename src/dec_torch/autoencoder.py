import torch
from torch import nn
from torch.utils.data import DataLoader

import pandas as pd

from abc import ABC, abstractmethod

from dataclasses import dataclass, asdict, replace
from typing import Optional

from dec_torch.training import train_model

import logging


logger = logging.getLogger(__name__)


_ACTIVATION_REGISTRY: dict[str, type[nn.Module] | None] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "linear": None
}


def get_activation_module(name: str) -> Optional[type[nn.Module]]:
    """Get an activation function module by name.

    Returns:
    Optional[nn.Module]: Instance of module or None if 'linear' is specified.

    Raises:
    KeyError: If the name is not found in the registry.
    """
    activation = _ACTIVATION_REGISTRY[name.lower()]
    if activation is None:
        return None
    return activation


def register_activation_module(name: str, object: type[nn.Module]):
    """Register a name for a custom activation function module."""
    _ACTIVATION_REGISTRY[name.lower()] = object


def list_activation_modules() -> set[str]:
    """List registered names of activation modules"""
    return set(_ACTIVATION_REGISTRY.keys())


@dataclass(frozen=True)
class CoderConfig:
    """Configuration for Coder (encoder or decoder) module

    Attributes:
    input_dim: Input size of the model.
    output_dim: Output size of the model.
    hidden_dims: Number of units for each hidden layer.
    input_dropout: Dropout probability of an element in the input.
    hidden_activation: Name of the module to be used in hidden layers.
    output_activation: Name of the module to be used in output layer.

    Registered activation functions are found via `list_activation_modules()`.
    Custom modules can be registered by calling `register_activation_module()`.

    Note: In PyTorch, Dropout only sets outputs to zero.
    """
    input_dim: int
    output_dim: int
    hidden_dims: Optional[list[int]] = None
    input_dropout: Optional[float] = None
    hidden_activation: str = "relu"
    output_activation: str = "relu"

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(config_dict: dict) -> "CoderConfig":
        return CoderConfig(**config_dict)


@dataclass(frozen=True)
class AutoEncoderConfig:
    """Configuration for AutoEncoder module"""
    encoder: CoderConfig
    decoder: CoderConfig

    def to_dict(self):
        return {
            "encoder": self.encoder.to_dict(),
            "decoder": self.decoder.to_dict()
        }

    @staticmethod
    def from_dict(config_dict: dict) -> "AutoEncoderConfig":
        return AutoEncoderConfig(
            encoder=CoderConfig.from_dict(config_dict["encoder"]),
            decoder=CoderConfig.from_dict(config_dict["decoder"]),
        )

    @staticmethod
    def build(
            input_dim: int,
            latent_dim: int,
            hidden_dims: Optional[list[int]] = None,
            input_dropout: Optional[float] = None,
            hidden_activation: str = "relu",
            encoder_output_activation: str = "relu",
            decoder_output_activation: str = "relu",
    ) -> "AutoEncoderConfig":
        """Initialize AutoEncoderConfig with a symmetrical configuration

        Arguments:
        input_dropout: Dropout probability of both encoder and decoder inputs.

        See `CoderConfig` for detailed explanation of each parameter.
        """
        shared_kwargs = {
            "hidden_dims": hidden_dims,
            "input_dropout": input_dropout,
            "hidden_activation": hidden_activation
        }
        encoder_config = CoderConfig(
            input_dim=input_dim,
            output_dim=latent_dim,
            output_activation=encoder_output_activation,
            **shared_kwargs
        )
        decoder_config = CoderConfig(
            input_dim=latent_dim,
            output_dim=input_dim,
            output_activation=decoder_output_activation,
            **shared_kwargs
        )
        return AutoEncoderConfig(
            encoder=encoder_config,
            decoder=decoder_config
        )


@dataclass(frozen=True)
class StackedAutoEncoderConfig:
    """Configuration for StackedAutoEncoder module"""
    autoencoders: list[AutoEncoderConfig]

    def to_dict(self):
        return {
            "autoencoders": [ae.to_dict() for ae in self.autoencoders]
        }

    @staticmethod
    def from_dict(config_dict: dict) -> "StackedAutoEncoderConfig":
        configs = [
            AutoEncoderConfig.from_dict(ae) for ae
            in config_dict["autoencoders"]
        ]
        return StackedAutoEncoderConfig(
            autoencoders=configs
        )

    @staticmethod
    def build(
            input_dim: int,
            latent_dims: list[int],
            hidden_dims: Optional[list[int]] = None,
            input_dropout: Optional[float] = None,
            hidden_activation: str = "relu",
            last_encoder_activation: str = "linear",
            last_decoder_activation: str = "linear"
    ) -> "StackedAutoEncoderConfig":
        """Initialize an SAE configuration with higher-level options.

        Arguments:
        hidden_dims: Number of hidden layer units of each encoder and decoder.
        input_dropout: Input dropout probability for each encoder and decoder.
        hidden_activation: Specifies activation function of each hidden layer.
        latent_dims: Specifies latent dimension of each subsequent autoencoder.
        last_encoder_activation: Activation of the last encoder (in last AE).
        last_decoder_activation: Activation of the last decoder (in first AE).

        Note:
        Last encoder and last decoder activations are expected to be non-ReLU
        to retain full information in final embedded space and recover negative
        values during reconstruction.
        """
        autoencoders = []
        prev_dim = input_dim

        def encoder_output_activation(layer_index):
            if layer_index == len(latent_dims) - 1:
                return last_encoder_activation
            return hidden_activation

        def decoder_output_activation(layer_index):
            if layer_index == 0:
                return last_decoder_activation
            return hidden_activation

        for i, latent_dim in enumerate(latent_dims):
            ae_config = AutoEncoderConfig.build(
                input_dim=prev_dim,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                input_dropout=input_dropout,
                hidden_activation=hidden_activation,
                encoder_output_activation=encoder_output_activation(i),
                decoder_output_activation=decoder_output_activation(i)
            )
            autoencoders.append(ae_config)
            prev_dim = latent_dim

        return StackedAutoEncoderConfig(autoencoders=autoencoders)

    def replace_input_dropout(
            self,
            new_dropout: Optional[float]
    ) -> "StackedAutoEncoderConfig":
        """Create a new instance by replacing all input_dropouts."""
        new_autoencoders = [
            replace(
                ae,
                encoder=replace(ae.encoder, input_dropout=new_dropout),
                decoder=replace(ae.decoder, input_dropout=new_dropout),
            )
            for ae in self.autoencoders
        ]
        return replace(self, autoencoders=new_autoencoders)


class Coder(nn.Module):
    """Generic template model that functions as an encoder or decoder."""
    def __init__(
            self,
            config: CoderConfig
    ):
        """Initialize a Coder with specified configuration."""
        super().__init__()

        self.config = config

        hidden_activation = get_activation_module(config.hidden_activation)
        output_activation = get_activation_module(config.output_activation)

        previous_dim = config.input_dim
        hidden_layers = []

        if config.hidden_dims is not None:
            for layer_dim in config.hidden_dims:
                hidden_layers.append(nn.Linear(previous_dim, layer_dim))
                if hidden_activation:
                    hidden_layers.append(hidden_activation())

                previous_dim = layer_dim

        output_layer = []
        output_layer.append(nn.Linear(previous_dim, config.output_dim))
        if output_activation is not None:
            output_layer.append(output_activation())

        self.hidden = nn.Sequential(*hidden_layers)
        self.output = nn.Sequential(*output_layer)

    def forward(self, x):
        x = self.hidden(x)
        x = self.output(x)

        return x

    def save(self, path: str, **kwargs):
        """Save the Coder model weights and configuration to path."""
        torch.save({
            "state_dict": self.state_dict(),
            "config": self.config.to_dict(),
            }, path, **kwargs)

    @staticmethod
    def load(path: str, **kwargs):
        """Load an Coder model from path.

        The underlying pickle file is expected to contain model configuration.
        """
        state = torch.load(path, **kwargs)
        config = CoderConfig.from_dict(state["config"])
        model = Coder(config)
        model.load_state_dict(state["state_dict"])
        return model


class BaseAutoEncoder(ABC):
    @property
    @abstractmethod
    def encoder(self) -> nn.Module:
        pass

    @property
    @abstractmethod
    def decoder(self) -> nn.Module:
        pass


class AutoEncoder(nn.Module, BaseAutoEncoder):
    """Generic autoencoder model"""
    def __init__(
            self,
            config: AutoEncoderConfig,
            encoder: Optional[Coder] = None,
            decoder: Optional[Coder] = None
    ):
        """Initialize an AE with specified configuration or existing modules.

        Arguments:
        config: Configuration is used if encoder or decoder is not provided.
        encoder: Use an existing encoder module (without deepcopy).
        decoder: Use an existing decoder module (without deepcopy).
        """
        super().__init__()

        self._encoder = encoder or Coder(config.encoder)
        self._decoder = decoder or Coder(config.decoder)

        self.config = AutoEncoderConfig(
            encoder=self._encoder.config,
            decoder=self._decoder.config
        )

    def forward(self, x):
        x = self._encoder(x)
        x = self._decoder(x)

        return x

    @property
    def encoder(self) -> nn.Module:
        return self._encoder

    @property
    def decoder(self) -> nn.Module:
        return self._decoder

    def fit(
            self,
            train_loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            loss_fn: nn.modules.loss._Loss,
            **kwargs
    ) -> pd.DataFrame:
        """Train the AutoEncoder to minimize reconstruction loss.

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

    def save(self, path: str, **kwargs):
        """Save the AutoEncoder model weights and configuration to path."""
        torch.save({
            "state_dict": self.state_dict(),
            "config": self.config.to_dict(),
            }, path, **kwargs)

    @staticmethod
    def load(path: str, **kwargs):
        """Load an AutoEncoder model from path.

        The underlying pickle file is expected to contain model configuration.
        """
        state = torch.load(path, **kwargs)
        config = AutoEncoderConfig.from_dict(state["config"])
        model = AutoEncoder(config)
        model.load_state_dict(state["state_dict"])
        return model


class StackedAutoEncoder(nn.Module, BaseAutoEncoder):
    """Generic stacked autoencoder (SAE) model"""
    def __init__(
            self,
            config: StackedAutoEncoderConfig,
    ):
        """Initialize an SAE with specified configuration."""
        super().__init__()

        self.config = config
        encoders = [Coder(cfg.encoder) for cfg in config.autoencoders]
        decoders = [Coder(cfg.decoder) for cfg in config.autoencoders]

        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

    def forward(self, x):
        for encoder in self.encoders:
            x = encoder(x)
        for decoder in self.decoders:
            x = decoder(x)

        return x

    @property
    def encoder(self) -> nn.Module:
        return nn.Sequential(*self.encoders)

    @property
    def decoder(self) -> nn.Module:
        return nn.Sequential(*self.decoders)

    def greedy_fit(
            self,
            train_loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            loss_fn: nn.modules.loss._Loss,
            **kwargs
    ) -> list[pd.DataFrame]:
        """Perform greedy layer-wise training on autoencoders.

        Returns:
        list[pandas.DataFrame]: Loss history for each autoencoder.

        Notes:
        See `.training.train_model()` for the details of the parameters.
        """
        device = next(self.parameters()).device

        def transform_fn(encoders: list[nn.Module], device=None):
            """Return a transform function from a list of encoders."""
            net = nn.Sequential(*encoders)
            if device:
                net.to(device)
            net.eval()

            def transform(x):
                with torch.no_grad():
                    return net(x)

            return transform

        coder_pairs = zip(self.encoders, reversed(self.decoders))
        trained_encoders = []
        history_autoencoders = []

        for i, (encoder, decoder) in enumerate(coder_pairs):
            logger.info("Training autoencoder " + str(i))

            config = AutoEncoderConfig(encoder=encoder.config,
                                       decoder=decoder.config)
            autoencoder = AutoEncoder(config, encoder, decoder)
            autoencoder = autoencoder.to(device)
            history = autoencoder.fit(train_loader,
                                      optimizer,
                                      loss_fn,
                                      **kwargs,
                                      transform=transform_fn(trained_encoders,
                                                             device=device))
            trained_encoders.append(encoder)
            history_autoencoders.append(history)

        return history_autoencoders

    def fit(
            self,
            train_loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            loss_fn: nn.modules.loss._Loss,
            **kwargs
    ) -> pd.DataFrame:
        """Perform global loss optimization of SAE.

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

    def save(self, path: str, **kwargs):
        """Save the SAE model weights and configuration to path."""
        torch.save({
            "state_dict": self.state_dict(),
            "config": self.config.to_dict(),
            }, path, **kwargs)

    @staticmethod
    def load(path: str, **kwargs):
        """Load an SAE model from path.

        The underlying pickle file is expected to contain model configuration.
        """
        state = torch.load(path, **kwargs)
        config = StackedAutoEncoderConfig.from_dict(state["config"])
        model = StackedAutoEncoder(config)
        model.load_state_dict(state["state_dict"])
        return model
