from torch import nn

from dataclasses import dataclass
from typing import Optional


_ACTIVATION_REGISTRY: dict[str, type[nn.Module] | None] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "linear": None
}


def get_activation_module(name: str) -> Optional[nn.Module]:
    """Get an activation function module by name.

    Returns:
    Optional[nn.Module]: Instance of module or None if 'linear' is specified.

    Raises:
    KeyError: If the name is not found in the registry.
    """
    activation = _ACTIVATION_REGISTRY[name.lower()]
    if activation is None:
        return None
    return activation()


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
    hidden_dims: list[int] = []
    input_dropout: Optional[float] = None
    hidden_activation: str = "relu"
    output_activation: str = "relu"


@dataclass(frozen=True)
class AutoEncoderConfig:
    """Configuration for AutoEncoder module"""
    encoder_config: CoderConfig
    decoder_config: CoderConfig

    @staticmethod
    def build(
            input_dim: int,
            latent_dim: int,
            hidden_dims: list[int] = [],
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
            encoder_config=encoder_config,
            decoder_config=decoder_config
        )


@dataclass(frozen=True)
class StackedAutoEncoderConfig:
    """Configuration for StackedAutoEncoder module"""
    autoencoder_configs: list[AutoEncoderConfig]

    @staticmethod
    def build(
            input_dim: int,
            latent_dims: list[int],
            hidden_dims: list[int],
            input_dropout: Optional[float] = None,
            hidden_activation: str = "relu",
            trailing_activation: str = "linear"
    ) -> "StackedAutoEncoderConfig":
        """Initialize an SAE with a higher-level configuration

        Arguments:
        hidden_dims: Number of hidden layer units of each encoder and decoder.
        input_dropout: Input dropout probability for each encoder and decoder.
        hidden_activation: Specifies activation function of each hidden layer.
        latent_dims: Specifies latent dimension of each subsequent autoencoder.
        trailing_activation: Activation of the first encoder and last decoder.
        """
        raise NotImplementedError()


class Coder(nn.Module):
    """Generic template model that functions as an encoder or decoder."""
    def __init__(
            self,
            config: CoderConfig
    ):
        """Initialize the model according to specified dimensions and layers.

        Args:
        input_dim: Input size of the model.
        output_dim: Output size of the model.
        hidden_dims: Number of units in each hidden layer.
        """
        super().__init__()

        self.config = config

        hidden_activation = get_activation_module(config.hidden_activation)
        output_activation = get_activation_module(config.output_activation)

        previous_dim = config.input_dim
        hidden_layers = []

        for layer_dim in config.hidden_dims:
            hidden_layers.append(nn.Linear(previous_dim, layer_dim))
            if hidden_activation:
                hidden_layers.append(hidden_activation())

            previous_dim = layer_dim

        output_layer = [nn.Linear(previous_dim, config.output_dim)]
        if output_activation:
            output_layer.append(output_activation())

        self.hidden = nn.Sequential(*hidden_layers)
        self.output = nn.Sequential(*output_layer)

    def forward(self, x):
        x = self.hidden(x)
        x = self.output(x)

        return x


class AutoEncoder(nn.Module):
    """Generic autoencoder model"""
    def __init__(
            self,
            config: AutoEncoderConfig,
    ):
        """Initialize the model according to specified dimensions and layers.

        Args:
        input_dim: Input and output size of the model.
        latent_dim: Size of the latent space vector.
        encoder_hidden_dims: Number of units in each hidden layer of encoder.
        decoder_hidden_dims: Number of units in each hidden leyer of decoder.
        """
        super().__init__()

        self.config = config

        self.encoder = Coder(config.encoder_config)
        self.decoder = Coder(config.decoder_config)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
