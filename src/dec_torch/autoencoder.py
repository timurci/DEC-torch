import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from dec_torch.training import train_ae_model

logger = logging.getLogger(__name__)


_ACTIVATION_REGISTRY: dict[str, type[nn.Module] | None] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "linear": None,
}


def get_activation_module(name: str) -> type[nn.Module] | None:
    """Get an activation function module by name.

    This function retrieves a registered activation function from the internal
    registry. Built-in activations include 'relu', 'tanh', 'sigmoid', and 'linear'.
    Custom activations can be registered using register_activation_module().

    Args:
        name: Name of the activation function (case-insensitive).

    Returns:
        Activation module class or None if 'linear' is specified.

    Raises:
        KeyError: If the name is not found in the registry.

    Example:
        >>> relu_class = get_activation_module('relu')
        >>> relu_instance = relu_class()
        >>> print(relu_instance)
        ReLU()
    """
    activation = _ACTIVATION_REGISTRY[name.lower()]
    if activation is None:
        return None
    return activation


def register_activation_module(name: str, module_type: type[nn.Module]) -> None:
    """Register a name for a custom activation function module.

    This allows using custom activation functions in encoder/decoder configurations
    by name instead of passing the class directly.

    Args:
        name: Name to register the activation under (stored lowercase).
        module_type: The activation module class to register.

    Example:
        >>> class CustomActivation(nn.Module):
        ...     def forward(self, x):
        ...         return x * torch.sigmoid(x)
        >>> register_activation_module('swish', CustomActivation)
        >>> print('swish' in list_activation_modules())
        True
    """
    _ACTIVATION_REGISTRY[name.lower()] = module_type


def list_activation_modules() -> set[str]:
    """List registered names of activation modules.

    Returns:
        Set of all currently registered activation module names.

    Example:
        >>> modules = list_activation_modules()
        >>> print(sorted(modules))
        ['linear', 'relu', 'sigmoid', 'tanh']
    """
    return set(_ACTIVATION_REGISTRY.keys())


@dataclass(frozen=True)
class CoderConfig:
    """Configuration for Coder (encoder or decoder) module.

    This dataclass stores all hyperparameters needed to construct a Coder
    module, which serves as a building block for encoders and decoders.

    Attributes:
        input_dim: Input size of the model.
        output_dim: Output size of the model.
        hidden_dims: Number of units for each hidden layer. If None, creates a
            single linear layer from input to output.
        input_dropout: Dropout probability of an element in the input. Applied
            after the input layer. Must be between 0.0 and 1.0.
        hidden_activation: Name of the module to be used in hidden layers. See
            list_activation_modules() for available options.
        output_activation: Name of the module to be used in output layer. See
            list_activation_modules() for available options.

    Note:
        In PyTorch, dropout will only set the output of neurons to zero,
        while still computing them. Therefore it does not improve
        the training time.
    """

    input_dim: int
    output_dim: int
    hidden_dims: list[int] | None = None
    input_dropout: float | None = None
    hidden_activation: str = "relu"
    output_activation: str = "relu"

    def to_dict(self) -> dict:
        """Return attributes as dict."""
        return asdict(self)

    @staticmethod
    def from_dict(config_dict: dict) -> "CoderConfig":
        """Construct config from a dict type."""
        return CoderConfig(**config_dict)


@dataclass(frozen=True)
class AutoEncoderConfig:
    """Configuration for AutoEncoder module.

    This dataclass pairs an encoder configuration with a decoder configuration
    to define a complete autoencoder architecture.

    Attributes:
        encoder: Configuration for the encoder module.
        decoder: Configuration for the decoder module.

    Example:
        >>> encoder_cfg = CoderConfig(input_dim=784, output_dim=128, hidden_dims=[500])
        >>> decoder_cfg = CoderConfig(input_dim=128, output_dim=784, hidden_dims=[500])
        >>> config = AutoEncoderConfig(encoder=encoder_cfg, decoder=decoder_cfg)
    """

    encoder: CoderConfig
    decoder: CoderConfig

    def to_dict(self) -> dict:
        """Return config in dict type."""
        return {"encoder": self.encoder.to_dict(), "decoder": self.decoder.to_dict()}

    @staticmethod
    def from_dict(config_dict: dict) -> "AutoEncoderConfig":
        """Construct config from dict type."""
        return AutoEncoderConfig(
            encoder=CoderConfig.from_dict(config_dict["encoder"]),
            decoder=CoderConfig.from_dict(config_dict["decoder"]),
        )

    @staticmethod
    def build(
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int] | None = None,
        input_dropout: float | None = None,
        hidden_activation: str = "relu",
        encoder_output_activation: str = "relu",
        decoder_output_activation: str = "relu",
    ) -> "AutoEncoderConfig":
        """Initialize AutoEncoderConfig with a symmetrical configuration.

        This convenience method creates both encoder and decoder configurations
        with matching architectures, making it easier to define standard
        autoencoders where the decoder mirrors the encoder.

        Args:
            input_dim: Input size of the autoencoder.
            latent_dim: Size of the latent representation.
            hidden_dims: Number of units for each hidden layer. These dimensions are
                used for both encoder and decoder.
            input_dropout: Dropout probability for both encoder and decoder inputs.
                Applied to the input of both modules.
            hidden_activation: Activation function for hidden layers in both
                encoder and decoder.
            encoder_output_activation: Activation function for encoder output.
            decoder_output_activation: Activation function for decoder output.

        Returns:
            A configuration for a symmetric autoencoder.

        Example:
            >>> config = AutoEncoderConfig.build(
            ...     input_dim=784,
            ...     latent_dim=128,
            ...     hidden_dims=[500, 250],
            ...     hidden_activation='relu',
            ...     encoder_output_activation='sigmoid',
            ...     decoder_output_activation='linear'
            ... )
            >>> print(config.encoder.output_dim)
            128
            >>> print(config.decoder.output_dim)
            784
        """
        shared_kwargs = {
            "hidden_dims": hidden_dims,
            "input_dropout": input_dropout,
            "hidden_activation": hidden_activation,
        }
        encoder_config = CoderConfig(
            input_dim=input_dim,
            output_dim=latent_dim,
            output_activation=encoder_output_activation,
            **shared_kwargs,
        )
        decoder_config = CoderConfig(
            input_dim=latent_dim,
            output_dim=input_dim,
            output_activation=decoder_output_activation,
            **shared_kwargs,
        )
        return AutoEncoderConfig(encoder=encoder_config, decoder=decoder_config)


@dataclass(frozen=True)
class StackedAutoEncoderConfig:
    """Configuration for StackedAutoEncoder module.

    This dataclass stores a list of AutoEncoderConfig objects, where
    each autoencoder's latent dimension becomes the input dimension for the
    next autoencoder in the stack.

    Attributes:
        autoencoders: List of autoencoder configurations in the order they will be
            stacked (from input to deepest latent space).

    Example:
        >>> # Create three autoencoders that progressively compress dimensions
        >>> ae1 = AutoEncoderConfig.build(input_dim=784, latent_dim=500)
        >>> ae2 = AutoEncoderConfig.build(input_dim=500, latent_dim=128)
        >>> ae3 = AutoEncoderConfig.build(input_dim=128, latent_dim=10)
        >>> config = StackedAutoEncoderConfig(autoencoders=[ae1, ae2, ae3])
    """

    autoencoders: list[AutoEncoderConfig]

    def to_dict(self):
        """Return config in a dict type."""
        return {"autoencoders": [ae.to_dict() for ae in self.autoencoders]}

    @staticmethod
    def from_dict(config_dict: dict) -> "StackedAutoEncoderConfig":
        """Construct config from a dict type."""
        configs = [
            AutoEncoderConfig.from_dict(ae) for ae in config_dict["autoencoders"]
        ]
        return StackedAutoEncoderConfig(autoencoders=configs)

    @staticmethod
    def build(
        input_dim: int,
        latent_dims: list[int],
        hidden_dims: list[int] | None = None,
        input_dropout: float | None = None,
        hidden_activation: str = "relu",
        last_encoder_activation: str = "linear",
        last_decoder_activation: str = "linear",
    ) -> "StackedAutoEncoderConfig":
        """Initialize an SAE configuration with higher-level options.

        This convenience method creates a stacked autoencoder by progressively
        building autoencoder configurations where each one's latent dimension
        becomes the input dimension for the next.

        Args:
            input_dim: Input dimension of the first autoencoder.
            latent_dims: Latent dimensions for each autoencoder in the stack. The
                length of this list determines the number of stacked autoencoders.
            hidden_dims: Number of hidden layer units for each encoder and decoder. If
                None, creates direct input-to-latent connections.
            input_dropout: Input dropout probability for each encoder and decoder.
                Applied to the input of each module.
            hidden_activation: Activation function for hidden layers in all encoders
                and decoders except the final ones.
            last_encoder_activation: Activation function for the final encoder's output
                layer. Typically set to 'linear' to preserve information.
            last_decoder_activation: Activation function for the final decoder's output
                layer. Typically set to 'linear' or 'sigmoid' for reconstruction.

        Returns:
            Configuration for a stacked autoencoder.

        Note:
            Last encoder and last decoder activations are typically set to be non-ReLU
            to retain full information in final embedded space and recover negative
            values during reconstruction.

        Example:
            >>> config = StackedAutoEncoderConfig.build(
            ...     input_dim=784,
            ...     latent_dims=[500, 128, 10],
            ...     hidden_dims=[1000, 500],
            ...     hidden_activation='relu',
            ...     last_encoder_activation='linear',
            ...     last_decoder_activation='linear'
            ... )
            >>> print(len(config.autoencoders))
        """
        autoencoders = []
        prev_dim = input_dim

        def encoder_output_activation(layer_index: int) -> str:
            if layer_index == len(latent_dims) - 1:
                return last_encoder_activation
            return hidden_activation

        def decoder_output_activation(layer_index: int) -> str:
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
                decoder_output_activation=decoder_output_activation(i),
            )
            autoencoders.append(ae_config)
            prev_dim = latent_dim

        return StackedAutoEncoderConfig(autoencoders=autoencoders)

    def replace_input_dropout(
        self, new_dropout: float | None
    ) -> "StackedAutoEncoderConfig":
        """Create a new instance by replacing all input_dropouts.

        This method creates a copy of the configuration with new dropout values
        for all encoders and decoders in the stacked autoencoder.

        Args:
            new_dropout: New dropout probability for all inputs. Set to None to remove
                dropout.

        Returns:
            New configuration with updated dropout values.

        Example:
            >>> config = StackedAutoEncoderConfig.build(
            ...     input_dim=784, latent_dims=[128, 10], input_dropout=0.2
            ... )
            >>> new_config = config.replace_input_dropout(0)
            >>> print(new_config.autoencoders[0].encoder.input_dropout)
            0.0
        """
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
    """Generic template model that functions as an encoder or decoder.

    This module creates a feed-forward neural network with configurable
    hidden layers, activations, and dropout. It can be used as either
    an encoder (compressing data) or decoder (reconstructing data).

    The architecture follows this pattern:
    Input -> [Linear -> Activation -> Dropout]* -> Linear -> Activation

    Args:
        config: Configuration object defining the architecture.

    Attributes:
        config: The configuration used to build this coder.
        hidden: Hidden layers of the network.
        output: Output layer of the network.

    Example:
        >>> config = CoderConfig(
        ...     input_dim=784,
        ...     output_dim=128,
        ...     hidden_dims=[500, 250],
        ...     hidden_activation='relu',
        ...     output_activation='linear'
        ... )
        >>> encoder = Coder(config)
        >>> print(encoder)
        Coder(
          (hidden): Sequential(...)
          (output): Sequential(...)
        )
    """

    def __init__(self, config: CoderConfig) -> None:
        """Initialize a Coder with specified configuration.

        Args:
            config: Configuration object defining the architecture.
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the coder.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).

        Example:
            >>> config = CoderConfig(input_dim=10, output_dim=5)
            >>> coder = Coder(config)
            >>> x = torch.randn(32, 10)
            >>> output = coder(x)
            >>> print(output.shape)
            torch.Size([32, 5])
        """
        x = self.hidden(x)

        return self.output(x)

    def save(self, path: str, **kwargs) -> None:
        """Save the Coder model weights and configuration to path.

        Saves both the model state dictionary and the configuration object,
        allowing full reconstruction of the model later.

        Args:
            path: Path where the model will be saved.
            **kwargs: Additional arguments passed to torch.save().

        Example:
            >>> coder = Coder(config)
            >>> coder.save('encoder.pth')
            >>> loaded_coder = Coder.load('encoder.pth')
        """
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self.config.to_dict(),
            },
            path,
            **kwargs,
        )

    @staticmethod
    def load(path: str, **kwargs) -> "Coder":
        """Load a Coder model from path.

        Loads a Coder model that was saved with `Coder.save()`. The file must contain
        both the model weights and the configuration.

        Args:
            path: Path to the saved model file.
            **kwargs: Additional arguments passed to torch.load().

        Returns:
            Loaded Coder model with saved weights and configuration.

        Example:
            >>> coder = Coder.load('encoder.pth')
            >>> print(coder.config.input_dim)
            784
        """
        state = torch.load(path, **kwargs)
        config = CoderConfig.from_dict(state["config"])
        model = Coder(config)
        model.load_state_dict(state["state_dict"])
        return model


class BaseAutoEncoder(ABC):
    """Abstract base class for autoencoders.

    This interface defines the common structure that all autoencoder
    implementations must follow, ensuring they provide access to both
    encoder and decoder components.

    Properties:
        encoder: The encoder module that compresses input to latent space.
        decoder: The decoder module that reconstructs from latent space.
    """

    @property
    @abstractmethod
    def encoder(self) -> nn.Module:
        """Get the encoder module of this autoencoder."""

    @property
    @abstractmethod
    def decoder(self) -> nn.Module:
        """Get the decoder module of this autoencoder."""


class AutoEncoder(nn.Module, BaseAutoEncoder):
    """Generic autoencoder model.

    A standard autoencoder consisting of an encoder and decoder pair.
    The encoder maps input to a latent representation, and the decoder
    maps the latent representation back to the original input space.

    Args:
        config: Configuration defining encoder and decoder architecture.
        encoder: Optional pre-built encoder (without deepcopy).
        decoder: Optional pre-built decoder (without deepcopy).

    Attributes:
        _encoder: The encoder module.
        _decoder: The decoder module.
        config: Configuration of this autoencoder.

    Example:
        >>> config = AutoEncoderConfig.build(
        ...     input_dim=784,
        ...     latent_dim=128,
        ...     hidden_dims=[500, 250]
        ... )
        >>> ae = AutoEncoder(config)
        >>> x = torch.randn(64, 784)
        >>> reconstructed = ae(x)
        >>> print(reconstructed.shape)
        torch.Size([64, 784])
    """

    def __init__(
        self,
        config: AutoEncoderConfig,
        encoder: Coder | None = None,
        decoder: Coder | None = None,
    ) -> None:
        """Initialize an AE with specified configuration or existing modules.

        Args:
            config: Configuration is used if encoder or decoder is not provided.
            encoder: Use an existing encoder module (without deepcopy).
            decoder: Use an existing decoder module (without deepcopy).

        Note:
            If either an encoder or decoder provided, its config will overwrite the
            provided config.
        """
        super().__init__()

        self._encoder = encoder or Coder(config.encoder)
        self._decoder = decoder or Coder(config.decoder)

        self.config = AutoEncoderConfig(
            encoder=self._encoder.config, decoder=self._decoder.config
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Reconstructed tensor of shape (batch_size, input_dim).

        Example:
            >>> ae = AutoEncoder(config)
            >>> x = torch.randn(32, 784)
            >>> x_recon = ae(x)
            >>> # Encoder and decoder can be accessed separately
            >>> z = ae.encoder(x)
            >>> print(z.shape)  # Latent representation
            torch.Size([32, 128])
        """
        return self._decoder(self._encoder(x))

    @property
    def encoder(self) -> nn.Module:
        """Get the encoder module of this autoencoder."""
        return self._encoder

    @property
    def decoder(self) -> nn.Module:
        """Get the decoder module of this autoencoder."""
        return self._decoder

    def fit(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.modules.loss._Loss,
        **kwargs,
    ) -> pd.DataFrame:
        """Train the AutoEncoder to minimize reconstruction loss.

        This method provides a convenient training loop for the autoencoder.
        It uses the training utilities from dec_torch.training module.

        Args:
            train_loader: DataLoader for training data.
            optimizer: Optimizer for model parameters.
            loss_fn: Loss function (typically MSELoss).
            **kwargs: Additional arguments passed to train_ae_model().

        Returns:
            Training and validation history with loss values.

        See Also:
            dec_torch.training.train_ae_model: Detailed parameter documentation.

        Example:
            >>> from torch.utils.data import DataLoader, TensorDataset
            >>> from torch import optim, nn
            >>> dataset = TensorDataset(torch.randn(1000, 784))
            >>> loader = DataLoader(dataset, batch_size=32)
            >>> optimizer = optim.Adam(ae.parameters(), lr=0.001)
            >>> loss_fn = nn.MSELoss()
            >>> history = ae.fit(loader, optimizer, loss_fn, n_epoch=50)
            >>> print(history.head())
        """
        device = next(self.parameters()).device
        if "device" not in kwargs:
            kwargs["device"] = device
        return train_ae_model(self, train_loader, optimizer, loss_fn, **kwargs)

    def save(self, path: str, **kwargs) -> None:
        """Save the AutoEncoder model weights and configuration to path.

        Saves both the model state dictionary and the configuration object,
        allowing full reconstruction of the model later.

        Args:
            path: Path where the model will be saved.
            **kwargs: Additional arguments passed to torch.save().

        See Also:
            AutoEncoder.load: Load a saved autoencoder.

        Example:
            >>> ae = AutoEncoder(config)
            >>> ae.save('autoencoder.pth')
            >>> loaded_ae = AutoEncoder.load('autoencoder.pth')
        """
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self.config.to_dict(),
            },
            path,
            **kwargs,
        )

    @staticmethod
    def load(path: str, **kwargs) -> "AutoEncoder":
        """Load an AutoEncoder model from path.

        Loads an AutoEncoder model that was saved with `AutoEncoder.save()`.
        The file must contain both the model weights and the configuration.

        Args:
            path: Path to the saved model file.
            **kwargs: Additional arguments passed to torch.load().

        Returns:
            Loaded autoencoder with saved weights and configuration.

        Example:
            >>> ae = AutoEncoder.load('autoencoder.pth')
            >>> print(ae.config.encoder.input_dim)
            784
        """
        state = torch.load(path, **kwargs)
        config = AutoEncoderConfig.from_dict(state["config"])
        model = AutoEncoder(config)
        model.load_state_dict(state["state_dict"])
        return model


class StackedAutoEncoder(nn.Module, BaseAutoEncoder):
    """Generic stacked autoencoder (SAE) model.

    A stacked autoencoder is functionally equivalent to a regular autoencoder,
    but is trained in two distinct phases to achieve better weight initialization
    and mitigate the vanishing gradient problem.

    Training Phases:
    1. Greedy layer-wise training: Each autoencoder is trained sequentially to
    initialize the weights effectively. For instance,
        - 1st layer (i.e, autoencoder) learns to reconstruct the raw input.
        - 2nd layer learns to reconstruct latent representation of the 1st layer.
        - 3rd layer leanrs to reconstruct latent representation of the 2nd layer.
    2. Global training: The entire stack is trained end-to-end as a single autoencoder.

    Attributes:
        config: Configuration of this stacked autoencoder.
        encoders: List of encoder modules.
        decoders: List of decoder modules (in reverse order).

    Example:
        >>> config = StackedAutoEncoderConfig.build(
        ...     input_dim=784,
        ...     latent_dims=[500, 128, 10],
        ... )
        >>> sae = StackedAutoEncoder(config)
        >>> x = torch.randn(64, 784)
        >>> x_recon = sae(x)
        >>> print(x_recon.shape)
        torch.Size([64, 784])
        >>> # Access individual layers
        >>> for i, encoder in enumerate(sae.encoders):
        ...     print(f"Encoder {i}: {encoder.config.output_dim}")
    """

    def __init__(
        self,
        config: StackedAutoEncoderConfig,
    ) -> None:
        """Initialize an SAE with specified configuration.

        Args:
            config: Configuration for stacked architecture.
        """
        super().__init__()

        self.config = config
        encoders = [Coder(cfg.encoder) for cfg in config.autoencoders]
        decoders = [Coder(cfg.decoder) for cfg in config.autoencoders]

        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the stacked autoencoder.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Reconstructed tensor of shape (batch_size, input_dim).
        """
        for encoder in self.encoders:
            x = encoder(x)
        for decoder in self.decoders:
            x = decoder(x)

        return x

    @property
    def encoder(self) -> nn.Module:
        """Get the stacked encoder module."""
        return nn.Sequential(*self.encoders)

    @property
    def decoder(self) -> nn.Module:
        """Get the stacked decoder module."""
        return nn.Sequential(*self.decoders)

    def greedy_fit(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.modules.loss._Loss,
        **kwargs,
    ) -> list[pd.DataFrame]:
        """Perform greedy layer-wise training on autoencoders.

        Trains each autoencoder sequentially, using the encoders from previously
        trained layers to transform the input. This allows hierarchical learning
        of features.

        Args:
            train_loader: DataLoader for training data.
            optimizer: Optimizer for model parameters.
            loss_fn: Loss function for reconstruction error.
            **kwargs: Additional arguments passed to train_ae_model().

        Returns:
            List of loss histories for each autoencoder, one DataFrame per layer.

        See Also:
            train_ae_model: Detailed parameter documentation.

        Example:
            >>> sae = StackedAutoEncoder(config)
            >>> optimizer = torch.optim.Adam(sae.parameters(), lr=0.001)
            >>> loss_fn = torch.nn.MSELoss()
            >>> histories = sae.greedy_fit(train_loader, optimizer, loss_fn, n_epoch=50)
            >>> print(f"Trained {len(histories)} layers")
        """
        device = next(self.parameters()).device
        if "device" not in kwargs:
            kwargs["device"] = device
        else:
            device = kwargs["device"]

        def transform_fn(encoders: list[nn.Module], device=None) -> Callable:
            """Return a transform function from a list of encoders."""
            net = nn.Sequential(*encoders)
            if device:
                net.to(device)
            net.eval()

            def transform(x: torch.Tensor) -> torch.Tensor:
                with torch.no_grad():
                    return net(x)

            return transform

        coder_pairs = zip(self.encoders, reversed(self.decoders), strict=True)
        trained_encoders = []
        history_autoencoders = []

        for i, (encoder, decoder) in enumerate(coder_pairs):
            logger.info("Training autoencoder %s", i)

            config = AutoEncoderConfig(encoder=encoder.config, decoder=decoder.config)
            autoencoder = AutoEncoder(config, encoder, decoder)
            autoencoder = autoencoder.to(device)
            history = autoencoder.fit(
                train_loader,
                optimizer,
                loss_fn,
                **kwargs,
                transform=transform_fn(trained_encoders, device=device),
            )
            trained_encoders.append(encoder)
            history_autoencoders.append(history)

        return history_autoencoders

    def fit(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.modules.loss._Loss,
        **kwargs,
    ) -> pd.DataFrame:
        """Perform global loss optimization of SAE.

        Fine-tunes the entire stacked autoencoder end-to-end after layer-wise
        pre-training. This allows all layers to adapt jointly.

        Args:
            train_loader: DataLoader for training data.
            optimizer: Optimizer for model parameters.
            loss_fn: Loss function for reconstruction error.
            **kwargs: Additional arguments passed to train_ae_model().

        Returns:
            Training and validation loss history.

        See Also:
            train_ae_model: Detailed parameter documentation.
            greedy_fit: For layer-wise pre-training.

        Example:
            > sae = StackedAutoEncoder(config)
            > optimizer = torch.optim.Adam(sae.parameters(), lr=0.001)
            > loss_fn = torch.nn.MSELoss()
            >
            > # Pre-train layer-wise
            > histories = sae.greedy_fit(train_loader, optimizer, loss_fn, n_epoch=50)
            >
            > # Fine-tune end-to-end
            > history = sae.fit(train_loader, optimizer, loss_fn, n_epoch=50)
        """
        device = next(self.parameters()).device
        if "device" not in kwargs:
            kwargs["device"] = device

        return train_ae_model(self, train_loader, optimizer, loss_fn, **kwargs)

    def save(self, path: str, **kwargs) -> None:
        """Save the SAE model weights and configuration to path.

        Args:
            path: Path where the model will be saved.
            **kwargs: Additional arguments passed to torch.save().
        """
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self.config.to_dict(),
            },
            path,
            **kwargs,
        )

    @staticmethod
    def load(path: str, **kwargs) -> "StackedAutoEncoder":
        """Load an SAE model from path.

        Args:
            path: Path to the saved model file.
            **kwargs: Additional arguments passed to torch.load().

        Returns:
            Loaded stacked autoencoder with saved weights and configuration.
        """
        state = torch.load(path, **kwargs)
        config = StackedAutoEncoderConfig.from_dict(state["config"])
        model = StackedAutoEncoder(config)
        model.load_state_dict(state["state_dict"])
        return model
