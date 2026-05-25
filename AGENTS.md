# DEC-torch: Deep Embedded Clustering in PyTorch

## Project Overview

DEC-torch is a modular PyTorch toolkit for Deep Embedded Clustering (DEC), an unsupervised clustering method that operates in the latent space of deep neural networks. The project implements the DEC algorithm as described in Xie et al. (2016), providing a clean separation between representation learning (handled by user-provided encoders) and clustering optimization.

### Key Philosophy
- **Modular Design**: DEC clustering is separate from representation learning
- **Plug-and-play**: Works with any PyTorch encoder (`nn.Module`)
- **Extensible**: All components can be customized or extended
- **Research-oriented**: Built for experimentation and research use cases

## Technology Stack

- **Language**: Python >= 3.10
- **Framework**: PyTorch >= 2.6.0
- **Build System**: setuptools (pyproject.toml)
- **Dependencies**:
  - Core: torch, pandas, scikit-learn
  - Visualization: seaborn, umap-learn
  - Development: No explicit testing framework detected

## Project Structure

```text
src/dec_torch/
├── __init__.py              # Package documentation, module organization
├── autoencoder.py           # Autoencoder implementations and configs
├── training.py              # Training loops and history tracking
├── dec/                     # Core DEC implementation
│   ├── __init__.py         # Module docs, exports DEC classes/functions
│   ├── dec.py              # Main DEC class and clustering logic
│   └── io.py               # DEC model persistence (save/load DEC models)
└── utils/                   # Utility modules
    ├── __init__.py         # Module documentation
    ├── data.py             # Data processing utilities
    └── visualization.py    # Plotting and visualization functions

examples/
└── example_mnist.ipynb     # Complete MNIST clustering workflow
```

## Core Architecture

### 1. DEC Module (`dec_torch.dec`)
- **DEC**: Main clustering model that wraps an encoder
- **KLDivLoss**: Custom KL divergence loss for clustering optimization
- **Cluster Initialization**: Multiple strategies (k-means trials, random)

### 2. Autoencoder Module (`dec_torch.autoencoder`)
- **Coder/CoderConfig**: Configurable encoder/decoder components
- **AutoEncoder/StackedAutoEncoder**: Built-in autoencoder implementations
- **Activation Registry**: Extensible activation function system

### 3. Training Module (`dec_torch.training`)
- **HistoryTracker**: Efficient performance logging with enum-based keys
- **Training Loops**: Generic epoch runners for autoencoders and DEC
- **Metrics System**: Flexible metric computation during training

### 4. Utilities (`dec_torch.utils`)
- **Data Processing**: Batch extraction and transformation utilities
- **Visualization**: Loss plotting and cluster visualization with UMAP/t-SNE

## Development Workflow

### Installation
```bash
git clone https://github.com/tururci/DEC-torch.git
cd DEC-torch
pip install -e .
```

### Typical Usage Pattern
1. **Pre-train Encoder**: Train autoencoder or use existing encoder
2. **Initialize Clusters**: Run k-means trials on encoder embeddings
3. **Train DEC**: Fine-tune clustering with KL divergence loss
4. **Evaluate**: Use built-in metrics and visualization tools

### Model I/O
- **Built-in Encoders**: Use `save()`/`load()` for DEC models with Coder-based encoders
- **Custom Encoders**: Use `save_generic()`/`load_generic()` for DEC models with any encoder
- **Important**: These functions save/load complete DEC models (encoder + cluster centroids), not standalone encoders/decoders
- **Flexibility**: Support for sequential encoders and device mapping

## Code Style Guidelines

### Documentation Style
- **Google Style Docstrings**: All docstrings follow Google Python Style Guide
  - Use "Args:" not "Arguments:" or "Parameters:"
  - Use "Returns:" not "Return:" or "Yields:"
  - No type information in docstrings (only in type hints)
  - No RST formatting (no `:class:`, `:func:`, backticks)
  - Use plain text for cross-references: "See train_ae_model() for details"
- **Type Hints**: Types specified in function signatures, not in docstrings
- **Examples**: Every public function/class includes usage examples with >>>
- **Notes/Warnings**: Use "Note:" and "Warning:" for important information

### Import Patterns
- Internal imports use relative imports within package
- External dependencies clearly separated
- Module-level logging with `logger = logging.getLogger(__name__)`

### Configuration Pattern
- Dataclass-based configurations (`@dataclass(frozen=True)`)
- `to_dict()`/`from_dict()` methods for serialization
- Registry pattern for extensible components (activations)

### Error Handling
- Assertion-based validation for internal consistency
- Clear error messages for unsupported operations
- Graceful handling of edge cases in data processing

## Testing Strategy

**Note**: No formal testing infrastructure detected. The project appears to rely on:
- Example notebook for integration testing
- Manual validation through the MNIST workflow
- Type hints and runtime assertions for validation

## Key Implementation Details

### Documentation Coverage
- **Module-level docs**: All __init__.py files contain package/module overviews
- **Class documentation**: Every public class has comprehensive docstrings with examples
- **Function documentation**: All public functions documented with Args, Returns, Examples
- **README**: Complete usage guide with API reference, best practices, and examples
- **Consistency**: All docstrings follow Google style format

### Clustering Algorithm
- Uses Student's t-distribution for soft assignments
- KL divergence between soft assignments and target distribution
- Target distribution updated iteratively to sharpen clusters

### Training Infrastructure
- Generic epoch runners work for both autoencoders and DEC
- History tracking with efficient enum-based key system
- Support for validation loops and early stopping

### Visualization Features
- Loss history plotting with seaborn integration
- 2D embedding visualization (UMAP, t-SNE, PCA)
- Cluster assignment tracking during training

## Security Considerations

- No explicit security measures implemented
- Standard PyTorch model loading/saving practices
- No input validation for file paths in I/O operations
- Relies on PyTorch's built-in security for model serialization

## Performance Notes

- Efficient history tracking using enums instead of strings
- Batch-wise processing with configurable data loaders
- Support for GPU acceleration through PyTorch
- Memory-efficient implementation of clustering operations

## Dependencies and Compatibility

- **PyTorch**: >= 2.6.0 (core dependency)
- **Python**: >= 3.10 (uses modern type hints and dataclasses)
- **scikit-learn**: For k-means initialization and metrics
- **pandas**: For data manipulation and history tracking
- **seaborn/umap**: Optional visualization dependencies