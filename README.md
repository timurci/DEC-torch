# DEC-torch

A modular PyTorch toolkit for Deep Embedded Clustering (DEC), an unsupervised clustering method via deep representation learning.

## Features

The core `DEC` class wraps a pre-trained encoder (any `nn.Module`) and trains to improve clustering in latent space. The package provides:

- **Plug-and-play with any encoder**: Use custom models, or built-in autoencoders
- **Multiple centroid initializations**: Compare k-means trials or use random centroid initialization
- **Robust model I/O**: Save/load utilities for both DEC and built-in autoencoders
- **Tracking & visualization**: History objects record losses/metrics; integrated cluster visualization
- **Extensible design**: All components are modular and can be swapped, extended, or customized

> **Note**: In the original DEC study ([Xie et al., 2016](https://doi.org/10.48550/arXiv.1511.06335)), "DEC" refers to the complete workflow combining representation learning *and* clustering. In this package, `DEC` refers only to the clustering model, under the assumption that suitable representation learning has already been performed in the encoder.

## Installation

Clone and install in editable mode:

```bash
git clone https://github.com/timurci/DEC-torch.git
cd DEC-torch
pip install -e .
```

## Minimal Usage Example

### 1. Initialize Cluster Centroids

```python
# Get embeddings from your encoder
# training_data: torch.Tensor of shape (n_samples, n_features)
embeddings = encoder_model(training_data)

# Run multiple k-means trials and select the best
centroids_list, scores_df = DEC.init_clusters_trials(
    embeddings, n_clusters=5, n_trials=20
)
best_centroids = centroids_list[scores_df.iloc[0].name]
```

### 2. Train DEC Model

```python
from dec_torch import DEC

dec_model = DEC(encoder=encoder_model, centroids=best_centroids)

# Built-in training loop
import torch
from dec_torch.dec.dec import KLDivLoss

loss_fn = KLDivLoss()
optimizer = torch.optim.SGD(dec_model.parameters(), lr=0.001)
history = dec_model.fit(
    training_loader,
    optimizer,
    loss_fn,
    val_loader=validation_loader
)

# Predict cluster labels
dec_model.eval()
with torch.no_grad():
    labels = torch.argmax(dec_model(new_data), dim=1)
```

### 3. Save and Load a DEC Model

**For built-in encoders extracted from `Autoencoder` or `StackedAutoencoder`:**
```python
from dec_torch.dec.io import save, load

save(dec_model, "encoder.pth", "centroids.pth")
dec_model = load("encoder.pth", "centroids.pth")
```

⚠️ Set `sequential_encoder=True` in `load()` if the encoder is extracted from `StackedAutoencoder`

**For custom encoders:**
```python
from dec_torch.dec.io import save_generic, load_generic

save_generic(dec_model, "encoder.pth", "centroids.pth")
dec_model = load_generic("encoder.pth", "centroids.pth", encoder_model)
```

Additional keyword arguments are passed to `torch.save` and `torch.load` respectively (e.g., add `map_location="cpu"` in `load()` or `load_generic()` for device mapping).

## License

MIT License (see [LICENSE](./LICENSE)).