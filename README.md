# DEC-torch

A modular PyTorch toolkit for Deep Embedded Clustering (DEC), an unsupervised clustering method via deep representation learning.

[![License](https://img.shields.io/github/license/timurci/DEC-torch?style=flat-square)](LICENSE)

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Representation Learning](#2-representation-learning)
  - [3. Cluster Initialization](#3-cluster-initialization)
  - [4. Training DEC](#4-training-dec)
  - [5. Model Persistence](#5-model-persistence)
- [Examples](#examples)

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

## Quick Start

Here's a minimal complete example:

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from dec_torch import DEC, init_clusters
from dec_torch.autoencoder import AutoEncoder, AutoEncoderConfig
from dec_torch.dec import KLDivLoss

# 1. Prepare data
data = torch.randn(1000, 784)  # Your dataset
dataset = TensorDataset(data)
train_loader = DataLoader(dataset, batch_size=64, shuffle=False)

# 2. Create and train an autoencoder for representation learning
config = AutoEncoderConfig.build(
    input_dim=784,
    latent_dim=128,
    hidden_dims=[500, 500, 2000],
    hidden_activation='relu',
    output_activation='linear'
)
ae = AutoEncoder(config)
optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

# Train the autoencoder
ae_history = ae.fit(train_loader, optimizer, loss_fn, n_epoch=50)
encoder = ae.encoder  # Extract trained encoder

# 3. Initialize cluster centroids using k-means
embeddings = encoder(data)
centroids = init_clusters(embeddings.detach().cpu(), n_clusters=10)

# 4. Create and train DEC model
dec_model = DEC(encoder=encoder, centroids=centroids, alpha=1.0)
dec_optimizer = torch.optim.SGD(dec_model.parameters(), lr=0.001)
dec_loss_fn = KLDivLoss()

# Train DEC
dec_history = dec_model.fit(train_loader, dec_optimizer, dec_loss_fn)

# 5. Get cluster predictions
dec_model.eval()
with torch.no_grad():
    cluster_assignments = torch.argmax(dec_model(data), dim=1)
```

## Detailed Usage

### 1. Data Preparation

For DEC training, your DataLoader should return input data only (or input-target pairs). For cluster assignment tracking, use `shuffle=False`:

```python
from torch.utils.data import DataLoader, TensorDataset

# For autoencoder pre-training (can shuffle)
train_loader_ae = DataLoader(dataset, batch_size=64, shuffle=True)

# For DEC training (should not shuffle to track reassignments)
train_loader_dec = DataLoader(dataset, batch_size=64, shuffle=False)
```

### 2. Representation Learning

#### Basic Autoencoder

```python
from dec_torch.autoencoder import AutoEncoder, AutoEncoderConfig

# Symmetric architecture
config = AutoEncoderConfig.build(
    input_dim=784,
    latent_dim=128,
    hidden_dims=[500, 500, 2000],
    hidden_activation='relu',
    output_activation='linear'
)
ae = AutoEncoder(config)
```

#### Stacked Autoencoder

```python
from dec_torch.autoencoder import StackedAutoEncoder, StackedAutoEncoderConfig

# Progressive compression
config = StackedAutoEncoderConfig.build(
    input_dim=784,
    latent_dims=[500, 128, 10],
    hidden_activation='relu',
    last_encoder_activation='sigmoid',
    last_decoder_activation='linear',
    input_dropout=0.2
)
sae = StackedAutoEncoder(config)

# Greedy layer-wise pre-training
optimizer = torch.optim.Adam(sae.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()
sae_history = sae.greedy_fit(train_loader, optimizer, loss_fn, n_epoch=50)

# Fine-tune end-to-end
config_fine_tune = config.replace_input_dropout(None)
sae_fine_tune = StackedAutoEncoder(fine_tune_cfg)  # New model without dropout
sae_fine_tune.load_state_dict(sae.state_dict())

sae_fine_tune_history = sae_fine_tune.fit(train_loader, optimizer, loss_fn, n_epoch=50)
encoder = sae_fine_tune.encoder  # Extract the stacked encoder
```

#### Custom Encoders

Any PyTorch module can be used as an encoder:

```python
import torchvision.models as models
from dec_torch import DEC

# Use a pre-trained ResNet
resnet = models.resnet18(pretrained=True)
# Modify the final layer for your latent dimension
resnet.fc = torch.nn.Linear(resnet.fc.in_features, 128)

# Use with DEC
centroids = torch.randn(10, 128)  # 10 clusters
dec_model = DEC(encoder=resnet, centroids=centroids)
```

### 3. Cluster Initialization

#### Multiple K-Means Trials (Recommended)

```python
from dec_torch.dec import init_clusters_trials
import pandas as pd

# Run 20 k-means trials and select the best
embeddings = encoder(data).detach().numpy()
centroids_list, scores_df = init_clusters_trials(
    embeddings,
    n_clusters=10,
    n_trials=20
)

# Analyze quality metrics
print(scores_df.head())
#        SIL       CH  SIL-rank  CH-rank  combined-rank
# run-id                                              
# 0     0.45    120.5       1.0      2.0            3.0
# 1     0.42    135.2       2.0      1.0            3.0

# Select best centroids
best_centroids = centroids_list[scores_df.iloc[0].name]
```

#### Single K-Means

```python
from dec_torch.dec import init_clusters

# Simple k-means initialization
centroids = init_clusters(embeddings, n_clusters=10)
```

#### Random Initialization (Not Recommended)

```python
from dec_torch.dec import init_clusters_random

# Random centroids - use only for testing
centroids = init_clusters_random(
    n_clusters=10,
    latent_dim=128,
    mean=0.0,
    std=1.0
)
```

### 4. Training DEC

```python
from dec_torch.dec import DEC, KLDivLoss

# Create DEC model
dec_model = DEC(
    encoder=encoder,
    centroids=initial_centroids,
    alpha=1.0  # degrees of freedom
)

# Loss function and optimizer
loss_fn = KLDivLoss(reduction="batchmean")
optimizer = torch.optim.SGD(
    dec_model.parameters(),
    lr=0.001,
    momentum=0.9
)

# Training parameters
history = dec_model.fit(
    train_loader=train_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    tolerance=0.01,  # Stop when <1% samples change clusters
    max_epoch=1000,  # Maximum epochs
    verbose=True
)
```

### 5. Model Persistence

#### For DEC models with built-in encoders

```python
from dec_torch.dec.io import save, load

# Save
dec_model.fit(train_loader, optimizer, loss_fn)
save(dec_model, "encoder.pth", "centroids.pth")

# Load later
dec_model = load("encoder.pth", "centroids.pth", alpha=1.0)

# For sequential encoders (StackedAutoEncoder)
save(dec_model, "encoder.pth", "centroids.pth")
dec_model = load(
    "encoder.pth",
    "centroids.pth",
    sequential_encoder=True,  # Important!
    alpha=1.0
)
```

#### For DEC models with custom encoders

```python
from dec_torch.dec.io import save_generic, load_generic

# Save
dec_model.fit(train_loader, optimizer, loss_fn)
save_generic(dec_model, "encoder.pth", "centroids.pth")

# Load (must provide initialized encoder)
encoder = MyCustomEncoder()  # Initialize with same architecture
dec_model = load_generic(
    "encoder.pth",
    "centroids.pth",
    encoder_instance=encoder,
    alpha=1.0,
    map_location="cpu"  # Optional device mapping
)
```

## Examples

### Visualizing Training Progress

```python
from dec_torch.utils.visualization import loss_plot, cluster_plot
import matplotlib.pyplot as plt

# Plot training curves
fig, ax = plt.subplots(figsize=(10, 6))  # Optional
loss_plot(dec_history, ax=ax)
ax.set_title("DEC Training Progress")
plt.show()

# Visualize clusters
embeddings, _ = extract_all_data(train_loader, transform=encoder)
with torch.no_grad():
    assignments = dec_model(embeddings).argmax(dim=1).numpy()
centroids = dec_model.centroids.detach().numpy()

cluster_plot(
    embeddings.numpy(),
    labels=assignments,
    centroids=centroids,
    reduction='umap',
)
plt.show()
```
