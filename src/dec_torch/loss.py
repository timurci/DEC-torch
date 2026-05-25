"""Custom loss functions for DEC-torch."""

import torch
from torch import nn


class KLDivLoss(nn.Module):
    """Custom KL Divergence Loss module for DEC training.

    This loss function computes the KL divergence between the soft assignment
    distribution (q) and the target distribution (p), which is used to train the
    DEC model. The module uses mathematically accurate reduction by default as
    suggested in PyTorch Documentation.

    The module automatically applies log transformation to Q as required by
    nn.functional.kl_div and adds a small epsilon for numerical stability.

    Args:
        reduction: Reduction method for the loss. Defaults to "batchmean" for
            correct KL divergence calculation.
        eps: Small constant added to Q before log transformation for numerical
            stability. Defaults to 1e-10.

    Attributes:
        reduction: The reduction method being used.
        eps: The epsilon value for numerical stability.

    Example:
        >>> loss_fn = KLDivLoss(reduction="batchmean", eps=1e-10)
        >>> q = torch.tensor([[0.1, 0.7, 0.2], [0.3, 0.3, 0.4]])
        >>> p = torch.tensor([[0.05, 0.85, 0.10], [0.4, 0.3, 0.3]])
        >>> loss = loss_fn(q, p)
        >>> print(loss.item())
        0.123...

    Note:
        See PyTorch documentation for nn.functional.kl_div for details about
        the reduction methods and the mathematical formulation.
    """

    def __init__(self, reduction: str = "batchmean", eps: float = 1e-10) -> None:
        """Initialize the KL Divergence loss function.

        Args:
            reduction: Reduction method for the loss. Defaults to "batchmean".
            eps: Epsilon for numerical stability. Defaults to 1e-10.
        """
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence between Q and P distributions.

        Args:
            q: Soft assignment distribution (batch_size, n_clusters).
            p: Target distribution (batch_size, n_clusters).

        Returns:
            KL divergence loss value.

        Note:
            Q is automatically transformed to log space and epsilon is added for
            numerical stability before computing the divergence.
        """
        q = (q + self.eps).log()  # kl_div requires Q to be in log space.
        return nn.functional.kl_div(q, p, reduction=self.reduction)
