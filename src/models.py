from __future__ import annotations

from typing import Iterable, List

import torch
from torch import Tensor, nn


class MLPRegressor(nn.Module):
    """
    Plain MLP regressor for tabular inputs.

    This is the baseline ERM model:
      - No gating
      - No interpretability-specific regularisation
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int] = (64, 64),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        prev_dim = input_dim

        # Build a standard MLP: (Linear -> ReLU -> optional Dropout) repeated
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        # Final scalar output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, d)

        Returns
        -------
        Tensor, shape (B,)
        """
        out: Tensor = self.net(x)      # (B, 1)
        return out.squeeze(-1)         # (B,)


class GatedMLRegressor(nn.Module):
    """
    MLP regressor with a global learnable input gate g in (0,1)^d.

    Forward pass (when use_gate=True):
        g = sigmoid(_gate)
        x_tilde = g ⊙ x
        y_hat = net(x_tilde)

    Interpretation:
      - Each g_j can be viewed as a global feature "throughput" coefficient.
      - Small g_j suppresses feature j; large g_j keeps feature j.

    Important caveat (relevant to your earlier findings):
      - In a generic MLP, g is not uniquely identifiable because the first-layer
        weights can often compensate (e.g., scaling g down and scaling weights up).
      - Therefore, a loss that only matches g to a pseudo-label z can sometimes be
        satisfied without strongly changing the learned function.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int] = (64, 64),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Raw gate parameters α; the actual gate is g = sigmoid(α)
        # Shape: (d,)
        self._gate = nn.Parameter(torch.ones(input_dim))

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, use_gate: bool = True) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, d)
        use_gate : bool
            If True, apply the global gate: x <- g ⊙ x.

        Returns
        -------
        Tensor, shape (B,)
        """
        if use_gate:
            g = self.g          # (d,)
            x = x * g           # broadcasted elementwise product, (B, d)

        out: Tensor = self.net(x)      # (B, 1)
        return out.squeeze(-1)         # (B,)

    @property
    def g(self) -> Tensor:
        """
        Current gating vector g in (0,1)^d, computed as sigmoid(_gate).
        """
        return torch.sigmoid(self._gate)