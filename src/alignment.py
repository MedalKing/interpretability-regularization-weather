from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, cast

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import torch
from torch import nn, Tensor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import permutation_importance
"""
Alignment / interpretability metrics (feature-importance part)

This file intentionally distinguishes:

(1) Parameter-level alignment:
    - Compare the learned gate vector g to the pseudo-label vector z.
    - This answers: "Did the gate parameter move toward the pseudo-label?"

(2) Function-level alignment:
    - Compare a *model-behaviour* importance vector to z.
    - Two proxies are used:
        - Occlusion importance: ablate feature j and measure prediction change.
        - Permutation importance: shuffle feature j and see how test loss changes.

In earlier experiments, parameter-level alignment could improve without
function-level alignment improving much. Function-level metrics are therefore
treated as the more reliable indicator of “feature usage”.
"""

@dataclass
class AlignmentMetrics:
    """
    Simple bundle of alignment metrics between two d-dimensional vectors.
    """
    l2_distance: float
    cosine_similarity: float
    spearman_rank_corr: float


def compute_alignment_metrics(g: np.ndarray, z: np.ndarray) -> AlignmentMetrics:
    """
    Compare two vectors g and z using:
      - L2 distance
      - cosine similarity
      - Spearman rank correlation

    Used for:
      - parameter-level alignment: g (gate) vs z (pseudo-labels)
      - function-level alignment: importance vector (occlusion/permutation) vs z
    """
    g64 = g.astype(np.float64).ravel()
    z64 = z.astype(np.float64).ravel()

    l2_dist = float(np.linalg.norm(g64 - z64))
    cos_sim = float(np.dot(g64, z64) / (np.linalg.norm(g64) * np.linalg.norm(z64) + 1e-8))

    spearman_res = spearmanr(g64, z64)
    if hasattr(spearman_res, "correlation"):
        rank_corr = float(getattr(spearman_res, "correlation"))
    else:
        corr_tuple = cast(Tuple[float, float], spearman_res)
        rank_corr = float(corr_tuple[0])

    return AlignmentMetrics(
        l2_distance=l2_dist,
        cosine_similarity=cos_sim,
        spearman_rank_corr=rank_corr,
    )

# scikit-learn's permutation_importance expects an estimator with a .predict(X) method.
# TorchRegressorWrapper adapts a PyTorch model to that interface without re-training it.
class TorchRegressorWrapper(BaseEstimator, RegressorMixin):
    """
    scikit-learn style wrapper so permutation_importance can call model.predict(X).

    Important:
      - This wrapper does NOT train the model. It assumes model is already trained.
    """
    def __init__(self, model: nn.Module, device: str = "cpu") -> None:
        self.model = model
        self.device = device

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "TorchRegressorWrapper":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            preds: Tensor = self.model(X_t)
        return preds.cpu().numpy().ravel()


def compute_perm_alignment(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    device: str = "cpu",
    n_repeats: int = 20,
    random_state: int = 0,
) -> AlignmentMetrics:
    """
    Compute alignment between z and permutation importances.

    permutation_importance returns a mean importance score per feature (how much the
    test loss worsens when the feature is shuffled). This is a function-level
    importance proxy.

    Note:
      - For small d this is fine; for larger d it becomes expensive.
    """
    # Permutation importance intuition:
    # - If shuffling feature j breaks predictions (loss increases), feature j matters.
    # - This is a black-box, function-level measure (not a parameter-level proxy).
    # - It is typically slower than occlusion, especially for many repeats or many features.
    wrapper = TorchRegressorWrapper(model, device=device)

    result = permutation_importance(
        wrapper,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="neg_mean_squared_error",
    )

    s = np.asarray(result["importances_mean"], dtype=np.float64).ravel()
    z64 = z.astype(np.float64).ravel()

    # For L2/cosine, compare L2-normalized versions (scale-invariant)
    eps = 1e-8
    s_unit = s / (np.linalg.norm(s) + eps)
    z_unit = z64 / (np.linalg.norm(z64) + eps)

    l2_dist = float(np.linalg.norm(s_unit - z_unit))
    cos_sim = float(np.dot(s_unit, z_unit))

    # For Spearman, compare the raw vectors (rank-based)
    spearman_res = spearmanr(s, z64)
    if hasattr(spearman_res, "correlation"):
        rank_corr = float(getattr(spearman_res, "correlation"))
    else:
        corr_tuple = cast(Tuple[float, float], spearman_res)
        rank_corr = float(corr_tuple[0])

    return AlignmentMetrics(
        l2_distance=l2_dist,
        cosine_similarity=cos_sim,
        spearman_rank_corr=rank_corr,
    )


def compute_occlusion_importance(
    model: nn.Module,
    X: np.ndarray,
    *,
    device: str = "cpu",
    batch_size: int = 256,
    baseline_value: float = 0.0,
    use_gate: bool | None = None,
) -> np.ndarray:
    """
    Compute function-level occlusion importance on a full dataset (evaluation only).

    For each feature j:
      - set x_j := baseline_value across the full dataset
      - measure average squared change in predictions:
            imp_j = mean( (f(x) - f(x with x_j masked))^2 )

    Returns a normalized importance vector (sum=1), shape (d,).

    Notes
    -----
    baseline_value=0.0 is appropriate for StandardScaler-normalized inputs.
    """
    model.eval()
    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape

    delta_sum = np.zeros(d, dtype=np.float64)
    count = 0

    with torch.no_grad():
        for start in range(0, n, batch_size):
            xb = torch.as_tensor(X[start : start + batch_size], device=device)
            preds: Tensor = _forward_model(model, xb, use_gate=use_gate)

            B = xb.shape[0]
            for j in range(d):
                xb_mask = xb.clone()
                xb_mask[:, j] = baseline_value
                preds_mask: Tensor = _forward_model(model, xb_mask, use_gate=use_gate)
                delta = (preds - preds_mask) ** 2
                delta_sum[j] += float(delta.sum().item())

            count += B

    imp = delta_sum / max(count, 1)
    imp = imp / (imp.sum() + 1e-8)
    return imp.astype(np.float64)

def _forward_model(
    model: nn.Module,
    x: Tensor,
    *,
    use_gate: bool | None = None,
) -> Tensor:
    """
    Forward helper that works for both:
      - MLPRegressor: model(x)
      - GatedMLRegressor: model(x, use_gate=...)
    """
    try:
        if use_gate is None:
            out: Tensor = model(x)
        else:
            out = model(x, use_gate=use_gate)
    except TypeError:
        out = model(x)

    return out

def compute_occlusion_alignment(
    model: nn.Module,
    X: np.ndarray,
    z: np.ndarray,
    *,
    device: str = "cpu",
    batch_size: int = 256,
    baseline_value: float = 0.0,
    use_gate: bool | None = None,
) -> AlignmentMetrics:
    """
    Compute alignment between z and global occlusion importance.
    """
    imp = compute_occlusion_importance(
        model=model,
        X=X,
        device=device,
        batch_size=batch_size,
        baseline_value=baseline_value,
        use_gate=use_gate,
    )

    z64 = np.asarray(z, dtype=np.float64).ravel()
    z64 = z64 / (z64.sum() + 1e-8)

    return compute_alignment_metrics(imp, z64)


def compute_e_star_scores(
    sweep_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    *,
    mse_col: str = "test_mse",
    cos_col: str = "occ_cosine_mean",
) -> pd.DataFrame:
    """
    Compute an EERM-inspired combined score (E_star) from:
      - L_e: normalized test risk (MSE)
      - H_e: normalized misalignment proxy derived from a cosine column

    This is a heuristic summary score: it becomes large when both (1-L_e) and (1-H_e)
    are large, and small when either is poor.
    """
    eps = 1e-8

    mse_vals = sweep_df[mse_col]
    L_min, L_max = float(mse_vals.min()), float(mse_vals.max())

    out = agg_df.copy()

    # Normalize risk to [0,1]
    if "test_mse_mean" in out.columns:
        out["L_e"] = (out["test_mse_mean"] - L_min) / (L_max - L_min + eps)
    else:
        out["L_e"] = (out["test_rmse_mean"] ** 2 - L_min) / (L_max - L_min + eps)

    # Normalize cosine alignment to [0,1] and define misalignment H_e = 1 - alignment
    cos_min, cos_max = float(out[cos_col].min()), float(out[cos_col].max())
    cos_norm = (out[cos_col] - cos_min) / (cos_max - cos_min + eps)
    out["H_e"] = 1.0 - cos_norm

    numerator = (1.0 - out["L_e"]) * (1.0 - out["H_e"]) * 2.0
    denominator = 2.0 - out["L_e"] - out["H_e"] + eps
    out["E_star"] = numerator / denominator

    return out