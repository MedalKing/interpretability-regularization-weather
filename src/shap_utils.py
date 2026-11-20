from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from src.alignment import AlignmentMetrics, compute_alignment_metrics

try:
    import shap as _shap
except ImportError:
    _shap = None


@dataclass
class ShapComputation:
    """
    Container for SHAP outputs for a regression model.
    """
    shap_values: np.ndarray               # shape (n_samples, d)
    expected_value: float
    importance_mean_abs: np.ndarray       # shape (d,)
    importance_mean_abs_norm: np.ndarray  # shape (d,)


class _ShapRegressionWrapper(nn.Module):
    """
    Wrap a regression model so SHAP always sees output shape (B, 1).

    For GatedMLRegressor we want to force use_gate=True.
    For plain MLPRegressor the wrapper falls back to model(x).
    """

    def __init__(self, model: nn.Module, *, use_gate: bool = True) -> None:
        super().__init__()
        self.model = model
        self.use_gate = use_gate

    def forward(self, x: Tensor) -> Tensor:
        try:
            out: Tensor = self.model(x, use_gate=self.use_gate)
        except TypeError:
            out = self.model(x)

        if out.ndim == 1:
            out = out.unsqueeze(-1)
        return out


def sample_rows(
    X: np.ndarray,
    n: int,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample n rows without replacement from X.
    Returns (X_sampled, indices).
    """
    X = np.asarray(X)
    n = min(n, len(X))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx], idx


def _unwrap_shap_values(raw: Any) -> np.ndarray:
    """
    Make SHAP outputs consistent for single-output regression.

    Depending on the SHAP explainer version/method, raw SHAP values can be:
      - ndarray of shape (n, d)
      - list with one ndarray
      - ndarray of shape (n, d, 1)
      - ndarray of shape (1, n, d)
    """
    if isinstance(raw, list):
        raw = raw[0]

    arr = np.asarray(raw, dtype=np.float64)

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    return arr.astype(np.float64)


def _unwrap_expected_value(raw: Any) -> float:
    arr = np.asarray(raw, dtype=np.float64).ravel()
    return float(arr[0])


def compute_global_shap_importance(
    shap_values: np.ndarray,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """
    Convert local SHAP values into a global importance vector via mean absolute SHAP.
    """
    shap_arr = np.asarray(shap_values, dtype=np.float64)

    if shap_arr.ndim == 3 and shap_arr.shape[-1] == 1:
        shap_arr = shap_arr[..., 0]

    imp = np.mean(np.abs(shap_arr), axis=0)

    if normalize:
        imp = imp / (imp.sum() + 1e-8)

    return imp.astype(np.float64)


def make_global_shap_table(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Create a per-feature table from SHAP values.
    """
    mean_abs = compute_global_shap_importance(shap_values, normalize=False)
    mean_abs_norm = compute_global_shap_importance(shap_values, normalize=True)

    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": mean_abs,
                "mean_abs_shap_norm": mean_abs_norm,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def compute_shap_alignment(
    shap_values: np.ndarray,
    z: np.ndarray,
) -> AlignmentMetrics:
    """
    Compare normalized global SHAP importance to the pseudo-label vector z.
    """
    imp = compute_global_shap_importance(shap_values, normalize=True)

    z64 = np.asarray(z, dtype=np.float64).ravel()
    z64 = z64 / (z64.sum() + 1e-8)

    return compute_alignment_metrics(imp, z64)


def compute_shap_values(
    model: nn.Module,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    *,
    device: str = "cpu",
    method: Literal["deep", "gradient", "kernel"] = "deep",
    use_gate: bool = True,
    kernel_nsamples: str | int = "auto",
) -> ShapComputation:
    """
    Compute SHAP values for a trained regression model.

    Parameters
    ----------
    model:
        Trained PyTorch model.
    X_background:
        Background/reference samples, shape (m, d).
    X_explain:
        Samples to explain, shape (n, d).
    device:
        "cpu" or "cuda".
    method:
        - "deep": SHAP DeepExplainer
        - "gradient": SHAP GradientExplainer
        - "kernel": SHAP KernelExplainer (slower, model-agnostic fallback)
    use_gate:
        For GatedMLRegressor, use_gate=True ensures explanations are for the gated model.
    """
    if _shap is None:
        raise ImportError(
            "The `shap` package is not installed. Install it with `pip install shap`."
        )

    X_background = np.asarray(X_background, dtype=np.float32)
    X_explain = np.asarray(X_explain, dtype=np.float32)

    model = model.to(device)
    model.eval()

    wrapped = _ShapRegressionWrapper(model, use_gate=use_gate).to(device)
    wrapped.eval()

    if method in {"deep", "gradient"}:
        background_t = torch.as_tensor(X_background, dtype=torch.float32, device=device)
        explain_t = torch.as_tensor(X_explain, dtype=torch.float32, device=device)

        # SHAP's PyTorch API supports tensors here, but some type stubs are too narrow.
        wrapped_any = cast(Any, wrapped)
        background_any = cast(Any, background_t)
        explain_any = cast(Any, explain_t)

        explainer: Any
        if method == "deep":
            explainer = _shap.DeepExplainer(wrapped_any, background_any)
        else:
            explainer = _shap.GradientExplainer(wrapped_any, background_any)

        raw_shap = explainer.shap_values(explain_any)
        shap_values = _unwrap_shap_values(raw_shap)

        expected_value_raw = getattr(explainer, "expected_value", None)
        if expected_value_raw is not None:
            expected_value = _unwrap_expected_value(expected_value_raw)
        else:
            with torch.no_grad():
                base_preds = wrapped(background_t).squeeze(-1).detach().cpu().numpy()
            expected_value = float(np.mean(base_preds))

    elif method == "kernel":
        def predict_fn(X_np: np.ndarray) -> np.ndarray:
            X_t = torch.as_tensor(X_np, dtype=torch.float32, device=device)
            wrapped.eval()
            with torch.no_grad():
                preds = wrapped(X_t).squeeze(-1)
            return preds.detach().cpu().numpy()

        explainer = _shap.KernelExplainer(predict_fn, X_background)
        raw_shap = explainer.shap_values(X_explain, nsamples=kernel_nsamples)
        shap_values = _unwrap_shap_values(raw_shap)
        expected_value = _unwrap_expected_value(getattr(explainer, "expected_value"))

    else:
        raise ValueError("method must be one of: 'deep', 'gradient', 'kernel'")

    importance_mean_abs = compute_global_shap_importance(shap_values, normalize=False)
    importance_mean_abs_norm = compute_global_shap_importance(shap_values, normalize=True)

    return ShapComputation(
        shap_values=shap_values,
        expected_value=expected_value,
        importance_mean_abs=importance_mean_abs,
        importance_mean_abs_norm=importance_mean_abs_norm,
    )