# src/training.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import copy

import numpy as np
from numpy.typing import ArrayLike

import pandas as pd
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from src.models import GatedMLRegressor
from src.data_loading import build_nextday_dataset, load_daily_clean_nuuksio

"""
High-level purpose of this file (feature-importance part)

This project trains a next-day temperature regressor on tabular inputs.

Three training regimes are implemented:

(1) Baseline ERM:
    Train a plain MLP to minimise mean squared error (MSE) on next-day tmax.

(2) Gated model + parameter-level gate matching:
    A learnable gate vector g in (0,1)^d multiplies inputs: x_tilde = g ⊙ x.
    Regulariser encourages g to match a pseudo-label vector z:
        L = MSE + λ_feat * ||g - z||^2
    Caveat: this regulariser matches a parameter, not necessarily the function's
    dependence on inputs. First-layer weights can sometimes compensate.

(3) Gated model + function-level occlusion regularisation:
    Importance is computed by ablation (how much does the prediction change if
    feature j is set to its baseline value?). Regulariser encourages the
    occlusion importance profile to match z:
        L = MSE + λ_feat * ||I_occ(x) - z||^2

A diagnostic mode also exists:
    Freeze network weights, train only the gate. Mainly confirms that the gate
    is actually connected to outputs.

Practical note on input scaling:
    Features are StandardScaler-normalised (fit on train only), so
    baseline_value=0.0 corresponds to "feature set to its mean".

New in this version:
    - optional gate-history tracking in all gated training loops
    - optional validation logging in gate-only diagnostics
    - histories now include total losses and can store per-feature g_j values
"""


# ---------------------------------------------------------------------------
# Config and small utilities
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainConfig:
    """
    Hyperparameters for training loops.

    Notes
    -----
    weight_decay is applied via AdamW (decoupled).
    Early stopping (if enabled) is based on validation loss.
    """
    lr: float
    batch_size: int
    eval_batch_size: int
    num_epochs: int
    weight_decay: float = 0.0
    early_stopping_patience: Optional[int] = None
    early_stopping_min_delta: float = 0.0


DEFAULT_TRAIN_CONFIG = TrainConfig(
    lr=1e-3,
    batch_size=64,
    eval_batch_size=256,
    num_epochs=100,
    weight_decay=1e-2,
    early_stopping_patience=10,
    early_stopping_min_delta=0.0,
)


@dataclass
class NextdayPreparedData:
    """
    Container for next-day regression data and loaders.

    The arrays X_* are scaled (StandardScaler fit on train only).
    """
    feature_cols: list[str]
    scaler: StandardScaler
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_loader: DataLoader[tuple[Tensor, ...]]
    val_loader: DataLoader[tuple[Tensor, ...]]
    test_loader: DataLoader[tuple[Tensor, ...]]


def prepare_nextday_regression_data(
    df_daily: pd.DataFrame,
    base_feature_cols: list[str],
    config: TrainConfig = DEFAULT_TRAIN_CONFIG,
    *,
    target_col: str = "tmax_next",
    add_rand_noise: bool = True,
    noise_col: str = "rand_noise",
    noise_seed: int = 42,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> NextdayPreparedData:
    """
    Build a next-day dataset with a temporal split and standardization.

    Steps:
      1) Sort by datetime
      2) Optionally add a random N(0,1) noise feature (sanity checks / stress tests)
      3) Build modelling dataframe where target is tmax shifted by -1
      4) Temporal split train/val/test
      5) Standardize X using StandardScaler fit on train only
      6) Wrap into TensorDatasets/DataLoaders
    """
    df = df_daily.sort_values("datetime").reset_index(drop=True).copy()
    feature_cols = list(base_feature_cols)

    if add_rand_noise:
        rng = np.random.default_rng(seed=noise_seed)
        df[noise_col] = rng.standard_normal(len(df))
        feature_cols.append(noise_col)

    df_model = build_nextday_dataset(df, feature_cols)

    n = len(df_model)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_df = df_model.iloc[:n_train].copy()
    val_df   = df_model.iloc[n_train : n_train + n_val].copy()
    test_df  = df_model.iloc[n_train + n_val :].copy()

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df[target_col].to_numpy(dtype=float)
    X_val   = val_df[feature_cols].to_numpy(dtype=float)
    y_val   = val_df[target_col].to_numpy(dtype=float)
    X_test  = test_df[feature_cols].to_numpy(dtype=float)
    y_test  = test_df[target_col].to_numpy(dtype=float)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    train_ds = make_tensor_dataset(X_train_s, y_train)
    val_ds   = make_tensor_dataset(X_val_s,   y_val)
    test_ds  = make_tensor_dataset(X_test_s,  y_test)

    train_loader: DataLoader[tuple[Tensor, ...]] = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True
    )
    val_loader: DataLoader[tuple[Tensor, ...]] = DataLoader(
        val_ds, batch_size=config.eval_batch_size, shuffle=False
    )
    test_loader: DataLoader[tuple[Tensor, ...]] = DataLoader(
        test_ds, batch_size=config.eval_batch_size, shuffle=False
    )

    return NextdayPreparedData(
        feature_cols=feature_cols,
        scaler=scaler,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        X_train=X_train_s,
        y_train=y_train,
        X_val=X_val_s,
        y_val=y_val,
        X_test=X_test_s,
        y_test=y_test,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )


def eval_loader_reg(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, ...]],
    device: str = "cpu",
) -> dict[str, float]:
    """
    Compute regression metrics (MSE, RMSE, MAE) on a DataLoader.
    """
    model.eval()
    preds_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []

    with torch.no_grad():
        for xb, yb in loader:
            preds: Tensor = model(xb.to(device))
            preds_all.append(preds.cpu().numpy())
            targets_all.append(yb.numpy())

    y_hat  = np.concatenate(preds_all).astype(np.float64).ravel()
    y_true = np.concatenate(targets_all).astype(np.float64).ravel()

    mse  = float(np.mean((y_hat - y_true) ** 2))
    mae  = float(np.mean(np.abs(y_hat - y_true)))
    rmse = float(np.sqrt(mse))
    return {"mse": mse, "rmse": rmse, "mae": mae}


def make_tensor_dataset(X: ArrayLike, y: ArrayLike) -> TensorDataset:
    """Convert numpy arrays to float32 tensors wrapped in a TensorDataset."""
    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    return TensorDataset(torch.from_numpy(X_arr), torch.from_numpy(y_arr))


# ---------------------------------------------------------------------------
# Private helpers shared across all training loops
# ---------------------------------------------------------------------------

class _EarlyStopper:
    """
    Encapsulates early-stopping bookkeeping so training loops don't repeat it.
    """

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_val: float = float("inf")
        self._epochs_no_improve: int = 0
        self.best_state: Optional[dict[str, Tensor]] = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """
        Record val_loss; save model state dict when val_loss improves.
        Returns True when training should stop.
        """
        if val_loss + self.min_delta < self.best_val:
            self.best_val = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self._epochs_no_improve = 0
            return False
        self._epochs_no_improve += 1
        return self._epochs_no_improve >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        """Load the best-seen state back into model (no-op if no state was saved)."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def _make_gated_optimizer(
    model: GatedMLRegressor,
    config: TrainConfig,
) -> torch.optim.AdamW:
    """
    AdamW with weight decay on network weights, but *zero* weight decay on _gate.

    Rationale: AdamW's decoupled weight decay pulls raw parameters toward zero.
    For _gate that would bias g = sigmoid(_gate) toward 0.5, which interferes
    with driving g toward a user-specified pseudo-label z.
    """
    gate_params  = [model._gate]
    other_params = [p for n, p in model.named_parameters() if n != "_gate"]
    return torch.optim.AdamW(
        [
            {"params": other_params, "weight_decay": config.weight_decay},
            {"params": gate_params,  "weight_decay": 0.0},
        ],
        lr=config.lr,
    )


def _should_log(epoch: int, num_epochs: int) -> bool:
    """True on the first epoch, every 10th epoch, and the final epoch."""
    return (epoch + 1) == 1 or (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs


def _append_gate_history(
    row: dict[str, float],
    model: GatedMLRegressor,
    gate_names: Optional[list[str]] = None,
) -> None:
    """
    Add the current gate values to a history row.

    Each gate value is stored under a key of the form ``g_<feature_name>``.
    """
    g_np = model.g.detach().cpu().numpy().astype(np.float64)
    if gate_names is None:
        gate_names = [str(j) for j in range(len(g_np))]
    for name, value in zip(gate_names, g_np):
        row[f"g_{name}"] = float(value)


# ---------------------------------------------------------------------------
# Baseline ERM training loop
# ---------------------------------------------------------------------------

def train_baseline(
    model: nn.Module,
    train_loader: DataLoader[tuple[Tensor, ...]],
    val_loader: Optional[DataLoader[tuple[Tensor, ...]]],
    config: TrainConfig,
    device: str = "cpu",
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    """
    Baseline ERM loop:
      - optimizer: AdamW
      - loss: MSE
      - early stopping on validation loss (if val_loader is provided)
    """
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    loss_fn = nn.MSELoss()
    history: List[Dict[str, float]] = []

    patience = config.early_stopping_patience
    if val_loader is not None and patience is not None:
        stopper: Optional[_EarlyStopper] = _EarlyStopper(
            patience,
            config.early_stopping_min_delta,
        )
    else:
        stopper = None

    for epoch in range(config.num_epochs):
        model.train()
        running_loss, n_train = 0.0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss: Tensor = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            n_train += xb.size(0)

        train_loss = running_loss / max(n_train, 1)

        val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            v_running, n_val = 0.0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    v_running += loss_fn(model(xb), yb).item() * xb.size(0)
                    n_val += xb.size(0)
            val_loss = v_running / max(n_val, 1)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if stopper is not None and not np.isnan(val_loss):
            if stopper.step(val_loss, model):
                print(f"Early stopping at epoch {epoch+1:03d} | best_val_loss={stopper.best_val:.4f}")
                break

        if _should_log(epoch, config.num_epochs):
            print(f"Epoch {epoch+1:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    if stopper is not None:
        stopper.restore_best(model)

    return model, history


# ---------------------------------------------------------------------------
# Parameter-level gate-matching regularizer
# ---------------------------------------------------------------------------

def train_gated_feature_importance(
    model: GatedMLRegressor,
    train_loader: DataLoader[tuple[Tensor, ...]],
    val_loader: Optional[DataLoader[tuple[Tensor, ...]]],
    z: Tensor,
    lambda_feat: float,
    config: TrainConfig,
    device: str = "cpu",
    *,
    track_gate_history: bool = False,
    gate_names: Optional[list[str]] = None,
) -> Tuple[GatedMLRegressor, List[Dict[str, float]]]:
    """
    Train a gated MLP with a parameter-level feature regulariser.

    Objective:
        L_total = MSE + lambda_feat * mean((g - z)^2)

    Early stopping monitors validation total loss:
        val_total = val_task + lambda_feat * val_feat

    Important:
      g is a parameter vector; matching g to z does not necessarily imply the
      function uses features accordingly when the rest of the network is
      trainable.
    """
    model.to(device)
    dtype = next(model.parameters()).dtype
    z = z.to(device=device, dtype=dtype)

    optimizer = _make_gated_optimizer(model, config)
    mse_loss  = nn.MSELoss()
    history: List[Dict[str, float]] = []

    patience = config.early_stopping_patience
    if val_loader is not None and patience is not None:
        stopper: Optional[_EarlyStopper] = _EarlyStopper(
            patience,
            config.early_stopping_min_delta,
        )
    else:
        stopper = None

    for epoch in range(config.num_epochs):
        model.train()
        running_task, running_feat, n_train = 0.0, 0.0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            task_loss: Tensor = mse_loss(model(xb, use_gate=True), yb)
            feat_loss: Tensor = torch.mean((model.g - z) ** 2)
            total_loss: Tensor = task_loss + lambda_feat * feat_loss
            total_loss.backward()
            optimizer.step()

            bs = xb.size(0)
            running_task += task_loss.item() * bs
            running_feat += feat_loss.item() * bs
            n_train += bs

        train_task = running_task / max(n_train, 1)
        train_feat = running_feat / max(n_train, 1)
        train_total = train_task + lambda_feat * train_feat

        val_task, val_feat, val_total = float("nan"), float("nan"), float("nan")
        if val_loader is not None:
            model.eval()
            v_task, v_feat, n_val = 0.0, 0.0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    bs = xb.size(0)
                    v_task += mse_loss(model(xb, use_gate=True), yb).item() * bs
                    v_feat += torch.mean((model.g - z) ** 2).item() * bs
                    n_val += bs
            val_task = v_task / max(n_val, 1)
            val_feat = v_feat / max(n_val, 1)
            val_total = val_task + lambda_feat * val_feat

        row: Dict[str, float] = {
            "epoch": float(epoch),
            "train_task_loss": train_task,
            "train_feat_loss": train_feat,
            "train_total_loss": train_total,
            "val_task_loss": val_task,
            "val_feat_loss": val_feat,
            "val_total_loss": val_total,
        }
        if track_gate_history:
            _append_gate_history(row, model, gate_names)
        history.append(row)

        if stopper is not None and not np.isnan(val_total):
            if stopper.step(val_total, model):
                print(f"Early stopping at epoch {epoch+1:03d} | best_val_total={stopper.best_val:.4f}")
                break

        if _should_log(epoch, config.num_epochs):
            print(
                f"Epoch {epoch+1:03d} | "
                f"train_task={train_task:.4f} train_feat={train_feat:.4f} train_total={train_total:.4f} | "
                f"val_task={val_task:.4f} val_feat={val_feat:.4f} val_total={val_total:.4f}"
            )

    if stopper is not None:
        stopper.restore_best(model)

    return model, history


# ---------------------------------------------------------------------------
# Gate-only training (diagnostic)
# ---------------------------------------------------------------------------

def set_trainable_gate_only(model: GatedMLRegressor) -> None:
    """
    Freeze all model parameters except the gate parameter.
    Used for diagnostic experiments to verify gate influence.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    model._gate.requires_grad_(True)


def train_gate_only(
    model: GatedMLRegressor,
    train_loader: DataLoader[tuple[Tensor, ...]],
    val_loader: Optional[DataLoader[tuple[Tensor, ...]]],
    z: Tensor,
    lambda_feat: float,
    config: TrainConfig,
    device: str = "cpu",
    *,
    track_gate_history: bool = False,
    gate_names: Optional[list[str]] = None,
) -> Tuple[GatedMLRegressor, List[Dict[str, float]]]:
    """
    Diagnostic training loop: freeze network weights, train only the gate.

    This loop is intended for diagnosis, not for the main experiments.
    It does not use early stopping, but it can log validation losses if
    val_loader is provided.

    New options
    -----------
    track_gate_history:
        If True, the history rows store the current gate values g_j.
    gate_names:
        Optional feature names used to name history columns as g_<feature>.
    """
    model.to(device)
    set_trainable_gate_only(model)

    dtype = next(model.parameters()).dtype
    z = z.to(device=device, dtype=dtype)

    optimizer = torch.optim.AdamW([model._gate], lr=config.lr, weight_decay=0.0)
    mse_loss  = nn.MSELoss()

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print("Trainable params:", trainable)
    assert trainable == ["_gate"]

    history: List[Dict[str, float]] = []

    for epoch in range(config.num_epochs):
        model.train()
        running_task, running_feat, n_train = 0.0, 0.0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            task_loss = mse_loss(model(xb, use_gate=True), yb)
            feat_loss = torch.mean((model.g - z) ** 2)
            total_loss = task_loss + lambda_feat * feat_loss
            total_loss.backward()
            optimizer.step()

            bs = xb.size(0)
            running_task += task_loss.item() * bs
            running_feat += feat_loss.item() * bs
            n_train += bs

        train_task = running_task / max(n_train, 1)
        train_feat = running_feat / max(n_train, 1)
        train_total = train_task + lambda_feat * train_feat

        val_task, val_feat, val_total = float("nan"), float("nan"), float("nan")
        if val_loader is not None:
            model.eval()
            v_task, v_feat, n_val = 0.0, 0.0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb, use_gate=True)
                    t = mse_loss(preds, yb).item()
                    f = torch.mean((model.g - z) ** 2).item()
                    bs = xb.size(0)
                    v_task += t * bs
                    v_feat += f * bs
                    n_val += bs
            val_task = v_task / max(n_val, 1)
            val_feat = v_feat / max(n_val, 1)
            val_total = val_task + lambda_feat * val_feat

        row: Dict[str, float] = {
            "epoch": float(epoch),
            "train_task_loss": train_task,
            "train_feat_loss": train_feat,
            "train_total_loss": train_total,
            "val_task_loss": val_task,
            "val_feat_loss": val_feat,
            "val_total_loss": val_total,
        }
        if track_gate_history:
            _append_gate_history(row, model, gate_names)
        history.append(row)

        if _should_log(epoch, config.num_epochs):
            print(
                f"Epoch {epoch+1:03d} | "
                f"train_task={train_task:.4f} train_feat={train_feat:.4f} train_total={train_total:.4f} | "
                f"val_task={val_task:.4f} val_feat={val_feat:.4f} val_total={val_total:.4f}"
            )

    return model, history


# ---------------------------------------------------------------------------
# Function-level feature importance regulariser (occlusion / ablation)
# ---------------------------------------------------------------------------

def _normalize_importance(v: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalize a nonnegative vector to sum to 1 (importance distribution)."""
    return v / (v.sum() + eps)


def batch_occlusion_importance(
    model: GatedMLRegressor,
    xb: Tensor,
    *,
    baseline_value: float = 0.0,
) -> Tensor:
    """
    Compute a function-level importance vector on a minibatch via feature ablation.

    Training-time counterpart of ``compute_occlusion_importance`` in alignment.py.
    This version returns a differentiable Tensor so gradients flow back through
    it into the regulariser loss.

    For each feature j:
      - create xb_mask by replacing xb[:, j] with baseline_value
      - compute mean squared change in predictions

    Returns a Tensor of shape (d,), normalized to sum to 1.

    Notes
    -----
    baseline_value=0.0 is correct when inputs are StandardScaler-normalised,
    because 0 corresponds to the feature mean.
    Cost: O(d) extra forward passes per batch.
    """
    preds = model(xb, use_gate=True)
    d = xb.shape[1]
    deltas: list[Tensor] = []

    for j in range(d):
        xb_mask = xb.clone()
        xb_mask[:, j] = baseline_value
        preds_mask = model(xb_mask, use_gate=True)
        deltas.append(torch.mean((preds - preds_mask) ** 2))

    imp = torch.stack(deltas)
    return _normalize_importance(imp)


def train_gated_feature_importance_occlusion(
    model: GatedMLRegressor,
    train_loader: DataLoader[tuple[Tensor, ...]],
    val_loader: Optional[DataLoader[tuple[Tensor, ...]]],
    z: Tensor,
    lambda_feat: float,
    config: TrainConfig,
    device: str = "cpu",
    *,
    baseline_value: float = 0.0,
    track_gate_history: bool = False,
    gate_names: Optional[list[str]] = None,
) -> Tuple[GatedMLRegressor, List[Dict[str, float]]]:
    """
    Train a gated model with a function-level (occlusion-based) regulariser.

    Objective:
        L_total = MSE + lambda_feat * mean((I_occ - z_norm)^2)

    Early stopping monitors val_total because alignment is part of the
    actual training objective.

    New options
    -----------
    track_gate_history:
        If True, the history rows store the current gate values g_j.
    gate_names:
        Optional feature names used to name history columns as g_<feature>.
    """
    model.to(device)
    dtype = next(model.parameters()).dtype
    z = z.to(device=device, dtype=dtype)
    z_norm = _normalize_importance(z)

    optimizer = _make_gated_optimizer(model, config)
    mse_loss  = nn.MSELoss()
    history: List[Dict[str, float]] = []

    patience = config.early_stopping_patience
    if val_loader is not None and patience is not None:
        stopper: Optional[_EarlyStopper] = _EarlyStopper(
            patience,
            config.early_stopping_min_delta,
        )
    else:
        stopper = None

    for epoch in range(config.num_epochs):
        model.train()
        running_task, running_feat, n_train = 0.0, 0.0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(xb, use_gate=True)
            task_loss = mse_loss(preds, yb)
            imp = batch_occlusion_importance(model, xb, baseline_value=baseline_value)
            feat_loss = torch.mean((imp - z_norm) ** 2)
            total_loss = task_loss + lambda_feat * feat_loss
            total_loss.backward()
            optimizer.step()

            bs = xb.size(0)
            running_task += task_loss.item() * bs
            running_feat += feat_loss.item() * bs
            n_train += bs

        train_task = running_task / max(n_train, 1)
        train_feat = running_feat / max(n_train, 1)
        train_total = train_task + lambda_feat * train_feat

        val_task, val_feat, val_total = float("nan"), float("nan"), float("nan")
        if val_loader is not None:
            model.eval()
            v_task, v_feat, n_val = 0.0, 0.0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb, use_gate=True)
                    t = mse_loss(preds, yb).item()
                    imp = batch_occlusion_importance(model, xb, baseline_value=baseline_value)
                    f = torch.mean((imp - z_norm) ** 2).item()
                    bs = xb.size(0)
                    v_task += t * bs
                    v_feat += f * bs
                    n_val  += bs
            val_task  = v_task / max(n_val, 1)
            val_feat  = v_feat / max(n_val, 1)
            val_total = val_task + lambda_feat * val_feat

        row: Dict[str, float] = {
            "epoch": float(epoch),
            "train_task_loss": train_task,
            "train_feat_loss": train_feat,
            "train_total_loss": train_total,
            "val_task_loss": val_task,
            "val_feat_loss": val_feat,
            "val_total_loss": val_total,
        }
        if track_gate_history:
            _append_gate_history(row, model, gate_names)
        history.append(row)

        if stopper is not None and not np.isnan(val_total):
            if stopper.step(val_total, model):
                print(f"Early stopping at epoch {epoch+1:03d} | best_val_total={stopper.best_val:.6f}")
                break

        if _should_log(epoch, config.num_epochs):
            print(
                f"Epoch {epoch+1:03d} | "
                f"train_task={train_task:.4f} train_feat={train_feat:.4f} train_total={train_total:.4f} | "
                f"val_task={val_task:.4f} val_feat={val_feat:.4f} val_total={val_total:.4f}"
            )

    if stopper is not None:
        stopper.restore_best(model)

    return model, history


# ---------------------------------------------------------------------------
# Canonical data-loading helper
# ---------------------------------------------------------------------------

def make_standard_experiment_data(
    project_root: Path | str,
    config: TrainConfig = DEFAULT_TRAIN_CONFIG,
    *,
    noise_seed: int = 42,
) -> NextdayPreparedData:
    """
    Canonical data-loading helper for next-day tmax experiments.

    Steps:
      1. Load nuuksio_daily_clean.csv from <project_root>/data/processed/
      2. Use the standard 6-feature base set plus rand_noise
      3. Temporal split + StandardScaler (fit on train only)
    """
    project_root = Path(project_root)
    df = load_daily_clean_nuuksio(
        project_root / "data" / "processed" / "nuuksio_daily_clean.csv"
    )
    feature_cols_base = ["tmin", "tavg", "tmax", "prcp", "dayofyear_sin", "dayofyear_cos"]
    return prepare_nextday_regression_data(
        df_daily=df,
        base_feature_cols=feature_cols_base,
        config=config,
        add_rand_noise=True,
        noise_col="rand_noise",
        noise_seed=noise_seed,
    )