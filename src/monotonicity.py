from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import json
import numpy as np
import pandas as pd
import torch
from numpy.typing import ArrayLike
from sklearn.neighbors import NearestNeighbors
from torch import Tensor, nn


SEASON_MAP = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}


# ---------------------------------------------------------------------------
# Generic model helpers
# ---------------------------------------------------------------------------

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


def predict_numpy(
    model: nn.Module,
    X_np: ArrayLike,
    *,
    device: str = "cpu",
    use_gate: bool | None = None,
    batch_size: int = 4096,
) -> np.ndarray:
    """
    Batched NumPy prediction helper for PyTorch regression models.
    """
    X_arr = np.asarray(X_np, dtype=np.float32)

    model = model.to(device)
    model.eval()

    preds_all: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X_arr), batch_size):
            xb = torch.as_tensor(X_arr[start : start + batch_size], dtype=torch.float32, device=device)
            preds = _forward_model(model, xb, use_gate=use_gate)
            preds_all.append(preds.detach().cpu().numpy().ravel())

    if not preds_all:
        return np.empty(0, dtype=np.float64)

    return np.concatenate(preds_all).astype(np.float64)


# ---------------------------------------------------------------------------
# Audit dataframe helpers
# ---------------------------------------------------------------------------

def make_split_audit_df(
    df_split: pd.DataFrame,
    X_scaled: ArrayLike,
    y_pred: ArrayLike,
    split_name: str,
    feature_cols: list[str],
    *,
    target_col: str = "tmax_next",
) -> pd.DataFrame:
    """
    Build a split-level audit dataframe containing:
      - original columns from df_split,
      - split label,
      - y_true,
      - baseline_pred,
      - scaled feature copies under <feature>__scaled.
    """
    out = df_split.copy().reset_index(drop=True)

    out["split"] = split_name
    out["y_true"] = out[target_col].astype(float)
    out["baseline_pred"] = np.asarray(y_pred, dtype=float).ravel()

    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"])
        out["month"] = out["datetime"].dt.month
        out["season"] = out["month"].map(SEASON_MAP)

    X_arr = np.asarray(X_scaled, dtype=float)
    for j, name in enumerate(feature_cols):
        out[f"{name}__scaled"] = X_arr[:, j]

    return out


# ---------------------------------------------------------------------------
# Pair construction helpers
# ---------------------------------------------------------------------------

def circular_dayofyear_gap(
    day_a: ArrayLike,
    day_b: ArrayLike,
    period: float = 365.25,
) -> float | np.ndarray:
    """
    Circular distance on day-of-year.

    Works for both scalars and arrays.
    """
    a = np.asarray(day_a, dtype=float)
    b = np.asarray(day_b, dtype=float)
    diff = np.abs(a - b)
    out = np.minimum(diff, period - diff)

    if out.ndim == 0:
        return float(out.item())
    return out.astype(np.float64)


def build_local_pairs(
    audit_df: pd.DataFrame,
    *,
    feature: str,
    sign: int,
    setting: dict[str, Any],
    feature_cols: list[str],
    split: str = "train",
    always_exclude_from_context: tuple[str, ...] = ("rand_noise",),
) -> pd.DataFrame:
    """
    Build local monotonicity comparison pairs within a split.

    Pair orientation:
      - row_hi has larger constrained feature value
      - row_lo has smaller constrained feature value

    The desired monotonicity direction is encoded by sign:
      +1 for nondecreasing
      -1 for nonincreasing

    Evaluation later checks:
      sign * (value_hi - value_lo) >= 0
    """
    df = audit_df.loc[audit_df["split"] == split].copy().reset_index(drop=True)

    cols = [
        "split",
        "feature",
        "sign",
        "direction",
        "setting",
        "row_hi_id",
        "row_lo_id",
        "feature_hi_scaled",
        "feature_lo_scaled",
        "delta_feature_scaled",
        "feature_hi_raw",
        "feature_lo_raw",
        "delta_feature_raw",
        "context_distance",
        "season_gap_days",
    ]

    if len(df) < 2:
        return pd.DataFrame(columns=cols)

    direction = "nondecreasing" if sign == 1 else "nonincreasing"

    setting_name = str(setting["setting"])
    k_neighbors = int(setting["k_neighbors"])
    context_max_dist = float(setting["context_max_dist"])
    min_feature_gap_scaled = float(setting["min_feature_gap_scaled"])
    max_pairs_per_anchor = int(setting["max_pairs_per_anchor"])

    max_day_gap = setting.get("max_dayofyear_gap_days", np.nan)
    if pd.isna(max_day_gap):
        max_day_gap = None
    else:
        max_day_gap = float(max_day_gap)

    feat_scaled_col = f"{feature}__scaled"

    context_features = [
        c for c in feature_cols
        if c != feature and c not in always_exclude_from_context
    ]
    context_scaled_cols = [f"{c}__scaled" for c in context_features]

    X_context = df[context_scaled_cols].to_numpy(dtype=float)
    if X_context.shape[1] == 0:
        X_context = np.zeros((len(df), 1), dtype=float)

    feat_scaled = df[feat_scaled_col].to_numpy(dtype=float)
    feat_raw = df[feature].to_numpy(dtype=float)
    dayofyear = df["dayofyear"].to_numpy(dtype=float)
    row_ids = df["row_id"].to_numpy(dtype=int)

    n_neighbors = min(len(df), k_neighbors + 1)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(X_context)
    dists, nbrs = nn.kneighbors(X_context, return_distance=True)

    records: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()

    for i in range(len(df)):
        accepted_here = 0

        for dist, j in zip(dists[i, 1:], nbrs[i, 1:]):  # skip self
            if dist > context_max_dist:
                continue

            season_gap = float(circular_dayofyear_gap(dayofyear[i], dayofyear[j]))
            if max_day_gap is not None and season_gap > max_day_gap:
                continue

            delta_scaled_ij = feat_scaled[i] - feat_scaled[j]
            if abs(delta_scaled_ij) < min_feature_gap_scaled:
                continue

            if delta_scaled_ij > 0:
                hi, lo = i, j
            else:
                hi, lo = j, i

            key = (int(row_ids[hi]), int(row_ids[lo]))
            if key in seen:
                continue

            seen.add(key)

            delta_feature_scaled = float(feat_scaled[hi] - feat_scaled[lo])
            delta_feature_raw = float(feat_raw[hi] - feat_raw[lo])

            records.append(
                {
                    "split": split,
                    "feature": feature,
                    "sign": int(sign),
                    "direction": direction,
                    "setting": setting_name,
                    "row_hi_id": int(row_ids[hi]),
                    "row_lo_id": int(row_ids[lo]),
                    "feature_hi_scaled": float(feat_scaled[hi]),
                    "feature_lo_scaled": float(feat_scaled[lo]),
                    "delta_feature_scaled": delta_feature_scaled,
                    "feature_hi_raw": float(feat_raw[hi]),
                    "feature_lo_raw": float(feat_raw[lo]),
                    "delta_feature_raw": delta_feature_raw,
                    "context_distance": float(dist),
                    "season_gap_days": season_gap,
                }
            )

            accepted_here += 1
            if accepted_here >= max_pairs_per_anchor:
                break

    if not records:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(records)
    out = out.sort_values(
        ["context_distance", "delta_feature_scaled"],
        ascending=[True, False],
    ).reset_index(drop=True)
    return out


def summarize_pairs(
    pairs_df: pd.DataFrame,
    *,
    split: str,
    feature: str,
    setting: str,
) -> dict[str, Any]:
    """
    Summarize pair quality diagnostics.
    """
    if pairs_df.empty:
        return {
            "split": split,
            "feature": feature,
            "setting": setting,
            "n_pairs": 0,
            "mean_context_distance": np.nan,
            "median_context_distance": np.nan,
            "mean_abs_feature_gap_scaled": np.nan,
            "median_abs_feature_gap_scaled": np.nan,
            "mean_abs_feature_gap_raw": np.nan,
            "median_abs_feature_gap_raw": np.nan,
        }

    return {
        "split": split,
        "feature": feature,
        "setting": setting,
        "n_pairs": int(len(pairs_df)),
        "mean_context_distance": float(pairs_df["context_distance"].mean()),
        "median_context_distance": float(pairs_df["context_distance"].median()),
        "mean_abs_feature_gap_scaled": float(pairs_df["delta_feature_scaled"].abs().mean()),
        "median_abs_feature_gap_scaled": float(pairs_df["delta_feature_scaled"].abs().median()),
        "mean_abs_feature_gap_raw": float(pairs_df["delta_feature_raw"].abs().mean()),
        "median_abs_feature_gap_raw": float(pairs_df["delta_feature_raw"].abs().median()),
    }


# ---------------------------------------------------------------------------
# Pairwise monotonicity audit metrics
# ---------------------------------------------------------------------------

def compute_pairwise_monotonicity_summary(
    audit_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    *,
    value_col: str,
    split: str,
    feature: str,
    setting: str,
    sign: int,
) -> dict[str, Any]:
    """
    Compute pairwise monotonicity metrics for either:
      - true labels: value_col='y_true'
      - model predictions: value_col='baseline_pred'
    """
    if pairs_df.empty:
        return {
            "split": split,
            "feature": feature,
            "setting": setting,
            "sign": int(sign),
            "value_col": value_col,
            "n_pairs": 0,
            "violation_rate": np.nan,
            "mean_hinge_violation": np.nan,
            "median_hinge_violation": np.nan,
            "p95_hinge_violation": np.nan,
            "mean_signed_diff": np.nan,
            "median_signed_diff": np.nan,
        }

    value_lookup = audit_df.set_index("row_id")[value_col]

    val_hi = value_lookup.loc[pairs_df["row_hi_id"].to_numpy()].to_numpy(dtype=float)
    val_lo = value_lookup.loc[pairs_df["row_lo_id"].to_numpy()].to_numpy(dtype=float)

    signed_diff = float(sign) * (val_hi - val_lo)
    hinge = np.maximum(0.0, -signed_diff)
    violations = signed_diff < 0.0

    return {
        "split": split,
        "feature": feature,
        "setting": setting,
        "sign": int(sign),
        "value_col": value_col,
        "n_pairs": int(len(pairs_df)),
        "violation_rate": float(np.mean(violations)),
        "mean_hinge_violation": float(np.mean(hinge)),
        "median_hinge_violation": float(np.median(hinge)),
        "p95_hinge_violation": float(np.quantile(hinge, 0.95)),
        "mean_signed_diff": float(np.mean(signed_diff)),
        "median_signed_diff": float(np.median(signed_diff)),
    }


def pairwise_metrics_from_predictions(
    y_pred: ArrayLike,
    pair_idx: ArrayLike,
    sign: int,
) -> dict[str, Any]:
    """
    Compute monotonicity metrics directly from predictions and local pair indices.

    pair_idx must have shape (n_pairs, 2) with columns:
      [row_hi_local_idx, row_lo_local_idx]
    """
    preds = np.asarray(y_pred, dtype=np.float64).ravel()
    idx = np.asarray(pair_idx, dtype=np.int64)

    if idx.size == 0:
        return {
            "n_pairs": 0,
            "violation_rate": np.nan,
            "mean_hinge_violation": np.nan,
            "median_hinge_violation": np.nan,
            "p95_hinge_violation": np.nan,
            "mean_signed_diff": np.nan,
            "median_signed_diff": np.nan,
        }

    val_hi = preds[idx[:, 0]]
    val_lo = preds[idx[:, 1]]

    signed_diff = float(sign) * (val_hi - val_lo)
    hinge = np.maximum(0.0, -signed_diff)
    violations = signed_diff < 0.0

    return {
        "n_pairs": int(len(idx)),
        "violation_rate": float(np.mean(violations)),
        "mean_hinge_violation": float(np.mean(hinge)),
        "median_hinge_violation": float(np.median(hinge)),
        "p95_hinge_violation": float(np.quantile(hinge, 0.95)),
        "mean_signed_diff": float(np.mean(signed_diff)),
        "median_signed_diff": float(np.median(signed_diff)),
    }


def pairwise_metrics_from_model(
    model: nn.Module,
    X_np: ArrayLike,
    pair_idx: ArrayLike,
    sign: int,
    *,
    device: str = "cpu",
    use_gate: bool | None = None,
) -> dict[str, Any]:
    """
    Convenience wrapper: predict first, then compute pairwise metrics.
    """
    preds = predict_numpy(model, X_np, device=device, use_gate=use_gate)
    return pairwise_metrics_from_predictions(preds, pair_idx, sign)


# ---------------------------------------------------------------------------
# Gradient-sign diagnostics
# ---------------------------------------------------------------------------

def compute_gradient_sign_audit(
    model: nn.Module,
    X_np: ArrayLike,
    *,
    split: str,
    constrained_feature: str,
    sign: int,
    feature_cols: list[str],
    device: str = "cpu",
    use_gate: bool | None = None,
    batch_size: int = 512,
) -> dict[str, Any]:
    """
    Compute local gradient-sign monotonicity diagnostics for one feature.
    """
    j = feature_cols.index(constrained_feature)
    X_arr = np.asarray(X_np, dtype=np.float32)

    signed_grads_all: list[np.ndarray] = []

    model = model.to(device)
    model.eval()

    for start in range(0, len(X_arr), batch_size):
        xb = torch.tensor(
            X_arr[start : start + batch_size],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        preds = _forward_model(model, xb, use_gate=use_gate)
        grads = torch.autograd.grad(preds.sum(), xb, create_graph=False)[0]
        signed_grads = float(sign) * grads[:, j].detach().cpu().numpy()
        signed_grads_all.append(signed_grads)

    if not signed_grads_all:
        return {
            "split": split,
            "feature": constrained_feature,
            "sign": int(sign),
            "n_points": 0,
            "grad_violation_rate": np.nan,
            "mean_signed_gradient": np.nan,
            "median_signed_gradient": np.nan,
            "p05_signed_gradient": np.nan,
            "p95_signed_gradient": np.nan,
        }

    signed_grads = np.concatenate(signed_grads_all).astype(np.float64)
    violations = signed_grads < 0.0

    return {
        "split": split,
        "feature": constrained_feature,
        "sign": int(sign),
        "n_points": int(len(signed_grads)),
        "grad_violation_rate": float(np.mean(violations)),
        "mean_signed_gradient": float(np.mean(signed_grads)),
        "median_signed_gradient": float(np.median(signed_grads)),
        "p05_signed_gradient": float(np.quantile(signed_grads, 0.05)),
        "p95_signed_gradient": float(np.quantile(signed_grads, 0.95)),
    }


# ---------------------------------------------------------------------------
# Qualitative inspection helpers
# ---------------------------------------------------------------------------

def select_representative_rows(
    df_split: pd.DataFrame,
    *,
    col: str = "y_true",
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> pd.DataFrame:
    """
    Pick rows whose target values are closest to the requested quantiles.
    """
    chosen_idx: list[int] = []
    used: set[int] = set()

    for q in quantiles:
        target = df_split[col].quantile(q)
        candidates = (df_split[col] - target).abs().sort_values().index.tolist()
        idx = next(i for i in candidates if i not in used)
        used.add(idx)
        chosen_idx.append(idx)

    return df_split.loc[chosen_idx].copy()


def make_counterfactual_sweep_df(
    model: nn.Module,
    *,
    base_row: pd.Series,
    feature_name: str,
    grid_orig: ArrayLike,
    feature_cols: list[str],
    scaler,
    device: str = "cpu",
    use_gate: bool | None = None,
) -> pd.DataFrame:
    """
    For one base row, vary one feature over a grid and record model predictions.
    """
    scaled_cols = [f"{c}__scaled" for c in feature_cols]
    base_x_scaled = base_row[scaled_cols].to_numpy(dtype=np.float32)

    grid_arr = np.asarray(grid_orig, dtype=np.float32).ravel()
    j = feature_cols.index(feature_name)

    X_cf = np.repeat(base_x_scaled[None, :], repeats=len(grid_arr), axis=0)
    X_cf[:, j] = (grid_arr - scaler.mean_[j]) / scaler.scale_[j]

    preds = predict_numpy(model, X_cf, device=device, use_gate=use_gate)

    return pd.DataFrame(
        {
            "feature_name": feature_name,
            "feature_value_orig": grid_arr.astype(float),
            "prediction": preds.astype(float),
            "base_row_id": int(base_row["row_id"]),
            "base_y_true": float(base_row["y_true"]),
            "base_pred": float(base_row["baseline_pred"]),
            "base_feature_value_orig": float(base_row[feature_name]),
        }
    )


# ---------------------------------------------------------------------------
# Pair-index conversion and artifact I/O
# ---------------------------------------------------------------------------

def pairs_to_local_index_array(
    pairs_df: pd.DataFrame,
    split_df: pd.DataFrame,
) -> np.ndarray:
    """
    Convert global row_id-based pairs into split-local index pairs.

    Output shape: (n_pairs, 2)
      - col 0: row_hi_local_idx
      - col 1: row_lo_local_idx
    """
    if pairs_df.empty:
        return np.empty((0, 2), dtype=np.int64)

    if "row_id" not in split_df.columns:
        raise KeyError("split_df must contain a 'row_id' column.")

    row_to_local = {int(rid): i for i, rid in enumerate(split_df["row_id"].tolist())}

    hi_idx = []
    lo_idx = []

    for hi_rid, lo_rid in zip(pairs_df["row_hi_id"], pairs_df["row_lo_id"]):
        hi_idx.append(row_to_local[int(hi_rid)])
        lo_idx.append(row_to_local[int(lo_rid)])

    return np.column_stack([hi_idx, lo_idx]).astype(np.int64)


def save_chosen_pair_artifacts(
    output_dir: Path | str,
    *,
    pair_store: Mapping[tuple[str, str, str], pd.DataFrame],
    best_settings_df: pd.DataFrame,
    split_dfs: Mapping[str, pd.DataFrame],
    constraints_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Save the chosen pair set for each feature and split, plus local index arrays.

    Files written:
      <output_dir>/chosen_pairs/<split>__<feature>__<setting>.csv
      <output_dir>/chosen_pairs/<split>__<feature>__<setting>__local_idx.npy
      <output_dir>/chosen_pairs_manifest.csv
    """
    output_dir = Path(output_dir)
    pair_dir = output_dir / "chosen_pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)

    sign_lookup: dict[str, float] = {}
    direction_lookup: dict[str, str] = {}

    if constraints_df is not None and not constraints_df.empty:
        if "feature" in constraints_df.columns and "sign" in constraints_df.columns:
            sign_lookup = {
                str(k): float(v)
                for k, v in constraints_df.set_index("feature")["sign"].items()
                if pd.notna(v)
            }

        if "feature" in constraints_df.columns and "direction" in constraints_df.columns:
            direction_lookup = {
                str(k): str(v)
                for k, v in constraints_df.set_index("feature")["direction"].items()
                if pd.notna(v)
            }

    records: list[dict[str, Any]] = []

    for row in best_settings_df.itertuples(index=False):
        feature = str(row.feature)
        setting = str(row.chosen_setting)

        sign_val = sign_lookup.get(feature, np.nan)
        direction_val = direction_lookup.get(feature, "")

        for split, split_df in split_dfs.items():
            key = (split, feature, setting)
            pairs_df = pair_store.get(key, pd.DataFrame())

            csv_name = f"{split}__{feature}__{setting}.csv"
            npy_name = f"{split}__{feature}__{setting}__local_idx.npy"

            csv_path = pair_dir / csv_name
            npy_path = pair_dir / npy_name

            pairs_df.to_csv(csv_path, index=False)
            local_idx = pairs_to_local_index_array(pairs_df, split_df)
            np.save(npy_path, local_idx)

            records.append(
                {
                    "split": split,
                    "feature": feature,
                    "setting": setting,
                    "sign": sign_val,
                    "direction": direction_val,
                    "n_pairs": int(len(pairs_df)),
                    "pairs_csv_path": str(Path("chosen_pairs") / csv_name),
                    "local_index_path": str(Path("chosen_pairs") / npy_name),
                }
            )

    manifest_df = pd.DataFrame(records)
    manifest_df.to_csv(output_dir / "chosen_pairs_manifest.csv", index=False)
    return manifest_df


def load_chosen_pair_artifacts(
    output_dir: Path | str,
) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    """
    Load the chosen-pair manifest and local pair-index arrays.

    Returns
    -------
    manifest_df
    pair_idx_store
        dict keyed by (split, feature) -> np.ndarray of shape (n_pairs, 2)
    """
    output_dir = Path(output_dir)
    manifest_path = output_dir / "chosen_pairs_manifest.csv"
    manifest_df = pd.read_csv(manifest_path)

    pair_idx_store: dict[tuple[str, str], np.ndarray] = {}
    for row in manifest_df.itertuples(index=False):
        arr = np.load(output_dir / str(row.local_index_path))
        pair_idx_store[(str(row.split), str(row.feature))] = arr.astype(np.int64)

    return manifest_df, pair_idx_store


def build_training_spec(
    final_spec_df: pd.DataFrame,
    *,
    keep_values: tuple[str, ...] = ("yes",),
) -> pd.DataFrame:
    """
    Build a compact training spec from final_monotonicity_spec.csv.

    By default, only rows with suggested_keep_for_training == 'yes' are kept.
    """
    spec = final_spec_df.loc[
        final_spec_df["suggested_keep_for_training"].isin(keep_values)
    ].copy()

    if spec.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "direction",
                "sign",
                "chosen_setting",
                "chosen_train_n_pairs",
                "train_label_violation_rate",
                "test_label_violation_rate",
                "test_baseline_violation_rate",
                "test_grad_violation_rate",
                "confidence",
                "rationale",
                "notes",
            ]
        )

    spec["sign"] = spec["sign"].astype(int)

    keep_cols = [
        "feature",
        "direction",
        "sign",
        "chosen_setting",
        "chosen_train_n_pairs",
        "train_label_violation_rate",
        "test_label_violation_rate",
        "test_baseline_violation_rate",
        "test_grad_violation_rate",
        "confidence",
        "rationale",
        "notes",
    ]
    return spec[keep_cols].reset_index(drop=True)


def save_training_spec(
    final_spec_df: pd.DataFrame,
    output_dir: Path | str,
    *,
    keep_values: tuple[str, ...] = ("yes",),
    name: str = "training_spec_main",
) -> pd.DataFrame:
    """
    Save a compact training spec as both CSV and JSON.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec_df = build_training_spec(final_spec_df, keep_values=keep_values)
    spec_df.to_csv(output_dir / f"{name}.csv", index=False)

    payload = {
        "keep_values": list(keep_values),
        "n_constraints": int(len(spec_df)),
        "constraints": spec_df.to_dict(orient="records"),
    }
    with open(output_dir / f"{name}.json", "w") as fh:
        json.dump(payload, fh, indent=2)

    return spec_df


def load_training_spec(
    output_dir: Path | str,
    *,
    name: str = "training_spec_main",
) -> pd.DataFrame:
    """
    Load a saved training spec CSV.
    """
    output_dir = Path(output_dir)
    return pd.read_csv(output_dir / f"{name}.csv")


def make_sign_dict(training_spec_df: pd.DataFrame) -> dict[str, int]:
    """
    Convert a training spec dataframe into a feature -> sign dict.
    """
    if "feature" not in training_spec_df.columns or "sign" not in training_spec_df.columns:
        raise KeyError("training_spec_df must contain 'feature' and 'sign' columns.")

    feature_series = training_spec_df["feature"].astype("string")
    sign_series = pd.to_numeric(training_spec_df["sign"], errors="raise").astype("int64")

    features: list[str] = [str(x) for x in feature_series.to_list()]
    signs_np = sign_series.to_numpy(dtype=np.int64)

    out: dict[str, int] = {}
    for i in range(len(features)):
        out[features[i]] = int(signs_np[i])

    return out


def make_pair_index_dict_from_manifest(
    manifest_df: pd.DataFrame,
    pair_idx_store: Mapping[tuple[str, str], np.ndarray],
    *,
    split: str,
    features: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """
    Build a feature -> pair-index-array dict for one split.

    Parameters
    ----------
    manifest_df : pd.DataFrame
        Output of load_chosen_pair_artifacts().
    pair_idx_store : dict
        Output of load_chosen_pair_artifacts().
    split : str
        One of {'train', 'val', 'test'}.
    features : list[str], optional
        If provided, keep only those features.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping feature -> local pair index array
    """
    sub = manifest_df.loc[manifest_df["split"] == split].copy()
    if features is not None:
        sub = sub.loc[sub["feature"].isin(features)].copy()

    out: dict[str, np.ndarray] = {}
    for row in sub.itertuples(index=False):
        out[str(row.feature)] = np.asarray(pair_idx_store[(str(row.split), str(row.feature))], dtype=np.int64)

    return out


# ---------------------------------------------------------------------------
# Training-time monotonicity losses
# ---------------------------------------------------------------------------

def sample_pair_index_array(
    pair_idx: ArrayLike,
    n_pairs: int | None,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Optionally subsample a pair-index array without replacement.
    """
    idx = np.asarray(pair_idx, dtype=np.int64)

    if idx.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    if idx.ndim != 2 or idx.shape[1] != 2:
        raise ValueError("pair_idx must have shape (n_pairs, 2).")

    if n_pairs is None or n_pairs >= len(idx):
        return idx

    if rng is None:
        rng = np.random.default_rng()

    chosen = rng.choice(len(idx), size=n_pairs, replace=False)
    return idx[chosen]


def pairwise_monotonicity_hinge_loss(
    model: nn.Module,
    X: Tensor,
    pair_idx: ArrayLike,
    sign: int,
    *,
    use_gate: bool | None = None,
    margin: float = 0.0,
    reduction: str = "mean",
) -> Tensor:
    """
    Pairwise hinge-style monotonicity loss.

    pair_idx columns:
      [row_hi_local_idx, row_lo_local_idx]

    Desired condition:
      sign * (pred_hi - pred_lo) >= margin

    With margin = 0, this is the basic monotonicity hinge.
    """
    idx = np.asarray(pair_idx, dtype=np.int64)

    if idx.size == 0:
        return torch.zeros((), device=X.device, dtype=X.dtype)

    if idx.ndim != 2 or idx.shape[1] != 2:
        raise ValueError("pair_idx must have shape (n_pairs, 2).")

    idx_t = torch.as_tensor(idx, dtype=torch.long, device=X.device)

    x_hi = X[idx_t[:, 0]]
    x_lo = X[idx_t[:, 1]]

    pred_hi = _forward_model(model, x_hi, use_gate=use_gate)
    pred_lo = _forward_model(model, x_lo, use_gate=use_gate)

    signed_diff = float(sign) * (pred_hi - pred_lo)
    hinge = torch.relu(margin - signed_diff)

    if reduction == "mean":
        return hinge.mean()
    if reduction == "sum":
        return hinge.sum()
    raise ValueError("reduction must be either 'mean' or 'sum'")


def combined_monotonicity_loss(
    model: nn.Module,
    X: Tensor,
    pair_index_dict: Mapping[str, ArrayLike],
    sign_dict: Mapping[str, int],
    *,
    use_gate: bool | None = None,
    margin: float = 0.0,
    feature_weights: Mapping[str, float] | None = None,
    max_pairs_per_feature: int | None = None,
    rng: np.random.Generator | None = None,
    reduction: str = "mean_over_features",
) -> tuple[Tensor, dict[str, float]]:
    """
    Combine feature-specific monotonicity losses.

    Parameters
    ----------
    pair_index_dict
        feature -> pair-index-array
    sign_dict
        feature -> sign (+1 / -1)

    Returns
    -------
    total_loss, detail_dict
    """
    losses: list[Tensor] = []
    details: dict[str, float] = {}

    for feature, pair_idx in pair_index_dict.items():
        if feature not in sign_dict:
            raise KeyError(f"Feature '{feature}' missing from sign_dict.")

        sampled_idx = sample_pair_index_array(pair_idx, max_pairs_per_feature, rng=rng)
        base_loss = pairwise_monotonicity_hinge_loss(
            model,
            X,
            sampled_idx,
            int(sign_dict[feature]),
            use_gate=use_gate,
            margin=margin,
            reduction="mean",
        )

        weight = 1.0
        if feature_weights is not None:
            weight = float(feature_weights.get(feature, 1.0))

        weighted_loss = weight * base_loss
        losses.append(weighted_loss)

        details[f"mono_loss_{feature}"] = float(base_loss.detach().cpu())
        details[f"mono_weight_{feature}"] = float(weight)
        details[f"mono_n_pairs_{feature}"] = float(len(sampled_idx))

    if not losses:
        zero = torch.zeros((), device=X.device, dtype=X.dtype)
        details["mono_loss_total"] = 0.0
        return zero, details

    stacked = torch.stack(losses)

    if reduction == "sum":
        total = stacked.sum()
    elif reduction == "mean_over_features":
        total = stacked.mean()
    else:
        raise ValueError("reduction must be either 'sum' or 'mean_over_features'")

    details["mono_loss_total"] = float(total.detach().cpu())
    return total, details