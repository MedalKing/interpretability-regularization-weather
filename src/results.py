# src/results.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


def save_experiment_results(
    results_dir: Path | str,
    model: nn.Module,
    config_dict: dict[str, Any],
    metrics: dict[str, dict[str, float]],
    *,
    alignment_metrics: dict[str, float] | None = None,
    extra_arrays: dict[str, np.ndarray] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> Path:
    """
    Save a standard set of experiment artefacts in one call.

    Files written
    -------------
    best_model.pt            model state dict (always)
    config.json              serialisable metadata (always)
    metrics.csv              train / val / test table (always)
    test_metrics.json        test split only, for quick access
                             (written only when "test" key is present)
    alignment_metrics.json   optional interpretability metrics
    history.csv              optional per-epoch history
    <name>.npy               one file per entry in extra_arrays (optional)
    """
    results_dir = Path(results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), results_dir / "best_model.pt")

    with open(results_dir / "config.json", "w") as fh:
        json.dump(config_dict, fh, indent=2)

    rows = [{"split": split, **m} for split, m in metrics.items()]
    pd.DataFrame(rows).to_csv(results_dir / "metrics.csv", index=False)

    if "test" in metrics:
        with open(results_dir / "test_metrics.json", "w") as fh:
            json.dump(metrics["test"], fh, indent=2)

    if alignment_metrics is not None:
        with open(results_dir / "alignment_metrics.json", "w") as fh:
            json.dump(alignment_metrics, fh, indent=2)

    if history is not None:
        pd.DataFrame(history).to_csv(results_dir / "history.csv", index=False)

    if extra_arrays is not None:
        for name, arr in extra_arrays.items():
            np.save(results_dir / f"{name}.npy", arr)

    return results_dir