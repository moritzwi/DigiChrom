"""Model versioning and persistence for DigiChrom."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def save_model(
    model: Any,
    preprocessor: Any,
    metrics: dict,
    model_name: str,
    models_dir: Path,
    custom_name: str | None = None,
    metadata: dict | None = None,
    X_sample: Any = None,
) -> Path:
    """Save a trained model with preprocessor, metadata, and an optional data sample.

    Args:
        X_sample: Optional scaled numpy array (e.g. X_train_scaled). When
            provided it is stored inside the joblib artifact so XAI can run
            without reloading the original data file.

    Returns:
        Path to the saved .joblib file.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    display_name = custom_name or f"{model_name}_{timestamp}"
    model_file = models_dir / f"{model_name}_{timestamp}.joblib"
    meta_file  = models_dir / f"{model_name}_{timestamp}_meta.json"

    artifact = {"model": model, "preprocessor": preprocessor}
    if X_sample is not None:
        artifact["X_sample"] = X_sample
    joblib.dump(artifact, model_file)

    meta = {
        "model_name":   model_name,
        "display_name": display_name,
        "timestamp":    timestamp,
        "saved_at":     datetime.now().isoformat(),
        "has_X_sample": X_sample is not None,
        "metrics": {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in metrics.items()
        },
    }
    if metadata:
        meta.update(metadata)

    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    return model_file


def get_available_models(models_dir: Path) -> pd.DataFrame:
    """List all saved models sorted newest-first.

    Returns:
        DataFrame with columns Name, Model, Timestamp, R², RMSE, MAE.
        Empty DataFrame if the directory doesn't exist or contains no models.
    """
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return pd.DataFrame()

    records = []
    for meta_file in sorted(models_dir.glob("*_meta.json"), reverse=True):
        try:
            with open(meta_file) as f:
                meta = json.load(f)
            records.append({
                "Name":      meta.get("display_name", meta.get("model_name", "")),
                "Model":     meta.get("model_name", ""),
                "Timestamp": meta.get("timestamp", ""),
                "R²":        meta.get("metrics", {}).get("r2", "N/A"),
                "RMSE":      meta.get("metrics", {}).get("rmse", "N/A"),
                "MAE":       meta.get("metrics", {}).get("mae", "N/A"),
                "X_sample":  meta.get("has_X_sample", False),
            })
        except Exception:
            continue

    return pd.DataFrame(records) if records else pd.DataFrame()


def load_model_by_name(
    display_name: str,
    models_dir: Path,
) -> tuple[Any, Any, dict] | None:
    """Load a saved model by its display name.

    The returned metadata dict may contain an ``"X_sample"`` key with the
    scaled training array if the model was saved with one.

    Returns:
        Tuple of (model, preprocessor, metadata) or None if not found.
    """
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return None

    for meta_file in models_dir.glob("*_meta.json"):
        try:
            with open(meta_file) as f:
                meta = json.load(f)
        except Exception:
            continue

        if meta.get("display_name") == display_name:
            model_file = meta_file.with_name(
                meta_file.name.replace("_meta.json", ".joblib")
            )
            if not model_file.exists():
                return None
            data = joblib.load(model_file)
            if "X_sample" in data:
                meta["X_sample"] = data["X_sample"]
            return data["model"], data["preprocessor"], meta

    return None


def delete_model(display_name: str, models_dir: Path) -> bool:
    """Delete a saved model and its metadata by display name."""
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return False

    for meta_file in models_dir.glob("*_meta.json"):
        try:
            with open(meta_file) as f:
                meta = json.load(f)
        except Exception:
            continue

        if meta.get("display_name") == display_name:
            model_file = meta_file.with_name(
                meta_file.name.replace("_meta.json", ".joblib")
            )
            try:
                meta_file.unlink(missing_ok=True)
                model_file.unlink(missing_ok=True)
                return True
            except Exception:
                return False

    return False
