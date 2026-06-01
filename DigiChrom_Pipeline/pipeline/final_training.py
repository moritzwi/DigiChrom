import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor, GradientBoostingRegressor,
    HistGradientBoostingRegressor, RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:
    from lightgbm import LGBMRegressor
    _HAS_LGB_FT = True
except Exception:
    _HAS_LGB_FT = False

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from .config import get_config
from .preprocessing import make_preprocessor

try:
    import torch as _torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    from catboost import CatBoostRegressor
    _HAS_CAT = True
except Exception:
    _HAS_CAT = False

try:
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False


def _build_model(model_name: str, params: dict, n_outputs: int = 1):
    """Instantiate a model by name, merging config defaults with tuned params.

    Args:
        model_name: One of 'linear', 'ridge', 'random_forest', 'xgboost',
            'catboost', 'mlp', etc.
        params: Hyperparameter overrides (e.g. best params from HP tuning).
        n_outputs: Number of target columns. When > 1, models that do not
            natively support multi-output are wrapped with MultiOutputRegressor.

    Returns:
        Unfitted estimator instance.

    Raises:
        ValueError: If model_name is not recognised.
    """
    from sklearn.multioutput import MultiOutputRegressor
    from .model_testing import TorchMLP
    base_params = dict(get_config().MODEL_DEFAULTS.get(model_name, {}))
    base_params.update(params)

    from .model_testing import _device_kwargs

    def _mo(m):
        return MultiOutputRegressor(m) if n_outputs > 1 else m

    if model_name == "linear":
        return LinearRegression()
    elif model_name == "ridge":
        return Ridge(**base_params)
    elif model_name == "cart":
        return DecisionTreeRegressor(**base_params)
    elif model_name == "gradient_boosting":
        return _mo(GradientBoostingRegressor(**base_params))
    elif model_name == "random_forest":
        return RandomForestRegressor(**base_params)
    elif model_name == "xgboost":
        return _mo(XGBRegressor(**{**base_params, **_device_kwargs("xgboost")}))
    elif model_name == "catboost":
        return _mo(CatBoostRegressor(**{**base_params, **_device_kwargs("catboost")}))
    elif model_name == "mlp":
        return TorchMLP(**base_params)
    elif model_name == "lightgbm":
        if not _HAS_LGB_FT:
            raise ImportError("lightgbm is not installed.")
        return _mo(LGBMRegressor(**{**base_params, **_device_kwargs("lightgbm")}))
    elif model_name == "tabnet":
        from .model_testing import TabNetRegressorWrapper
        return TabNetRegressorWrapper(**base_params)
    elif model_name == "tab_cnn":
        from .model_testing import TabCNNRegressor
        return TabCNNRegressor(**base_params)
    elif model_name == "ft_transformer":
        from .model_testing import FTTransformerRegressor
        return FTTransformerRegressor(**base_params)
    elif model_name == "saint":
        from .model_testing import SAINTRegressor
        return SAINTRegressor(**base_params)
    elif model_name == "deep_gbm":
        from .model_testing import DeepGBMRegressor
        return DeepGBMRegressor(**base_params)
    elif model_name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(**base_params)
    elif model_name == "extra_trees":
        return ExtraTreesRegressor(**base_params)
    elif model_name == "elasticnet":
        return ElasticNet(**base_params)
    elif model_name == "svr":
        return _mo(SVR(**base_params))
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_final(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict = None,
    preprocessor=None,
) -> tuple:
    """Fit a model on the full training set.

    Args:
        model_name: Name of the model to train.
        X_train: Training feature DataFrame.
        y_train: Training target Series.
        best_params: Hyperparameter overrides. Defaults to empty dict (uses
            config defaults only).
        preprocessor: Unfitted sklearn Pipeline. Defaults to make_preprocessor().

    Returns:
        Tuple of (fitted_model, fitted_preprocessor).
    """
    best_params = best_params or {}
    preprocessor = preprocessor or make_preprocessor()

    y_arr = y_train.values if hasattr(y_train, "values") else np.asarray(y_train)
    n_outputs = 1 if y_arr.ndim == 1 else y_arr.shape[1]
    model = _build_model(model_name, best_params, n_outputs=n_outputs)
    X_scaled = preprocessor.fit_transform(X_train.values)
    model.fit(X_scaled, y_arr)

    print(f"[final_training] Trained {model_name} on {len(X_train)} samples")
    return model, preprocessor


def _build_classifier(model_name: str, params: dict):
    """Instantiate a classifier by name, merging config defaults with tuned params.

    Mirrors _build_model() for regression. Supports all classifiers from
    get_classifiers() including deep learning variants.

    Args:
        model_name: One of the keys in config.CLASSIFIER_DEFAULTS.
        params: Hyperparameter overrides (e.g. best params from HP tuning).

    Returns:
        Unfitted classifier instance.

    Raises:
        ValueError: If model_name is not recognised.
    """
    base_params = dict(get_config().CLASSIFIER_DEFAULTS.get(model_name, {}))
    base_params.update(params)

    if model_name == "logistic":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**base_params)
    elif model_name == "cart":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(**{k: v for k, v in base_params.items()
                                         if k != "criterion"})
    elif model_name == "c50":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(**base_params)
    elif model_name == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(**base_params)
    elif model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**base_params)
    elif model_name == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(**base_params)
    elif model_name == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(**base_params)
    elif model_name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(**{**base_params, **_device_kwargs("xgboost")},
                             eval_metric="logloss")
    elif model_name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**{**base_params, **_device_kwargs("catboost")})
    elif model_name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**{**base_params, **_device_kwargs("lightgbm")})
    elif model_name == "mlp":
        from .model_testing import TorchMLPClassifier
        return TorchMLPClassifier(**base_params)
    elif model_name == "tab_cnn":
        from .model_testing import TabCNNClassifier
        return TabCNNClassifier(**base_params)
    elif model_name == "ft_transformer":
        from .model_testing import FTTransformerClassifier
        return FTTransformerClassifier(**base_params)
    elif model_name == "saint":
        from .model_testing import SAINTClassifier
        return SAINTClassifier(**base_params)
    elif model_name == "deep_gbm":
        from .model_testing import DeepGBMClassifier
        return DeepGBMClassifier(**base_params)
    elif model_name == "tabnet":
        from .model_testing import TabNetClassifierWrapper
        return TabNetClassifierWrapper(**base_params)
    else:
        raise ValueError(
            f"Unknown classifier: {model_name}. "
            f"Available: {list(get_config().CLASSIFIER_DEFAULTS.keys())}"
        )


def train_final_classifier(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict = None,
    preprocessor=None,
) -> tuple:
    """Fit a classifier on the full training set for binary classification."""
    best_params  = best_params or {}
    preprocessor = preprocessor or make_preprocessor()
    model = _build_classifier(model_name, best_params)
    X_scaled = preprocessor.fit_transform(X_train.values)
    model.fit(X_scaled, y_train.values)
    print(f"[final_training] Trained classifier {model_name} on {len(X_train)} samples")
    return model, preprocessor


def eval_final_classifier(model, preprocessor, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate a fitted classifier. Returns accuracy, F1, AUC, classification report dict,
    and confusion matrix (as nested list) for downstream display."""
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        f1_score, roc_auc_score,
    )
    X_scaled = preprocessor.transform(X_test.values)
    preds    = model.predict(X_scaled)
    m = {
        "accuracy":              float(accuracy_score(y_test, preds)),
        "f1":                    float(f1_score(y_test, preds, zero_division=0)),
        "n_test":                len(y_test),
        "classification_report": classification_report(y_test, preds,
                                                        zero_division=0, output_dict=True),
        "confusion_matrix":      confusion_matrix(y_test, preds).tolist(),
    }
    if hasattr(model, "predict_proba"):
        try:
            probs    = model.predict_proba(X_scaled)[:, 1]
            m["auc"] = float(roc_auc_score(y_test, probs))
        except Exception:
            m["auc"] = float("nan")
    return m


def eval_final(
    model,
    preprocessor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    bootstrap_n: int = 1000,
    ci_alpha: float = 0.05,
) -> dict:
    """Evaluate a fitted model on a held-out test set.

    Computes point metrics (R², RMSE, MAE) plus bootstrap confidence intervals.

    Args:
        model: Fitted estimator.
        preprocessor: Fitted preprocessing pipeline.
        X_test: Test feature DataFrame.
        y_test: Test target Series.
        bootstrap_n: Number of bootstrap resamples for CI computation.
        ci_alpha: Significance level; CI covers (1 - ci_alpha) probability mass.

    Returns:
        Dictionary with keys 'r2', 'rmse', 'mae', 'n_test', and per-metric
        '_ci_low' / '_ci_high' keys (e.g. 'r2_ci_low', 'r2_ci_high').
    """
    X_scaled = preprocessor.transform(X_test.values)
    preds    = model.predict(X_scaled)
    y_arr    = y_test.values

    m = {
        "r2":     float(r2_score(y_arr, preds)),
        "rmse":   float(np.sqrt(mean_squared_error(y_arr, preds))),
        "mae":    float(mean_absolute_error(y_arr, preds)),
        "n_test": len(y_arr),
    }

    # Bootstrap CIs
    rng     = np.random.default_rng(get_config().RANDOM_SEED)
    n       = len(y_arr)
    r2s, rmses, maes = [], [], []
    for _ in range(bootstrap_n):
        idx  = rng.integers(0, n, size=n)
        yb, pb = y_arr[idx], preds[idx]
        r2s.append(r2_score(yb, pb))
        rmses.append(float(np.sqrt(mean_squared_error(yb, pb))))
        maes.append(float(mean_absolute_error(yb, pb)))

    lo, hi = ci_alpha / 2, 1 - ci_alpha / 2
    m.update({
        "r2_ci_low":   float(np.quantile(r2s,   lo)),
        "r2_ci_high":  float(np.quantile(r2s,   hi)),
        "rmse_ci_low": float(np.quantile(rmses,  lo)),
        "rmse_ci_high":float(np.quantile(rmses,  hi)),
        "mae_ci_low":  float(np.quantile(maes,   lo)),
        "mae_ci_high": float(np.quantile(maes,   hi)),
    })
    return m


def save_model(model, preprocessor, model_name: str, save_dir=None) -> None:
    """Persist a fitted model and its preprocessor to disk.

    MLP models additionally save PyTorch state-dict weights alongside the
    joblib pickle.

    Args:
        model: Fitted estimator.
        preprocessor: Fitted preprocessing pipeline.
        model_name: Filename stem (e.g. 'catboost' → 'catboost.pkl').
        save_dir: Directory to write to. Defaults to config.MODELS_DIR.
    """
    save_dir = Path(save_dir or get_config().MODELS_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    if _HAS_TORCH and hasattr(model, "model_") and model.model_ is not None:
        _torch.save({k: v.cpu() for k, v in model.model_.state_dict().items()},
                    save_dir / f"{model_name}_weights.pt")
        joblib.dump({"wrapper": model, "preprocessor": preprocessor},
                    save_dir / f"{model_name}.pkl")
        print(f"[final_training] Saved MLP → {save_dir / model_name}.pkl + _weights.pt")
        return

    joblib.dump({"model": model, "preprocessor": preprocessor},
                save_dir / f"{model_name}.pkl")
    print(f"[final_training] Saved {model_name} → {save_dir / f'{model_name}.pkl'}")


def load_model(model_name: str, load_dir=None) -> tuple:
    """Load a previously saved model and preprocessor from disk.

    Args:
        model_name: Filename stem used when saving (e.g. 'catboost').
        load_dir: Directory to read from. Defaults to config.MODELS_DIR.

    Returns:
        Tuple of (model, preprocessor).
    """
    load_dir = Path(load_dir or get_config().MODELS_DIR)
    artifact = joblib.load(load_dir / f"{model_name}.pkl")
    if "wrapper" in artifact:
        return artifact["wrapper"], artifact["preprocessor"]
    return artifact["model"], artifact["preprocessor"]


def save_metrics(metrics: dict, model_name: str) -> None:
    """Write evaluation metrics to a JSON file in config.REPORTS_DIR.

    Args:
        metrics: Dictionary of metric name → value.
        model_name: Used to name the output file, e.g. 'catboost' →
            'final_metrics_catboost.json'.
    """
    path = get_config().REPORTS_DIR / f"final_metrics_{model_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[final_training] Metrics → {path}")


if __name__ == "__main__":
    from .data_loading import load_clean
    from .preprocessing import split_xy, train_test

    df, _ = load_clean()
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test(X, y)

    model, pre = train_final("catboost", X_train, y_train)
    metrics = eval_final(model, pre, X_test, y_test)
    print(metrics)
    save_model(model, pre, "catboost")
    save_metrics(metrics, "catboost")
