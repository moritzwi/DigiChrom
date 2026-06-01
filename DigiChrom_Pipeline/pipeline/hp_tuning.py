import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor, GradientBoostingRegressor,
    HistGradientBoostingRegressor, RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .model_testing import _device_kwargs

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from .config import get_config
from .model_testing import TorchMLP
from .preprocessing import cross_val_splits, make_preprocessor

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except Exception:
    print("optuna not available, HP tuning disabled")
    _HAS_OPTUNA = False

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
    from lightgbm import LGBMRegressor
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

try:
    from .model_testing import TabNetRegressorWrapper, _HAS_TABNET
except Exception:
    _HAS_TABNET = False

try:
    from .model_testing import (
        TabCNNRegressor, FTTransformerRegressor,
        SAINTRegressor, DeepGBMRegressor, _HAS_TORCH,
    )
except Exception:
    _HAS_TORCH = False

TUNABLE_MODELS = {"ridge", "random_forest", "elasticnet", "svr",
                  "extra_trees", "hist_gradient_boosting"}


class _EarlyStoppingCallback:
    """Stop Optuna study after `patience` consecutive trials with no improvement.

    A warmup of `n_warmup` completed trials is mandatory before early stopping
    can trigger, giving the sampler time to explore the space first.
    """

    def __init__(self, patience: int, n_warmup: int = 10) -> None:
        self.patience  = patience
        self.n_warmup  = n_warmup
        self._best     = None
        self._no_improve = 0

    def __call__(self, study, trial) -> None:
        import optuna
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        n_done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        if n_done < self.n_warmup:
            return
        if self._best is None or study.best_value < self._best:
            self._best      = study.best_value
            self._no_improve = 0
        else:
            self._no_improve += 1
        if self._no_improve >= self.patience:
            study.stop()


def _augment_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    aug_ratio: float,
    noise_std: float = 0.05,
    perturb_y: bool = True,
) -> tuple:
    """Add aug_ratio * N Gaussian-noise copies to a training fold.

    Augmentation is done on raw (unscaled) data before the preprocessor runs.
    Feature noise scale = noise_std * per-feature std; target noise scale =
    noise_std * 0.3 * target std, matching GaussianNoiseAugmenter exactly.
    Set perturb_y=False for classification targets to keep discrete labels intact.
    """
    rng = np.random.RandomState(get_config().RANDOM_SEED)
    n_aug = max(1, int(len(X_tr) * aug_ratio))
    idx = rng.choice(len(X_tr), size=n_aug, replace=True)
    X_syn = X_tr[idx] + rng.normal(0, noise_std, size=(n_aug, X_tr.shape[1])) * X_tr.std(axis=0)
    if perturb_y:
        y_syn = y_tr[idx] + rng.normal(0, noise_std * 0.3, size=n_aug) * float(y_tr.std())
    else:
        y_syn = y_tr[idx]
    return np.vstack([X_tr, X_syn]), np.concatenate([y_tr, y_syn])


def _objective_factory(
    model_name: str,
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    cv_splits: list,
    aug_ratio: float = 0.0,
    n_outputs: int = 1,
) -> "callable":
    """Build an Optuna objective function for the given model.

    The objective trains the model over cv_splits and returns the mean RMSE
    across folds. A fresh preprocessor is created per fold. When aug_ratio > 0,
    Gaussian noise augmentation is applied only to each fold's training split
    (never the validation split) to avoid data leakage.

    Args:
        model_name: One of 'ridge', 'random_forest', 'xgboost', 'catboost',
            'mlp'.
        X_arr: Feature matrix as numpy array (original, non-augmented).
        y_arr: Target vector as numpy array (original, non-augmented).
        cv_splits: List of (train_idx, val_idx) index tuples.
        aug_ratio: Fraction of training-fold rows to add as synthetic samples
            (e.g. 0.5 adds 50% synthetic rows). Default 0 = no augmentation.

    Returns:
        Callable that accepts an optuna.Trial and returns mean CV RMSE.

    Raises:
        ValueError: If model_name is not recognised.
    """
    def objective(trial: "optuna.Trial") -> float:
        if model_name == "ridge":
            params = {"alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True)}
            model = Ridge(**params)

        elif model_name == "cart":
            params = {
                "max_depth":        trial.suggest_int("max_depth", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "min_samples_split":trial.suggest_int("min_samples_split", 2, 20),
                "random_state":     get_config().RANDOM_SEED,
            }
            model = DecisionTreeRegressor(**params)

        elif model_name == "gradient_boosting":
            params = {
                "n_estimators":   trial.suggest_int("n_estimators", 50, 500),
                "max_depth":      trial.suggest_int("max_depth", 2, 8),
                "learning_rate":  trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample":      trial.suggest_float("subsample", 0.5, 1.0),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "random_state":   get_config().RANDOM_SEED,
            }
            model = GradientBoostingRegressor(**params)

        elif model_name == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "random_state": get_config().RANDOM_SEED,
            }
            model = RandomForestRegressor(**params)

        elif model_name == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "random_state": get_config().RANDOM_SEED,
                "verbosity": 0,
            }
            model = XGBRegressor(**params, **_device_kwargs("xgboost"))

        elif model_name == "catboost":
            params = {
                "iterations": trial.suggest_int("iterations", 100, 1000),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
                "random_seed": get_config().RANDOM_SEED,
                "verbose": 0,
            }
            model = CatBoostRegressor(**params, **_device_kwargs("catboost"))
        elif model_name == "mlp":
            params = {
                "hidden_sizes": trial.suggest_categorical(
                    "hidden_sizes", [[64, 32], [128, 64], [256, 128, 64]]
                ),
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "epochs": trial.suggest_int("epochs", 50, 150),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            }
            model = TorchMLP(**params)

        elif model_name == "elasticnet":
            params = {
                "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
                "max_iter": 2000,
            }
            model = ElasticNet(**params)

        elif model_name == "svr":
            params = {
                "C": trial.suggest_float("C", 1e-2, 100.0, log=True),
                "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
                "kernel": trial.suggest_categorical("kernel", ["rbf", "linear"]),
            }
            model = SVR(**params)

        elif model_name == "extra_trees":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "random_state": get_config().RANDOM_SEED,
            }
            model = ExtraTreesRegressor(**params)

        elif model_name == "hist_gradient_boosting":
            params = {
                "max_iter": trial.suggest_int("max_iter", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            }
            model = HistGradientBoostingRegressor(**params)

        elif model_name == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "random_state": get_config().RANDOM_SEED,
                "verbose": -1,
            }
            model = LGBMRegressor(**params, **_device_kwargs("lightgbm"))

        elif model_name == "tab_cnn":
            params = {
                "n_filters":   trial.suggest_categorical("n_filters", [32, 64, 128]),
                "kernel_size": trial.suggest_int("kernel_size", 2, 5),
                "n_layers":    trial.suggest_int("n_layers", 1, 4),
                "dropout":     trial.suggest_float("dropout", 0.0, 0.5),
                "lr":          trial.suggest_float("lr", 1e-4, 1e-2, log=True),                "epochs":      trial.suggest_int("epochs", 30, 150),
                "batch_size":  trial.suggest_categorical("batch_size", [16, 32, 64]),
            }
            model = TabCNNRegressor(**params)

        elif model_name == "ft_transformer":
            d_token = trial.suggest_categorical("d_token", [32, 64, 128])
            params = {
                "d_token":    d_token,
                "n_heads":    trial.suggest_categorical("n_heads", [2, 4, 8]),
                "n_layers":   trial.suggest_int("n_layers", 1, 4),
                "dropout":    trial.suggest_float("dropout", 0.0, 0.4),
                "lr":         trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "epochs":      trial.suggest_int("epochs", 30, 150),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            }
            model = FTTransformerRegressor(**params)

        elif model_name == "saint":
            params = {
                "d_token":    trial.suggest_categorical("d_token", [16, 32, 64]),
                "n_heads":    trial.suggest_categorical("n_heads", [2, 4]),
                "n_layers":   trial.suggest_int("n_layers", 1, 3),
                "dropout":    trial.suggest_float("dropout", 0.0, 0.4),
                "lr":         trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "epochs":      trial.suggest_int("epochs", 30, 150),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
            }
            model = SAINTRegressor(**params)

        elif model_name == "deep_gbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth":    trial.suggest_int("max_depth", 2, 6),
                "hidden_size":  trial.suggest_categorical("hidden_size", [32, 64, 128]),
                "dropout":      trial.suggest_float("dropout", 0.0, 0.4),
                "lr":           trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "epochs":       50,
                "batch_size":   trial.suggest_categorical("batch_size", [16, 32, 64]),
            }
            model = DeepGBMRegressor(**params)

        elif model_name == "tabnet":
            params = {
                "n_d": trial.suggest_int("n_d", 8, 64),
                "n_steps": trial.suggest_int("n_steps", 3, 8),
                "gamma": trial.suggest_float("gamma", 1.0, 2.0),
                "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-2, log=True),
                "max_epochs": trial.suggest_int("max_epochs", 50, 150),
                "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
            }
            model = TabNetRegressorWrapper(**params)

        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Wrap with MultiOutputRegressor for models that don't support multi-output natively
        if n_outputs > 1 and model_name in (
            "gradient_boosting", "hist_gradient_boosting", "xgboost", "catboost", "lightgbm", "svr"
        ):
            from sklearn.multioutput import MultiOutputRegressor
            model = MultiOutputRegressor(model)

        rmses = []
        for train_idx, val_idx in cv_splits:
            X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
            y_tr, y_val = y_arr[train_idx], y_arr[val_idx]
            if aug_ratio > 0.0:
                X_tr, y_tr = _augment_fold(X_tr, y_tr, aug_ratio)
            pre = make_preprocessor()
            X_tr_s = pre.fit_transform(X_tr)
            X_val_s = pre.transform(X_val)
            model.fit(X_tr_s, y_tr)
            preds = model.predict(X_val_s)
            rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
        return float(np.mean(rmses))

    return objective


def tune(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: list = None,
    n_trials: int = 100,
    n_outputs: int = 1,
) -> dict:
    """Run an Optuna hyperparameter search for a single model.

    Args:
        model_name: Name of the model to tune (e.g. 'catboost').
        X: Feature DataFrame.
        y: Target Series.
        cv_splits: Pre-computed CV splits. Defaults to cross_val_splits(X, y).
        n_trials: Number of Optuna trials to run.

    Returns:
        Dictionary of best hyperparameter name → value.

    Raises:
        ImportError: If optuna is not installed.
    """
    if not _HAS_OPTUNA:
        raise ImportError("optuna required for HP tuning")

    if cv_splits is None:
        cv_splits = cross_val_splits(X, y)

    X_arr, y_arr = X.values, y.values

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=get_config().RANDOM_SEED),
    )
    study.optimize(
        _objective_factory(model_name, X_arr, y_arr, cv_splits, n_outputs=n_outputs),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best = study.best_params
    print(f"[hp_tuning] {model_name}: best RMSE={study.best_value:.4f}, params={best}")
    return best

def tune_all(
    X: pd.DataFrame,
    y,
    n_trials: int = 100,
    save_path=None,
    force: bool = False,
    aug_ratio: float = 0.0,
    patience: int = 20,
    n_outputs: int = 1,
) -> tuple:
    """Tune all non-baseline models and save results to disk.

    Skips a model silently if its dependency is unavailable.
    If save_path already exists and force=False, resumes from partial cache:
    models already present in the file are skipped; only missing ones are tuned.

    Args:
        X: Feature DataFrame (original, non-augmented).
        y: Target Series (original, non-augmented).
        n_trials: Max number of Optuna trials per model.
        save_path: JSON path to write/read results. Defaults to
            config.REPORTS_DIR / 'hp_tuning.json'.
        force: If True, rerun HPO even if cached results exist.
        aug_ratio: Fraction of each fold's training rows to add as Gaussian-
            noise synthetic samples (e.g. 0.5 = +50%). Augmentation happens
            inside the CV loop on the training split only — the validation
            split always uses original data. Default 0 = no augmentation.
        patience: Stop tuning a model early after this many consecutive trials
            with no improvement in CV RMSE. A warmup of 10 trials is applied
            before early stopping can trigger. Default 20.

    Returns:
        Tuple of (params, scores) where params maps model name → best
        hyperparameter dict and scores maps model name → best CV RMSE.
    """
    out = Path(save_path or get_config().REPORTS_DIR / "hp_tuning.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    cv_splits_shared = cross_val_splits(X, y)

    tunable = ["ridge", "random_forest", "elasticnet", "svr",
               "cart", "gradient_boosting",
               "extra_trees", "hist_gradient_boosting"]
    if _HAS_XGB:
        tunable.append("xgboost")
    if _HAS_CAT:
        tunable.append("catboost")
    if _HAS_LGB:
        tunable.append("lightgbm")
    if _HAS_TORCH:
        tunable.extend(["mlp", "tab_cnn", "ft_transformer", "saint", "deep_gbm"])
    if _HAS_TABNET:
        tunable.append("tabnet")

    params, scores = {}, {}
    if out.exists() and not force:
        try:
            with open(out) as f:
                cached = json.load(f)
            params  = cached.get("params", {})
            scores  = cached.get("scores", {})
        except json.JSONDecodeError:
            print(f"[hp_tuning] WARNING: {out} is corrupted — starting fresh.")
            params, scores = {}, {}
        already = set(params.keys())
        tunable = [m for m in tunable if m not in already]
        if not tunable:
            print(f"[hp_tuning] All models already tuned. Loaded from {out}")
            return params, scores
        if already:
            print(f"[hp_tuning] Resuming: {len(already)} done, {len(tunable)} remaining: {tunable}")

    print(f"[hp_tuning] Tuning {len(tunable)} models: {tunable}")
    for name in tunable:
        try:
            print(f"[hp_tuning] Tuning {name}...")
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=get_config().RANDOM_SEED),
            )
            y_arr = y.values if hasattr(y, "values") else np.array(y)
            study.optimize(
                _objective_factory(name, X.values, y_arr, cv_splits_shared, aug_ratio, n_outputs),
                n_trials=n_trials,
                callbacks=[_EarlyStoppingCallback(patience=patience)],
                show_progress_bar=False,
            )
            params[name] = study.best_params
            scores[name] = float(study.best_value)
            print(f"[hp_tuning] {name}: best RMSE={study.best_value:.4f}, params={study.best_params}")
            # Atomic write: temp file → rename so a crash never corrupts the cache
            tmp = out.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump({"params": params, "scores": scores}, f, indent=2)
            tmp.replace(out)
        except Exception as e:
            print(f"[hp_tuning] Skipping {name}: {e}")

    print(f"[hp_tuning] Saved results → {out}")
    return params, scores


def _clf_objective_factory(
    model_name: str,
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    cv_splits: list,
    aug_ratio: float = 0.0,
) -> "callable":
    """Build an Optuna objective for a classifier. Minimises negative mean CV F1.

    Model names match get_classifiers() keys so that tune_classifiers() results
    map directly to train_final_classifier() without translation.
    When aug_ratio > 0, Gaussian augmentation is applied only to each fold's
    training split; the validation split always uses original data.
    """
    from sklearn.metrics import f1_score

    def objective(trial: "optuna.Trial") -> float:
        if model_name == "logistic":
            from sklearn.linear_model import LogisticRegression
            params = {"C": trial.suggest_float("C", 1e-3, 100.0, log=True), "max_iter": 1000}
            model = LogisticRegression(**params)

        elif model_name == "cart":
            from sklearn.tree import DecisionTreeClassifier
            params = {
                "max_depth":         trial.suggest_int("max_depth", 2, 20),
                "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "random_state":      get_config().RANDOM_SEED,
            }
            model = DecisionTreeClassifier(**params)

        elif model_name == "c50":
            from sklearn.tree import DecisionTreeClassifier
            params = {
                "max_depth":         trial.suggest_int("max_depth", 2, 20),
                "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "criterion":         "entropy",
                "random_state":      get_config().RANDOM_SEED,
            }
            model = DecisionTreeClassifier(**params)

        elif model_name == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier
            params = {
                "n_estimators":      trial.suggest_int("n_estimators", 50, 500),
                "max_depth":         trial.suggest_int("max_depth", 2, 8),
                "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
                "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 20),
                "random_state":      get_config().RANDOM_SEED,
            }
            model = GradientBoostingClassifier(**params)

        elif model_name == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "random_state": get_config().RANDOM_SEED,
            }
            model = RandomForestClassifier(**params)

        elif model_name == "extra_trees":
            from sklearn.ensemble import ExtraTreesClassifier
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "random_state": get_config().RANDOM_SEED,
            }
            model = ExtraTreesClassifier(**params)

        elif model_name == "hist_gradient_boosting":
            from sklearn.ensemble import HistGradientBoostingClassifier
            params = {
                "max_iter": trial.suggest_int("max_iter", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            }
            model = HistGradientBoostingClassifier(**params)

        elif model_name == "xgboost":
            from xgboost import XGBClassifier
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "random_state": get_config().RANDOM_SEED,
                "verbosity": 0,
            }
            model = XGBClassifier(**params, **_device_kwargs("xgboost"))

        elif model_name == "catboost":
            from catboost import CatBoostClassifier
            params = {
                "iterations": trial.suggest_int("iterations", 100, 1000),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
                "random_seed": get_config().RANDOM_SEED,
                "verbose": 0,
            }
            model = CatBoostClassifier(**params, **_device_kwargs("catboost"))

        elif model_name == "lightgbm":
            from lightgbm import LGBMClassifier
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "random_state": get_config().RANDOM_SEED,
                "verbose": -1,
            }
            model = LGBMClassifier(**params, **_device_kwargs("lightgbm"))

        elif model_name == "mlp":
            from .model_testing import TorchMLPClassifier
            params = {
                "hidden_sizes": trial.suggest_categorical(
                    "hidden_sizes", [[64, 32], [128, 64], [256, 128, 64]]
                ),
                "dropout":    trial.suggest_float("dropout", 0.0, 0.5),
                "lr":         trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "epochs":     trial.suggest_int("epochs", 50, 150),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            }
            model = TorchMLPClassifier(**params)

        elif model_name == "tab_cnn":
            from .model_testing import TabCNNClassifier
            params = {
                "n_filters":   trial.suggest_categorical("n_filters", [32, 64, 128]),
                "kernel_size": trial.suggest_int("kernel_size", 2, 5),
                "n_layers":    trial.suggest_int("n_layers", 1, 4),
                "dropout":     trial.suggest_float("dropout", 0.0, 0.5),
                "lr":          trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "epochs":      trial.suggest_int("epochs", 30, 150),
                "batch_size":  trial.suggest_categorical("batch_size", [16, 32, 64]),
            }
            model = TabCNNClassifier(**params)

        elif model_name == "ft_transformer":
            from .model_testing import FTTransformerClassifier
            d_token = trial.suggest_categorical("d_token", [32, 64, 128])
            params = {
                "d_token":    d_token,
                "n_heads":    trial.suggest_categorical("n_heads", [2, 4, 8]),
                "n_layers":   trial.suggest_int("n_layers", 1, 4),
                "dropout":    trial.suggest_float("dropout", 0.0, 0.4),
                "lr":         trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "epochs":     trial.suggest_int("epochs", 30, 150),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            }
            model = FTTransformerClassifier(**params)

        elif model_name == "saint":
            from .model_testing import SAINTClassifier
            params = {
                "d_token":    trial.suggest_categorical("d_token", [16, 32, 64]),
                "n_heads":    trial.suggest_categorical("n_heads", [2, 4]),
                "n_layers":   trial.suggest_int("n_layers", 1, 3),
                "dropout":    trial.suggest_float("dropout", 0.0, 0.4),
                "lr":         trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "epochs":     trial.suggest_int("epochs", 30, 150),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
            }
            model = SAINTClassifier(**params)

        elif model_name == "deep_gbm":
            from .model_testing import DeepGBMClassifier
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth":    trial.suggest_int("max_depth", 2, 6),
                "hidden_size":  trial.suggest_categorical("hidden_size", [32, 64, 128]),
                "dropout":      trial.suggest_float("dropout", 0.0, 0.4),
                "lr":           trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "epochs":       50,
                "batch_size":   trial.suggest_categorical("batch_size", [16, 32, 64]),
            }
            model = DeepGBMClassifier(**params)

        elif model_name == "tabnet":
            from .model_testing import TabNetClassifierWrapper
            params = {
                "n_d":           trial.suggest_int("n_d", 8, 64),
                "n_steps":       trial.suggest_int("n_steps", 3, 8),
                "gamma":         trial.suggest_float("gamma", 1.0, 2.0),
                "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-2, log=True),
                "max_epochs":    trial.suggest_int("max_epochs", 50, 150),
                "batch_size":    trial.suggest_categorical("batch_size", [256, 512, 1024]),
            }
            model = TabNetClassifierWrapper(**params)

        else:
            raise ValueError(f"Unknown classifier: {model_name}")

        f1s = []
        for train_idx, val_idx in cv_splits:
            X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
            y_tr, y_val = y_arr[train_idx], y_arr[val_idx]
            if aug_ratio > 0.0:
                X_tr, y_tr = _augment_fold(X_tr, y_tr, aug_ratio, perturb_y=False)
            pre = make_preprocessor()
            X_tr_s = pre.fit_transform(X_tr)
            X_val_s = pre.transform(X_val)
            model.fit(X_tr_s, y_tr)
            preds = model.predict(X_val_s)
            f1s.append(f1_score(y_val, preds, zero_division=0))
        return -float(np.mean(f1s))  # minimise negative F1

    return objective


def tune_classifiers(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 100,
    cv_splits: list = None,
    save_path=None,
    force: bool = False,
    aug_ratio: float = 0.0,
    patience: int = 20,
) -> tuple:
    """Tune all classifiers via Optuna. Objective = maximise mean CV F1.

    Parallel to tune_all(): resumes from partial cache unless force=True, saves
    intermediate results after each model, includes DL classifiers when torch
    is available. Model name keys match get_classifiers() for direct use in
    train_final_classifier().

    Args:
        X: Feature DataFrame (original, non-augmented).
        y: Binary label Series (0/1, original).
        n_trials: Max number of Optuna trials per classifier.
        cv_splits: Pre-computed CV splits. Defaults to cross_val_splits(X, y).
        save_path: JSON path to write results. Defaults to config.REPORTS_DIR /
            'hp_clf.json'.
        force: If True, rerun HPO even if cached results exist.
        aug_ratio: In-fold augmentation ratio (see tune_all).
        patience: Stop tuning a classifier early after this many consecutive
            trials with no improvement in CV F1. Default 20.

    Returns:
        Tuple of (params, scores) where params maps classifier name → best
        hyperparameter dict and scores maps classifier name → best CV F1.
    """
    if not _HAS_OPTUNA:
        raise ImportError("optuna required for HP tuning")

    out = Path(save_path or get_config().REPORTS_DIR / "hp_clf.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    if cv_splits is None:
        cv_splits = cross_val_splits(X, y)

    X_arr, y_arr = X.values, y.values

    tunable = ["logistic", "cart", "c50", "gradient_boosting",
               "random_forest", "extra_trees", "hist_gradient_boosting"]
    if _HAS_XGB:
        tunable.append("xgboost")
    if _HAS_CAT:
        tunable.append("catboost")
    if _HAS_LGB:
        tunable.append("lightgbm")
    if _HAS_TORCH:
        tunable.extend(["mlp", "tab_cnn", "ft_transformer", "saint", "deep_gbm"])
    try:
        from .model_testing import _HAS_TABNET_CLF as _htc
        if _htc:
            tunable.append("tabnet")
    except Exception:
        pass

    params, scores = {}, {}
    if out.exists() and not force:
        try:
            with open(out) as f:
                cached = json.load(f)
            params  = cached.get("params", {})
            scores  = cached.get("scores", {})
        except json.JSONDecodeError:
            print(f"[hp_tuning] WARNING: {out} is corrupted — starting fresh.")
            params, scores = {}, {}
        already = set(params.keys())
        tunable = [m for m in tunable if m not in already]
        if not tunable:
            print(f"[hp_tuning] All classifiers already tuned. Loaded from {out}")
            return params, scores
        if already:
            print(f"[hp_tuning] Resuming: {len(already)} done, {len(tunable)} remaining: {tunable}")

    print(f"[hp_tuning] Tuning {len(tunable)} classifiers: {tunable}")
    for name in tunable:
        try:
            print(f"[hp_tuning] Tuning classifier {name}...")
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=get_config().RANDOM_SEED),
            )
            study.optimize(
                _clf_objective_factory(name, X_arr, y_arr, cv_splits, aug_ratio),
                n_trials=n_trials,
                callbacks=[_EarlyStoppingCallback(patience=patience)],
                show_progress_bar=False,
            )
            best_f1 = -study.best_value
            params[name] = study.best_params
            scores[name] = float(best_f1)
            print(f"[hp_tuning] {name}: best F1={best_f1:.4f}, params={study.best_params}")
            # Atomic write: temp file → rename so a crash never corrupts the cache
            tmp = out.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump({"params": params, "scores": scores}, f, indent=2)
            tmp.replace(out)
        except Exception as e:
            print(f"[hp_tuning] Skipping classifier {name}: {e}")

    print(f"[hp_tuning] Saved classifier results → {out}")
    return params, scores


if __name__ == "__main__":
    from .data_loading import load_clean
    from .preprocessing import split_xy
    df, _ = load_clean()
    X, y = split_xy(df)
    results = tune_all(X, y, n_trials=30)
    print(results)


# ── Post-HPO Model Ranking ────────────────────────────────────────────────────

def rank_models_reg(
    best_params_reg_orig: dict,
    scores_reg_orig: dict,
    best_params_reg_aug: dict,
    scores_reg_aug: dict,
    X_train: pd.DataFrame,
    y_train,
    cv_splits: list,
    aug_ratio: float = 0.5,
    top_n: int = 3,
    n_outputs: int = 1,
) -> tuple:
    """Select orig/aug per model via HPO scores, then re-evaluate with fixed params.

    Two-step process:
    1. Orig vs aug: pick whichever HPO CV score is better (fast, uses cached scores).
    2. Fresh CV with fixed best params → unbiased mean RMSE ± std per model.

    Args:
        best_params_reg_orig: model → best HPs from original-data HPO.
        scores_reg_orig: model → best CV RMSE from original-data HPO.
        best_params_reg_aug: model → best HPs from augmented-data HPO.
        scores_reg_aug: model → best CV RMSE from augmented-data HPO.
        X_train: Original (non-augmented) feature DataFrame.
        y_train: Original target Series.
        cv_splits: CV splits built on X_train.
        aug_ratio: In-fold augmentation ratio (used when data_source == "aug").
        top_n: Number of top models to return as ensemble candidates.

    Returns:
        Tuple of (ranking_df, top_n_models, best_model, best_params, best_data_source)
        where ranking_df has columns [model, mean_cv_rmse, std_cv_rmse, data_source].
    """
    from .final_training import _build_model

    all_models = set(list(best_params_reg_orig.keys()) + list(best_params_reg_aug.keys()))

    # Step 1: orig vs aug via HPO scores
    best_params: dict = {}
    best_data_source: dict = {}
    for m in all_models:
        rmse_orig = scores_reg_orig.get(m, float("inf"))
        rmse_aug  = scores_reg_aug.get(m,  float("inf"))
        if rmse_aug < rmse_orig:
            best_params[m]      = best_params_reg_aug[m]
            best_data_source[m] = "aug"
        else:
            best_params[m]      = best_params_reg_orig[m]
            best_data_source[m] = "orig"

    # Step 2: fresh CV re-evaluation with fixed params
    X_arr, y_arr = X_train.values, y_train.values
    rows = []
    for m in all_models:
        try:
            use_aug = best_data_source[m] == "aug"
            rmses = []
            for train_idx, val_idx in cv_splits:
                X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
                y_tr, y_val = y_arr[train_idx], y_arr[val_idx]
                if use_aug:
                    X_tr, y_tr = _augment_fold(X_tr, y_tr, aug_ratio)
                pre = make_preprocessor()
                X_tr_s  = pre.fit_transform(X_tr)
                X_val_s = pre.transform(X_val)
                model = _build_model(m, best_params[m], n_outputs=n_outputs)
                model.fit(X_tr_s, y_tr)
                preds = model.predict(X_val_s)
                rmses.append(float(np.sqrt(mean_squared_error(y_val, preds))))
            rows.append({
                "model":        m,
                "mean_cv_rmse": float(np.mean(rmses)),
                "std_cv_rmse":  float(np.std(rmses)),
                "data_source":  best_data_source[m],
            })
            print(f"[rank_reg] {m}: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}  ({best_data_source[m]})")
        except Exception as e:
            print(f"[rank_reg] Skipping {m}: {e}")

    ranking_df  = pd.DataFrame(rows).sort_values("mean_cv_rmse").reset_index(drop=True)
    top_n_models = ranking_df["model"].head(top_n).tolist()
    best_model   = ranking_df["model"].iloc[0]
    return ranking_df, top_n_models, best_model, best_params, best_data_source


def rank_models_clf(
    best_params_clf_orig: dict,
    scores_clf_orig: dict,
    best_params_clf_aug: dict,
    scores_clf_aug: dict,
    X_train: pd.DataFrame,
    y_train_cls: pd.Series,
    cv_splits: list,
    aug_ratio: float = 0.5,
    top_n: int = 3,
) -> tuple:
    """Select orig/aug per classifier via HPO scores, then re-evaluate with fixed params.

    Mirrors rank_models_reg for classification. Metric: mean CV F1 (higher = better).

    Returns:
        Tuple of (ranking_df, top_n_models, best_clf, best_params_clf, best_data_source_clf)
        where ranking_df has columns [model, mean_cv_f1, std_cv_f1, data_source].
    """
    from sklearn.metrics import f1_score as _f1
    from .final_training import _build_classifier

    all_models = set(list(best_params_clf_orig.keys()) + list(best_params_clf_aug.keys()))

    best_params: dict = {}
    best_data_source: dict = {}
    for c in all_models:
        f1_orig = scores_clf_orig.get(c, -float("inf"))
        f1_aug  = scores_clf_aug.get(c,  -float("inf"))
        if f1_aug > f1_orig:
            best_params[c]      = best_params_clf_aug[c]
            best_data_source[c] = "aug"
        else:
            best_params[c]      = best_params_clf_orig[c]
            best_data_source[c] = "orig"

    X_arr, y_arr = X_train.values, y_train_cls.values
    rows = []
    for c in all_models:
        try:
            use_aug = best_data_source[c] == "aug"
            f1s = []
            for train_idx, val_idx in cv_splits:
                X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
                y_tr, y_val = y_arr[train_idx], y_arr[val_idx]
                if use_aug:
                    X_tr, y_tr = _augment_fold(X_tr, y_tr, aug_ratio, perturb_y=False)
                pre = make_preprocessor()
                X_tr_s  = pre.fit_transform(X_tr)
                X_val_s = pre.transform(X_val)
                clf = _build_classifier(c, best_params[c])
                clf.fit(X_tr_s, y_tr)
                preds = clf.predict(X_val_s)
                f1s.append(float(_f1(y_val, preds, zero_division=0)))
            rows.append({
                "model":       c,
                "mean_cv_f1":  float(np.mean(f1s)),
                "std_cv_f1":   float(np.std(f1s)),
                "data_source": best_data_source[c],
            })
            print(f"[rank_clf] {c}: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}  ({best_data_source[c]})")
        except Exception as e:
            print(f"[rank_clf] Skipping {c}: {e}")

    ranking_df   = pd.DataFrame(rows).sort_values("mean_cv_f1", ascending=False).reset_index(drop=True)
    top_n_models = ranking_df["model"].head(top_n).tolist()
    best_clf     = ranking_df["model"].iloc[0]
    return ranking_df, top_n_models, best_clf, best_params, best_data_source


# ── CV Ensemble Evaluation ────────────────────────────────────────────────────

def cv_eval_ensembles_reg(
    top_models: list,
    best_params_reg: dict,
    best_data_source: dict,
    X_train: pd.DataFrame,
    y_train,
    cv_splits: list,
    aug_ratio: float = 0.5,
    n_outputs: int = 1,
) -> pd.DataFrame:
    """CV evaluation of averaging / weighted / stacking ensembles for regression.

    Each base model is trained on its own optimal data (augmented or original)
    as determined by best_data_source, so models that benefited from aug get it
    and models that didn't are left on original data.

    Stacking uses original (non-augmented) data for ALL models during the inner
    cross_val_predict step so that meta-feature row counts align across models.

    Args:
        top_models: Ordered list of base model names (from rank_models_reg).
        best_params_reg: model → best HP dict.
        best_data_source: model → 'aug' or 'orig' (from rank_models_reg).
        X_train: Original feature DataFrame.
        y_train: Original target Series.
        cv_splits: CV splits on X_train.
        aug_ratio: Augmentation ratio applied when best_data_source[m]=='aug'.

    Returns:
        DataFrame with columns [ensemble, mean_cv_rmse, std_cv_rmse].
    """
    from sklearn.model_selection import cross_val_predict as _cvp
    from sklearn.linear_model import Ridge
    from .final_training import _build_model

    X_arr, y_arr = X_train.values, y_train.values
    avg_rmses, w_rmses, stk_rmses = [], [], []

    for train_idx, val_idx in cv_splits:
        X_tr_raw, X_val = X_arr[train_idx], X_arr[val_idx]
        y_tr_raw, y_val = y_arr[train_idx], y_arr[val_idx]

        # Per-model augmentation for averaging / weighted
        base_preds_val = {}
        fold_rmses     = {}
        for mname in top_models:
            use_aug_m = best_data_source.get(mname) == "aug"
            X_tr_m, y_tr_m = (_augment_fold(X_tr_raw, y_tr_raw, aug_ratio)
                              if use_aug_m else (X_tr_raw, y_tr_raw))
            pre_m     = make_preprocessor()
            X_tr_m_s  = pre_m.fit_transform(X_tr_m)
            X_val_s_m = pre_m.transform(X_val)
            m = _build_model(mname, best_params_reg.get(mname, {}), n_outputs=n_outputs)
            m.fit(X_tr_m_s, y_tr_m)
            p = m.predict(X_val_s_m)
            base_preds_val[mname] = p
            fold_rmses[mname]     = float(np.sqrt(mean_squared_error(y_val, p)))

        # Averaging
        avg_pred = np.mean(list(base_preds_val.values()), axis=0)
        avg_rmses.append(float(np.sqrt(mean_squared_error(y_val, avg_pred))))

        # Weighted (1 / individual RMSE)
        inv = {m: 1.0 / max(fold_rmses[m], 1e-9) for m in top_models}
        total = sum(inv.values())
        w_pred = sum(inv[m] / total * base_preds_val[m] for m in top_models)
        w_rmses.append(float(np.sqrt(mean_squared_error(y_val, w_pred))))

        # Stacking: original data for all models so meta-feature rows align
        pre_stk    = make_preprocessor()
        X_tr_stk_s = pre_stk.fit_transform(X_tr_raw)
        meta_X_tr  = np.column_stack([
            _cvp(_build_model(mname, best_params_reg.get(mname, {}), n_outputs=n_outputs),
                 X_tr_stk_s, y_tr_raw, cv=3)
            for mname in top_models
        ])
        meta_learner = Ridge(alpha=1.0)
        meta_learner.fit(meta_X_tr, y_tr_raw)
        meta_X_val = np.column_stack([base_preds_val[m] for m in top_models])
        stk_pred   = meta_learner.predict(meta_X_val)
        stk_rmses.append(float(np.sqrt(mean_squared_error(y_val, stk_pred))))

    rows = [
        {"ensemble": "averaging", "mean_cv_rmse": float(np.mean(avg_rmses)), "std_cv_rmse": float(np.std(avg_rmses))},
        {"ensemble": "weighted",  "mean_cv_rmse": float(np.mean(w_rmses)),   "std_cv_rmse": float(np.std(w_rmses))},
        {"ensemble": "stacking",  "mean_cv_rmse": float(np.mean(stk_rmses)), "std_cv_rmse": float(np.std(stk_rmses))},
    ]
    return pd.DataFrame(rows).sort_values("mean_cv_rmse").reset_index(drop=True)


def cv_eval_ensembles_clf(
    top_models: list,
    best_params_clf: dict,
    best_data_source: dict,
    X_train: pd.DataFrame,
    y_train_cls: pd.Series,
    cv_splits: list,
    aug_ratio: float = 0.5,
) -> pd.DataFrame:
    """CV evaluation of averaging / weighted / stacking ensembles for classification.

    Mirrors cv_eval_ensembles_reg. Each base classifier is trained on its own
    optimal data (aug or orig) per best_data_source. Stacking uses original data
    for all classifiers during inner cross_val_predict so row counts align.

    Returns:
        DataFrame with columns [ensemble, mean_cv_f1, std_cv_f1].
    """
    from sklearn.metrics import f1_score as _f1
    from sklearn.model_selection import cross_val_predict as _cvp
    from sklearn.linear_model import LogisticRegression
    from .final_training import _build_classifier

    X_arr, y_arr = X_train.values, y_train_cls.values
    avg_f1s, w_f1s, stk_f1s = [], [], []

    for train_idx, val_idx in cv_splits:
        X_tr_raw, X_val = X_arr[train_idx], X_arr[val_idx]
        y_tr_raw, y_val = y_arr[train_idx], y_arr[val_idx]

        # Per-model augmentation for averaging / weighted
        base_preds_val = {}
        fold_f1s       = {}
        for cname in top_models:
            use_aug_c = best_data_source.get(cname) == "aug"
            X_tr_c, y_tr_c = (_augment_fold(X_tr_raw, y_tr_raw, aug_ratio, perturb_y=False)
                              if use_aug_c else (X_tr_raw, y_tr_raw))
            pre_c     = make_preprocessor()
            X_tr_c_s  = pre_c.fit_transform(X_tr_c)
            X_val_s_c = pre_c.transform(X_val)
            c = _build_classifier(cname, best_params_clf.get(cname, {}))
            c.fit(X_tr_c_s, y_tr_c)
            p = c.predict(X_val_s_c)
            base_preds_val[cname] = p
            fold_f1s[cname]       = float(_f1(y_val, p, zero_division=0))

        # Averaging (majority vote via mean > 0.5)
        avg_pred = (np.mean(list(base_preds_val.values()), axis=0) > 0.5).astype(int)
        avg_f1s.append(float(_f1(y_val, avg_pred, zero_division=0)))

        # Weighted (by individual F1)
        total = sum(max(fold_f1s[c], 1e-9) for c in top_models)
        w_raw  = sum(fold_f1s[cname] / total * base_preds_val[cname] for cname in top_models)
        w_pred = (w_raw > 0.5).astype(int)
        w_f1s.append(float(_f1(y_val, w_pred, zero_division=0)))

        # Stacking: original data for all classifiers so meta-feature rows align
        pre_stk    = make_preprocessor()
        X_tr_stk_s = pre_stk.fit_transform(X_tr_raw)
        meta_X_tr  = np.column_stack([
            _cvp(_build_classifier(cname, best_params_clf.get(cname, {})),
                 X_tr_stk_s, y_tr_raw, cv=3)
            for cname in top_models
        ])
        meta_learner = LogisticRegression(max_iter=1000)
        meta_learner.fit(meta_X_tr, y_tr_raw)
        meta_X_val = np.column_stack([base_preds_val[c] for c in top_models])
        stk_pred   = meta_learner.predict(meta_X_val)
        stk_f1s.append(float(_f1(y_val, stk_pred, zero_division=0)))

    rows = [
        {"ensemble": "averaging", "mean_cv_f1": float(np.mean(avg_f1s)), "std_cv_f1": float(np.std(avg_f1s))},
        {"ensemble": "weighted",  "mean_cv_f1": float(np.mean(w_f1s)),   "std_cv_f1": float(np.std(w_f1s))},
        {"ensemble": "stacking",  "mean_cv_f1": float(np.mean(stk_f1s)), "std_cv_f1": float(np.std(stk_f1s))},
    ]
    return pd.DataFrame(rows).sort_values("mean_cv_f1", ascending=False).reset_index(drop=True)
