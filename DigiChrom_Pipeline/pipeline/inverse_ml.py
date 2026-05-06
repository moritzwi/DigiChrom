"""Inverse ML: find process parameters that produce a desired chromium thickness.

Given a set of fixed parameter values, a list of 2–3 free variables, and a
target thickness, this module searches for the optimal values of the free
variables using either Bayesian optimisation (Optuna) or gradient-based
input optimisation (PyTorch, for TorchMLP models only).

Example usage::

    from pipeline.inverse_ml import find_inputs, get_bounds_from_data

    bounds = get_bounds_from_data(X_train, free_vars=["Deposition time [min]"])
    solutions = find_inputs(
        model=trained_model,
        preprocessor=fitted_preprocessor,
        feature_names=list(X.columns),
        fixed_params={"pH": 7.0, "Temperature [°C]": 50.0, ...},
        free_vars=["Deposition time [min]", "Current density [A/dm²]"],
        target_thickness=25.0,
        bounds=bounds,
        n_solutions=5,
        method="bayesian",
    )
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from .config import get_config

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except Exception:
    _HAS_OPTUNA = False

try:
    import torch

    def _torch_device() -> "torch.device":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def _build_input_vector(
    fixed_params: dict,
    free_values: dict,
    feature_names: list,
) -> np.ndarray:
    """Assemble a full feature vector from fixed and free parameter dicts.

    Args:
        fixed_params: Mapping of feature name → fixed value.
        free_values: Mapping of free variable name → candidate value.
        feature_names: Ordered list of all feature names.

    Returns:
        1-D numpy array of length len(feature_names).
    """
    row = {**fixed_params, **free_values}
    # Features not explicitly set get 0.0; the preprocessor's imputer will handle NaN,
    # but 0 is safer for models saved without an imputer step.
    return np.array([float(row.get(f, 0.0)) for f in feature_names], dtype=float)


def _bayesian_search(
    model,
    preprocessor,
    feature_names: list,
    fixed_params: dict,
    free_vars: list,
    target: float,
    bounds: dict,
    n_trials: int,
    n_solutions: int,
) -> pd.DataFrame:
    """Search for optimal free-variable values via Bayesian optimisation.

    Args:
        model: Fitted estimator with a predict method.
        preprocessor: Fitted sklearn preprocessing pipeline.
        feature_names: Ordered list of all feature names.
        fixed_params: Fixed feature values.
        free_vars: Names of the variables to optimise.
        target: Desired target thickness in µm.
        bounds: Dict mapping each free variable to a (min, max) tuple.
        n_trials: Number of Optuna trials.
        n_solutions: Number of top solutions to return.

    Returns:
        DataFrame with one row per solution, sorted by absolute error.

    Raises:
        ImportError: If optuna is not installed.
    """
    if not _HAS_OPTUNA:
        raise ImportError("optuna required for bayesian method")

    # target may be a scalar (single output) or a list/array (multi-output)
    target_arr = np.atleast_1d(np.array(target, dtype=float))
    is_multi = target_arr.ndim == 1 and len(target_arr) > 1

    collected = []

    def objective(trial: "optuna.Trial") -> float:
        free_vals = {v: trial.suggest_float(v, bounds[v][0], bounds[v][1]) for v in free_vars}
        x = _build_input_vector(fixed_params, free_vals, feature_names).reshape(1, -1)
        raw = model.predict(preprocessor.transform(x))[0]
        pred_arr = np.atleast_1d(np.array(raw, dtype=float))
        error = float(np.sqrt(np.mean((pred_arr - target_arr) ** 2)))
        row = {**free_vals}
        if is_multi:
            for i, p in enumerate(pred_arr):
                row[f"predicted_{i}"] = float(p)
        else:
            row["predicted_thickness"] = float(pred_arr[0])
        row["error"] = error
        collected.append(row)
        return error

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=get_config().RANDOM_SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return pd.DataFrame(collected).sort_values("error").head(n_solutions).reset_index(drop=True)


def _gradient_search(
    model,
    preprocessor,
    feature_names: list,
    fixed_params: dict,
    free_vars: list,
    target: float,
    bounds: dict,
    n_solutions: int,
    n_steps: int = 500,
    lr: float = 0.01,
) -> pd.DataFrame:
    """Search for optimal free-variable values via gradient descent through the MLP.

    Runs n_solutions * 3 random restarts and returns the best n_solutions.
    Free variables are optimised in scaled space and clipped to their
    training-data bounds after each step.

    Args:
        model: Fitted TorchMLP instance with a model_ attribute.
        preprocessor: Fitted sklearn preprocessing pipeline.
        feature_names: Ordered list of all feature names.
        fixed_params: Fixed feature values.
        free_vars: Names of the variables to optimise.
        target: Desired target thickness in µm.
        bounds: Dict mapping each free variable to a (min, max) tuple.
        n_solutions: Number of best solutions to return.
        n_steps: Gradient-descent steps per restart.
        lr: Adam learning rate for the input optimisation.

    Returns:
        DataFrame with one row per solution, sorted by absolute error.

    Raises:
        ImportError: If torch is not installed.
    """
    if not _HAS_TORCH:
        raise ImportError("torch required for gradient method")

    _is_ens = hasattr(model, "models") and isinstance(getattr(model, "models", None), dict)
    if _is_ens or not hasattr(model, "model_"):
        raise ValueError(
            "Gradient search requires a TorchMLP with a model_ attribute. "
            "Use method='bayesian' for ensembles and non-MLP models."
        )

    results = []
    free_indices = [feature_names.index(v) for v in free_vars]

    for _ in range(n_solutions * 3):
        x_np = np.zeros(len(feature_names), dtype=np.float32)
        for f, val in fixed_params.items():
            if f in feature_names:
                x_np[feature_names.index(f)] = val
        for v in free_vars:
            lo, hi = bounds[v]
            x_np[feature_names.index(v)] = np.random.uniform(lo, hi)

        x_scaled_init = preprocessor.transform(x_np.reshape(1, -1)).astype(np.float32)
        device = next(model.model_.parameters()).device
        x_t = torch.tensor(x_scaled_init, requires_grad=False, device=device)
        free_t = torch.tensor(
            [x_scaled_init[0, i] for i in free_indices],
            requires_grad=True,
            dtype=torch.float32,
            device=device,
        )
        optimizer = torch.optim.Adam([free_t], lr=lr)

        for _ in range(n_steps):
            optimizer.zero_grad()
            x_full = x_t.clone()
            for k, idx in enumerate(free_indices):
                x_full[0, idx] = free_t[k]
            loss = (model.model_(x_full).squeeze() - target) ** 2
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for k, v in enumerate(free_vars):
                    lo, hi = bounds[v]
                    lo_s = float(preprocessor.transform(
                        np.array([[lo if j == free_indices[k] else 0
                                   for j in range(len(feature_names))]])
                    )[0, free_indices[k]])
                    hi_s = float(preprocessor.transform(
                        np.array([[hi if j == free_indices[k] else 0
                                   for j in range(len(feature_names))]])
                    )[0, free_indices[k]])
                    free_t[k].clamp_(min(lo_s, hi_s), max(lo_s, hi_s))

        with torch.no_grad():
            x_full = x_t.clone()
            for k, idx in enumerate(free_indices):
                x_full[0, idx] = free_t[k]
            final_pred = float(model.model_(x_full).squeeze())

        x_inv = preprocessor.inverse_transform(x_full.detach().cpu().numpy())[0]
        row = {v: float(x_inv[feature_names.index(v)]) for v in free_vars}
        row["predicted_thickness"] = final_pred
        row["error"] = abs(final_pred - target)
        results.append(row)

    return pd.DataFrame(results).sort_values("error").head(n_solutions).reset_index(drop=True)


def get_bounds_from_data(X: pd.DataFrame, free_vars: list) -> dict:
    """Derive min/max bounds for free variables from the training data.

    Args:
        X: Training feature DataFrame.
        free_vars: List of feature names to derive bounds for.

    Returns:
        Dictionary mapping each variable name to a (min, max) tuple.
    """
    return {v: (float(X[v].min()), float(X[v].max())) for v in free_vars if v in X.columns}


def find_inputs(
    model,
    preprocessor,
    feature_names: list,
    fixed_params: dict,
    free_vars: list,
    target_thickness: float,
    bounds: dict = None,
    X_train=None,
    n_solutions: int = 5,
    method: str = "bayesian",
    n_trials: int = 300,
) -> pd.DataFrame:
    """Find process parameters that produce a desired chromium layer thickness.

    All features must be accounted for via either fixed_params or free_vars.

    Args:
        model: Fitted estimator.
        preprocessor: Fitted sklearn preprocessing pipeline.
        feature_names: Ordered list of all feature names (must match training).
        fixed_params: Feature name → value for all non-free features.
        free_vars: Names of the 1–3 variables to optimise.
        target_thickness: Desired chromium thickness in µm.
        bounds: Min/max bounds per free variable. If None, derived automatically
            from X_train.
        X_train: Training feature DataFrame. Used to derive bounds when bounds
            is None.
        n_solutions: Number of candidate solutions to return.
        method: Search strategy; 'bayesian' (Optuna, works for all models) or
            'gradient' (PyTorch backprop, requires TorchMLP).
        n_trials: Number of Optuna trials (only used when method='bayesian').

    Returns:
        DataFrame with columns for each free variable plus
        'predicted_thickness' and 'error', sorted ascending by error.

    Raises:
        ValueError: If bounds cannot be determined or any free variable is
            missing bounds.
        ValueError: If method is not 'bayesian' or 'gradient'.
    """
    if bounds is None:
        if X_train is None:
            raise ValueError("Provide either bounds or X_train to derive bounds automatically.")
        bounds = get_bounds_from_data(X_train, free_vars)
        print(f"[inverse_ml] Auto-derived bounds from training data: {bounds}")

    missing_bounds = [v for v in free_vars if v not in bounds]
    if missing_bounds:
        raise ValueError(f"No bounds for free vars: {missing_bounds}")

    unset = set(feature_names) - set(fixed_params.keys()) - set(free_vars)
    if unset:
        raise ValueError(f"Not all features accounted for. Missing: {unset}")

    print(f"[inverse_ml] Searching for inputs → target={target_thickness}µm, "
          f"free vars={free_vars}, method={method}")

    if method == "bayesian":
        df = _bayesian_search(
            model, preprocessor, feature_names, fixed_params,
            free_vars, target_thickness, bounds, n_trials, n_solutions,
        )
    elif method == "gradient":
        df = _gradient_search(
            model, preprocessor, feature_names, fixed_params,
            free_vars, target_thickness, bounds, n_solutions,
        )
    else:
        raise ValueError(f"Unknown method: '{method}'. Use 'bayesian' or 'gradient'.")

    print(f"[inverse_ml] Top solution: {df.iloc[0].to_dict()}")
    return df


if __name__ == "__main__":
    from .data_loading import load_clean
    from .final_training import train_final
    from .preprocessing import split_xy, train_test

    df, _ = load_clean()
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test(X, y)
    feature_names = list(X.columns)

    model, pre = train_final("catboost", X_train, y_train)

    free = ["Deposition time [min]", "Current density [A/dm²]"]
    fixed = {f: float(X_test.iloc[0][f]) for f in feature_names if f not in free}
    bounds = get_bounds_from_data(X, free)

    results = find_inputs(model, pre, feature_names, fixed, free, target_thickness=20.0, bounds=bounds)
    print(results)
