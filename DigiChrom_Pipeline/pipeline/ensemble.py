"""Ensemble / fusion methods for combining multiple trained models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ─────────────────────────────────────────────────────────────────────────────
# Regression ensembles
# ─────────────────────────────────────────────────────────────────────────────

class AveragingEnsemble:
    """Simple unweighted average of predictions from multiple fitted models."""

    def __init__(self, models: dict, task: str = "regression") -> None:
        self.models = models  # {name: fitted_model}
        self.task = task

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.stack([m.predict(X) for m in self.models.values()], axis=1)
        avg = preds.mean(axis=1)
        if self.task == "classification":
            return (avg > 0.5).astype(int)
        return avg

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average predict_proba across base models (classifiers only)."""
        probas = np.stack(
            [m.predict_proba(X) for m in self.models.values()], axis=0
        )
        return probas.mean(axis=0)

    def __repr__(self):
        return f"AveragingEnsemble(models={list(self.models.keys())})"


class WeightedEnsemble:
    """Weighted average; weights are proportional to CV R² scores."""

    def __init__(self, models: dict, weights: dict, task: str = "regression") -> None:
        self.models  = models
        self.weights = weights  # {name: weight}
        self.task = task

    def predict(self, X: np.ndarray) -> np.ndarray:
        total = sum(self.weights.values())
        preds = sum(
            self.weights[n] / total * m.predict(X)
            for n, m in self.models.items()
            if n in self.weights
        )
        if self.task == "classification":
            return (preds > 0.5).astype(int)
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Weighted average of predict_proba (classifiers only)."""
        total = sum(self.weights.values())
        result = None
        for n, m in self.models.items():
            if n not in self.weights:
                continue
            w = self.weights[n] / total
            p = m.predict_proba(X)
            result = p * w if result is None else result + p * w
        return result

    @classmethod
    def from_cv_results(
        cls,
        models: dict,
        cv_results: pd.DataFrame,
        metric: str = "r2",
    ) -> "WeightedEnsemble":
        """Build weights from cross-validation scores (higher score → higher weight).

        Args:
            models: Dict of fitted base models.
            cv_results: DataFrame with a 'model' column and a metric column.
            metric: Column name to use as weight source. Use 'r2' for regression,
                'f1' or 'auc' for classification.
        """
        score_mean = cv_results.groupby("model")[metric].mean()
        weights = {}
        for name in models:
            score = float(score_mean.get(name, 0.0))
            weights[name] = max(score, 0.0)        # clamp negatives to zero
        if sum(weights.values()) == 0:
            weights = {n: 1.0 for n in models}    # fallback: uniform
        return cls(models, weights)

    def __repr__(self):
        return f"WeightedEnsemble(weights={self.weights})"


class StackingEnsemble:
    """Meta-learner stacking: base model predictions → Ridge meta-learner.

    For classification, set task='classification' to use LogisticRegression.
    """

    def __init__(self, base_models: dict, task: str = "regression") -> None:
        self.base_models  = base_models
        self.task         = task
        self.meta_learner = (LogisticRegression(max_iter=1000)
                             if task == "classification"
                             else Ridge(alpha=1.0))
        self.fitted_      = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackingEnsemble":
        meta_X = self._base_predict(X)
        self.meta_learner.fit(meta_X, y)
        self.fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.meta_learner.predict(self._base_predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.meta_learner, "predict_proba"):
            return self.meta_learner.predict_proba(self._base_predict(X))
        raise NotImplementedError("Meta-learner has no predict_proba")

    def _base_predict(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([m.predict(X) for m in self.base_models.values()])

    def __repr__(self):
        return f"StackingEnsemble(base={list(self.base_models.keys())}, meta={self.meta_learner})"


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_ensemble(
    ensemble,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str = "regression",
) -> dict:
    """Evaluate a fitted ensemble on test data."""
    preds = ensemble.predict(X_test)
    if task == "regression":
        return {
            "r2":   float(r2_score(y_test, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "mae":  float(mean_absolute_error(y_test, preds)),
        }
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    # Averaging/weighted ensembles return floats; round to 0/1 for accuracy/f1
    preds_int = (preds > 0.5).astype(int)
    m = {
        "accuracy": float(accuracy_score(y_test, preds_int)),
        "f1":       float(f1_score(y_test, preds_int, zero_division=0)),
    }
    if hasattr(ensemble, "predict_proba"):
        try:
            m["auc"] = float(roc_auc_score(y_test, ensemble.predict_proba(X_test)[:, 1]))
        except Exception:
            pass
    return m


def compare_ensembles(
    base_models: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cv_results: pd.DataFrame = None,
    task: str = "regression",
) -> pd.DataFrame:
    """Compare all ensemble strategies on test data. Returns metrics DataFrame."""
    ensembles = {
        "averaging": AveragingEnsemble(base_models),
    }
    if cv_results is not None:
        ensembles["weighted"] = WeightedEnsemble.from_cv_results(base_models, cv_results)
    stacking = StackingEnsemble(base_models, task=task)
    stacking.fit(X_train, y_train)
    ensembles["stacking"] = stacking

    rows = []
    for name, ens in ensembles.items():
        m = evaluate_ensemble(ens, X_test, y_test, task=task)
        rows.append({"ensemble": name, **m})
    return pd.DataFrame(rows)
