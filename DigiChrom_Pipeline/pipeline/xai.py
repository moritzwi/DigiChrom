import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import permutation_importance as sk_perm

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from .config import get_config

try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False


def shap_analysis(
    model,
    X_scaled: np.ndarray,
    feature_names: list,
    save_path=None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Compute SHAP values and save a summary bar + beeswarm plot.

    Uses TreeExplainer for tree-based models and KernelExplainer otherwise.
    Saves mean |SHAP| per feature to config.REPORTS_DIR / 'shap_importance.csv'.

    Args:
        model: Fitted estimator.
        X_scaled: Scaled feature matrix (numpy array).
        feature_names: Ordered list of feature names matching X_scaled columns.
        save_path: Output path for the PDF. Defaults to config.FIGURES_DIR /
            'shap_summary.pdf'.

    Returns:
        Tuple of (shap_values, importance_df) where shap_values is a 2-D
        array of shape (n_samples, n_features) and importance_df is a
        DataFrame with columns ['feature', 'mean_abs_shap'].

    Raises:
        ImportError: If shap is not installed.
    """
    if not _HAS_SHAP:
        raise ImportError("shap required")

    save_path = save_path or get_config().FIGURES_DIR / "shap_summary.pdf"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    model_type = type(model).__name__.lower()
    if any(t in model_type for t in ["catboost", "xgb", "forest", "tree"]):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
    else:
        background = shap.sample(X_scaled, min(100, len(X_scaled)))
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_scaled, nsamples=100)

    X_df = pd.DataFrame(X_scaled, columns=feature_names)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.sca(axes[0])
    shap.summary_plot(shap_values, X_df, plot_type="bar", show=False)
    axes[0].set_title("SHAP Feature Importance (mean |SHAP|)")
    plt.sca(axes[1])
    shap.summary_plot(shap_values, X_df, show=False)
    axes[1].set_title("SHAP Beeswarm")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[xai] Saved SHAP summary → {save_path}")

    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance.to_csv(get_config().REPORTS_DIR / "shap_importance.csv", index=False)

    return shap_values, importance


def pdp_plots(
    model,
    X_scaled: np.ndarray,
    feature_names: list,
    top_n: int = 8,
    save_path=None,
) -> matplotlib.figure.Figure:
    """Compute and save partial dependence plots for the top-N features.

    Feature ranking is based on mean |SHAP| if shap is available, otherwise
    the first top_n features are used.

    Args:
        model: Fitted estimator with a predict method.
        X_scaled: Scaled feature matrix (numpy array).
        feature_names: Ordered list of feature names.
        top_n: Number of features to plot.
        save_path: Output path for the PDF. Defaults to config.FIGURES_DIR /
            'pdp_plots.pdf'.

    Returns:
        Matplotlib Figure object.
    """
    save_path = save_path or get_config().FIGURES_DIR / "pdp_plots.pdf"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    if _HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_scaled)
            top_idx = np.argsort(np.abs(sv).mean(axis=0))[::-1][:top_n]
        except Exception:
            top_idx = list(range(min(top_n, len(feature_names))))
    else:
        top_idx = list(range(min(top_n, len(feature_names))))

    features_to_plot = [int(i) for i in top_idx]
    ncols = 4
    nrows = (len(features_to_plot) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()

    PartialDependenceDisplay.from_estimator(
        model, X_scaled, features=features_to_plot,
        feature_names=feature_names, ax=axes[:len(features_to_plot)],
        line_kw={"color": "#4C72B0"},
    )

    for j in range(len(features_to_plot), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Partial Dependence Plots (Top Features by SHAP)", fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[xai] Saved PDP plots → {save_path}")
    return fig


def permutation_importance(
    model,
    X_scaled: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    save_path=None,
) -> pd.DataFrame:
    """Compute permutation importance and save a horizontal bar chart.

    Results are also written to config.REPORTS_DIR / 'permutation_importance.csv'.

    Args:
        model: Fitted estimator.
        X_scaled: Scaled feature matrix (numpy array).
        y: True target values (numpy array).
        feature_names: Ordered list of feature names.
        save_path: Output path for the PDF. Defaults to config.FIGURES_DIR /
            'permutation_importance.pdf'.

    Returns:
        DataFrame with columns ['feature', 'importance_mean', 'importance_std'],
        sorted descending by importance_mean.
    """
    save_path = save_path or get_config().FIGURES_DIR / "permutation_importance.pdf"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    result = sk_perm(model, X_scaled, y, n_repeats=10,
                     random_state=get_config().RANDOM_SEED, scoring="r2")
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.35)))
    ax.barh(imp_df["feature"][::-1], imp_df["importance_mean"][::-1],
            xerr=imp_df["importance_std"][::-1], color="#4C72B0", ecolor="gray")
    ax.set_xlabel("Mean decrease in R²")
    ax.set_title("Permutation Importance")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[xai] Saved permutation importance → {save_path}")

    imp_df.to_csv(get_config().REPORTS_DIR / "permutation_importance.csv", index=False)
    return imp_df


def shap_analysis_classifier(
    model,
    X_scaled: np.ndarray,
    feature_names: list,
    save_path=None,
) -> tuple:
    """Compute SHAP values for a binary classifier using predict_proba[:,1].

    Uses TreeExplainer for tree-based models, KernelExplainer otherwise.
    Saves mean |SHAP| per feature to config.REPORTS_DIR / 'shap_importance_clf.csv'.

    Args:
        model: Fitted binary classifier with predict_proba.
        X_scaled: Scaled feature matrix (numpy array).
        feature_names: Ordered list of feature names.
        save_path: Output PDF path. Defaults to config.FIGURES_DIR /
            'shap_summary_clf.pdf'.

    Returns:
        Tuple of (shap_values, importance_df).
    """
    if not _HAS_SHAP:
        raise ImportError("shap required")

    save_path = save_path or get_config().FIGURES_DIR / "shap_summary_clf.pdf"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    model_type = type(model).__name__.lower()
    if any(t in model_type for t in ["catboost", "xgb", "forest", "tree", "gradient"]):
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_scaled)
            # For binary classifiers TreeExplainer may return list [neg_class, pos_class]
            shap_values = sv[1] if isinstance(sv, list) else sv
        except Exception:
            background = shap.sample(X_scaled, min(100, len(X_scaled)))
            explainer = shap.KernelExplainer(
                lambda x: model.predict_proba(x)[:, 1], background
            )
            shap_values = explainer.shap_values(X_scaled, nsamples=100)
    else:
        background = shap.sample(X_scaled, min(100, len(X_scaled)))
        explainer = shap.KernelExplainer(
            lambda x: model.predict_proba(x)[:, 1], background
        )
        shap_values = explainer.shap_values(X_scaled, nsamples=100)

    X_df = pd.DataFrame(X_scaled, columns=feature_names)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.sca(axes[0])
    shap.summary_plot(shap_values, X_df, plot_type="bar", show=False)
    axes[0].set_title("SHAP Feature Importance — Classification (mean |SHAP|)")
    plt.sca(axes[1])
    shap.summary_plot(shap_values, X_df, show=False)
    axes[1].set_title("SHAP Beeswarm — P(thick)")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[xai] Saved classifier SHAP summary → {save_path}")

    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance.to_csv(get_config().REPORTS_DIR / "shap_importance_clf.csv", index=False)

    return shap_values, importance


def permutation_importance_classifier(
    model,
    X_scaled: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    scoring: str = "f1",
    save_path=None,
) -> pd.DataFrame:
    """Permutation importance for a binary classifier.

    Args:
        model: Fitted binary classifier.
        X_scaled: Scaled feature matrix (numpy array).
        y: True binary labels (numpy array).
        feature_names: Ordered list of feature names.
        scoring: sklearn scoring string, e.g. 'f1', 'roc_auc', 'accuracy'.
        save_path: Output PDF path. Defaults to config.FIGURES_DIR /
            'permutation_importance_clf.pdf'.

    Returns:
        DataFrame with columns ['feature', 'importance_mean', 'importance_std'].
    """
    save_path = save_path or get_config().FIGURES_DIR / "permutation_importance_clf.pdf"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    result = sk_perm(model, X_scaled, y, n_repeats=10,
                     random_state=get_config().RANDOM_SEED, scoring=scoring)
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.35)))
    ax.barh(imp_df["feature"][::-1], imp_df["importance_mean"][::-1],
            xerr=imp_df["importance_std"][::-1], color="#DD8452", ecolor="gray")
    ax.set_xlabel(f"Mean decrease in {scoring}")
    ax.set_title("Permutation Importance — Classification")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[xai] Saved classifier permutation importance → {save_path}")

    imp_df.to_csv(get_config().REPORTS_DIR / "permutation_importance_clf.csv", index=False)
    return imp_df


def roc_curve_plot(
    model,
    X_scaled: np.ndarray,
    y: np.ndarray,
    label: str = "",
    save_path=None,
) -> float:
    """Plot ROC curve for a binary classifier and return AUC.

    Args:
        model: Fitted binary classifier with predict_proba.
        X_scaled: Scaled feature matrix (numpy array).
        y: True binary labels (numpy array).
        label: Legend label for the curve (e.g. model name).
        save_path: Output PDF path. Defaults to config.FIGURES_DIR / 'roc_curve.pdf'.

    Returns:
        AUC score (float).
    """
    from sklearn.metrics import RocCurveDisplay, roc_auc_score

    save_path = save_path or get_config().FIGURES_DIR / "roc_curve.pdf"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_scaled)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_scaled)
    else:
        raise ValueError("Model has neither predict_proba nor decision_function")

    auc = float(roc_auc_score(y, y_score))

    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y, y_score, name=label or "Model", ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_title(f"ROC Curve  (AUC = {auc:.4f})")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[xai] Saved ROC curve → {save_path}  AUC={auc:.4f}")
    return auc


def ice_plots(
    model,
    X_scaled: np.ndarray,
    feature_names: list,
    top_n: int = 6,
    save_path=None,
) -> matplotlib.figure.Figure:
    """ICE (Individual Conditional Expectation) + PDP overlay for top-N features.

    Each line is one sample's predicted response as the feature varies.
    The bold line is the mean (PDP). Features are selected by SHAP importance
    when available, otherwise by index.

    Args:
        model: Fitted estimator with a predict method.
        X_scaled: Scaled feature matrix (numpy array).
        feature_names: Ordered list of feature names.
        top_n: Number of features to plot.
        save_path: Output PDF path. Defaults to config.FIGURES_DIR / 'ice_plots.pdf'.

    Returns:
        Matplotlib Figure object.
    """
    save_path = save_path or get_config().FIGURES_DIR / "ice_plots.pdf"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    if _HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_scaled)
            top_idx = list(np.argsort(np.abs(sv).mean(axis=0))[::-1][:top_n])
        except Exception:
            top_idx = list(range(min(top_n, len(feature_names))))
    else:
        top_idx = list(range(min(top_n, len(feature_names))))

    ncols = min(3, len(top_idx))
    nrows = (len(top_idx) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5))
    axes = np.array(axes).flatten()

    PartialDependenceDisplay.from_estimator(
        model, X_scaled, features=top_idx,
        feature_names=feature_names, ax=axes[:len(top_idx)],
        kind="both",
        subsample=min(200, len(X_scaled)),
        ice_lines_kw={"color": "#aec6e8", "alpha": 0.3, "linewidth": 0.7},
        pd_line_kw={"color": "#d62728", "linewidth": 2.5, "label": "PDP"},
    )

    for j in range(len(top_idx), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("ICE Plots (Individual Conditional Expectation + PDP)", fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[xai] Saved ICE plots → {save_path}")
    return fig


def ale_plots(
    model,
    X_scaled: np.ndarray,
    feature_names: list,
    top_n: int = 6,
    n_bins: int = 20,
    save_path=None,
) -> matplotlib.figure.Figure:
    """Accumulated Local Effects (ALE) plots for top-N features.

    ALE is computed manually: for each bin the local effect is the average
    change in prediction when the feature moves from the bin's lower to upper
    edge, keeping all other features fixed. Effects are then accumulated.

    Args:
        model: Fitted estimator with a predict method.
        X_scaled: Scaled feature matrix (numpy array).
        feature_names: Ordered list of feature names.
        top_n: Number of features to plot.
        n_bins: Number of bins per feature.
        save_path: Output PDF path. Defaults to config.FIGURES_DIR / 'ale_plots.pdf'.

    Returns:
        Matplotlib Figure object.
    """
    save_path = save_path or get_config().FIGURES_DIR / "ale_plots.pdf"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    if _HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_scaled)
            top_idx = list(np.argsort(np.abs(sv).mean(axis=0))[::-1][:top_n])
        except Exception:
            top_idx = list(range(min(top_n, len(feature_names))))
    else:
        top_idx = list(range(min(top_n, len(feature_names))))

    ncols = min(3, len(top_idx))
    nrows = (len(top_idx) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5))
    axes = np.array(axes).flatten()

    for plot_i, feat_idx in enumerate(top_idx):
        x_col    = X_scaled[:, feat_idx]
        quantiles = np.quantile(x_col, np.linspace(0, 1, n_bins + 1))
        quantiles = np.unique(quantiles)
        if len(quantiles) < 2:
            axes[plot_i].set_visible(False)
            continue

        ale_vals = np.zeros(len(quantiles) - 1)
        for b in range(len(quantiles) - 1):
            mask = (x_col >= quantiles[b]) & (x_col < quantiles[b + 1])
            if mask.sum() == 0:
                continue
            X_lo = X_scaled[mask].copy()
            X_hi = X_scaled[mask].copy()
            X_lo[:, feat_idx] = quantiles[b]
            X_hi[:, feat_idx] = quantiles[b + 1]
            ale_vals[b] = np.mean(model.predict(X_hi) - model.predict(X_lo))

        ale_cumsum = np.concatenate([[0], np.cumsum(ale_vals)])
        ale_cumsum -= ale_cumsum.mean()
        # x_plot and y_plot have the same length: len(quantiles)
        x_plot = quantiles
        y_plot = ale_cumsum

        ax = axes[plot_i]
        ax.plot(x_plot, y_plot, color="#4C72B0", linewidth=2)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.fill_between(x_plot, 0, y_plot, alpha=0.15, color="#4C72B0")
        ax.set_xlabel(feature_names[feat_idx])
        ax.set_ylabel("ALE")
        ax.set_title(feature_names[feat_idx])

    for j in range(len(top_idx), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("ALE Plots (Accumulated Local Effects)", fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[xai] Saved ALE plots → {save_path}")
    return fig


def shap_waterfall_plots(
    model,
    X_scaled: np.ndarray,
    feature_names: list,
    n_samples: int = 3,
    save_path=None,
) -> None:
    """SHAP waterfall plots for individual samples (best/worst/median predictions).

    Args:
        model: Fitted estimator.
        X_scaled: Scaled feature matrix (numpy array).
        feature_names: Ordered list of feature names.
        n_samples: Number of individual samples to plot (taken from best, median, worst).
        save_path: Output PDF path. Defaults to config.FIGURES_DIR / 'shap_waterfall.pdf'.
    """
    if not _HAS_SHAP:
        raise ImportError("shap required")

    save_path = save_path or get_config().FIGURES_DIR / "shap_waterfall.pdf"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    model_type = type(model).__name__.lower()
    if any(t in model_type for t in ["catboost", "xgb", "forest", "tree", "boost", "lgb"]):
        explainer  = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        ev          = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev = float(ev[0]) if hasattr(ev, "__len__") else float(ev)
    else:
        background = shap.sample(X_scaled, min(100, len(X_scaled)))
        explainer  = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_scaled, nsamples=100)
        ev          = float(explainer.expected_value)

    # pick representative sample indices
    preds     = model.predict(X_scaled)
    sorted_idx = np.argsort(preds)
    n          = len(preds)
    picks      = [sorted_idx[0], sorted_idx[n // 2], sorted_idx[-1]][:n_samples]
    labels     = ["Lowest prediction", "Median prediction", "Highest prediction"][:n_samples]

    for i, (idx, lbl) in enumerate(zip(picks, labels)):
        sv_row = shap_values[idx]
        expl   = shap.Explanation(
            values=sv_row,
            base_values=ev,
            data=X_scaled[idx],
            feature_names=feature_names,
        )
        fig_wf, ax_wf = plt.subplots(figsize=(10, max(4, len(feature_names) * 0.35)))
        plt.sca(ax_wf)
        shap.waterfall_plot(expl, max_display=15, show=False)
        ax_wf.set_title(f"SHAP Waterfall — {lbl}")
        plt.tight_layout()
        out = Path(str(save_path).replace(".pdf", f"_{i}.pdf"))
        fig_wf.savefig(out, bbox_inches="tight")
        plt.close(fig_wf)
        print(f"[xai] Saved SHAP waterfall → {out}")


def shap_dependence_plots(
    model,
    X_scaled: np.ndarray,
    feature_names: list,
    top_n: int = 6,
    save_path=None,
) -> None:
    """SHAP dependence plots with automatic interaction-feature coloring.

    For each of the top-N features (by mean |SHAP|), plots the feature value
    on x vs. the SHAP value on y, colored by the feature that interacts most.

    Args:
        model: Fitted estimator.
        X_scaled: Scaled feature matrix (numpy array).
        feature_names: Ordered list of feature names.
        top_n: Number of features to plot.
        save_path: Output PDF path. Defaults to config.FIGURES_DIR / 'shap_dependence.pdf'.
    """
    if not _HAS_SHAP:
        raise ImportError("shap required")

    save_path = save_path or get_config().FIGURES_DIR / "shap_dependence.pdf"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    model_type = type(model).__name__.lower()
    if any(t in model_type for t in ["catboost", "xgb", "forest", "tree", "boost", "lgb"]):
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
    else:
        background  = shap.sample(X_scaled, min(100, len(X_scaled)))
        explainer   = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_scaled, nsamples=100)

    top_idx = list(np.argsort(np.abs(shap_values).mean(axis=0))[::-1][:top_n])
    X_df    = pd.DataFrame(X_scaled, columns=feature_names)

    ncols = min(3, len(top_idx))
    nrows = (len(top_idx) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5))
    axes = np.array(axes).flatten()

    for plot_i, feat_idx in enumerate(top_idx):
        ax = axes[plot_i]
        plt.sca(ax)
        shap.dependence_plot(
            feat_idx, shap_values, X_df,
            interaction_index="auto", show=False, ax=ax,
        )
        ax.set_title(feature_names[feat_idx])

    for j in range(len(top_idx), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("SHAP Dependence Plots (with interaction coloring)", fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[xai] Saved SHAP dependence plots → {save_path}")


def shap_interaction_matrix(
    model,
    X_scaled: np.ndarray,
    feature_names: list,
    save_path=None,
) -> pd.DataFrame:
    """Compute and visualise the full SHAP interaction matrix (TreeExplainer only).

    Saves the top-50 pairwise interaction strengths to a CSV. Falls back to
    a warning if the model does not support interaction values.

    Args:
        model: Fitted tree-based estimator (TreeExplainer required).
        X_scaled: Scaled feature matrix (numpy array).
        feature_names: Ordered list of feature names.
        save_path: Output PDF path. Defaults to config.FIGURES_DIR / 'shap_interactions.pdf'.

    Returns:
        DataFrame with columns ['feature_a', 'feature_b', 'mean_abs_interaction'],
        sorted descending. Empty DataFrame on failure.
    """
    if not _HAS_SHAP:
        raise ImportError("shap required")

    save_path = save_path or get_config().FIGURES_DIR / "shap_interactions.pdf"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        explainer   = shap.TreeExplainer(model)
        shap_inter  = explainer.shap_interaction_values(X_scaled)
    except Exception as e:
        print(f"[xai] SHAP interactions not available for this model: {e}")
        return pd.DataFrame()

    # shap_inter shape: (n_samples, n_features, n_features)
    mean_inter = np.abs(shap_inter).mean(axis=0)
    n_feat     = len(feature_names)

    # Heatmap of off-diagonal interactions
    mask = ~np.eye(n_feat, dtype=bool)
    disp = mean_inter.copy()
    disp[~mask] = 0

    fig, ax = plt.subplots(figsize=(max(6, n_feat * 0.6), max(5, n_feat * 0.55)))
    im = ax.imshow(disp, cmap="Blues")
    ax.set_xticks(range(n_feat)); ax.set_xticklabels(feature_names, rotation=90, fontsize=7)
    ax.set_yticks(range(n_feat)); ax.set_yticklabels(feature_names, fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("SHAP Interaction Matrix (mean |interaction|)")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[xai] Saved SHAP interaction matrix → {save_path}")

    # Build pairwise DataFrame (upper triangle only, excl. diagonal)
    rows = []
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            rows.append({
                "feature_a":             feature_names[i],
                "feature_b":             feature_names[j],
                "mean_abs_interaction":  float(mean_inter[i, j]),
            })
    df_inter = (pd.DataFrame(rows)
                .sort_values("mean_abs_interaction", ascending=False)
                .reset_index(drop=True))

    top50 = df_inter.head(50)
    top50.to_csv(get_config().REPORTS_DIR / "shap_interaction_pairs.csv", index=False)

    # Save full matrix CSV
    pd.DataFrame(mean_inter, index=feature_names, columns=feature_names).to_csv(
        get_config().REPORTS_DIR / "shap_interactions_matrix.csv"
    )
    np.save(get_config().REPORTS_DIR / "all_shap_interactions_raw.npy", shap_inter)
    print(f"[xai] Saved SHAP interaction CSVs → reports/")
    return df_inter


def learning_curve_plot(
    model,
    X_scaled: np.ndarray,
    y: np.ndarray,
    feature_names: list = None,
    scoring: str = "r2",
    cv: int = 5,
    n_jobs: int = -1,
    save_path=None,
) -> matplotlib.figure.Figure:
    """Plot sklearn learning curves (training set size vs. train/val score).

    Args:
        model: Unfitted (or cloned) estimator.
        X_scaled: Scaled feature matrix (numpy array).
        y: Target values.
        feature_names: Unused — kept for API consistency.
        scoring: sklearn scoring string.
        cv: Cross-validation folds.
        n_jobs: Parallel jobs for sklearn learning_curve.
        save_path: Output PDF path. Defaults to config.FIGURES_DIR / 'learning_curve.pdf'.

    Returns:
        Matplotlib Figure object.
    """
    from sklearn.model_selection import learning_curve
    from sklearn.base import clone

    save_path = save_path or get_config().FIGURES_DIR / "learning_curve.pdf"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        m = clone(model)
    except Exception:
        m = model

    train_sizes, train_scores, val_scores = learning_curve(
        m, X_scaled, y,
        train_sizes=np.linspace(0.1, 1.0, 8),
        cv=cv, scoring=scoring,
        n_jobs=n_jobs, random_state=get_config().RANDOM_SEED,
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_mean, "o-", color="#4C72B0", label="Train")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color="#4C72B0")
    ax.plot(train_sizes, val_mean, "s-", color="#DD8452", label="Validation")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color="#DD8452")
    ax.set_xlabel("Training set size")
    ax.set_ylabel(scoring.upper())
    ax.set_title(f"Learning Curve ({type(model).__name__})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[xai] Saved learning curve → {save_path}")
    return fig


def show_tree(
    model,
    feature_names: list,
    max_depth: int = 4,
    save_path=None,
) -> matplotlib.figure.Figure:
    """Visualise a decision tree (CART / C5.0-style models).

    Works for any sklearn DecisionTreeRegressor / DecisionTreeClassifier.
    For ensemble models, plots the first estimator.

    Args:
        model: Fitted tree-based estimator.
        feature_names: Ordered list of feature names.
        max_depth: Maximum depth of the displayed tree (for readability).
        save_path: Output PDF path. Defaults to config.FIGURES_DIR / 'decision_tree.pdf'.

    Returns:
        Matplotlib Figure object.

    Raises:
        ValueError: If the model has no tree structure to display.
    """
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

    save_path = save_path or get_config().FIGURES_DIR / "decision_tree.pdf"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    tree = None
    if isinstance(model, (DecisionTreeRegressor, DecisionTreeClassifier)):
        tree = model
    elif hasattr(model, "estimators_"):
        first = model.estimators_[0]
        if isinstance(first, (DecisionTreeRegressor, DecisionTreeClassifier)):
            tree = first
        elif hasattr(first, "estimators_"):
            tree = first.estimators_[0]
    elif hasattr(model, "estimators"):
        first = model.estimators[0]
        if isinstance(first, (DecisionTreeRegressor, DecisionTreeClassifier)):
            tree = first

    if tree is None:
        raise ValueError(
            f"show_tree: cannot extract a decision tree from {type(model).__name__}. "
            "Only DecisionTreeRegressor/Classifier and single-estimator ensembles are supported."
        )

    n_nodes = tree.tree_.node_count
    fig_w   = max(12, min(40, n_nodes * 0.6))
    fig_h   = max(6, max_depth * 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    plot_tree(
        tree,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        max_depth=max_depth,
        ax=ax,
        fontsize=8,
    )
    ax.set_title(
        f"Decision Tree — {type(model).__name__}  "
        f"(criterion={getattr(tree, 'criterion', '?')}, "
        f"depth≤{tree.get_depth()})",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[xai] Saved decision tree → {save_path}")
    return fig


def confusion_matrix_plot(
    model,
    X_scaled: np.ndarray,
    y: np.ndarray,
    class_names: list = None,
    save_path=None,
) -> matplotlib.figure.Figure:
    """Plot a normalised + raw confusion matrix for a classifier.

    Args:
        model: Fitted binary or multiclass classifier.
        X_scaled: Scaled feature matrix (numpy array).
        y: True labels (numpy array).
        class_names: Display labels for classes. Defaults to string integers.
        save_path: Output PDF path. Defaults to config.FIGURES_DIR / 'confusion_matrix.pdf'.

    Returns:
        Matplotlib Figure object.
    """
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    save_path   = save_path or get_config().FIGURES_DIR / "confusion_matrix.pdf"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    preds       = model.predict(X_scaled)
    classes     = class_names or sorted(set(y))
    cm_raw      = confusion_matrix(y, preds)
    cm_norm     = confusion_matrix(y, preds, normalize="true")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ConfusionMatrixDisplay(cm_raw,  display_labels=classes).plot(ax=ax1, colorbar=False)
    ax1.set_title("Confusion Matrix (counts)")
    ConfusionMatrixDisplay(cm_norm, display_labels=classes).plot(ax=ax2, colorbar=False)
    ax2.set_title("Confusion Matrix (normalised)")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[xai] Saved confusion matrix → {save_path}")
    return fig


def classification_report_plot(
    model,
    X_scaled: np.ndarray,
    y: np.ndarray,
    class_names: list = None,
    save_path=None,
) -> pd.DataFrame:
    """Render the per-class classification report as a heatmap and save as CSV.

    Args:
        model: Fitted classifier.
        X_scaled: Scaled feature matrix (numpy array).
        y: True labels (numpy array).
        class_names: Display labels for classes.
        save_path: Output PDF path. Defaults to config.FIGURES_DIR / 'classification_report.pdf'.

    Returns:
        DataFrame with precision, recall, f1-score, support per class.
    """
    from sklearn.metrics import classification_report

    save_path = save_path or get_config().FIGURES_DIR / "classification_report.pdf"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    preds   = model.predict(X_scaled)
    report  = classification_report(
        y, preds,
        target_names=[str(c) for c in (class_names or sorted(set(y)))],
        zero_division=0, output_dict=True,
    )
    df_rep  = pd.DataFrame(report).T
    df_rep.to_csv(get_config().REPORTS_DIR / "classification_report.csv")

    # Heatmap (metrics only, skip accuracy/macro/weighted rows for the main cells)
    metric_rows = [r for r in df_rep.index if r not in ("accuracy",)]
    df_heat     = df_rep.loc[metric_rows, ["precision", "recall", "f1-score"]]

    fig, ax = plt.subplots(figsize=(6, max(3, len(metric_rows) * 0.5)))
    im = ax.imshow(df_heat.values.astype(float), vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(df_heat.columns)))
    ax.set_xticklabels(df_heat.columns)
    ax.set_yticks(range(len(df_heat.index)))
    ax.set_yticklabels(df_heat.index)
    for i in range(len(df_heat.index)):
        for j in range(len(df_heat.columns)):
            v = df_heat.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="black" if 0.3 < float(v) < 0.85 else "white")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Classification Report")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[xai] Saved classification report → {save_path}")
    return df_rep


def bootstrap_ci_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    ci_alpha: float = 0.05,
    save_path=None,
) -> matplotlib.figure.Figure:
    """Plot predicted vs. true values with bootstrap confidence band.

    Draws the scatter of predictions, the identity line, and a shaded CI band
    around the identity computed via bootstrap resampling.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions.
        n_bootstrap: Number of bootstrap resamples.
        ci_alpha: Significance level for the CI (0.05 → 95 % CI).
        save_path: Output PDF path. Defaults to config.FIGURES_DIR / 'bootstrap_ci.pdf'.

    Returns:
        Matplotlib Figure object.
    """
    from sklearn.metrics import r2_score as _r2

    save_path = save_path or get_config().FIGURES_DIR / "bootstrap_ci.pdf"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    rng   = np.random.default_rng(get_config().RANDOM_SEED)
    n     = len(y_true)
    lo_q, hi_q = ci_alpha / 2, 1 - ci_alpha / 2

    # Bootstrap per-sample prediction intervals
    residuals = y_pred - y_true
    boot_residuals = np.array([
        rng.choice(residuals, size=n, replace=True) for _ in range(n_bootstrap)
    ])
    ci_lo = y_pred + np.quantile(boot_residuals, lo_q, axis=0)
    ci_hi = y_pred + np.quantile(boot_residuals, hi_q, axis=0)

    sort_idx = np.argsort(y_true)
    r2       = float(_r2(y_true, y_pred))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.fill_between(y_true[sort_idx], ci_lo[sort_idx], ci_hi[sort_idx],
                    alpha=0.2, color="#4C72B0",
                    label=f"{int((1 - ci_alpha) * 100)}% CI")
    ax.scatter(y_true, y_pred, s=18, alpha=0.6, color="#4C72B0", zorder=3)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", linewidth=1, label="Perfect fit")
    ax.set_xlabel("True values")
    ax.set_ylabel("Predicted values")
    ax.set_title(f"Predicted vs. True  (R² = {r2:.4f})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[xai] Saved bootstrap CI plot → {save_path}")
    return fig


# ── CV XAI helpers ───────────────────────────────────────────────────────────

def _is_ensemble(model) -> bool:
    return hasattr(model, "models") and isinstance(getattr(model, "models", None), dict)


def _score_model(model, X: np.ndarray, y: np.ndarray, scoring: str, task: str) -> float:
    """Compute a single scalar score for any model/ensemble without sklearn validation."""
    from sklearn.metrics import r2_score, f1_score, roc_auc_score, accuracy_score
    if scoring == "r2":
        return float(r2_score(y, np.atleast_1d(model.predict(X))))
    if scoring in ("f1", "f1_macro", "f1_weighted"):
        return float(f1_score(y, np.atleast_1d(model.predict(X)), average=scoring.split("_")[1] if "_" in scoring else "binary", zero_division=0))
    if scoring == "roc_auc":
        proba = np.atleast_1d(model.predict_proba(X)[:, 1])
        return float(roc_auc_score(y, proba))
    if scoring == "accuracy":
        return float(accuracy_score(y, np.atleast_1d(model.predict(X))))
    raise ValueError(f"Unsupported scoring: {scoring!r}")


def _manual_perm_importance(
    model, X: np.ndarray, y: np.ndarray,
    scoring: str, task: str, n_repeats: int = 5, random_state: int = 42,
) -> np.ndarray:
    """Model-agnostic permutation importance; returns importances_mean array."""
    rng  = np.random.RandomState(random_state)
    base = _score_model(model, X, y, scoring, task)
    means = np.zeros(X.shape[1])
    for fi in range(X.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, fi] = rng.permutation(X_perm[:, fi])
            scores.append(_score_model(model, X_perm, y, scoring, task))
        means[fi] = base - float(np.mean(scores))
    return means


def compute_shap_values(
    model,
    X_scaled: np.ndarray,
    task: str = "regression",
) -> np.ndarray:
    """Compute SHAP values for a fitted model on a scaled feature matrix.

    Public wrapper around the internal fold helper. Use this to get raw SHAP
    values for shap_dependence_plots. Tries TreeExplainer for tree-based single
    models, falls back to KernelExplainer for ensembles and DL models.

    Args:
        model: Fitted estimator or ensemble.
        X_scaled: Scaled feature matrix (numpy array).
        task: 'regression' or 'classification'.

    Returns:
        2-D SHAP values array of shape (n_samples, n_features).
    """
    if not _HAS_SHAP:
        raise ImportError("shap required")
    return _compute_shap_fold(model, X_scaled, task)


def shap_dependence_plots(
    shap_values: np.ndarray,
    X_scaled: np.ndarray,
    feature_names: list,
    top_n: int = 6,
    interaction_feature: str = None,
    task: str = "regression",
    save_path=None,
) -> plt.Figure:
    """SHAP dependence plots: SHAP value vs feature value for top-N features.

    Each panel shows one feature's SHAP value on the y-axis against its scaled
    value on the x-axis. Points are coloured by an interaction feature if
    specified, otherwise by the feature's own value.

    Args:
        shap_values: 2-D array (n_samples, n_features).
        X_scaled: Scaled feature matrix used to compute shap_values.
        feature_names: Feature names corresponding to columns.
        top_n: Number of top features (by mean |SHAP|) to plot.
        interaction_feature: Feature name to use for point colouring.
            If None, colours by the feature's own value.
        task: 'regression' or 'classification' (affects colormap).
        save_path: Output PDF path. Defaults to config.FIGURES_DIR /
            'shap_dependence_{task}.pdf'.

    Returns:
        Matplotlib Figure.
    """
    if not _HAS_SHAP:
        raise ImportError("shap required for shap_dependence_plots")

    save_path = save_path or get_config().FIGURES_DIR / f"shap_dependence_{task}.pdf"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    importance_order = np.argsort(np.abs(shap_values).mean(axis=0))[::-1]
    top_idx          = importance_order[:top_n]
    top_features     = [feature_names[i] for i in top_idx]

    int_idx = None
    if interaction_feature and interaction_feature in feature_names:
        int_idx = feature_names.index(interaction_feature)

    ncols    = min(3, top_n)
    nrows    = (top_n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5))
    axes_flat = np.array(axes).flatten() if top_n > 1 else np.array([axes])
    cmap      = "coolwarm" if task == "regression" else "RdYlGn"

    for ax, feat_name, feat_i in zip(axes_flat, top_features, top_idx):
        color_vals = X_scaled[:, int_idx] if int_idx is not None else X_scaled[:, feat_i]
        sc = ax.scatter(
            X_scaled[:, feat_i], shap_values[:, feat_i],
            c=color_vals, cmap=cmap, alpha=0.7, s=18, edgecolors="none",
        )
        ax.axhline(0, color="gray", lw=0.8, linestyle="--")
        ax.set_xlabel(feat_name, fontsize=9)
        ax.set_ylabel("SHAP value", fontsize=9)
        ax.set_title(feat_name, fontsize=10)
        clabel = interaction_feature if int_idx is not None else feat_name
        plt.colorbar(sc, ax=ax, label=clabel, pad=0.02)

    for ax in axes_flat[top_n:]:
        ax.set_visible(False)

    title = "SHAP Dependence Plots — " + ("Regression" if task == "regression" else "Classification")
    if interaction_feature:
        title += f"\n(colour = {interaction_feature})"
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[xai] Saved SHAP dependence plots → {save_path}")
    return fig


def _compute_shap_fold(model, X_val_s: np.ndarray, task: str = "regression") -> np.ndarray:
    """SHAP values on one fold's validation split. Works for single models and ensembles."""
    model_type = type(model).__name__.lower()
    use_tree = (
        not _is_ensemble(model)
        and any(t in model_type for t in ["catboost", "xgb", "forest", "tree", "gradient", "lgbm", "boost"])
    )
    if use_tree:
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_val_s, check_additivity=False)
            if isinstance(sv, list):
                sv = sv[1] if task == "classification" else sv[0]
            return np.atleast_2d(sv)
        except Exception:
            pass
    background = shap.sample(X_val_s, min(50, len(X_val_s)))
    if task == "regression":
        fn = lambda x: np.atleast_1d(model.predict(x).astype(float))
    else:
        fn = lambda x: np.atleast_1d(model.predict_proba(x)[:, 1])
    sv = shap.KernelExplainer(fn, background).shap_values(X_val_s, nsamples=50)
    return np.atleast_2d(sv)


def shap_analysis_cv(
    model,
    preprocessor,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: list,
    feature_names: list,
    task: str = "regression",
    save_path=None,
    label: str = "",
) -> pd.DataFrame:
    """CV SHAP: mean ± std |SHAP| by evaluating each fold's validation split.

    Uses the already-fitted model + preprocessor — no re-training per fold.
    The std across folds captures how stable the explanation is on different
    data subsets. Works with single models and ensemble wrappers.

    Args:
        model: Fitted estimator or ensemble (predict / predict_proba).
        preprocessor: Fitted sklearn preprocessing pipeline.
        X: Raw (unscaled) training DataFrame.
        y: Target Series.
        cv_splits: List of (train_idx, val_idx) tuples.
        feature_names: Ordered list of feature names.
        task: 'regression' or 'classification'.
        save_path: Output PDF path.
        label: Extra label for filenames/titles.

    Returns:
        DataFrame with columns [feature, mean_abs_shap, std_abs_shap].
    """
    if not _HAS_SHAP:
        raise ImportError("shap required")

    tag    = "reg" if task == "regression" else "clf"
    suffix = f"_{label}" if label else ""
    save_path = save_path or get_config().FIGURES_DIR / f"shap_cv_{tag}{suffix}.pdf"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    X_arr = X.values
    fold_importances = []

    for fold_idx, (_, val_idx) in enumerate(cv_splits):
        try:
            X_val_s = preprocessor.transform(X_arr[val_idx])
            sv = _compute_shap_fold(model, X_val_s, task)
            fold_importances.append(np.abs(sv).mean(axis=0))
        except Exception as e:
            print(f"[xai] shap_analysis_cv fold {fold_idx + 1} skipped: {e}")

    if not fold_importances:
        raise RuntimeError("[xai] shap_analysis_cv: all folds failed")

    mean_shap = np.mean(fold_importances, axis=0)
    std_shap  = np.std(fold_importances,  axis=0)

    importance = pd.DataFrame({
        "feature":       feature_names,
        "mean_abs_shap": mean_shap,
        "std_abs_shap":  std_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    color = "#4C72B0" if task == "regression" else "#DD8452"
    rev   = importance.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(5, len(feature_names) * 0.4)))
    ax.barh(rev["feature"], rev["mean_abs_shap"], xerr=rev["std_abs_shap"],
            color=color, ecolor="gray", capsize=3, alpha=0.85)
    ax.set_xlabel("Mean |SHAP|  (±1 std across folds)")
    lbl = f" — {label}" if label else ""
    ax.set_title(f"CV SHAP  [{len(cv_splits)} folds]  {task.capitalize()}{lbl}")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[xai] Saved SHAP-CV → {save_path}")

    importance.to_csv(get_config().REPORTS_DIR / f"shap_cv_{tag}{suffix}.csv", index=False)
    return importance


def pdp_plots_cv(
    model,
    preprocessor,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: list,
    feature_names: list,
    task: str = "regression",
    top_n: int = 8,
    top_features: list = None,
    grid_resolution: int = 50,
    save_path=None,
) -> dict:
    """CV PDP: mean ± std partial dependence by marginalising over each fold's val set.

    For each fold, the held-out validation split is used as the marginalisation
    distribution → one PDP curve per fold → mean ± std bands. Works with single
    models and ensembles.

    Args:
        model: Fitted estimator or ensemble.
        preprocessor: Fitted sklearn preprocessing pipeline.
        X: Raw training DataFrame (original scale).
        y: Target Series.
        cv_splits: List of (train_idx, val_idx) tuples.
        feature_names: Ordered list of feature names.
        task: 'regression' or 'classification'.
        top_n: Number of top features to plot (ranked by SHAP if available).
        top_features: Explicit feature list — overrides SHAP ranking.
        grid_resolution: Grid points per feature.
        save_path: Output PDF path.

    Returns:
        Dict mapping feature name → (grid, mean_pdp, std_pdp).
    """
    tag = "reg" if task == "regression" else "clf"
    save_path = save_path or get_config().FIGURES_DIR / f"pdp_cv_{tag}.pdf"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    X_arr = X.values

    if top_features is None:
        if _HAS_SHAP:
            try:
                X_all_s = preprocessor.transform(X_arr)
                sv = _compute_shap_fold(model, X_all_s, task)
                top_idx = np.argsort(np.abs(sv).mean(axis=0))[::-1][:top_n]
                top_features = [feature_names[i] for i in top_idx]
                plt.close("all")
            except Exception:
                top_features = feature_names[:top_n]
        else:
            top_features = feature_names[:top_n]

    feat_grids = {
        feat: np.linspace(
            float(X_arr[:, feature_names.index(feat)].min()),
            float(X_arr[:, feature_names.index(feat)].max()),
            grid_resolution,
        )
        for feat in top_features
    }
    pdp_curves = {f: [] for f in top_features}

    for fold_idx, (_, val_idx) in enumerate(cv_splits):
        X_val = X_arr[val_idx]
        for feat in top_features:
            try:
                fi   = feature_names.index(feat)
                grid = feat_grids[feat]
                y_pdp = np.empty(len(grid))
                for gi, val in enumerate(grid):
                    X_tmp = X_val.copy()
                    X_tmp[:, fi] = val
                    X_tmp_s = preprocessor.transform(X_tmp)
                    if task == "regression":
                        y_pdp[gi] = np.atleast_1d(model.predict(X_tmp_s)).mean()
                    else:
                        y_pdp[gi] = np.atleast_1d(model.predict_proba(X_tmp_s)[:, 1]).mean()
                pdp_curves[feat].append(y_pdp)
            except Exception as e:
                print(f"[xai] pdp_plots_cv fold {fold_idx + 1} / '{feat}': {e}")

    n_feat    = len(top_features)
    ncols     = min(4, n_feat)
    nrows     = (n_feat + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes_flat = np.array(axes).flatten() if n_feat > 1 else np.array([axes])
    color     = "#4C72B0" if task == "regression" else "#DD8452"
    results: dict = {}

    for ax, feat in zip(axes_flat, top_features):
        if not pdp_curves[feat]:
            ax.set_visible(False)
            continue
        grid     = feat_grids[feat]
        mat      = np.array(pdp_curves[feat])
        mean_pdp = mat.mean(axis=0)
        std_pdp  = mat.std(axis=0)
        results[feat] = (grid, mean_pdp, std_pdp)
        ax.plot(grid, mean_pdp, color=color, lw=2, label="Mean")
        ax.fill_between(grid, mean_pdp - std_pdp, mean_pdp + std_pdp,
                        alpha=0.25, color=color, label="±1 std")
        ax.set_xlabel(feat, fontsize=9)
        ax.set_ylabel("Partial dep.", fontsize=9)
        ax.set_title(feat, fontsize=10)
        ax.legend(fontsize=8)

    for ax in axes_flat[n_feat:]:
        ax.set_visible(False)

    label_str = "Regression" if task == "regression" else "Classification"
    fig.suptitle(f"PDP with fold uncertainty  [{len(cv_splits)} folds]  {label_str}", fontsize=12)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[xai] Saved PDP-CV → {save_path}")
    return results


def permutation_importance_cv(
    model,
    preprocessor,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: list,
    feature_names: list,
    task: str = "regression",
    scoring: str = None,
    save_path=None,
) -> pd.DataFrame:
    """CV permutation importance: mean ± std across fold validation splits.

    Permutation importance is computed independently on each fold's validation
    set → mean ± std captures fold-level variability. Works with single models
    and ensembles.

    Args:
        model: Fitted estimator or ensemble.
        preprocessor: Fitted sklearn preprocessing pipeline.
        X: Raw training DataFrame.
        y: Target Series.
        cv_splits: List of (train_idx, val_idx) tuples.
        feature_names: Ordered list of feature names.
        task: 'regression' or 'classification' (sets default scoring).
        scoring: sklearn scoring string. Defaults to 'r2' for regression, 'f1' for clf.
        save_path: Output PDF path.

    Returns:
        DataFrame with columns [feature, importance_mean, importance_std].
    """
    if scoring is None:
        scoring = "r2" if task == "regression" else "f1"

    tag = "reg" if task == "regression" else "clf"
    save_path = save_path or get_config().FIGURES_DIR / f"permutation_importance_cv_{tag}.pdf"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    X_arr, y_arr = X.values, y.values
    fold_means = []

    use_manual = _is_ensemble(model) or not hasattr(model, "fit")

    for fold_idx, (_, val_idx) in enumerate(cv_splits):
        try:
            X_val_s = preprocessor.transform(X_arr[val_idx])
            y_val   = y_arr[val_idx]
            if use_manual:
                fold_means.append(
                    _manual_perm_importance(model, X_val_s, y_val, scoring, task,
                                            n_repeats=5, random_state=get_config().RANDOM_SEED)
                )
            else:
                result = sk_perm(model, X_val_s, y_val, n_repeats=5,
                                 random_state=get_config().RANDOM_SEED, scoring=scoring)
                fold_means.append(result.importances_mean)
        except Exception as e:
            print(f"[xai] permutation_importance_cv fold {fold_idx + 1} skipped: {e}")

    if not fold_means:
        raise RuntimeError("[xai] permutation_importance_cv: all folds failed")

    mat   = np.array(fold_means)
    means = mat.mean(axis=0)
    stds  = mat.std(axis=0)

    imp_df = pd.DataFrame({
        "feature":          feature_names,
        "importance_mean":  means,
        "importance_std":   stds,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    color = "#4C72B0" if task == "regression" else "#DD8452"
    rev   = imp_df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.35)))
    ax.barh(rev["feature"], rev["importance_mean"], xerr=rev["importance_std"],
            color=color, ecolor="gray", capsize=3, alpha=0.85)
    ax.set_xlabel(f"Mean decrease in {scoring}  (±1 std across folds)")
    ax.set_title(f"CV Permutation Importance  [{len(cv_splits)} folds]  {task.capitalize()}")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[xai] Saved CV permutation importance → {save_path}")

    imp_df.to_csv(get_config().REPORTS_DIR / f"permutation_importance_cv_{tag}.csv", index=False)
    return imp_df


if __name__ == "__main__":
    from .data_loading import load_clean
    from .final_training import train_final
    from .preprocessing import split_xy, train_test

    df, _ = load_clean()
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test(X, y)
    feature_names = list(X.columns)

    model, pre = train_final("catboost", X_train, y_train)
    X_test_scaled = pre.transform(X_test.values)

    shap_analysis(model, X_test_scaled, feature_names)
    pdp_plots(model, X_test_scaled, feature_names)
    permutation_importance(model, X_test_scaled, y_test.values, feature_names)
