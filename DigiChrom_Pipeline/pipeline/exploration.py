import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).parent.parent))
from .config import get_config


def correlation_matrix(df: pd.DataFrame, save_path=None) -> matplotlib.figure.Figure:
    """Plot and save the lower-triangle correlation heatmap for all numeric columns.

    Args:
        df: DataFrame containing at least the feature and target columns.
        save_path: Output path for the PDF. Defaults to config.FIGURES_DIR /
            'correlation_matrix.pdf'.

    Returns:
        Matplotlib Figure object.
    """
    save_path = save_path or get_config().FIGURES_DIR / "correlation_matrix.pdf"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, square=True, linewidths=0.5, ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, pad=12)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[exploration] Saved correlation matrix → {save_path}")
    return fig


def feature_distributions(df: pd.DataFrame, save_path=None) -> matplotlib.figure.Figure:
    """Plot histograms for all feature columns defined in config.FEATURE_COLS.

    Args:
        df: DataFrame containing the feature columns.
        save_path: Output path for the PDF. Defaults to config.FIGURES_DIR /
            'feature_distributions.pdf'.

    Returns:
        Matplotlib Figure object.
    """
    save_path = save_path or get_config().FIGURES_DIR / "feature_distributions.pdf"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    cols = [c for c in get_config().FEATURE_COLS if c in df.columns]
    n = len(cols)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        axes[i].hist(df[col].dropna(), bins=30, edgecolor="white", color="#4C72B0")
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xlabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[exploration] Saved distributions → {save_path}")
    return fig


def target_vs_features(
    df: pd.DataFrame, save_path=None, top_n: int = 12
) -> matplotlib.figure.Figure:
    """Plot scatter plots of the target against the top-N most correlated features.

    Args:
        df: DataFrame containing feature and target columns.
        save_path: Output path for the PDF. Defaults to config.FIGURES_DIR /
            'target_vs_features.pdf'.
        top_n: Number of features to include, ranked by absolute Pearson
            correlation with the target.

    Returns:
        Matplotlib Figure object.
    """
    save_path = save_path or get_config().FIGURES_DIR / "target_vs_features.pdf"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    target = get_config().TARGET_COL
    feat_cols = [c for c in get_config().FEATURE_COLS if c in df.columns]

    corr_with_target = (
        df[feat_cols + [target]].corr()[target].drop(target).abs().sort_values(ascending=False)
    )
    top_features = corr_with_target.head(top_n).index.tolist()

    ncols = 4
    nrows = (len(top_features) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()

    for i, col in enumerate(top_features):
        axes[i].scatter(df[col], df[target], alpha=0.4, s=15, color="#4C72B0")
        axes[i].set_xlabel(col, fontsize=8)
        axes[i].set_ylabel(target, fontsize=8)
        r = df[[col, target]].corr().iloc[0, 1]
        axes[i].set_title(f"r = {r:.2f}", fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Target vs. Top {top_n} Features (by |r|)", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[exploration] Saved target vs features → {save_path}")
    return fig


if __name__ == "__main__":
    from .data_loading import load_clean
    df, _ = load_clean()
    correlation_matrix(df)
    feature_distributions(df)
    target_vs_features(df)
