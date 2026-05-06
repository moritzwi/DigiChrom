import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, RepeatedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from .config import get_config


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a clean DataFrame into feature matrix X and target series y.

    Includes any '<col>_was_missing' indicator columns produced by
    data_loading.add_missing_indicators() for columns listed in
    config.INDICATOR_COLS. Columns entirely NaN are silently dropped.

    Returns:
        Tuple of (X, y) where X is a DataFrame of features and y is a Series
        of target values.
    """
    cfg = get_config()
    feat_cols = [c for c in cfg.FEATURE_COLS if c in df.columns]

    # Append indicator columns if present in the DataFrame
    indicator_cols = getattr(cfg, "INDICATOR_COLS", [])
    for col in indicator_cols:
        ind = f"{col}_was_missing"
        if ind in df.columns and ind not in feat_cols:
            feat_cols.append(ind)

    X = df[feat_cols].copy()
    all_nan = [c for c in X.columns if X[c].isna().all()]
    if all_nan:
        X = X.drop(columns=all_nan)

    # One-hot encode any non-numeric column (handles object, string[pyarrow], category, etc.)
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False, dtype=float)
        print(f"[preprocessing] OHE applied to: {cat_cols} → {X.shape[1]} total features")

    target = cfg.TARGET_COL
    if isinstance(target, list):
        y = df[target].copy()           # DataFrame → multi-output
    else:
        y = df[target].copy()           # Series → single-output
    return X, y


def make_preprocessor() -> Pipeline:
    """Build a fresh, unfitted sklearn preprocessing pipeline.

    Returns:
        An unfitted sklearn Pipeline with mean imputation then standard scaling.
    """
    # SimpleImputer must come before StandardScaler so NaNs are filled first
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # noqa: F821
        ("scaler", StandardScaler()),
    ])


def train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = None,
    random_state: int = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split X and y into train and test sets.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        test_size: Fraction reserved for test. Defaults to config.TEST_SIZE.
        random_state: Random seed. Defaults to config.RANDOM_SEED.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    test_size = test_size or get_config().TEST_SIZE
    random_state = random_state or get_config().RANDOM_SEED
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def cross_val_splits(
    X: pd.DataFrame, y: pd.Series, n_folds: int = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate stratified KFold index splits for cross-validation.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        n_folds: Number of folds. Defaults to config.CV_FOLDS.

    Returns:
        List of (train_indices, val_indices) tuples, one per fold.
    """
    n_folds = n_folds or get_config().CV_FOLDS
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=get_config().RANDOM_SEED)
    return list(kf.split(X, y))


def repeated_cross_val_splits(
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = None,
    n_repeats: int = 3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate repeated KFold splits for more stable CV estimates.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        n_folds: Number of folds per repeat. Defaults to config.CV_FOLDS.
        n_repeats: Number of independent repetitions.

    Returns:
        List of (train_indices, val_indices) tuples across all repeats.
    """
    n_folds = n_folds or get_config().CV_FOLDS
    rkf = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats,
                        random_state=get_config().RANDOM_SEED)
    return list(rkf.split(X, y))


def stratified_cross_val_splits(
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = None,
    n_bins: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate KFold splits stratified by binned target values (for regression).

    Bins the continuous target into n_bins quantile-based categories and uses
    StratifiedKFold to keep the target distribution consistent across folds.

    Args:
        X: Feature DataFrame.
        y: Continuous target Series.
        n_folds: Number of folds. Defaults to config.CV_FOLDS.
        n_bins: Number of bins for stratification labels.

    Returns:
        List of (train_indices, val_indices) tuples.
    """
    from sklearn.model_selection import StratifiedKFold
    n_folds   = n_folds or get_config().CV_FOLDS
    bin_edges = np.quantile(y, np.linspace(0, 1, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    labels    = np.digitize(y, bin_edges[1:-1])
    skf       = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                 random_state=get_config().RANDOM_SEED)
    return list(skf.split(X, labels))


def add_polynomial_features(
    X: pd.DataFrame,
    degree: int = 2,
    interaction_only: bool = False,
    include_bias: bool = False,
) -> pd.DataFrame:
    """Add polynomial and interaction features to X.

    Args:
        X: Feature DataFrame.
        degree: Polynomial degree.
        interaction_only: If True, only interaction terms (no powers).
        include_bias: If True, include a bias column of ones.

    Returns:
        New DataFrame with original + polynomial features.
    """
    poly       = PolynomialFeatures(degree=degree, interaction_only=interaction_only,
                                     include_bias=include_bias)
    X_poly     = poly.fit_transform(X.values)
    feat_names = poly.get_feature_names_out(list(X.columns))
    return pd.DataFrame(X_poly, columns=feat_names, index=X.index)


def select_k_best_features(
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 10,
) -> tuple[pd.DataFrame, list[str]]:
    """Select the k best features by F-statistic (ANOVA / f_regression).

    Args:
        X: Feature DataFrame.
        y: Target Series.
        k: Number of features to retain. Clamped to n_features.

    Returns:
        Tuple of (filtered DataFrame, list of selected column names).
    """
    k         = min(k, X.shape[1])
    selector  = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X.values, y.values)
    mask      = selector.get_support()
    sel_cols  = [c for c, m in zip(X.columns, mask) if m]
    return X[sel_cols].copy(), sel_cols


def align_features(X: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Align X to the exact columns a fitted preprocessor expects.

    Missing columns are added as NaN; extra columns are dropped; order matches
    feature_cols. This is needed when a saved model was trained on a different
    column set than the current session data.
    """
    X = X.copy()
    for col in feature_cols:
        if col not in X.columns:
            X[col] = np.nan
    return X[feature_cols]


def feature_bounds(X: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Compute the min/max range for each feature column.

    Args:
        X: Feature DataFrame (typically the training set).

    Returns:
        Dictionary mapping column name to (min, max) tuple.
    """
    return {col: (float(X[col].min()), float(X[col].max())) for col in X.columns}


if __name__ == "__main__":
    from .data_loading import load_clean
    df, _ = load_clean()
    X, y = split_xy(df)
    print(f"X: {X.shape}, y: {y.shape}")
    splits = cross_val_splits(X, y)
    print(f"CV splits: {len(splits)}")
