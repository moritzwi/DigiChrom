import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from .config import get_config


def parse_censored_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace censored strings like '<5' with half the detection limit (2.5).

    Reads CENSORED_COLS from config: dict mapping column name to the censoring
    prefix character (typically '<'). Unrecognised strings become NaN.
    If CENSORED_COLS is not defined in config, returns df unchanged.
    """
    censored_cols = getattr(get_config(), "CENSORED_COLS", {})
    if not censored_cols:
        return df
    df = df.copy()
    for col, symbol in censored_cols.items():
        if col not in df.columns:
            continue
        def _parse(val, sym=symbol):
            if isinstance(val, str) and val.strip().startswith(sym):
                m = re.search(r"[\d.]+", val)
                return float(m.group()) / 2 if m else np.nan
            return val
        df[col] = df[col].apply(_parse)
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add a binary '<col>_was_missing' column for each column in INDICATOR_COLS.

    Call this BEFORE any dropna so that the indicator captures the original NaN.
    The indicator (1 = step was skipped / value absent, 0 = present) is then
    treated as a regular feature; the original column is imputed normally.
    If INDICATOR_COLS is not defined in config, returns df unchanged.
    """
    indicator_cols = getattr(get_config(), "INDICATOR_COLS", [])
    if not indicator_cols:
        return df
    df = df.copy()
    for col in indicator_cols:
        if col in df.columns:
            df[f"{col}_was_missing"] = df[col].isna().astype(int)
    return df


def load_raw(path=None) -> pd.DataFrame:
    """Load the raw Excel file and replace missing-value sentinels with NaN.

    Args:
        path: Path to the Excel file. Defaults to config.DATA_PATH.

    Returns:
        Raw DataFrame with sentinel values replaced by pd.NA.
    """
    path = path or get_config().DATA_PATH
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    df.replace(get_config().MISSING_SENTINEL, pd.NA, inplace=True)
    return df


def validate_data(df: pd.DataFrame) -> dict:
    """Compute a data-quality report for the relevant feature and target columns.

    Args:
        df: Raw DataFrame as returned by load_raw.

    Returns:
        Dictionary with keys:
            total_rows (int): Number of rows in df.
            rows_with_nan (int): Rows containing at least one NaN in relevant cols.
            rows_after_drop (int): Rows remaining after dropping NaN rows.
            missing_columns (list[str]): Expected columns absent from df.
            nan_per_feature (dict[str, int]): NaN count per column (only cols > 0).
    """
    target = get_config().TARGET_COL
    target_cols = target if isinstance(target, list) else [target]
    relevant_cols = get_config().FEATURE_COLS + target_cols
    existing = [c for c in relevant_cols if c in df.columns]
    missing_cols = [c for c in relevant_cols if c not in df.columns]

    sub = df[existing]
    nan_counts = sub.isna().sum()
    rows_with_nan = sub.isna().any(axis=1).sum()

    return {
        "total_rows": len(df),
        "rows_with_nan": int(rows_with_nan),
        "rows_after_drop": int(len(df) - rows_with_nan),
        "missing_columns": missing_cols,
        "nan_per_feature": nan_counts[nan_counts > 0].to_dict(),
    }


def load_clean(path=None) -> tuple[pd.DataFrame, dict]:
    """Load, validate, and clean the dataset, keeping only relevant columns.

    Processing order:
      1. Parse censored strings (e.g. '<5' → 2.5) for CENSORED_COLS.
      2. Add binary '_was_missing' indicator columns for INDICATOR_COLS before
         any row is dropped, so the indicator captures the original NaN.
      3. Drop rows where NaN remains in non-optional feature columns or the
         target. Columns listed in INDICATOR_COLS are optional — their NaN is
         imputed later by the preprocessor.

    Args:
        path: Path to the Excel file. Defaults to config.DATA_PATH.

    Returns:
        Tuple of (df_clean, report) where df_clean contains feature + indicator
        + target columns with NaN only in INDICATOR_COLS, and report is the
        dict from validate_data.
    """
    cfg = get_config()
    df = load_raw(path)
    df = parse_censored_values(df)
    df = add_missing_indicators(df)

    report = validate_data(df)

    indicator_cols = getattr(cfg, "INDICATOR_COLS", [])
    all_feat_cols  = cfg.FEATURE_COLS + [f"{c}_was_missing" for c in indicator_cols if c in df.columns]
    target_cols    = cfg.TARGET_COL if isinstance(cfg.TARGET_COL, list) else [cfg.TARGET_COL]
    relevant_cols  = [c for c in all_feat_cols + target_cols if c in df.columns]

    # Only require non-null in non-optional features and the target
    required_cols = [c for c in relevant_cols if c not in indicator_cols]
    df_clean = df[relevant_cols].dropna(subset=required_cols).reset_index(drop=True)

    n_dropped = report["total_rows"] - len(df_clean)
    print(f"[data_loading] {report['total_rows']} rows loaded, "
          f"{n_dropped} dropped → {len(df_clean)} clean rows")
    if indicator_cols:
        n_ind = sum(df_clean[c].isna().sum() for c in indicator_cols if c in df_clean.columns)
        print(f"[data_loading] {len(indicator_cols)} indicator column(s); "
              f"{n_ind} imputable NaN(s) retained across them")
    if report["missing_columns"]:
        print(f"[data_loading] WARNING: columns not found: {report['missing_columns']}")

    return df_clean, report


if __name__ == "__main__":
    df, report = load_clean()
    print(df.head())
    print(report)
