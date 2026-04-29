import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from .config import get_config


def load_raw(path=None) -> pd.DataFrame:
    """Load the raw Excel file and replace missing-value sentinels with NaN.

    Args:
        path: Path to the Excel file. Defaults to config.DATA_PATH.

    Returns:
        Raw DataFrame with sentinel values replaced by pd.NA.
    """
    path = path or get_config().DATA_PATH
    df = pd.read_excel(path)
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
    relevant_cols = get_config().FEATURE_COLS + [get_config().TARGET_COL]
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

    Drops rows that contain any NaN in feature or target columns and resets
    the index.

    Args:
        path: Path to the Excel file. Defaults to config.DATA_PATH.

    Returns:
        Tuple of (df_clean, report) where df_clean contains only feature and
        target columns with no missing values, and report is the dict from
        validate_data.
    """
    df = load_raw(path)
    report = validate_data(df)

    relevant_cols = [c for c in get_config().FEATURE_COLS + [get_config().TARGET_COL] if c in df.columns]
    df_clean = df[relevant_cols].dropna().reset_index(drop=True)

    print(f"[data_loading] {report['total_rows']} rows loaded, "
          f"{report['rows_with_nan']} dropped → {len(df_clean)} clean rows")
    if report["missing_columns"]:
        print(f"[data_loading] WARNING: columns not found: {report['missing_columns']}")

    return df_clean, report


if __name__ == "__main__":
    df, report = load_clean()
    print(df.head())
    print(report)
