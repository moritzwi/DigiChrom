"""Feature mapping and column validation for Excel import."""

from __future__ import annotations

from typing import Any

import pandas as pd
from difflib import SequenceMatcher

from pipeline.feature_registry import FEATURE_COLS, TARGET_COL


def suggest_column_mapping(
    excel_cols: list[str],
    target_cols: list[str],
) -> dict[str, str | None]:
    """Suggest mapping from Excel columns to target columns using fuzzy matching.
    
    Args:
        excel_cols: Column names from uploaded Excel file.
        target_cols: Expected column names (e.g., FEATURE_COLS + [TARGET_COL]).
        
    Returns:
        Dictionary mapping target column -> excel column (or None if not found).
    """
    def similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    mapping = {}
    used_cols = set()
    
    # First pass: exact matches (case-insensitive)
    for target in target_cols:
        for excel_col in excel_cols:
            if target.lower() == excel_col.lower() and excel_col not in used_cols:
                mapping[target] = excel_col
                used_cols.add(excel_col)
                break
    
    # Only suggest fuzzy matches for non-exact matches
    remaining_targets = [t for t in target_cols if t not in mapping]
    remaining_excel = [e for e in excel_cols if e not in used_cols]
    
    for target in remaining_targets:
        if remaining_excel:
            best_match = max(
                remaining_excel,
                key=lambda e: similarity(target, e)
            )
            score = similarity(target, best_match)
            # Only suggest if similarity > 60%
            if score > 0.6:
                mapping[target] = best_match
                remaining_excel.remove(best_match)
            else:
                mapping[target] = None
        else:
            mapping[target] = None
    
    return mapping


def apply_mapping(
    df: pd.DataFrame,
    mapping: dict[str, str | None],
) -> tuple[pd.DataFrame, list[str]]:
    """Apply column mapping to dataframe.
    
    Args:
        df: DataFrame with original column names.
        mapping: Mapping from target names to source names.
        
    Returns:
        Tuple of (mapped_df, errors) where errors lists missing columns.
    """
    errors = []
    rename_dict = {}
    
    for target, source in mapping.items():
        if source is None:
            errors.append(target)
        elif source in df.columns:
            rename_dict[source] = target
        else:
            errors.append(f"{target} (source '{source}' not found)")
    
    df_mapped = df.rename(columns=rename_dict)
    return df_mapped, errors


def get_mapping_help() -> str:
    """Generate help text for feature mapping."""
    text = (
        "# Feature Mapping Hilfe\n\n"
        "Falls deine Excel-Spalten anders benannt sind, kannst du hier "
        "ein Mapping vornehmen.\n\n"
        "## Erforderliche Features:\n\n"
    )
    
    for i, col in enumerate(FEATURE_COLS, 1):
        text += f"{i}. `{col}`\n"
    
    text += f"\n## Target (Zielgröße):\n- `{TARGET_COL}`\n\n"
    
    text += (
        "## Beispiel:\n"
        "Wenn deine Excel-Spalte 'Temperatur [°C]' heißt, aber das System "
        "`Temperature [°C]` erwartet, müsst du diese Spalte beim Hochladen "
        "mappen.\n"
    )
    
    return text
