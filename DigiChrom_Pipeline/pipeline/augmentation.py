"""Data augmentation for tabular data: Gaussian noise."""

from __future__ import annotations

import numpy as np
import pandas as pd

class GaussianNoiseAugmenter:
    """Simple augmentation: add Gaussian noise to features.

    Useful as a lightweight fallback when CTGAN is not available.
    Target column is perturbed with smaller noise to preserve label fidelity.
    """

    def __init__(self, noise_std: float = 0.05, random_state: int = 42) -> None:
        self.noise_std    = noise_std
        self.random_state = random_state

    def augment(
        self,
        df: pd.DataFrame,
        target_col: str,
        n_synthetic: int,
        feature_noise_std: float = None,
        target_noise_std: float = None,
    ) -> pd.DataFrame:
        """Generate n_synthetic rows by perturbing randomly chosen real rows."""
        rng  = np.random.RandomState(self.random_state)
        fns  = feature_noise_std or self.noise_std
        tns  = target_noise_std  or (self.noise_std * 0.3)

        idx      = rng.choice(len(df), size=n_synthetic, replace=True)
        syn      = df.iloc[idx].copy().reset_index(drop=True)
        feat_cols = [c for c in df.columns if c != target_col]

        stds = df[feat_cols].std().values
        syn[feat_cols] += rng.normal(0, fns, size=(n_synthetic, len(feat_cols))) * stds
        syn[target_col] += rng.normal(0, tns, size=n_synthetic) * float(df[target_col].std())
        return pd.concat([df, syn], ignore_index=True)


def augment_dataset(
    df: pd.DataFrame,
    target_col: str,
    n_synthetic: int = 200,
    method: str = "gaussian",
    **kwargs,
) -> pd.DataFrame:
    """Augment a dataset with synthetic rows.

    Args:
        df: Original DataFrame.
        target_col: Name of the target column.
        n_synthetic: Number of synthetic rows to generate.
        method: 'ctgan' (requires ctgan package) or 'gaussian'.

    Returns:
        Augmented DataFrame.
    """
    if method == "ctgan":
        aug = CTGANAugmenter(**kwargs)
        return aug.augment(df, n_synthetic)
    if method == "gaussian":
        aug = GaussianNoiseAugmenter(**kwargs)
        return aug.augment(df, target_col, n_synthetic)
    raise ValueError(f"Unknown method: '{method}'. Use 'ctgan' or 'gaussian'.")
