"""Central registry of all features used in the DigiChrom pipeline.

Each entry in FEATURE_REGISTRY defines a feature with its column name,
physical unit, feature type, default inverse-ML role, and data-driven
statistics (min, max, mean, std) computed from the full cleaned dataset
(revision.xlsx, 441 rows).

The data_min / data_max values serve as default bounds for the Inverse ML
search; they can be overridden by the user at call time.

Usage::

    from pipeline.feature_registry import FEATURE_REGISTRY, FEATURE_COLS, TARGET_COL

    bounds = {k: (v["data_min"], v["data_max"]) for k, v in FEATURE_REGISTRY.items()}
    print(FEATURE_COLS)   # ordered list of column names
"""

TARGET_COL = "Thickness Cr [µm]"

TARGET_META = {
    "display_name": "Chromium Thickness",
    "unit": "µm",
    "dtype": "continuous",
    "data_min": 0.013,
    "data_max": 1.368,
    "data_mean": 0.310,
    "data_std": 0.196,
    "description": "Deposited chromium layer thickness — prediction target",
}

FEATURE_REGISTRY: dict[str, dict] = {

    # ------------------------------------------------------------------
    # Process Parameters
    # ------------------------------------------------------------------
    "pH": {
        "display_name": "pH",
        "unit": "-",
        "dtype": "continuous",
        "feature_type": "process",
        "default_role": "variable",
        "required": True,
        "data_min": 3.0,
        "data_max": 3.7,
        "data_mean": 3.44,
        "data_std": 0.179,
        "description": "Electrolyte bath pH",
    },

    "Deposition time [min]": {
        "display_name": "Deposition Time",
        "unit": "min",
        "dtype": "continuous",
        "feature_type": "process",
        "default_role": "variable",
        "required": True,
        "data_min": 1.0,
        "data_max": 20.0,
        "data_mean": 9.16,
        "data_std": 4.08,
        "description": "Duration of the electroplating process",
    },

    "Temperature [°C]": {
        "display_name": "Temperature",
        "unit": "°C",
        "dtype": "continuous",
        "feature_type": "process",
        "default_role": "variable",
        "required": True,
        "data_min": 50.0,
        "data_max": 60.0,
        "data_mean": 55.19,
        "data_std": 3.23,
        "description": "Bath temperature during deposition",
    },

    "Current density [A/dm²]": {
        "display_name": "Current Density",
        "unit": "A/dm²",
        "dtype": "continuous",
        "feature_type": "process",
        "default_role": "variable",
        "required": True,
        "data_min": 2.8,
        "data_max": 11.6,
        "data_mean": 7.05,
        "data_std": 2.09,
        "description": "Applied current density during electroplating",
    },

    # ------------------------------------------------------------------
    # Bath Chemistry
    # ------------------------------------------------------------------
    "Chromium [g/L]": {
        "display_name": "Chromium",
        "unit": "g/L",
        "dtype": "continuous",
        "feature_type": "bath",
        "default_role": "variable",
        "required": True,
        "data_min": 9.59,
        "data_max": 11.51,
        "data_mean": 10.26,
        "data_std": 0.31,
        "description": "Chromium ion concentration in the bath",
    },

    "Buffer [g/L]": {
        "display_name": "Buffer",
        "unit": "g/L",
        "dtype": "continuous",
        "feature_type": "bath",
        "default_role": "fixed",
        "required": True,
        "data_min": 69.57,
        "data_max": 92.87,
        "data_mean": 89.91,
        "data_std": 2.47,
        "description": "Buffer substance concentration",
    },

    "Complexing agent [mL/L]": {
        "display_name": "Complexing Agent",
        "unit": "mL/L",
        "dtype": "continuous",
        "feature_type": "bath",
        "default_role": "fixed",
        "required": True,
        "data_min": 37.06,
        "data_max": 50.0,
        "data_mean": 39.64,
        "data_std": 1.81,
        "description": "Complexing agent volume concentration",
    },

    "Additive [mL/L]": {
        "display_name": "Additive",
        "unit": "mL/L",
        "dtype": "continuous",
        "feature_type": "bath",
        "default_role": "fixed",
        "required": True,
        "data_min": 15.46,
        "data_max": 30.3,
        "data_mean": 22.65,
        "data_std": 3.67,
        "description": "Additive volume concentration",
    },

    "Supporting electrolyte [g/L]": {
        "display_name": "Supporting Electrolyte",
        "unit": "g/L",
        "dtype": "continuous",
        "feature_type": "bath",
        "default_role": "fixed",
        "required": True,
        "data_min": 174.35,
        "data_max": 176.43,
        "data_mean": 175.21,
        "data_std": 0.56,
        "description": "Supporting electrolyte concentration",
    },

    "Brigthener [mL/L]": {
        "display_name": "Brightener",
        "unit": "mL/L",
        "dtype": "continuous",
        "feature_type": "bath",
        "default_role": "variable",
        "required": True,
        "data_min": 1.2,
        "data_max": 5.37,
        "data_mean": 2.53,
        "data_std": 0.51,
        "description": "Brightener additive volume concentration",
    },

    # ------------------------------------------------------------------
    # Contamination
    # ------------------------------------------------------------------
    "Nickel [mg/L]": {
        "display_name": "Nickel",
        "unit": "mg/L",
        "dtype": "continuous",
        "feature_type": "contamination",
        "default_role": "fixed",
        "required": True,
        "data_min": 0.0,
        "data_max": 131.7,
        "data_mean": 4.67,
        "data_std": 11.04,
        "description": "Nickel impurity concentration in the bath",
    },

    "Iron [mg/L]": {
        "display_name": "Iron",
        "unit": "mg/L",
        "dtype": "continuous",
        "feature_type": "contamination",
        "default_role": "fixed",
        "required": True,
        "data_min": 0.0,
        "data_max": 46.56,
        "data_mean": 2.63,
        "data_std": 4.34,
        "description": "Iron impurity concentration in the bath",
    },

    "Copper [mg/L]": {
        "display_name": "Copper",
        "unit": "mg/L",
        "dtype": "continuous",
        "feature_type": "contamination",
        "default_role": "fixed",
        "required": True,
        "data_min": 0.15,
        "data_max": 8.36,
        "data_mean": 3.02,
        "data_std": 2.81,
        "description": "Copper impurity concentration in the bath",
    },

    # ------------------------------------------------------------------
    # Bath Lifecycle
    # ------------------------------------------------------------------
    "Bath age [Ah/L]": {
        "display_name": "Bath Age",
        "unit": "Ah/L",
        "dtype": "continuous",
        "feature_type": "lifecycle",
        "default_role": "fixed",
        "required": True,
        "data_min": 0.0,
        "data_max": 30.9,
        "data_mean": 7.93,
        "data_std": 7.05,
        "description": "Cumulative charge passed through the bath per litre",
    },

    "Last reactivation [days]": {
        "display_name": "Last Reactivation",
        "unit": "days",
        "dtype": "continuous",
        "feature_type": "lifecycle",
        "default_role": "fixed",
        "required": True,
        "data_min": 0.0,
        "data_max": 96.0,
        "data_mean": 33.35,
        "data_std": 26.91,
        "description": "Days since the bath was last reactivated",
    },

    "Service life [days]": {
        "display_name": "Service Life",
        "unit": "days",
        "dtype": "continuous",
        "feature_type": "lifecycle",
        "default_role": "fixed",
        "required": True,
        "data_min": 0.0,
        "data_max": 41.0,
        "data_mean": 4.13,
        "data_std": 5.02,
        "description": "Current service life of the bath in days",
    },

    # ------------------------------------------------------------------
    # Geometry / Substrate
    # ------------------------------------------------------------------
    "Gloss Nickel layer [%]": {
        "display_name": "Gloss Nickel Layer",
        "unit": "%",
        "dtype": "continuous",
        "feature_type": "geometry",
        "default_role": "fixed",
        "required": True,
        "data_min": 0.0,
        "data_max": 1.0,
        "data_mean": 0.76,
        "data_std": 0.29,
        "description": "Gloss fraction of the underlying nickel layer (0 = matt, 1 = gloss)",
    },

    "Sample Area [dm²]": {
        "display_name": "Sample Area",
        "unit": "dm²",
        "dtype": "continuous",
        "feature_type": "geometry",
        "default_role": "fixed",
        "required": True,
        "data_min": 0.71,
        "data_max": 1.2,
        "data_mean": 1.11,
        "data_std": 0.19,
        "description": "Surface area of the sample being plated",
    },

    "Anode Area [dm²]": {
        "display_name": "Anode Area",
        "unit": "dm²",
        "dtype": "continuous",
        "feature_type": "geometry",
        "default_role": "fixed",
        "required": True,
        "data_min": 0.84,
        "data_max": 6.44,
        "data_mean": 2.26,
        "data_std": 2.44,
        "description": "Total active anode area",
    },

    "Volume [L]": {
        "display_name": "Volume",
        "unit": "L",
        "dtype": "continuous",
        "feature_type": "geometry",
        "default_role": "fixed",
        "required": True,
        "data_min": 1.0,
        "data_max": 14.14,
        "data_mean": 11.79,
        "data_std": 5.04,
        "description": "Electrolyte bath volume",
    },
}

FEATURE_COLS: list[str] = list(FEATURE_REGISTRY.keys())


def bounds_dict() -> dict[str, tuple[float, float]]:
    """Return data-driven (min, max) bounds for every feature.

    Returns:
        Dictionary mapping feature name to (data_min, data_max) tuple.
    """
    return {name: (meta["data_min"], meta["data_max"]) for name, meta in FEATURE_REGISTRY.items()}


def summary_df():
    """Build a summary DataFrame of all feature metadata including the target.

    Returns:
        pandas DataFrame with columns ['name', 'display_name', 'unit',
        'feature_type', 'default_role', 'data_min', 'data_max', 'data_mean',
        'data_std', 'description'], one row per feature plus the target row.
    """
    import pandas as pd

    rows = []
    for name, meta in FEATURE_REGISTRY.items():
        rows.append({"name": name, **meta})

    target_row = {"name": TARGET_COL, "feature_type": "target", "default_role": "-", **TARGET_META}
    rows.append(target_row)

    return pd.DataFrame(rows)[[
        "name", "display_name", "unit", "feature_type", "default_role",
        "data_min", "data_max", "data_mean", "data_std", "description",
    ]]


if __name__ == "__main__":
    df = summary_df()
    print(df.to_string(index=False))
