"""
DigiChrom Pipeline — Konfigurationsvorlage
==========================================
Kopiere diese Datei und benenne sie nach deinem Standort / Datensatz,
z.B. config_mein_Datensatz.py, config_HS_Aalen.py, etc.
Die standortspezifischen Dateien werden von Git ignoriert.

Neue Daten mit anderen Features / Target:
    Passe TARGET_COL, FEATURE_COLS und META_COLS unten direkt an.
    Die Pipeline liest alles über get_config() — kein weiterer Code muss
    geändert werden, solange die Spaltennamen mit der Excel-Datei übereinstimmen.
    Für volle Metadaten (Einheiten, Bounds für Inverse-ML) kannst du optional
    eine eigene feature_registry_<name>.py anlegen und hier importieren.
"""
import sys
from pathlib import Path

# Pfade — bei PyInstaller-Builds landet OUTPUT_DIR im Home-Verzeichnis
if getattr(sys, "frozen", False):
    _SRC_DIR = Path(sys._MEIPASS)
    DATA_PATH = _SRC_DIR / "data" / "daten.xlsx"   # <-- Dateiname anpassen
    OUTPUT_DIR = Path.home() / "DigiChrom"
else:
    _SRC_DIR = Path(__file__).parent
    DATA_PATH = _SRC_DIR / "data" / "daten.xlsx"   # <-- Dateiname anpassen
    OUTPUT_DIR = _SRC_DIR / "outputs"

BASE_DIR    = _SRC_DIR
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR  = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Experiment-Parameter
RANDOM_SEED      = 42
CV_FOLDS         = 5
TEST_SIZE        = 0.15
MISSING_SENTINEL = -999999

# ── Features & Target ───────────────────────────────────────────────────────
# Direkt hier anpassen — Spaltennamen müssen exakt mit der Excel-Datei übereinstimmen.
# Die Pipeline liest ausschließlich über get_config().FEATURE_COLS / TARGET_COL.

# Single-output:  TARGET_COL = "Thickness Cr [µm]"
# Multi-output:   TARGET_COL = ["Thickness Cr [µm]", "Hardness [HV]"]
TARGET_COL = "Thickness Cr [µm]"   # <-- Zielspalte(n) anpassen

FEATURE_COLS = [
    # Prozessparameter
    "pH",
    "Deposition time [min]",
    "Temperature [°C]",
    "Current density [A/dm²]",
    # Badchemie
    "Chromium [g/L]",
    "Buffer [g/L]",
    "Complexing agent [mL/L]",
    "Additive [mL/L]",
    "Supporting electrolyte [g/L]",
    "Brigthener [mL/L]",
    # Kontamination
    "Nickel [mg/L]",
    "Iron [mg/L]",
    "Copper [mg/L]",
    # Lebenszyklus
    "Bath age [Ah/L]",
    "Last reactivation [days]",
    "Service life [days]",
    # Geometrie
    "Gloss Nickel layer [%]",
    "Sample Area [dm²]",
    "Anode Area [dm²]",
    "Volume [L]",
    # <-- neue Features einfach hier eintragen oder alte entfernen
]

# ── Fehlende Werte als Information ──────────────────────────────────────────
# NaN in diesen Spalten bedeutet "Schritt wurde ausgelassen", nicht "Datenfehler".
# Für jede Spalte wird automatisch eine binäre "<Spalte>_was_missing"-Spalte
# erstellt (1 = fehlend/ausgelassen, 0 = vorhanden) und als Feature genutzt.
# Die Originalspalte wird anschließend normal imputiert (Median).
# Leer lassen oder weglassen, wenn alle NaN Datenfehler sind.
INDICATOR_COLS = [
    # "Duty-Cycle",
    # "Reactivation step",
]

# ── Zensierte Messwerte ──────────────────────────────────────────────────────
# Werte unterhalb der Nachweisgrenze (z.B. "<5") werden auf halbe Grenze gesetzt.
# Dict: Spaltenname → Präfix-Zeichen. Leer lassen wenn nicht benötigt.
CENSORED_COLS = {
    # "Cr(VI)": "<",
}

# Spalten, die keine Features/Target sind (werden beim Laden ignoriert)
META_COLS = [
    "Experiment ID",
    "Experiment",
    "Partner",
    "Date",
    "Time",
    # weitere Meta-Spalten hier eintragen
]

# ── Modell-Defaults (Regression) ────────────────────────────────────────────
MODEL_DEFAULTS = {
    "ridge":                  {"alpha": 1.0},
    "elasticnet":             {"alpha": 1.0, "l1_ratio": 0.5},
    "cart":                   {"max_depth": None, "min_samples_leaf": 1,
                               "random_state": RANDOM_SEED},
    "gradient_boosting":      {"n_estimators": 200, "learning_rate": 0.05,
                               "max_depth": 3, "random_state": RANDOM_SEED},
    "random_forest":          {"n_estimators": 200, "max_depth": None,
                               "random_state": RANDOM_SEED, "n_jobs": -1},
    "extra_trees":            {"n_estimators": 200, "max_depth": None,
                               "random_state": RANDOM_SEED, "n_jobs": -1},
    "hist_gradient_boosting": {"max_iter": 200, "learning_rate": 0.05,
                               "max_depth": None, "random_state": RANDOM_SEED},
    "xgboost":                {"n_estimators": 200, "learning_rate": 0.05,
                               "max_depth": 6, "random_state": RANDOM_SEED,
                               "verbosity": 0},
    "catboost":               {"iterations": 500, "learning_rate": 0.05,
                               "depth": 6, "random_seed": RANDOM_SEED,
                               "verbose": 0},
    "lightgbm":               {"n_estimators": 200, "learning_rate": 0.05,
                               "max_depth": -1, "num_leaves": 31,
                               "random_state": RANDOM_SEED, "verbosity": -1,
                               "n_jobs": -1},
    "mlp":                    {"hidden_sizes": [128, 64], "dropout": 0.2,
                               "lr": 1e-3, "epochs": 200, "batch_size": 32},
    "tab_cnn":                {"n_filters": 64, "kernel_size": 3, "n_layers": 2,
                               "dropout": 0.3, "lr": 1e-3, "epochs": 200,
                               "batch_size": 32},
    "ft_transformer":         {"d_token": 64, "n_heads": 8, "n_layers": 3,
                               "dropout": 0.1, "lr": 1e-3, "epochs": 200,
                               "batch_size": 32},
    "saint":                  {"d_token": 32, "n_heads": 4, "n_layers": 2,
                               "dropout": 0.1, "lr": 1e-3, "epochs": 200,
                               "batch_size": 64},
    "deep_gbm":               {"n_estimators": 200, "max_depth": 4,
                               "hidden_size": 64, "dropout": 0.2, "lr": 1e-3,
                               "epochs": 100, "batch_size": 32},
    "tabnet":                 {"n_d": 8, "n_a": 8, "n_steps": 3, "gamma": 1.3,
                               "n_independent": 1, "n_shared": 1,
                               "batch_size": 48, "virtual_batch_size": 16,
                               "seed": RANDOM_SEED, "verbose": 0,
                               "max_epochs": 200, "patience": 30},
}

# ── Classifier-Defaults (Klassifikation) ────────────────────────────────────
CLASSIFIER_DEFAULTS = {
    "logistic":               {"C": 1.0, "max_iter": 1000, "n_jobs": -1},
    "cart":                   {"max_depth": None, "min_samples_leaf": 1,
                               "random_state": RANDOM_SEED},
    "c50":                    {"max_depth": None, "min_samples_leaf": 1,
                               "criterion": "entropy", "random_state": RANDOM_SEED},
    "gradient_boosting":      {"n_estimators": 200, "learning_rate": 0.05,
                               "max_depth": 3, "random_state": RANDOM_SEED},
    "random_forest":          {"n_estimators": 200, "max_depth": None,
                               "random_state": RANDOM_SEED, "n_jobs": -1},
    "extra_trees":            {"n_estimators": 200, "max_depth": None,
                               "random_state": RANDOM_SEED, "n_jobs": -1},
    "hist_gradient_boosting": {"max_iter": 200, "learning_rate": 0.05,
                               "max_depth": None},
    "xgboost":                {"n_estimators": 200, "learning_rate": 0.05,
                               "max_depth": 6, "random_state": RANDOM_SEED,
                               "verbosity": 0},
    "catboost":               {"iterations": 500, "learning_rate": 0.05,
                               "depth": 6, "random_seed": RANDOM_SEED,
                               "verbose": 0},
    "lightgbm":               {"n_estimators": 200, "learning_rate": 0.05,
                               "num_leaves": 31, "random_state": RANDOM_SEED,
                               "verbosity": -1, "n_jobs": -1},
    "mlp":                    {"hidden_sizes": [128, 64], "dropout": 0.2,
                               "lr": 1e-3, "epochs": 200, "batch_size": 32},
    "tab_cnn":                {"n_filters": 64, "kernel_size": 3, "n_layers": 2,
                               "dropout": 0.3, "lr": 1e-3, "epochs": 200,
                               "batch_size": 32},
    "ft_transformer":         {"d_token": 64, "n_heads": 8, "n_layers": 3,
                               "dropout": 0.1, "lr": 1e-3, "epochs": 200,
                               "batch_size": 32},
    "saint":                  {"d_token": 32, "n_heads": 4, "n_layers": 2,
                               "dropout": 0.1, "lr": 1e-3, "epochs": 200,
                               "batch_size": 64},
    "deep_gbm":               {"n_estimators": 200, "max_depth": 4,
                               "hidden_size": 64, "dropout": 0.2, "lr": 1e-3,
                               "epochs": 100, "batch_size": 32},
    "tabnet":                 {"n_d": 8, "n_a": 8, "n_steps": 3, "gamma": 1.3,
                               "n_independent": 1, "n_shared": 1,
                               "batch_size": 48, "virtual_batch_size": 16,
                               "seed": RANDOM_SEED, "verbose": 0,
                               "max_epochs": 200, "patience": 30},
}
