"""DigiChrom — Streamlit UI for the chromium plating ML pipeline."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import datetime as dt

# ── Path & output dirs ───────────────────────────────────────────────────────
_HERE = Path(sys._MEIPASS) if getattr(sys, "frozen", False) else Path(__file__).parent
sys.path.insert(0, str(_HERE))

_MODELS_DIR   = _HERE / "app" / "models"
_FIGURES_DIR  = _HERE / "app" / "figures"
_REPORTS_DIR  = _HERE / "app" / "reports"
_PROFILES_DIR = _HERE / "app" / "profiles"
for _d in (_MODELS_DIR, _FIGURES_DIR, _REPORTS_DIR, _PROFILES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Bootstrap pipeline config ─────────────────────────────────────────────────
# All pipeline functions call get_config() — provide a mutable SimpleNamespace.
from pipeline import config as _pipe_config

_MODEL_DEFAULTS = {
    "ridge":                  {"alpha": 1.0},
    "elasticnet":             {"alpha": 0.1, "l1_ratio": 0.5},
    "cart":                   {"max_depth": 5, "random_state": 42},
    "gradient_boosting":      {"n_estimators": 200, "max_depth": 4, "random_state": 42},
    "random_forest":          {"n_estimators": 200, "max_depth": None, "random_state": 42},
    "extra_trees":            {"n_estimators": 200, "random_state": 42},
    "hist_gradient_boosting": {"max_iter": 200, "random_state": 42},
    "xgboost":                {"n_estimators": 200, "max_depth": 6, "random_state": 42, "verbosity": 0},
    "catboost":               {"iterations": 500, "depth": 6, "verbose": 0, "random_seed": 42},
    "lightgbm":               {"n_estimators": 200, "num_leaves": 31, "random_state": 42, "verbose": -1},
    "mlp":                    {"hidden_sizes": [128, 64], "dropout": 0.2, "lr": 1e-3,
                               "epochs": 200, "batch_size": 32},
    "tab_cnn": {}, "ft_transformer": {}, "saint": {}, "deep_gbm": {}, "tabnet": {},
    "svr": {"C": 1.0, "kernel": "rbf"},
}

_APPCONF = SimpleNamespace(
    RANDOM_SEED=42, CV_FOLDS=5, TEST_SIZE=0.15, MISSING_SENTINEL=-999999,
    FEATURE_COLS=[], TARGET_COL="", INDICATOR_COLS=[], CENSORED_COLS={},
    MODEL_DEFAULTS=_MODEL_DEFAULTS, CLASSIFIER_DEFAULTS={},
    DATA_PATH=None, BASE_DIR=_HERE, OUTPUT_DIR=_HERE,
    FIGURES_DIR=_FIGURES_DIR, MODELS_DIR=_MODELS_DIR, REPORTS_DIR=_REPORTS_DIR,
)
_pipe_config.set_config(_APPCONF)

from pipeline.preprocessing import train_test, cross_val_splits, align_features
from pipeline.model_testing import get_models, evaluate_all, get_classifiers, evaluate_classifiers
from pipeline.final_training import (train_final, eval_final,
                                      train_final_classifier, eval_final_classifier)
from pipeline import hp_tuning, xai
from pipeline.inverse_ml import find_inputs
from pipeline import model_management

try:
    import shap as _shap
    _HAS_SHAP = True
except Exception:
    _shap = None
    _HAS_SHAP = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DigiChrom", page_icon="🔬",
    layout="wide", initial_sidebar_state="expanded",
)

# ── Profile helpers ───────────────────────────────────────────────────────────
def _list_profiles() -> list[str]:
    return sorted(p.stem for p in _PROFILES_DIR.glob("*.json"))

def _save_profile(name: str, data: dict) -> None:
    (_PROFILES_DIR / f"{name}.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

def _load_profile(name: str) -> dict | None:
    p = _PROFILES_DIR / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

def _delete_profile(name: str) -> None:
    p = _PROFILES_DIR / f"{name}.json"
    if p.exists():
        p.unlink()

# ── Session state ─────────────────────────────────────────────────────────────
_SS_DEFAULTS: dict = {
    "df": None, "X": None, "y": None,
    "X_train": None, "X_test": None, "y_train": None, "y_test": None,
    "raw_df": None, "raw_filename": None,
    "cv_results": None, "best_model_name": None,
    "final_model": None, "final_preprocessor": None,
    "final_metrics": None, "final_X_sample": None,
    "xai_result": None,
    "inverse_results": None,
    "custom_feature_cols": None,
    "custom_target_col":   None,
    "task": "regression",
    "y_class_train": None, "y_class_test": None,
    "median_threshold": None,
    "nan_sentinel": -999999,
    "indicator_cols": [],
    "censored_cols": {},
    "median_threshold_override": None,
    "is_multi_output": False,
    "target_names": [],
    "_pending_profile": None,
    "original_feature_cols": [],
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Small helpers ─────────────────────────────────────────────────────────────
def _fc() -> list[str]:
    return st.session_state.custom_feature_cols or []

def _tc():
    return st.session_state.custom_target_col or ""

def _n_outputs() -> int:
    tc = _tc()
    return len(tc) if isinstance(tc, list) else (1 if tc else 0)

def _parse_sentinel(s: str):
    try:
        return int(s)
    except (ValueError, TypeError):
        try:
            return float(s)
        except (ValueError, TypeError):
            return None

def _parse_censored_str(s: str) -> dict:
    result = {}
    for part in s.split(","):
        part = part.strip()
        if ":" in part:
            col, sym = part.split(":", 1)
            col = col.strip()
            sym = sym.strip()
            if col:
                result[col] = sym
    return result

def _apply_nan_sentinel(raw: pd.DataFrame, sentinel) -> pd.DataFrame:
    if sentinel is not None:
        try:
            sv = float(sentinel)
            raw = raw.replace(sv, np.nan)
        except (ValueError, TypeError):
            pass
    return raw

def _apply_censored(raw: pd.DataFrame, censored: dict) -> pd.DataFrame:
    for col, sym in censored.items():
        if col not in raw.columns:
            continue
        def _parse(val, s=sym):
            if isinstance(val, str) and val.strip().startswith(s):
                m = re.search(r"[\d.]+", val)
                return float(m.group()) / 2 if m else np.nan
            return val
        raw = raw.copy()
        raw[col] = raw[col].apply(_parse)
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    return raw

def _store_data(df: pd.DataFrame, feature_cols: list[str], target_col,
                task: str, indicator_cols: list[str],
                threshold_override=None) -> None:
    is_multi = isinstance(target_col, list) and len(target_col) > 1
    tc_list  = target_col if isinstance(target_col, list) else [target_col]

    _APPCONF.FEATURE_COLS    = feature_cols
    _APPCONF.TARGET_COL      = target_col
    _APPCONF.INDICATOR_COLS  = indicator_cols
    _APPCONF.CENSORED_COLS   = st.session_state.censored_cols

    avail_feat = [c for c in feature_cols if c in df.columns]
    avail_tgt  = [c for c in tc_list if c in df.columns]
    wdf = df[avail_feat + avail_tgt].copy()

    for ind_col in indicator_cols:
        if ind_col in df.columns:
            wdf[f"{ind_col}_was_missing"] = df[ind_col].isna().astype(int)

    wdf = wdf.dropna(subset=avail_tgt).reset_index(drop=True)

    x_cols = [c for c in avail_feat if c in wdf.columns]
    for ind_col in indicator_cols:
        ind_name = f"{ind_col}_was_missing"
        if ind_name in wdf.columns and ind_name not in x_cols:
            x_cols.append(ind_name)

    X = wdf[x_cols].copy()
    for _col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[_col]):
            _converted = pd.to_numeric(
                X[_col].astype(str).str.replace(",", ".", regex=False), errors="coerce"
            )
            if _converted.notna().mean() >= 0.8:
                X[_col] = _converted
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False, dtype=float)

    y = wdf[target_col].copy() if is_multi else wdf[tc_list[0]].copy()
    X_tr, X_te, y_tr, y_te = train_test(X, y)

    if task == "classification" and not is_multi:
        med = float(threshold_override) if threshold_override is not None else float(y.median())
        y_cls_tr = (y_tr >= med).astype(int)
        y_cls_te = (y_te >= med).astype(int)
    else:
        med = None
        y_cls_tr = None
        y_cls_te = None

    st.session_state.update({
        "df": wdf, "X": X, "y": y,
        "X_train": X_tr, "X_test": X_te, "y_train": y_tr, "y_test": y_te,
        "y_class_train": y_cls_tr, "y_class_test": y_cls_te,
        "median_threshold": med, "median_threshold_override": threshold_override,
        "custom_feature_cols": list(X.columns), "custom_target_col": target_col,
        "original_feature_cols": list(x_cols),
        "task": task, "indicator_cols": indicator_cols,
        "is_multi_output": is_multi,
        "target_names": target_col if is_multi else [tc_list[0]],
        "cv_results": None, "best_model_name": None,
        "final_model": None, "final_preprocessor": None,
        "final_metrics": None, "final_X_sample": None,
        "xai_result": None, "inverse_results": None,
    })


def _build_param_df() -> pd.DataFrame:
    X_ref = st.session_state.X_train
    rows = []
    for feat in _fc():
        if X_ref is not None and feat in X_ref.columns:
            mn  = float(X_ref[feat].min())
            mx  = float(X_ref[feat].max())
            avg = float(X_ref[feat].mean())
        else:
            mn, mx, avg = 0.0, 1.0, 0.5
        rows.append({
            "Feature": feat, "Role": "Variable",
            "Fixed Value": round(avg, 4),
            "Min": round(mn, 4), "Max": round(mx, 4),
        })
    return pd.DataFrame(rows)


def _load_saved(name: str):
    result = model_management.load_model_by_name(name, _MODELS_DIR)
    if result is None:
        return None, None, None
    return result


# ── Auto-load most recent saved model on startup ──────────────────────────────
if st.session_state.final_model is None:
    try:
        _avail = model_management.get_available_models(_MODELS_DIR)
        if not _avail.empty:
            _mdl, _pre, _meta = _load_saved(_avail.iloc[0]["Name"])
            if _mdl is not None:
                st.session_state.update({
                    "final_model":        _mdl,
                    "final_preprocessor": _pre,
                    "final_metrics":      _meta.get("metrics", {}),
                    "best_model_name":    _meta.get("model_name", ""),
                    "final_X_sample":     _meta.get("X_sample"),
                })
    except Exception:
        pass


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("DigiChrom Machine Learning Pipeline")
    st.divider()
    st.markdown("**Status**")
    st.markdown(f"{'✅' if st.session_state.df is not None else '⚪'} Data loaded")
    st.markdown(f"{'✅' if st.session_state.cv_results is not None else '⚪'} CV evaluated")
    st.markdown(f"{'✅' if st.session_state.final_model is not None else '⚪'} Model available")
    if st.session_state.df is not None:
        lbl = "multi-output" if st.session_state.is_multi_output else st.session_state.task
        st.caption(f"{len(st.session_state.df)} samples · {len(_fc())} features · {lbl}")
    st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_data, tab_train, tab_xai, tab_inv, tab_feat = st.tabs([
    "Data",
    "Model Training",
    "Feature Importance",
    "Inverse ML",
    "Data Overview",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA
# ════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.header("Data")

    # ── Saved profiles ─────────────────────────────────────────────────────
    profiles = _list_profiles()
    with st.expander("💾  Saved Configurations", expanded=bool(profiles)):
        if profiles:
            pc1, pc2, pc3 = st.columns([3, 1, 1])
            sel_prof = pc1.selectbox(
                "Profile", profiles, label_visibility="collapsed", key="profile_load_sel",
            )
            if pc2.button("Load", use_container_width=True, key="btn_load_prof"):
                prof = _load_profile(sel_prof)
                if prof:
                    st.session_state["_pending_profile"] = prof
                    st.toast(f"Profile '{sel_prof}' ready — upload a file and click Load data.",
                             icon="📂")
            if pc3.button("Delete", use_container_width=True, key="btn_del_prof",
                          type="secondary"):
                _delete_profile(sel_prof)
                st.toast(f"Deleted '{sel_prof}'", icon="🗑")
                st.rerun()
        else:
            st.caption("No saved profiles yet. Configure columns below and save a profile.")

    st.divider()

    # ── File upload ────────────────────────────────────────────────────────
    uploaded = st.file_uploader("Upload Excel or CSV", type=["xlsx", "xls", "csv"])

    if uploaded is not None:
        if st.session_state.raw_filename != uploaded.name:
            with st.spinner("Reading file…"):
                try:
                    if uploaded.name.endswith(".csv"):
                        try:
                            raw = pd.read_csv(uploaded)
                        except Exception:
                            uploaded.seek(0)
                            raw = pd.read_csv(uploaded, sep=None, engine="python",
                                              on_bad_lines="skip")
                    else:
                        raw = pd.read_excel(uploaded)
                    raw.columns = raw.columns.str.strip()
                    st.session_state.raw_df       = raw
                    st.session_state.raw_filename = uploaded.name
                except Exception as exc:
                    st.error(f"Error reading file: {exc}")
                    st.session_state.raw_df = None

        raw = st.session_state.raw_df
        if raw is not None:
            all_cols = list(raw.columns)
            pprof    = st.session_state.get("_pending_profile")

            with st.form("col_config_form"):
                st.subheader("Column Configuration")

                # NaN sentinel
                sentinel_str = st.text_input(
                    "Fehlender-Wert-Sentinel (ersetzt mit NaN vor dem Laden)",
                    value=str(pprof["nan_sentinel"]) if pprof and "nan_sentinel" in pprof
                          else str(st.session_state.nan_sentinel),
                    help="Zeilen, die diesen Wert enthalten, werden als fehlende Werte behandelt. Default: -999999",
                )

                st.divider()
                fc_top, fc_bot = st.columns(2)

                # Target column(s)
                default_tgt = pprof.get("target_col", all_cols[-1]) if pprof else all_cols[-1]
                if isinstance(default_tgt, str):
                    default_tgt = [default_tgt]
                valid_tgt = [c for c in default_tgt if c in all_cols]

                sel_target = fc_top.multiselect(
                    "Target column(s) — 1 for regression/classification, 2+ for multi-output regression",
                    options=all_cols,
                    default=valid_tgt or [all_cols[-1]],
                )

                _task_opts    = ["regression", "classification"]
                _default_task = pprof.get("task", "regression") if pprof else "regression"
                _task_disabled = len(sel_target) > 1
                sel_task = fc_bot.radio(
                    "Task",
                    _task_opts,
                    index=_task_opts.index(_default_task) if _default_task in _task_opts else 0,
                    horizontal=True,
                    disabled=_task_disabled
                )
                if _task_disabled:
                    sel_task = "regression"

                # Features
                non_target    = [c for c in all_cols if c not in sel_target]
                default_feats = pprof.get("feature_cols", []) if pprof else []
                valid_feats   = [c for c in default_feats if c in non_target]
                sel_features  = st.multiselect(
                    "Feature columns",
                    options=non_target,
                    key="sel_feature_cols",
                    default=default_feats,
                )

                st.divider()
                
                # Indicator columns
                default_ind = pprof.get("indicator_cols", []) if pprof else []
                valid_ind   = [c for c in default_ind if c in sel_features]
                sel_ind = st.multiselect(
                    "Optionale Merkmale — NaN bedeutet, dass der Schritt übersprungen wurde (kein Fehler)",
                    options=non_target,
                    default=default_ind,
                    key="sel_indicator_cols",
                    help="Diese Spalten behalten ihre NaN-Zeilen. Es wird eine binäre Indikatorspalte "
                        "hinzugefügt.",
                )

                # Censored columns
                default_cens = pprof.get("censored_cols", {}) if pprof else {}
                cens_default_str = ", ".join(
                    f"{col}:{sym}" for col, sym in default_cens.items()
                )
                cens_input = st.text_input(
                    "Zensierte Spalten (Format: Spalte1:<, Spalte2:<)",
                    value=cens_default_str,
                    help="Werte wie '<5' werden durch die Hälfte der Nachweisgrenze (2.5) ersetzt. "
                        "Geben Sie Spaltenname : Präfix ein, getrennt durch Kommata.",
                )

                # Classification threshold
                thr_override = None
                if sel_task == "classification" and len(sel_target) == 1:
                    _default_thr = pprof.get("median_threshold") if pprof else None
                    use_custom_thr = st.checkbox(
                        "Override classification threshold (default: median of target)",
                        value=_default_thr is not None,
                    )
                    if use_custom_thr:
                        thr_override = st.number_input(
                            "Threshold — samples ≥ threshold → class 1",
                            value=float(_default_thr) if _default_thr is not None else 0.0,
                            format="%.4f",
                        )

                st.divider()

                # Profile save row (inside the form)
                ps1, ps2 = st.columns([3, 1])
                profile_name = ps1.text_input(
                    "Speicherung des Proils unter diesem Namen",
                    value=pprof.get("_name", "") if pprof else "",
                    placeholder="e.g.  my_dataset_configuration",
                )
                save_profile_btn = ps2.form_submit_button("💾  Save profile")

                load_btn = st.form_submit_button(
                    "✓  Daten mit dieser Konfiguration laden", type="primary",
                )

            # ── Form submission handlers ───────────────────────────────────
            if save_profile_btn and not profile_name.strip():
                st.error("Bitte einen Namen für das Profil angeben.")
                st.session_state["_pending_profile"] = {
                    "feature_cols":    sel_features,
                    "target_col":      sel_target if len(sel_target) > 1 else
                                       (sel_target[0] if sel_target else ""),
                    "task":            sel_task,
                    "nan_sentinel":    _parse_sentinel(sentinel_str),
                    "indicator_cols":  sel_ind,
                    "censored_cols":   _parse_censored_str(cens_input),
                    "median_threshold": thr_override,
                }
            elif save_profile_btn and profile_name.strip():
                st.toast(f"Selected Features: {', '.join(sel_features)}", icon="📋")
                _saved = {
                    "_name":           profile_name.strip(),
                    "feature_cols":    sel_features,
                    "target_col":      sel_target if len(sel_target) > 1 else
                                       (sel_target[0] if sel_target else ""),
                    "task":            sel_task,
                    "nan_sentinel":    _parse_sentinel(sentinel_str),
                    "indicator_cols":  sel_ind,
                    "censored_cols":   _parse_censored_str(cens_input),
                    "median_threshold": thr_override,
                }
                _save_profile(profile_name.strip(), _saved)
                st.session_state["_pending_profile"] = _saved
                st.toast(f"✅ Profile '{profile_name.strip()}' saved", icon="💾")
                #st.rerun()

            if load_btn:
                if not sel_features:
                    st.error("Select at least one feature column.")
                elif not sel_target:
                    st.error("Select at least one target column.")
                else:
                    sentinel_val  = _parse_sentinel(sentinel_str)
                    censored_dict = _parse_censored_str(cens_input)
                    st.session_state.nan_sentinel  = sentinel_val
                    st.session_state.censored_cols = censored_dict
                    target_val = sel_target if len(sel_target) > 1 else sel_target[0]
                    # Preserve current form state so multiselects don't reset to "all" on rerun
                    st.session_state["_pending_profile"] = {
                        "feature_cols":    sel_features,
                        "target_col":      target_val,
                        "task":            sel_task,
                        "nan_sentinel":    sentinel_val,
                        "indicator_cols":  sel_ind,
                        "censored_cols":   censored_dict,
                        "median_threshold": thr_override,
                    }

                    proc = _apply_nan_sentinel(raw.copy(), sentinel_val)
                    proc = _apply_censored(proc, censored_dict)
                    _store_data(
                        proc,
                        feature_cols=sel_features,
                        target_col=target_val,
                        task=sel_task,
                        indicator_cols=sel_ind,
                        threshold_override=thr_override,
                    )
                    dropped = len(raw) - len(st.session_state.df)
                    note = (f"✅ Loaded {len(st.session_state.df)} samples · "
                            f"{len(_fc())} features · target(s): {', '.join(sel_target)}")
                    if dropped:
                        note += f"  ({dropped} rows dropped — missing target)"
                    st.toast(note, icon="✅")
                    #st.rerun()

    # ── Data summary ───────────────────────────────────────────────────────
    if st.session_state.df is not None:
        df_   = st.session_state.df
        tc_   = _tc()
        task_ = st.session_state.task

        st.divider()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total samples", len(df_))
        c2.metric("Train",    len(st.session_state.X_train))
        c3.metric("Test",     len(st.session_state.X_test))
        c4.metric("Features", len(_fc()))
        c5.metric("Mode", "multi-output" if st.session_state.is_multi_output else task_)

        if task_ == "classification" and not st.session_state.is_multi_output:
            med = st.session_state.median_threshold
            tc_str = tc_ if isinstance(tc_, str) else tc_[0]
            if med is not None and tc_str in df_.columns:
                n1 = int((df_[tc_str] >= med).sum())
                n0 = int((df_[tc_str] <  med).sum())
                src = "manual" if st.session_state.median_threshold_override else "median"
                st.caption(
                    f"Classification threshold ({src}): **{med:.4f}** → "
                    f"class 0: {n0} · class 1: {n1}"
                )

        tc_list_ = tc_ if isinstance(tc_, list) else [tc_]
        show_cols = [c for c in _fc() if c in df_.columns][:15] + \
                    [c for c in tc_list_ if c in df_.columns]
        with st.expander("Data preview (first 50 rows)", expanded=True):
            st.dataframe(df_[show_cols].head(50), use_container_width=True)
        with st.expander("Feature statistics"):
            st.dataframe(
                df_[[c for c in _fc() if c in df_.columns]].describe().T.round(3),
                use_container_width=True,
            )

        # Target histograms
        n_tgt = len(tc_list_)
        hist_cols = st.columns(min(n_tgt, 3))
        for i, tname in enumerate(tc_list_):
            if tname in df_.columns:
                with hist_cols[i % len(hist_cols)]:
                    fig_h, ax_h = plt.subplots(figsize=(4, 2.5))
                    ax_h.hist(df_[tname].dropna(), bins=30,
                              color="#4C72B0", edgecolor="white", alpha=0.85)
                    ax_h.set_title(tname, fontsize=9); ax_h.set_ylabel("Count", fontsize=8)
                    ax_h.tick_params(labelsize=8)
                    plt.tight_layout()
                    st.pyplot(fig_h, use_container_width=False, clear_figure=True)
                    plt.close(fig_h)
    else:
        st.info("Lade eine Excel- oder CSV-Datei hoch, um zu starten.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL TRAINING
# ════════════════════════════════════════════════════════════════════════════
with tab_train:
    st.header("Model Training")

    if st.session_state.df is None:
        st.warning("Zuerst Daten laden (→ **Data** tab).")
    else:
        _is_clf     = st.session_state.task == "classification" and not st.session_state.is_multi_output
        _n_out      = _n_outputs()
        _mdl_pool   = get_classifiers() if _is_clf else get_models(n_outputs=_n_out)

        # ── Cross-Validation ───────────────────────────────────────────────
        st.subheader("Cross-Validation")
        with st.expander("ℹ️ Was ist Cross-Validation?", expanded=False):
            st.markdown(
                "**Cross-Validation** testet mehrere Modelle fair auf deinen Daten. "
                "Die Daten werden in *k* Teile aufgeteilt — jedes Modell wird k-mal trainiert "
                "und auf dem jeweils zurückgehaltenen Teil bewertet. "
                "Das gibt ein realistisches Bild, wie gut ein Modell auf neuen Daten funktioniert. "
                "**R²** nahe 1 = gut, **RMSE/MAE** kleiner = besser."
            )

        _STD_MODELS = [m for m in _mdl_pool if m not in
                       ("tab_cnn", "ft_transformer", "saint", "deep_gbm", "tabnet")]
        _DL_MODELS  = [m for m in _mdl_pool if m in
                       ("tab_cnn", "ft_transformer", "saint", "deep_gbm", "tabnet")]

        selected = st.multiselect(
            "Modelle auswählen",
            options=_STD_MODELS,
            default=_STD_MODELS[:6],
            help="Wähle die Modelle, die gegeneinander verglichen werden sollen. "
                 "Mehr Modelle = länger, aber bessere Übersicht.",
        )
        with st.expander("🔬 Erweiterte Modelle (Deep Learning)", expanded=False):
            st.caption("Diese Modelle sind komplexer und brauchen deutlich länger.")
            if _DL_MODELS:
                adv_sel = st.multiselect("Deep-Learning-Modelle", options=_DL_MODELS, default=[])
                selected = selected + adv_sel
            else:
                st.info(
                    "Keine Deep-Learning-Modelle verfügbar. "
                    "Installiere **PyTorch** (`pip install torch`) und starte die App neu, "
                    "um Tab-CNN, FT-Transformer, SAINT und Deep-GBM zu aktivieren."
                )

        cv_col1, cv_col2, cv_col3, cv_col4 = st.columns([1, 1, 1, 2])
        n_folds   = int(cv_col1.number_input(
            "CV Folds", 2, 10, 5, 1,
            help="Anzahl der Aufteilungen. 5 ist ein guter Standardwert.",
        ))
        use_pfhpo = cv_col2.toggle(
            "Per-fold HPO", value=False,
            help="Optimiert die Modellparameter in jeder Falte automatisch. Langsamer, aber genauer.",
        )
        # use_aug   = cv_col3.toggle(
        #     "Augmentierung", value=False,
        #     help="Erzeugt synthetische Trainingsdaten durch leichtes Verrauschen "
        #          "der vorhandenen Proben. Hilfreich bei kleinen Datensätzen.",
        # )
        pfhpo_trials = int(cv_col4.slider(
            "HPO-Versuche pro Fold", 10, 100, 30, step=5, disabled=not use_pfhpo,
        ))
        # aug_ratio = 0.0
        # if use_aug:
        #     aug_ratio = float(cv_col4.slider(
        #         "Augmentierungsfaktor", 0.1, 2.0, 0.5, step=0.1,
        #         help="0.5 = 50% zusätzliche synthetische Proben, 1.0 = Datensatz verdoppeln.",
        #     ))

        if st.button("▶  Cross-Validation starten", type="primary"):
            if not selected:
                st.warning("Mindestens ein Modell auswählen.")
            else:
                X_tr = st.session_state.X_train
                y_tr = st.session_state.y_class_train if _is_clf else st.session_state.y_train
                splits = cross_val_splits(X_tr, y_tr, n_folds=n_folds)
                bar    = st.progress(0, text="Initialisiere…")
                parts  = []
                for i, name in enumerate(selected):
                    bar.progress(i / len(selected), text=f"Evaluiere {name}…")
                    mdl_sub = {name: _mdl_pool[name]}
                    if _is_clf:
                        parts.append(evaluate_classifiers(
                            X_tr, y_tr, models=mdl_sub, cv_splits=splits,
                            per_fold_hpo=use_pfhpo, n_hpo_trials=pfhpo_trials,
                        ))
                    else:
                        parts.append(evaluate_all(
                            X_tr, y_tr, models=mdl_sub, cv_splits=splits,
                            per_fold_hpo=use_pfhpo, n_hpo_trials=pfhpo_trials,
                            #aug_ratio=aug_ratio,
                        ))
                bar.progress(1.0, text="Fertig!")
                cv = pd.concat(parts, ignore_index=True)
                sort_col = "accuracy" if _is_clf else "rmse"
                best = (cv.groupby("model")[sort_col].mean().idxmax() if _is_clf
                        else cv.groupby("model")[sort_col].mean().idxmin())
                st.session_state.cv_results      = cv
                st.session_state.best_model_name = best
                st.success(f"Bestes Modell: **{best}**")

        if st.session_state.cv_results is not None:
            cv = st.session_state.cv_results
            metrics_avail = [c for c in ["r2", "rmse", "mae", "accuracy", "f1", "auc"]
                             if c in cv.columns]
            summary = cv.groupby("model")[metrics_avail].mean().round(4)
            if "rmse" in summary.columns:
                summary = summary.sort_values("rmse")
            elif "accuracy" in summary.columns:
                summary = summary.sort_values("accuracy", ascending=False)
            hi_cols = [c for c in ["r2", "accuracy", "f1", "auc"] if c in summary.columns]
            lo_cols = [c for c in ["rmse", "mae"]                  if c in summary.columns]
            styled  = summary.style
            if hi_cols: styled = styled.highlight_max(subset=hi_cols, color="#c8f7c5")
            if lo_cols: styled = styled.highlight_min(subset=lo_cols, color="#c8f7c5")
            st.dataframe(styled, use_container_width=True)

            n_m = len(metrics_avail)
            fig_cv, axes = plt.subplots(1, n_m, figsize=(min(4 * n_m, 8), 3.5))
            if n_m == 1:
                axes = [axes]
            for ax, metric in zip(axes, metrics_avail):
                order = cv.groupby("model")[metric].median().sort_values(
                    ascending=(metric not in ("r2", "accuracy", "f1", "auc"))).index
                ax.boxplot([cv[cv["model"] == m][metric].values for m in order],
                           labels=order, patch_artist=True)
                ax.set_title(metric.upper(), fontsize=9)
                ax.tick_params(axis="x", rotation=40, labelsize=7)
            fig_cv.suptitle("Cross-Validation Vergleich", fontsize=11)
            plt.tight_layout()
            col_plot, _ = st.columns(2)
            col_plot.pyplot(fig_cv, use_container_width=True, clear_figure=True)
            plt.close(fig_cv)

        # ── Final Model ────────────────────────────────────────────────────
        st.divider()
        st.subheader("Final Model")
        model_opts  = list(_mdl_pool.keys())
        best_name   = st.session_state.best_model_name
        default_idx = model_opts.index(best_name) if best_name in model_opts else 0
        chosen = st.selectbox("Model to train on full training set", model_opts, index=default_idx)

        fin_col1, fin_col2, fin_col3 = st.columns([1, 1, 2])
        run_hpo      = fin_col1.toggle("HPO (Optuna)", value=False,
                                       help="Automatische Hyperparameter-Suche mit Optuna.")
        # use_aug_fin  = fin_col2.toggle("Augmentierung", value=False,
        #                                help="Synthetische Datenpunkte für das finale Training erzeugen.")
        n_trials     = fin_col3.slider("HPO-Versuche", 10, 200, 50, step=10, disabled=not run_hpo)
        # aug_ratio_fin = 0.0
        # if use_aug_fin:
        #     aug_ratio_fin = float(fin_col3.slider(
        #         "Augmentierungsfaktor (final)", 0.1, 2.0, 0.5, step=0.1,
        #         help="0.5 = 50% zusätzliche synthetische Proben.",
        #     ))

        if st.button("▶  Finales Modell trainieren", type="primary"):
            with st.spinner(f"Training {chosen}…"):
                try:
                    best_params: dict = {}
                    if run_hpo and chosen not in ("linear", "logistic"):
                        try:
                            best_params = hp_tuning.tune(
                                chosen, st.session_state.X_train,
                                st.session_state.y_train, n_trials=n_trials,
                                n_outputs=_n_outputs(),
                            )
                            st.info(f"Best params: {best_params}")
                        except Exception as hpo_exc:
                            st.warning(f"HPO skipped: {hpo_exc}")

                    y_tr = st.session_state.y_class_train if _is_clf else st.session_state.y_train
                    y_te = st.session_state.y_class_test  if _is_clf else st.session_state.y_test

                    if _is_clf:
                        mdl_, pre_ = train_final_classifier(chosen, st.session_state.X_train, y_tr)
                        # if aug_ratio_fin > 0.0:
                        #     _rng   = np.random.default_rng(42)
                        #     _Xs    = pre_.transform(st.session_state.X_train.values)
                        #     _n_aug = max(1, int(len(_Xs) * aug_ratio_fin))
                        #     _idx   = _rng.integers(0, len(_Xs), size=_n_aug)
                        #     _noise = _rng.normal(0, 0.02, size=(_n_aug, _Xs.shape[1]))
                        #     _Xs_aug = np.vstack([_Xs, _Xs[_idx] + _noise])
                        #     _ys_aug = np.concatenate([y_tr.values, y_tr.values[_idx]])
                        #     mdl_.fit(_Xs_aug, _ys_aug)
                        met = eval_final_classifier(mdl_, pre_, st.session_state.X_test, y_te)
                    else:
                        mdl_, pre_ = train_final(chosen, st.session_state.X_train, y_tr,
                                                 best_params=best_params)
                        # if aug_ratio_fin > 0.0:
                        #     _rng   = np.random.default_rng(42)
                        #     _Xs    = pre_.transform(st.session_state.X_train.values)
                        #     _n_aug = max(1, int(len(_Xs) * aug_ratio_fin))
                        #     _idx   = _rng.integers(0, len(_Xs), size=_n_aug)
                        #     _noise = _rng.normal(0, 0.02, size=(_n_aug, _Xs.shape[1]))
                        #     _Xs_aug = np.vstack([_Xs, _Xs[_idx] + _noise])
                        #     _ys_aug = np.concatenate([y_tr.values, _rng.normal(
                        #         y_tr.values[_idx], np.abs(y_tr.values).mean() * 0.02)])
                        #     mdl_.fit(_Xs_aug, _ys_aug)
                        met = eval_final(mdl_, pre_, st.session_state.X_test, y_te)

                    X_sample = pre_.transform(st.session_state.X_train.values)
                    st.session_state.update({
                        "final_model": mdl_, "final_preprocessor": pre_,
                        "final_metrics": met, "best_model_name": chosen,
                        "final_X_sample": X_sample,
                        "xai_result": None,
                    })
                    if _is_clf:
                        st.success(f"Trained **{chosen}** — Accuracy={met.get('accuracy', '?'):.4f}")
                    else:
                        # Multi-output: met may contain per-target metrics
                        r2_str = f"R²={met['r2']:.4f}" if met.get("r2") is not None else ""
                        st.success(f"Trained **{chosen}**  {r2_str}")
                except Exception as exc:
                    st.error(f"Training failed: {exc}")

        if st.session_state.final_metrics:
            m = st.session_state.final_metrics
            if _is_clf:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy",     f"{m['accuracy']:.4f}" if m.get("accuracy") is not None else "—")
                c2.metric("F1",           f"{m['f1']:.4f}"       if m.get("f1")       is not None else "—")
                c3.metric("AUC",          f"{m['auc']:.4f}"      if m.get("auc")      is not None else "—")
                c4.metric("Test Samples", int(m["n_test"]) if m.get("n_test") is not None else "—")

                if m.get("confusion_matrix"):
                    from sklearn.metrics import ConfusionMatrixDisplay
                    with st.expander("Confusion Matrix", expanded=False):
                        cm_arr = np.array(m["confusion_matrix"])
                        fig_cm, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
                        ConfusionMatrixDisplay(cm_arr).plot(ax=ax1, colorbar=False)
                        ax1.set_title("Counts")
                        cm_norm = cm_arr.astype(float) / cm_arr.sum(axis=1, keepdims=True)
                        ConfusionMatrixDisplay(np.round(cm_norm, 2)).plot(ax=ax2, colorbar=False)
                        ax2.set_title("Normalised")
                        plt.tight_layout()
                        st.pyplot(fig_cm, clear_figure=True)
                        plt.close(fig_cm)

                if m.get("classification_report"):
                    with st.expander("Classification Report", expanded=False):
                        st.dataframe(pd.DataFrame(m["classification_report"]).T.round(4),
                                     use_container_width=True)
            else:
                # Regression / multi-output
                if st.session_state.is_multi_output:
                    # Show per-target metrics if available, else aggregated
                    tnames = st.session_state.target_names
                    per_tgt = m.get("per_target", {})
                    if per_tgt:
                        met_df = pd.DataFrame(per_tgt).T.round(4)
                        st.dataframe(met_df, use_container_width=True)
                    else:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Test R² (avg)",   f"{m['r2']:.4f}"   if m.get("r2")   else "—")
                        c2.metric("Test RMSE (avg)", f"{m['rmse']:.4f}" if m.get("rmse") else "—")
                        c3.metric("Test MAE (avg)",  f"{m['mae']:.4f}"  if m.get("mae")  else "—")
                else:
                    c1, c2, c3, c4 = st.columns(4)
                    r2_s   = f"{m['r2']:.4f}"   if m.get("r2")   is not None else "—"
                    rmse_s = f"{m['rmse']:.4f}" if m.get("rmse") is not None else "—"
                    mae_s  = f"{m['mae']:.4f}"  if m.get("mae")  is not None else "—"
                    if m.get("r2_ci_low") is not None:
                        r2_s   += f"  [{m['r2_ci_low']:.3f}, {m['r2_ci_high']:.3f}]"
                        rmse_s += f"  [{m['rmse_ci_low']:.4f}, {m['rmse_ci_high']:.4f}]"
                        mae_s  += f"  [{m['mae_ci_low']:.4f}, {m['mae_ci_high']:.4f}]"
                    c4.metric("Test Samples", int(m["n_test"]) if m.get("n_test") is not None else "—")
                    c1.metric("Test R²",      r2_s)
                    c2.metric("Test RMSE",    rmse_s)
                    c3.metric("Test MAE",     mae_s)

                    with st.expander("Predicted vs. True", expanded=False):
                        try:
                            _fc_saved = (list(st.session_state.X_train.columns)
                                         if st.session_state.X_train is not None else _fc())
                            X_te_al = align_features(st.session_state.X_test, _fc_saved)
                            X_te_sc = st.session_state.final_preprocessor.transform(X_te_al.values)
                            y_te_np = st.session_state.y_test.values
                            y_pr_np = st.session_state.final_model.predict(X_te_sc)
                            fig_bci = xai.bootstrap_ci_plot(y_te_np, y_pr_np)
                            st.pyplot(fig_bci, clear_figure=True, use_container_width=False)
                            plt.close(fig_bci)
                        except Exception as exc:
                            st.warning(f"Plot failed: {exc}")

            # Save / delete
            st.divider()
            if not _is_clf and m.get("r2") is not None and m["r2"] < 0.6:
                st.warning(f"⚠️ R² = {m['r2']:.4f} — model quality is low (< 0.6).")
            col_name, col_save, col_del = st.columns([2, 1, 1])
            model_save_name = col_name.text_input(
                "Save as", value=f"{chosen}_{dt.datetime.now().strftime('%Y%m%d')}",
            )
            if col_save.button("💾 Save", use_container_width=True):
                try:
                    _x_sample = st.session_state.final_X_sample
                    if _x_sample is None and st.session_state.X_train is not None:
                        _x_sample = st.session_state.final_preprocessor.transform(
                            st.session_state.X_train.values
                        )
                    _feat_cols = (list(st.session_state.X_train.columns)
                                  if st.session_state.X_train is not None else _fc())
                    model_management.save_model(
                        st.session_state.final_model,
                        st.session_state.final_preprocessor,
                        st.session_state.final_metrics,
                        st.session_state.best_model_name,
                        _MODELS_DIR,
                        custom_name=model_save_name,
                        X_sample=_x_sample,
                        metadata={
                            "feature_cols":          _feat_cols,
                            "original_feature_cols": st.session_state.get("original_feature_cols", []),
                            "target_col":            _tc(),
                            "target_names":          st.session_state.target_names,
                            "task":                  st.session_state.task,
                            "is_multi_output":       st.session_state.is_multi_output,
                            "median_threshold":      st.session_state.median_threshold,
                        },
                    )
                    st.toast(f"✅ Saved as '{model_save_name}'", icon="💾")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Save failed: {exc}")
            if col_del.button("🗑 Clear", use_container_width=True):
                st.session_state.update({
                    "final_model": None, "final_preprocessor": None,
                    "final_metrics": None, "final_X_sample": None,
                    "best_model_name": None,
                })
                st.rerun()

        # ── Saved Models ───────────────────────────────────────────────────
        st.divider()
        st.subheader("Saved Models")
        avail = model_management.get_available_models(_MODELS_DIR)
        if avail.empty:
            st.info("No saved models yet.")
        else:
            st.dataframe(avail.drop(columns=["X_sample"], errors="ignore"),
                         use_container_width=True)
            del_name = st.selectbox("Delete model", ["—"] + avail["Name"].tolist(),
                                    key="del_model_select")
            if del_name != "—" and st.button("🗑 Delete selected", type="secondary"):
                if model_management.delete_model(del_name, _MODELS_DIR):
                    st.success(f"Deleted '{del_name}'")
                    st.rerun()
                else:
                    st.error("Delete failed.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════════════════════
with tab_xai:
    st.header("Feature Importance")

    avail_xai = model_management.get_available_models(_MODELS_DIR)
    if avail_xai.empty:
        st.info("No saved models found. Train and save a model first (→ **Model Training** tab).")
    else:
        sel_xai = st.selectbox("Saved model", avail_xai["Name"].tolist(),
                               key="xai_model_select")
        mdl_xai, pre_xai, meta_xai = _load_saved(sel_xai)

        if mdl_xai is None:
            st.error(f"Could not load '{sel_xai}'")
        else:
            X_xai            = meta_xai.get("X_sample")
            y_xai            = None
            src_label        = ""
            _model_feat_cols = meta_xai.get("feature_cols", _fc())

            _n_model_feats = len(_model_feat_cols)

            def _transform_test():
                X_al = align_features(st.session_state.X_test, _model_feat_cols)
                return pre_xai.transform(X_al.values)

            def _test_compatible() -> bool:
                """Return True if current X_test can be aligned to model's features."""
                if st.session_state.X_test is None:
                    return False
                try:
                    X_al = align_features(st.session_state.X_test, _model_feat_cols)
                    return X_al.shape[1] == _n_model_feats
                except Exception:
                    return False

            _feat_mismatch = st.session_state.X_test is not None and not _test_compatible()
            if _feat_mismatch:
                st.warning(
                    f"Das geladene Modell wurde mit **{_n_model_feats} Features** trainiert, "
                    f"die aktuellen Daten haben **{st.session_state.X_test.shape[1]} Features**. "
                    "Bitte trainiere ein neues Modell mit der aktuellen Datenkonfiguration. "
                    "Die gespeicherte Trainingsstichprobe wird für die Analyse verwendet."
                )

            if st.session_state.X_test is not None and not _feat_mismatch:
                if X_xai is not None:
                    use_test = st.checkbox("Aktuellen Test-Datensatz statt gespeicherter Stichprobe verwenden",
                                           value=True)
                    if use_test:
                        X_xai     = _transform_test()
                        y_xai     = (st.session_state.y_class_test.values
                                     if st.session_state.task == "classification"
                                     else st.session_state.y_test.values)
                        src_label = f"geladener Test-Datensatz ({len(y_xai)} Proben)"
                    else:
                        src_label = f"gespeicherte Trainingsstichprobe ({X_xai.shape[0]} Proben)"
                else:
                    X_xai     = _transform_test()
                    y_xai     = (st.session_state.y_class_test.values
                                 if st.session_state.task == "classification"
                                 else st.session_state.y_test.values)
                    src_label = "geladener Test-Datensatz"
            elif X_xai is not None:
                src_label = f"gespeicherte Trainingsstichprobe ({X_xai.shape[0]} Proben)"
            else:
                st.warning("Keine Daten verfügbar. Daten laden oder Modell neu trainieren.")

            if X_xai is not None:
                st.caption(f"Model: **{meta_xai.get('model_name', sel_xai)}** · {src_label}")

                # Feature filter — lets users remove columns from analysis
                _orig_feat_cols = meta_xai.get("original_feature_cols") or _model_feat_cols
                with st.expander("🔧 Features für Analyse auswählen", expanded=False):
                    _orig_filter = st.multiselect(
                        "Aktive Features",
                        options=_orig_feat_cols,
                        default=_orig_feat_cols,
                        key="xai_feat_filter",
                    )
                if set(_orig_filter) != set(_orig_feat_cols):
                    _removed = set(_orig_feat_cols) - set(_orig_filter)
                    def _ohe_belongs_to(col, orig_names):
                        for name in orig_names:
                            if col == name or col.startswith(f"{name}_"):
                                return True
                        return False
                    _keep_idx = [
                        i for i, col in enumerate(_model_feat_cols)
                        if not _ohe_belongs_to(col, _removed)
                    ]
                    X_xai            = X_xai[:, _keep_idx]
                    _model_feat_cols = [_model_feat_cols[i] for i in _keep_idx]

                _METHOD_INFO = {
                    "SHAP Summary": (
                        "Zeigt, wie stark jedes Merkmal die Vorhersage beeinflusst. "
                        "**Bar-Plot**: mittlere absolute SHAP-Werte — breite Balken = großer Einfluss. "
                        "**Beeswarm**: Punkte zeigen den Einfluss jedes einzelnen Datenpunkts; "
                        "rot = hoher Merkmalswert, blau = niedriger Merkmalswert."
                    ),
                    "SHAP Dependence": (
                        "Zeigt, wie sich der SHAP-Wert eines Merkmals mit seinem Wert verändert. "
                        "Jeder Punkt ist eine Probe. Eine klare Steigung zeigt einen linearen Effekt; "
                        "Streuung deutet auf Wechselwirkungen mit anderen Merkmalen hin."
                    ),
                    "SHAP Waterfall": (
                        "Erklärt **eine einzelne Vorhersage** Schritt für Schritt. "
                        "Rote Balken treiben die Vorhersage nach oben, blaue Balken nach unten. "
                        "Der Ausgangspunkt ist der Durchschnitt aller Vorhersagen."
                    ),
                    "SHAP Interactions (tree models)": (
                        "Zeigt Wechselwirkungen zwischen Merkmalspaaren. "
                        "Starke Farben (rot/blau) bedeuten, dass diese zwei Merkmale gemeinsam "
                        "einen größeren Effekt haben als jedes für sich allein. "
                    ),
                    "ICE Plots": (
                        "**Individual Conditional Expectation**: Jede Linie zeigt, wie die Vorhersage "
                        "für *eine* Probe reagiert, wenn das gewählte Merkmal variiert wird. "
                        "Parallele Linien = gleichmäßiger Effekt; gekreuzte Linien = Wechselwirkungen."
                    ),
                    "ALE Plots": (
                        "**Accumulated Local Effects**: Wie ändert sich die Vorhersage im Durchschnitt, "
                        "wenn das Merkmal leicht erhöht wird? Robuster als PDPs bei korrelierten Merkmalen. "
                        "Eine steigende Kurve = mehr Merkmal → höhere Vorhersage."
                    ),
                    "PDP Plots": (
                        "**Partial Dependence Plot**: Zeigt den durchschnittlichen Effekt eines Merkmals "
                        "auf die Vorhersage, während alle anderen Merkmale auf ihrem Mittelwert gehalten werden. "
                        "Einfach zu lesen, aber kann bei korrelierten Merkmalen irreführend sein."
                    ),
                    "Permutation Importance": (
                        "Misst, wie sehr sich die Modellgüte verschlechtert, wenn ein Merkmal "
                        "zufällig durchgemischt wird (und damit seinen Informationsgehalt verliert). "
                        "Großer Abfall = das Merkmal ist wichtig. Fehlerbalken zeigen die Streuung über Wiederholungen."
                    ),
                    "Learning Curve": (
                        "Zeigt, wie sich Trainings- und Validierungsfehler verändern, "
                        "wenn mehr Trainingsdaten hinzugefügt werden. "
                        "Wenn beide Kurven konvergieren und immer noch weit vom Optimum entfernt sind, "
                        "helfen mehr Daten. Wenn nur der Trainingsfehler niedrig ist, liegt Überanpassung vor."
                    ),
                    "Decision Tree Visualisation": (
                        "Visualisiert die Entscheidungsregeln des Modells als Baumdiagramm. "
                        "Jeder Knoten stellt eine Ja/Nein-Frage zu einem Merkmal. "
                        "Farbe = Vorhersagewert, Zahlen = Anzahl Proben. "
                        "Nur für Baum-Modelle (CART, C5.0) verfügbar."
                    ),
                    "Confusion Matrix (classification)": (
                        "Zeigt, wie oft das Modell richtig und falsch liegt. "
                        "Die Diagonale (links oben nach rechts unten) sind korrekte Vorhersagen. "
                        "Werte außerhalb der Diagonale sind Fehler."
                    ),
                    "Classification Report (classification)": (
                        "Detaillierte Tabelle der Klassifikationsgüte pro Klasse: "
                        "**Precision** = Anteil richtiger Positiver unter allen als positiv klassifizierten. "
                        "**Recall** = Anteil erkannter Positiver. "
                        "**F1** = harmonisches Mittel aus Precision und Recall."
                    ),
                }
                _USES_TOP_N = {
                    "SHAP Dependence",
                    "ICE Plots", "ALE Plots", "PDP Plots",
                }

                if st.session_state.task == "classification":
                    _METHODS = [
                        "SHAP Summary",
                        "SHAP Dependence",
                        "SHAP Interactions (tree models)",
                        "ICE Plots",
                        "ALE Plots",
                        "PDP Plots",
                        "Permutation Importance",
                        "Decision Tree Visualisation",
                        "Confusion Matrix (classification)",
                        "Classification Report (classification)",
                    ]
                else:
                    _METHODS = [
                        "SHAP Summary",
                        "SHAP Dependence",
                        "SHAP Interactions (tree models)",
                        "ICE Plots",
                        "ALE Plots",
                        "PDP Plots",
                        "Permutation Importance"                    ]
                method = st.selectbox("Analysemethode", _METHODS)
                top_n_xai = 6
                if method in _USES_TOP_N:
                    top_n_xai = st.slider(
                        "Top-N Features", 3, min(15, len(_model_feat_cols)), 6,
                        key="xai_top_n",
                    )
                if method == "Decision Tree Visualisation":
                    st.session_state["tree_max_depth"] = int(
                        st.slider("Maximale Tiefe", 1, 6, 2, key="xai_tree_depth")
                    )

                _needs_labels = method in (
                    "Permutation Importance", "Confusion Matrix (classification)", "Classification Report (classification)",
                )

                disabled_btn = _needs_labels and y_xai is None
                if disabled_btn:
                    st.warning("This method needs ground-truth labels. Load data in **Data** tab.")
                
                if st.button("▶  Compute", type="primary", disabled=disabled_btn):
                    st.session_state.xai_result = None  # always clear previous result
                    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
                    with st.spinner(f"Running {method}…"):
                        try:
                            if method == "SHAP Summary":
                                if not _HAS_SHAP:
                                    st.error("`shap` not installed.")
                                else:
                                    mtype = type(mdl_xai).__name__.lower()
                                    tree_kw = ("catboost", "xgb", "forest", "tree", "boost", "lgb")
                                    if any(t in mtype for t in tree_kw):
                                        explainer = _shap.TreeExplainer(mdl_xai)
                                        sv = explainer.shap_values(X_xai)
                                    else:
                                        bg = _shap.sample(X_xai, min(100, len(X_xai)))
                                        explainer = _shap.KernelExplainer(mdl_xai.predict, bg)
                                        sv = explainer.shap_values(X_xai, nsamples=100)
                                    imp_df = (pd.DataFrame({
                                        "Feature": _model_feat_cols,
                                        "Mean |SHAP|": np.abs(sv).mean(axis=0),
                                    }).sort_values("Mean |SHAP|", ascending=False)
                                      .reset_index(drop=True))
                                    X_df = pd.DataFrame(X_xai, columns=_model_feat_cols)
                                    h = max(4, min(10, len(_model_feat_cols) * 0.32))
                                    plt.figure(figsize=(7, h))
                                    _shap.summary_plot(sv, X_df, plot_type="bar", show=False)
                                    fig_bar = plt.gcf()
                                    plt.tight_layout()
                                    plt.close(fig_bar)
                                    plt.figure(figsize=(7, h))
                                    _shap.summary_plot(sv, X_df, show=False)
                                    fig_bee = plt.gcf()
                                    plt.tight_layout()
                                    plt.close(fig_bee)
                                    st.session_state.xai_result = {
                                        "method": method,
                                        "figs": [fig_bar, fig_bee],
                                        "imp": imp_df,
                                    }

                            elif method == "SHAP Dependence":
                                if not _HAS_SHAP:
                                    st.error("`shap` not installed.")
                                else:
                                    sv = xai.compute_shap_values(mdl_xai, X_xai)
                                    fig = xai.shap_dependence_plots(
                                        sv, X_xai, _model_feat_cols, top_n=top_n_xai,
                                    )
                                    st.session_state.xai_result = {"method": method, "fig": fig}

                            # elif method == "SHAP Waterfall":
                            #     if not _HAS_SHAP:
                            #         st.error("`shap` not installed.")
                            #     else:
                            #         figs = xai.shap_waterfall_plots(
                            #             mdl_xai, X_xai, _model_feat_cols)
                            #         st.session_state.xai_result = {"method": method, "figs": figs}

                            elif method == "SHAP Interactions (tree models)":
                                if not _HAS_SHAP:
                                    st.error("`shap` not installed.")
                                else:
                                    fig_inter, df_inter = xai.shap_interaction_matrix(
                                        mdl_xai, X_xai, _model_feat_cols)
                                    st.session_state.xai_result = {
                                        "method": method,
                                        "fig": fig_inter,
                                        "data": df_inter,
                                    }

                            elif method == "ICE Plots":
                                fig = xai.ice_plots(mdl_xai, X_xai, _model_feat_cols,
                                                    top_n=top_n_xai)
                                st.session_state.xai_result = {"method": method, "fig": fig}

                            elif method == "ALE Plots":
                                fig = xai.ale_plots(mdl_xai, X_xai, _model_feat_cols,
                                                    top_n=top_n_xai)
                                st.session_state.xai_result = {"method": method, "fig": fig}

                            elif method == "PDP Plots":
                                fig = xai.pdp_plots(mdl_xai, X_xai, _model_feat_cols,
                                                    top_n=top_n_xai)
                                st.session_state.xai_result = {"method": method, "fig": fig}

                            elif method == "Permutation Importance":
                                imp_df = xai.permutation_importance(
                                    mdl_xai, X_xai, y_xai, _model_feat_cols)
                                n_f = len(_model_feat_cols)
                                fig_pi, ax_pi = plt.subplots(figsize=(8, max(4, n_f * 0.32)))
                                ax_pi.barh(imp_df["feature"][::-1],
                                           imp_df["importance_mean"][::-1],
                                           xerr=imp_df["importance_std"][::-1],
                                           color="#4C72B0", ecolor="gray", alpha=0.85)
                                ax_pi.set_xlabel("Mean decrease in R²")
                                ax_pi.set_title("Permutation Importance")
                                plt.tight_layout()
                                plt.close(fig_pi)
                                st.session_state.xai_result = {
                                    "method": method, "fig": fig_pi, "imp": imp_df,
                                }

                            # elif method == "Learning Curve":
                            #     from sklearn.base import clone as _clone
                            #     try:
                            #         m_c = _clone(mdl_xai)
                            #     except Exception:
                            #         m_c = mdl_xai
                            #     fig = xai.learning_curve_plot(m_c, X_xai, y_xai,
                            #                                   _model_feat_cols)
                            #     st.session_state.xai_result = {"method": method, "fig": fig}

                            elif method == "Decision Tree Visualisation":
                                _td = st.session_state.get("tree_max_depth", 2)
                                fig = xai.show_tree(mdl_xai, _model_feat_cols, max_depth=_td)
                                # also generate readable text export
                                from sklearn.tree import (
                                    DecisionTreeRegressor, DecisionTreeClassifier, export_text
                                )
                                _tree_obj = None
                                if isinstance(mdl_xai, (DecisionTreeRegressor, DecisionTreeClassifier)):
                                    _tree_obj = mdl_xai
                                elif hasattr(mdl_xai, "estimators_"):
                                    _tree_obj = mdl_xai.estimators_[0]
                                _tree_txt = (
                                    export_text(_tree_obj, feature_names=_model_feat_cols,
                                                max_depth=_td)
                                    if _tree_obj is not None else ""
                                )
                                st.session_state.xai_result = {
                                    "method": method, "fig": fig, "data": _tree_txt,
                                }

                            elif method == "Confusion Matrix (classification)":
                                fig = xai.confusion_matrix_plot(mdl_xai, X_xai, y_xai)
                                st.session_state.xai_result = {"method": method, "fig": fig}

                            elif method == "Classification Report (classification)":
                                fig, df_rep = xai.classification_report_plot(
                                    mdl_xai, X_xai, y_xai)
                                st.session_state.xai_result = {
                                    "method": method, "fig": fig, "data": df_rep,
                                }

                        except Exception as exc:
                            st.error(f"{method} failed: {exc}")

                # ── Render results (persistent across reruns) ──────────────
                r = st.session_state.get("xai_result")
                if r:
                    m_name = r.get("method", "")

                    # Show contextual explanation above the plot
                    _info_txt = _METHOD_INFO.get(m_name, "")
                    if _info_txt:
                        st.info(_info_txt)

                    if m_name == "SHAP Summary" and r.get("figs"):
                        col_l, col_r = st.columns(2)
                        col_l.subheader("Mean |SHAP| – Bar")
                        col_l.pyplot(r["figs"][0], use_container_width=True)
                        col_r.subheader("SHAP Beeswarm")
                        col_r.pyplot(r["figs"][1], use_container_width=True)
                    elif r.get("figs"):
                        for fig_ in r["figs"]:
                            st.pyplot(fig_, use_container_width=False)
                    elif r.get("fig") is not None:
                        if m_name == "Decision Tree Visualisation":
                            st.pyplot(r["fig"], use_container_width=True)
                        else:
                            st.pyplot(r["fig"], use_container_width=False)

                    if r.get("imp") is not None:
                        imp = r["imp"]
                        st.dataframe(imp.round(5), use_container_width=True)
                        fname = ("shap_importance.csv" if "SHAP" in m_name
                                 else "permutation_importance.csv")
                        st.download_button(
                            f"⬇  {fname}", imp.to_csv(index=False).encode(),
                            fname, "text/csv",
                        )

                    if r.get("data") is not None:
                        data = r["data"]
                        if m_name == "Decision Tree Visualisation" and isinstance(data, str) and data:
                            st.subheader("Textuelle Darstellung")
                            st.code(data, language=None)
                        elif hasattr(data, "head"):
                            st.dataframe(data.head(20).round(6), use_container_width=True)
                            fname = ("shap_interaction_pairs.csv"
                                     if "Interaction" in m_name else "xai_data.csv")
                            st.download_button(
                                f"⬇  {fname}", data.to_csv(index=False).encode(),
                                fname, "text/csv",
                            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — INVERSE ML
# ════════════════════════════════════════════════════════════════════════════
with tab_inv:
    st.header("Inverse ML")
    st.caption(
        "Find process parameters that produce desired target values. "
        "Bounds are derived automatically from the training data."
    )

    avail_inv = model_management.get_available_models(_MODELS_DIR)
    if avail_inv.empty:
        st.info("No saved models found. Train and save a model first (→ **Model Training** tab).")
    else:
        sel_inv = st.selectbox("Saved model", avail_inv["Name"].tolist(),
                               key="inv_model_select")
        mdl_inv, pre_inv, meta_inv = _load_saved(sel_inv)
        _inv_feat_cols     = meta_inv.get("feature_cols", _fc()) if meta_inv else _fc()
        _inv_target_names  = meta_inv.get("target_names", ["target"]) if meta_inv else ["target"]
        _inv_is_multi      = meta_inv.get("is_multi_output", False) if meta_inv else False

        if mdl_inv is None:
            st.error(f"Could not load '{sel_inv}'")
        else:
            with st.expander("ℹ️ Was ist Inverse ML?", expanded=False):
                st.markdown(
                    "Das Modell wurde trainiert, um aus Prozessparametern die "
                    "Schichtdicke vorherzusagen. **Inverses ML dreht das um**: "
                    "Du gibst eine gewünschte Schichtdicke vor, und die Software "
                    "sucht automatisch nach Prozessparameterkombinationen, die diesen Wert ergeben. "
                    "Die Suche nutzt **Bayessche Optimierung** — eine schlaue Suche, "
                    "die mit wenigen Versuchen gute Lösungen findet."
                )

            # ── Target value(s) ────────────────────────────────────────────
            st.subheader("Gewünschter Zielwert")
            target_vals: list[float] = []
            t_cols = st.columns(max(len(_inv_target_names), 1))
            for i, tname in enumerate(_inv_target_names):
                val = t_cols[i % len(t_cols)].number_input(
                    tname, value=0.0, format="%.4f", key=f"inv_target_{i}",
                    help="Welchen Wert soll das Modell erzeugen?",
                )
                target_vals.append(val)

            # ── Interactive parameter editor ────────────────────────────────
            st.divider()
            st.subheader("Parameter konfigurieren")
            st.caption(
                "Hake Parameter ab, die berücksichtigt werden sollen. "
                "**Fest** = bekannter Wert, der nicht verändert wird. "
                "**Variabel** = der Parameter wird vom Algorithmus optimiert."
            )

            X_ref = st.session_state.X_train

            def _feat_stats(feat):
                if X_ref is not None and feat in X_ref.columns:
                    lo   = float(X_ref[feat].min())
                    hi   = float(X_ref[feat].max())
                    mean = float(X_ref[feat].mean())
                    return lo, hi, mean
                return 0.0, 1.0, 0.5

            free_vars, fixed_params, bounds = [], {}, {}

            for feat in _inv_feat_cols:
                lo, hi, mean = _feat_stats(feat)
                c_inc, c_name, c_role, c_val = st.columns([0.5, 2.5, 1.2, 4])

                include = c_inc.checkbox(
                    "", value=True, key=f"inv_inc_{feat}",
                    help="Parameter in die Suche einbeziehen",
                )
                c_name.markdown(f"**{feat}**", help=f"Trainingsbereich: [{lo:.3g}, {hi:.3g}]")

                if not include:
                    fixed_params[feat] = mean
                    continue

                is_var = c_role.toggle("Variabel", value=False, key=f"inv_var_{feat}")

                if is_var:
                    free_vars.append(feat)
                    if abs(hi - lo) < 1e-9:
                        bounds[feat] = (lo, lo)
                        c_val.caption(f"Konstant: {lo:.4g}")
                    else:
                        rng_vals = c_val.slider(
                            "Bereich", min_value=float(lo), max_value=float(hi),
                            value=(float(lo), float(hi)), key=f"inv_rng_{feat}",
                            format="%.4g",
                        )
                        bounds[feat] = (rng_vals[0], rng_vals[1])
                else:
                    fixed_params[feat] = mean
                    if abs(hi - lo) < 1e-9:
                        c_val.caption(f"Konstant: {lo:.4g}")
                        fixed_params[feat] = lo
                    else:
                        fixed_params[feat] = c_val.slider(
                            "Wert", min_value=float(lo), max_value=float(hi),
                            value=float(mean), key=f"inv_fix_{feat}",
                            format="%.4g",
                        )

            if not free_vars:
                st.warning("Mindestens einen Parameter auf **Variabel** setzen.")

            st.divider()
            col_ta, col_tb = st.columns(2)
            n_trials_inv    = int(col_ta.number_input(
                "Suchversuche (Bayessch)", 50, 2000, 300, 50,
                help="Mehr Versuche = bessere Lösung, aber länger. 300 ist ein guter Start.",
            ))
            n_solutions_inv = int(col_tb.number_input(
                "Anzahl Lösungen", 1, 20, 5, 1,
                help="Wie viele der besten Lösungen sollen angezeigt werden?",
            ))

            if st.button("▶  Parameter suchen", type="primary", disabled=not free_vars):
                with st.spinner(f"Bayessche Suche — {n_trials_inv} Versuche…"):
                    try:
                        target_arg = (target_vals if _inv_is_multi and len(target_vals) > 1
                                      else target_vals[0])
                        results = find_inputs(
                            model=mdl_inv, preprocessor=pre_inv,
                            feature_names=_inv_feat_cols,
                            fixed_params=fixed_params, free_vars=free_vars,
                            target_thickness=target_arg,
                            bounds=bounds,
                            n_solutions=n_solutions_inv,
                            method="bayesian", n_trials=n_trials_inv,
                        )
                        st.session_state.inverse_results = results
                        st.success(
                            f"{len(results)} Lösungen gefunden — "
                            f"bester Fehler: **{results['error'].iloc[0]:.4f}**"
                        )
                    except Exception as exc:
                        st.error(f"Optimierung fehlgeschlagen: {exc}")

            if st.session_state.inverse_results is not None:
                res = st.session_state.inverse_results.copy()
                for col in res.columns:
                    if col not in ("error",) and not col.startswith("predicted"):
                        res[col] = res[col].round(4)
                res["error"] = res["error"].map("{:.5f}".format)
                st.subheader("Beste Lösungen")
                st.dataframe(res, use_container_width=True)
                st.download_button(
                    "⬇  solutions.csv",
                    st.session_state.inverse_results.to_csv(index=False).encode(),
                    "inverse_ml_solutions.csv", "text/csv",
                )


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — COLUMN OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab_feat:
    st.header("Column Overview")

    if st.session_state.df is None:
        st.info("Load data first (→ **Data** tab).")
    else:
        df_   = st.session_state.df
        fc_   = _fc()
        tc_   = _tc()
        tc_list_ = tc_ if isinstance(tc_, list) else ([tc_] if tc_ else [])

        # Feature table
        feat_rows = []
        for col in fc_:
            if col not in df_.columns:
                continue
            is_ind  = col.endswith("_was_missing")
            is_cat  = not pd.api.types.is_numeric_dtype(df_[col])
            role    = "indicator" if is_ind else ("categorical" if is_cat else "numeric")
            is_opt  = col in st.session_state.indicator_cols
            is_cens = col in st.session_state.censored_cols

            row = {
                "Column":     col,
                "Type":       role,
                "Optional":   "✓" if is_opt  else "",
                "Censored":   "✓" if is_cens else "",
                "N valid":    int(df_[col].notna().sum()),
                "N missing":  int(df_[col].isna().sum()),
            }
            if not is_cat and not is_ind and col in df_.columns:
                row["Min"]  = round(float(df_[col].min()),  4)
                row["Max"]  = round(float(df_[col].max()),  4)
                row["Mean"] = round(float(df_[col].mean()), 4)
                row["Std"]  = round(float(df_[col].std()),  4)
            else:
                row["Min"] = row["Max"] = row["Mean"] = row["Std"] = ""

            feat_rows.append(row)

        feat_df = pd.DataFrame(feat_rows)
        st.subheader(f"Features ({len(feat_df)})")
        st.dataframe(feat_df, use_container_width=True, height=420)

        # Target(s) summary
        st.divider()
        st.subheader(f"Target(s): {', '.join(tc_list_)}")
        tgt_rows = []
        for tname in tc_list_:
            if tname not in df_.columns:
                continue
            s = df_[tname].dropna()
            tgt_rows.append({
                "Column":   tname,
                "N valid":  int(s.notna().sum()),
                "Min":      round(float(s.min()),    4),
                "Max":      round(float(s.max()),    4),
                "Mean":     round(float(s.mean()),   4),
                "Std":      round(float(s.std()),    4),
                "Median":   round(float(s.median()), 4),
            })
        if tgt_rows:
            st.dataframe(pd.DataFrame(tgt_rows), use_container_width=True)

        # Config summary
        st.divider()
        st.subheader("Active Configuration")
        cfg_items = {
            "Task":             st.session_state.task if not st.session_state.is_multi_output
                                else "multi-output regression",
            "NaN sentinel":     str(st.session_state.nan_sentinel),
            "Optional features": ", ".join(st.session_state.indicator_cols) or "—",
            "Censored columns":  ", ".join(
                f"{k} ({v})" for k, v in st.session_state.censored_cols.items()
            ) or "—",
            "Threshold":        str(st.session_state.median_threshold) or "—",
        }
        for k, v in cfg_items.items():
            st.markdown(f"**{k}:** {v}")
