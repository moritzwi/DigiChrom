"""DigiChrom — Streamlit UI for the chromium plating ML pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import datetime as dt

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = Path(sys._MEIPASS) if getattr(sys, "frozen", False) else Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "pipeline"))

import DigiChrom_Pipeline.config_TU_Ilmenau as config_TU_Ilmenau
from pipeline.feature_registry import (
    FEATURE_REGISTRY, FEATURE_COLS, TARGET_COL, TARGET_META,
)
from pipeline.data_loading import load_clean
from pipeline.preprocessing import split_xy, train_test, cross_val_splits, align_features
from pipeline.model_testing import (get_models, evaluate_all,
                                     get_classifiers, evaluate_classifiers)
from pipeline.final_training import (train_final, eval_final,
                                      train_final_classifier, eval_final_classifier)
from pipeline import hp_tuning
from pipeline import xai
from pipeline.inverse_ml import find_inputs
from pipeline import model_management
from pipeline import feature_mapping
from pipeline import ensemble as _ensemble

try:
    import shap as _shap
    _HAS_SHAP = True
except Exception:
    _shap = None
    _HAS_SHAP = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DigiChrom",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────────────────────
for _k, _v in {
    "df": None, "X": None, "y": None,
    "X_train": None, "X_test": None, "y_train": None, "y_test": None,
    "raw_df": None, "raw_filename": None,
    "cv_results": None, "best_model_name": None,
    "final_model": None, "final_preprocessor": None,
    "final_metrics": None, "final_X_sample": None,
    "shap_values": None, "shap_importance": None,
    "shap_X": None, "perm_importance": None,
    "inverse_results": None,
    # dynamic feature/target columns (None = use FEATURE_REGISTRY defaults)
    "custom_feature_cols": None,
    "custom_target_col":   None,
    # task: "regression" | "classification"
    "task": "regression",
    # for classification: binary y derived from median split
    "y_class_train": None, "y_class_test": None,
    "median_threshold": None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Helpers ───────────────────────────────────────────────────────────────────
def _fc() -> list[str]:
    """Active feature columns (custom > registry default)."""
    return st.session_state.custom_feature_cols or FEATURE_COLS

def _tc() -> str:
    """Active target column (custom > registry default)."""
    return st.session_state.custom_target_col or TARGET_COL

def _store_data(df: pd.DataFrame, feature_cols: list[str] = None, target_col: str = None) -> None:
    feat_cols  = feature_cols or _fc()
    tgt_col    = target_col   or _tc()
    # subset df to selected cols (handle missing gracefully)
    available  = [c for c in feat_cols if c in df.columns]
    all_nan    = [c for c in available if df[c].isna().all()]
    available  = [c for c in available if c not in all_nan]
    X          = df[available].copy()
    y          = df[tgt_col].copy()
    X_tr, X_te, y_tr, y_te = train_test(X, y)
    # classification labels via median split
    med        = float(y.median())
    y_class_tr = (y_tr >= med).astype(int)
    y_class_te = (y_te >= med).astype(int)
    st.session_state.update({
        "df": df, "X": X, "y": y,
        "X_train": X_tr, "X_test": X_te, "y_train": y_tr, "y_test": y_te,
        "y_class_train": y_class_tr, "y_class_test": y_class_te,
        "median_threshold": med,
        "cv_results": None, "best_model_name": None,
        "final_model": None, "final_preprocessor": None,
        "final_metrics": None, "final_X_sample": None,
        "shap_values": None, "shap_importance": None, "shap_X": None,
        "perm_importance": None, "inverse_results": None,
    })


def _build_param_df(feature_cols: list[str] = None) -> pd.DataFrame:
    """Build Inverse ML parameter table from model feature list or FEATURE_REGISTRY."""
    cols = feature_cols or _fc()
    rows = []
    X_ref = st.session_state.X_train  # may be None if no data loaded
    for feat in cols:
        if feat in FEATURE_REGISTRY:
            meta = FEATURE_REGISTRY[feat]
            rows.append({
                "Feature":      feat,
                "Display Name": meta["display_name"],
                "Unit":         meta.get("unit", ""),
                "Category":     meta.get("feature_type", ""),
                "Role":         "Variable" if meta["default_role"] == "variable" else "Fixed",
                "Fixed Value":  float(meta["data_mean"]),
                "Min":          float(meta["data_min"]),
                "Max":          float(meta["data_max"]),
            })
        else:
            # Dynamic feature: derive bounds from training data if available
            if X_ref is not None and feat in X_ref.columns:
                mn  = float(X_ref[feat].min())
                mx  = float(X_ref[feat].max())
                avg = float(X_ref[feat].mean())
            else:
                mn, mx, avg = 0.0, 1.0, 0.5
            rows.append({
                "Feature":      feat,
                "Display Name": feat,
                "Unit":         "",
                "Category":     "custom",
                "Role":         "Variable",
                "Fixed Value":  avg,
                "Min":          mn,
                "Max":          mx,
            })
    return pd.DataFrame(rows)


def _load_saved(name: str):
    """Returns (model, preprocessor, meta) or (None, None, None)."""
    result = model_management.load_model_by_name(name, config_TU_Ilmenau.MODELS_DIR)
    if result is None:
        return None, None, None
    return result


# ── Auto-load most recent saved model on startup ──────────────────────────────
if st.session_state.final_model is None:
    try:
        _avail = model_management.get_available_models(config_TU_Ilmenau.MODELS_DIR)
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
    st.title("🔬 DigiChrom")
    st.caption("Chromium Plating ML Pipeline")
    st.divider()
    st.markdown("**Status**")
    st.markdown(f"{'✅' if st.session_state.df is not None else '⚪'} Data loaded")
    st.markdown(f"{'✅' if st.session_state.cv_results is not None else '⚪'} CV evaluated")
    st.markdown(f"{'✅' if st.session_state.final_model is not None else '⚪'} Model ready")
    if st.session_state.df is not None:
        st.caption(f"{len(st.session_state.df)} samples · {len(_fc())} features")
    if st.session_state.final_metrics:
        m = st.session_state.final_metrics
        if isinstance(m.get("r2"), float):
            st.divider()
            st.markdown("**Test Metrics**")
            st.caption(
                f"R² = {m['r2']:.3f}\n"
                f"RMSE = {m['rmse']:.4f} µm\n"
                f"MAE  = {m['mae']:.4f} µm"
            )
    st.divider()
    st.caption("PhD Project · DigiChrom")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_data, tab_train, tab_xai, tab_inv, tab_feat = st.tabs([
    "📂  Data",
    "🤖  Model Training",
    "📊  Feature Importance",
    "🔄  Inverse ML",
    "🏷️  Features",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA
# ════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.header("Data")

    col_up, col_demo = st.columns([3, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Upload Excel or CSV",
            type=["xlsx", "xls", "csv"],
        )
    with col_demo:
        st.write(""); st.write("")
        if st.button("Use demo data", use_container_width=True):
            with st.spinner("Loading…"):
                try:
                    demo_df, _ = load_clean()
                    _store_data(demo_df)
                    st.session_state.raw_df = None
                    st.session_state.raw_filename = None
                    st.success(f"Loaded {len(demo_df)} samples.")
                except Exception as exc:
                    st.error(f"Could not load demo data: {exc}")

    if uploaded is not None:
        # Only re-read when a new file is uploaded (cache by filename)
        if st.session_state.raw_filename != uploaded.name:
            with st.spinner("Reading file…"):
                try:
                    raw = (
                        pd.read_csv(uploaded)
                        if uploaded.name.endswith(".csv")
                        else pd.read_excel(uploaded)
                    )
                    st.session_state.raw_df = raw
                    st.session_state.raw_filename = uploaded.name
                except Exception as exc:
                    st.error(f"Error reading file: {exc}")
                    st.session_state.raw_df = None

        raw = st.session_state.raw_df
        if raw is not None:
            raw = raw.replace(config_TU_Ilmenau.MISSING_SENTINEL, np.nan)
            all_cols = list(raw.columns)

            # ── Column configuration form ──────────────────────────────────
            with st.form("col_config_form"):
                st.subheader("Column Configuration")
                cc1, cc2 = st.columns(2)

                # Target column
                default_tgt = TARGET_COL if TARGET_COL in all_cols else all_cols[-1]
                sel_target  = cc1.selectbox(
                    "Target column", all_cols,
                    index=all_cols.index(default_tgt) if default_tgt in all_cols else len(all_cols) - 1,
                )

                # Task
                sel_task = cc2.radio("Task", ["regression", "classification"],
                                     horizontal=True)

                # Feature columns
                default_feats = [c for c in FEATURE_COLS if c in all_cols] or \
                                [c for c in all_cols if c != sel_target]
                sel_features  = st.multiselect(
                    "Feature columns",
                    options=[c for c in all_cols if c != sel_target],
                    default=default_feats[:min(len(default_feats), 20)],
                )

                col_cfg_submit = st.form_submit_button("✓ Load with this configuration")

            if col_cfg_submit:
                if not sel_features:
                    st.error("Select at least one feature column.")
                else:
                    df_filtered = raw.dropna(subset=[sel_target]).reset_index(drop=True)
                    # Fill missing feature cols with NaN
                    for fc in sel_features:
                        if fc not in df_filtered.columns:
                            df_filtered[fc] = np.nan
                    st.session_state.custom_target_col   = sel_target
                    st.session_state.custom_feature_cols = sel_features
                    st.session_state.task                = sel_task
                    _store_data(df_filtered, feature_cols=sel_features, target_col=sel_target)
                    dropped = len(raw) - len(df_filtered)
                    note = f"✅ Loaded {len(df_filtered)} samples · {len(sel_features)} features · task={sel_task}"
                    if dropped:
                        note += f" ({dropped} rows dropped — missing target)"
                    st.toast(note, icon="✅")
                    st.rerun()

    if st.session_state.df is not None:
        df      = st.session_state.df
        fc_     = _fc()
        tc_     = _tc()
        task_   = st.session_state.task

        st.divider()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total samples", len(df))
        c2.metric("Train", len(st.session_state.X_train))
        c3.metric("Test",  len(st.session_state.X_test))
        c4.metric("Features", len(fc_))
        c5.metric("Task", task_)

        if task_ == "classification" and st.session_state.median_threshold is not None:
            med = st.session_state.median_threshold
            n1  = int((df[tc_] >= med).sum())
            n0  = int((df[tc_] <  med).sum())
            st.caption(f"Classification threshold (median): **{med:.4f}** → class 0: {n0} · class 1: {n1}")

        show_cols = [c for c in fc_ if c in df.columns] + [tc_]
        with st.expander("Data preview (first 50 rows)", expanded=True):
            st.dataframe(df[show_cols].head(50), use_container_width=True)
        with st.expander("Feature statistics"):
            feat_show = [c for c in fc_ if c in df.columns]
            st.dataframe(df[feat_show].describe().T.round(3), use_container_width=True)

        col_h, _ = st.columns(2)
        with col_h:
            fig_h, ax_h = plt.subplots(figsize=(6, 3.5))
            ax_h.hist(df[tc_].dropna(), bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
            ax_h.set_xlabel(tc_); ax_h.set_ylabel("Count")
            ax_h.set_title("Target Distribution")
            plt.tight_layout()
            st.pyplot(fig_h, clear_figure=True)
            plt.close(fig_h)
    else:
        st.info("Upload a file or click **Use demo data** to get started.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL TRAINING
# ════════════════════════════════════════════════════════════════════════════
with tab_train:
    st.header("Model Training")

    if st.session_state.df is None:
        st.warning("Load data first (→ **Data** tab).")
    else:
        _is_clf = st.session_state.task == "classification"

        # ── Cross-Validation ───────────────────────────────────────────────
        st.subheader("Cross-Validation")
        _mdl_pool = get_classifiers() if _is_clf else get_models()
        all_model_names = list(_mdl_pool.keys())
        selected = st.multiselect(
            "Models to evaluate", options=all_model_names, default=all_model_names[:6],
        )
        cv_col1, cv_col2, cv_col3 = st.columns([1, 1, 2])
        n_folds        = int(cv_col1.number_input("CV Folds", 2, 10, config_TU_Ilmenau.CV_FOLDS, 1))
        use_pfhpo      = cv_col2.toggle(
            "Per-fold HPO",
            value=False,
            help="Run a separate Optuna study for each CV fold. "
                 "More accurate but much slower (n_folds × n_trials evaluations).",
        )
        pfhpo_trials   = int(cv_col3.slider(
            "HPO trials per fold", 10, 100, 30, step=5,
            disabled=not use_pfhpo,
        ))

        if st.button("▶  Run Cross-Validation", type="primary"):
            if not selected:
                st.warning("Select at least one model.")
            else:
                X_tr = st.session_state.X_train
                y_tr = st.session_state.y_class_train if _is_clf else st.session_state.y_train
                splits = cross_val_splits(X_tr, y_tr, n_folds=n_folds)
                bar    = st.progress(0, text="Initializing…")
                parts  = []
                for i, name in enumerate(selected):
                    bar.progress(i / len(selected), text=f"Evaluating {name}…")
                    mdl_subset = {name: _mdl_pool[name]}
                    if _is_clf:
                        parts.append(evaluate_classifiers(
                            X_tr, y_tr, models=mdl_subset, cv_splits=splits,
                            per_fold_hpo=use_pfhpo, n_hpo_trials=pfhpo_trials,
                        ))
                    else:
                        parts.append(evaluate_all(
                            X_tr, y_tr, models=mdl_subset, cv_splits=splits,
                            per_fold_hpo=use_pfhpo, n_hpo_trials=pfhpo_trials,
                        ))
                bar.progress(1.0, text="Done!")
                cv = pd.concat(parts, ignore_index=True)
                sort_col = "accuracy" if _is_clf else "rmse"
                best = cv.groupby("model")[sort_col].mean().idxmax() if _is_clf \
                    else cv.groupby("model")[sort_col].mean().idxmin()
                st.session_state.cv_results      = cv
                st.session_state.best_model_name = best
                st.success(f"Best model: **{best}**")

        if st.session_state.cv_results is not None:
            cv = st.session_state.cv_results
            metrics_avail = [c for c in ["r2", "rmse", "mae", "accuracy", "f1", "auc"]
                             if c in cv.columns]
            summary = cv.groupby("model")[metrics_avail].mean().round(4)
            if "rmse" in summary.columns:
                summary = summary.sort_values("rmse")
            elif "accuracy" in summary.columns:
                summary = summary.sort_values("accuracy", ascending=False)

            hi_cols  = [c for c in ["r2", "accuracy", "f1", "auc"] if c in summary.columns]
            lo_cols  = [c for c in ["rmse", "mae"]                  if c in summary.columns]
            styled = summary.style
            if hi_cols: styled = styled.highlight_max(subset=hi_cols, color="#c8f7c5")
            if lo_cols: styled = styled.highlight_min(subset=lo_cols, color="#c8f7c5")
            st.dataframe(styled, use_container_width=True)

            n_metrics = len(metrics_avail)
            fig_cv, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
            if n_metrics == 1:
                axes = [axes]
            for ax, metric in zip(axes, metrics_avail):
                order   = cv.groupby("model")[metric].median().sort_values(
                    ascending=(metric not in ("r2", "accuracy", "f1", "auc"))).index
                data_cv = [cv[cv["model"] == m][metric].values for m in order]
                ax.boxplot(data_cv, labels=order, patch_artist=True)
                ax.set_title(metric.upper())
                ax.tick_params(axis="x", rotation=35)
            fig_cv.suptitle("Cross-Validation Comparison", fontsize=13)
            plt.tight_layout()
            st.pyplot(fig_cv, clear_figure=True)
            plt.close(fig_cv)

        # ── Final Model ────────────────────────────────────────────────────
        st.divider()
        st.subheader("Final Model")
        model_opts  = list(_mdl_pool.keys())
        best_name   = st.session_state.best_model_name
        default_idx = model_opts.index(best_name) if best_name in model_opts else 0
        chosen = st.selectbox("Model to train on full training set", model_opts, index=default_idx)

        hpo_col, trials_col = st.columns([1, 2])
        run_hpo  = hpo_col.toggle("HPO (Optuna)", value=False)
        n_trials = trials_col.slider("Trials", 10, 200, 50, step=10, disabled=not run_hpo)

        if st.button("▶  Train Final Model", type="primary"):
            with st.spinner(f"Training {chosen}…"):
                try:
                    best_params: dict = {}
                    if run_hpo and chosen not in ("linear", "logistic"):
                        try:
                            best_params = hp_tuning.tune(
                                chosen, st.session_state.X_train,
                                st.session_state.y_train, n_trials=n_trials,
                            )
                            st.info(f"Best params: {best_params}")
                        except Exception as hpo_exc:
                            st.warning(f"HPO skipped: {hpo_exc}")

                    y_tr = st.session_state.y_class_train if _is_clf else st.session_state.y_train
                    y_te = st.session_state.y_class_test  if _is_clf else st.session_state.y_test

                    if _is_clf:
                        mdl_, pre_ = train_final_classifier(chosen, st.session_state.X_train, y_tr)
                        met        = eval_final_classifier(mdl_, pre_, st.session_state.X_test, y_te)
                    else:
                        mdl_, pre_ = train_final(chosen, st.session_state.X_train, y_tr,
                                                 best_params=best_params)
                        met        = eval_final(mdl_, pre_, st.session_state.X_test, y_te)

                    X_sample = pre_.transform(st.session_state.X_train.values)
                    st.session_state.update({
                        "final_model": mdl_, "final_preprocessor": pre_,
                        "final_metrics": met, "best_model_name": chosen,
                        "final_X_sample": X_sample,
                        "shap_values": None, "shap_importance": None,
                        "shap_X": None, "perm_importance": None,
                    })
                    primary_metric = f"Accuracy={met['accuracy']:.4f}" if _is_clf \
                        else f"R²={met['r2']:.4f}"
                    st.success(f"Trained **{chosen}** — {primary_metric}")
                except Exception as exc:
                    st.error(f"Training failed: {exc}")

        if st.session_state.final_metrics:
            m = st.session_state.final_metrics
            if _is_clf:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{m['accuracy']:.4f}" if m.get("accuracy") is not None else "—")
                c2.metric("F1",       f"{m['f1']:.4f}"       if m.get("f1")       is not None else "—")
                c3.metric("AUC",      f"{m['auc']:.4f}"      if m.get("auc")      is not None else "—")
                c4.metric("Test Samples", int(m["n_test"]) if m.get("n_test") is not None else "—")

                # Confusion matrix inline
                if m.get("confusion_matrix"):
                    with st.expander("Confusion Matrix", expanded=False):
                        import matplotlib.ticker as _ticker
                        from sklearn.metrics import ConfusionMatrixDisplay
                        cm_arr = np.array(m["confusion_matrix"])
                        fig_cm, (ax_cm1, ax_cm2) = plt.subplots(1, 2, figsize=(8, 3))
                        ConfusionMatrixDisplay(cm_arr).plot(ax=ax_cm1, colorbar=False)
                        ax_cm1.set_title("Counts")
                        cm_norm = cm_arr.astype(float) / cm_arr.sum(axis=1, keepdims=True)
                        ConfusionMatrixDisplay(np.round(cm_norm, 2)).plot(ax=ax_cm2, colorbar=False)
                        ax_cm2.set_title("Normalised")
                        plt.tight_layout()
                        st.pyplot(fig_cm, clear_figure=True)
                        plt.close(fig_cm)

                if m.get("classification_report"):
                    with st.expander("Classification Report", expanded=False):
                        rep_dict = m["classification_report"]
                        rep_df   = pd.DataFrame(rep_dict).T
                        st.dataframe(rep_df.round(4), use_container_width=True)
            else:
                c1, c2, c3, c4 = st.columns(4)
                r2_str   = f"{m['r2']:.4f}"   if m.get("r2")   is not None else "—"
                rmse_str = f"{m['rmse']:.4f} µm" if m.get("rmse") is not None else "—"
                mae_str  = f"{m['mae']:.4f} µm"  if m.get("mae")  is not None else "—"
                if m.get("r2_ci_low") is not None:
                    r2_str   += f"  [{m['r2_ci_low']:.3f}, {m['r2_ci_high']:.3f}]"
                    rmse_str += f"  [{m['rmse_ci_low']:.4f}, {m['rmse_ci_high']:.4f}]"
                    mae_str  += f"  [{m['mae_ci_low']:.4f}, {m['mae_ci_high']:.4f}]"
                c1.metric("Test R²",      r2_str)
                c2.metric("Test RMSE",    rmse_str)
                c3.metric("Test MAE",     mae_str)
                c4.metric("Test Samples", int(m["n_test"]) if m.get("n_test") is not None else "—")

                # Bootstrap CI scatter plot
                if (st.session_state.final_model is not None
                        and st.session_state.final_preprocessor is not None
                        and st.session_state.X_test is not None):
                    with st.expander("Predicted vs. True with Bootstrap CI", expanded=False):
                        try:
                            from pipeline.preprocessing import align_features as _align
                            _fc_saved = (list(st.session_state.X_train.columns)
                                         if st.session_state.X_train is not None else _fc())
                            X_te_al = _align(st.session_state.X_test, _fc_saved)
                            X_te_sc = st.session_state.final_preprocessor.transform(X_te_al.values)
                            y_te_np = (st.session_state.y_class_test.values if _is_clf
                                       else st.session_state.y_test.values)
                            y_pr_np = st.session_state.final_model.predict(X_te_sc)
                            fig_bci = xai.bootstrap_ci_plot(y_te_np, y_pr_np)
                            st.pyplot(fig_bci, clear_figure=True)
                            plt.close(fig_bci)
                        except Exception as exc:
                            st.warning(f"Bootstrap CI plot failed: {exc}")

            st.divider()
            _r2 = m.get("r2")
            if not _is_clf and _r2 is not None and _r2 < 0.6:
                st.warning(f"⚠️ R² = {_r2:.4f} — model quality is low (< 0.6). Saving is not recommended.")
            col_name, col_save, col_del = st.columns([2, 1, 1])
            model_save_name = col_name.text_input(
                "Save as",
                value=f"{chosen}_{dt.datetime.now().strftime('%Y%m%d')}",
            )
            if col_save.button("💾 Save", use_container_width=True):
                try:
                    _x_sample = st.session_state.final_X_sample
                    if (
                        _x_sample is None
                        and st.session_state.X_train is not None
                        and st.session_state.final_preprocessor is not None
                    ):
                        _x_sample = st.session_state.final_preprocessor.transform(
                            st.session_state.X_train.values
                        )
                    _feat_cols = list(st.session_state.X_train.columns) if st.session_state.X_train is not None else _fc()
                    model_management.save_model(
                        st.session_state.final_model,
                        st.session_state.final_preprocessor,
                        st.session_state.final_metrics,
                        st.session_state.best_model_name,
                        config_TU_Ilmenau.MODELS_DIR,
                        custom_name=model_save_name,
                        X_sample=_x_sample,
                        metadata={
                            "feature_cols": _feat_cols,
                            "target_col":   _tc(),
                            "task":         st.session_state.task,
                            "median_threshold": st.session_state.median_threshold,
                        },
                    )
                    st.toast(f"✅ Saved as '{model_save_name}'", icon="💾")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Save failed: {exc}")
            if col_del.button("🗑 Clear", use_container_width=True):
                st.session_state.update({
                    "final_model": None, "final_preprocessor": None,
                    "final_metrics": None, "final_X_sample": None, "best_model_name": None,
                })
                st.rerun()

        # ── Saved Models ───────────────────────────────────────────────────
        st.divider()
        st.subheader("Saved Models")
        avail = model_management.get_available_models(config_TU_Ilmenau.MODELS_DIR)
        if avail.empty:
            st.info("No saved models yet.")
        else:
            st.dataframe(avail.drop(columns=["X_sample"], errors="ignore"),
                         use_container_width=True)
            del_name = st.selectbox("Delete model", ["—"] + avail["Name"].tolist(),
                                    key="del_model_select")
            if del_name != "—" and st.button("🗑 Delete selected", type="secondary"):
                if model_management.delete_model(del_name, config_TU_Ilmenau.MODELS_DIR):
                    st.success(f"Deleted '{del_name}'")
                    st.rerun()
                else:
                    st.error("Delete failed.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — FEATURE IMPORTANCE
# Only uses saved models. X_sample (scaled training data) stored in the
# artifact enables XAI without reloading the original data file.
# ════════════════════════════════════════════════════════════════════════════
with tab_xai:
    st.header("Feature Importance")

    avail_xai = model_management.get_available_models(config_TU_Ilmenau.MODELS_DIR)
    if avail_xai.empty:
        st.info("No saved models found. Train and save a model first (→ **Model Training** tab).")
    else:
        sel_xai = st.selectbox(
            "Saved model",
            avail_xai["Name"].tolist(),
            key="xai_model_select",
        )
        mdl_xai, pre_xai, meta_xai = _load_saved(sel_xai)

        if mdl_xai is None:
            st.error(f"Could not load '{sel_xai}'")
        else:
            # ── Resolve data source ────────────────────────────────────────
            # Priority: loaded test set > stored training sample > nothing
            X_xai   = meta_xai.get("X_sample")   # may be None
            y_xai   = None
            src_label = ""
            _model_feat_cols = meta_xai.get("feature_cols", _fc())

            def _transform_test():
                """Align session X_test to the model's feature set, then transform."""
                X_aligned = align_features(st.session_state.X_test, _model_feat_cols)
                return pre_xai.transform(X_aligned.values)

            if st.session_state.X_test is not None:
                # Data is loaded — always prefer the test set for XAI
                if X_xai is not None:
                    use_test = st.checkbox(
                        "Use loaded test set instead of stored training sample", value=True,
                    )
                    if use_test:
                        X_xai     = _transform_test()
                        y_xai     = st.session_state.y_test.values
                        src_label = f"loaded test set ({len(y_xai)} samples)"
                    else:
                        src_label = f"stored training sample ({X_xai.shape[0]} samples)"
                else:
                    X_xai     = _transform_test()
                    y_xai     = st.session_state.y_test.values
                    src_label = f"loaded test set ({len(y_xai)} samples)"
            elif X_xai is not None:
                src_label = f"stored training sample ({X_xai.shape[0]} samples)"
            else:
                st.warning(
                    "No data available for this model. "
                    "Either load data in the **Data** tab, "
                    "or re-train and save the model to include a training sample."
                )

            if X_xai is not None:
                st.caption(
                    f"Model: **{meta_xai.get('model_name', sel_xai)}** · "
                    f"Data source: {src_label}"
                )

                _METHODS = [
                    "SHAP Summary",
                    "SHAP Dependence",
                    "SHAP Waterfall",
                    "SHAP Interactions (tree models)",
                    "ICE Plots",
                    "ALE Plots",
                    "PDP Plots",
                    "Permutation Importance",
                    "Learning Curve",
                    "Decision Tree Visualisation",
                    "Confusion Matrix (classification)",
                    "Classification Report (classification)",
                ]
                method = st.selectbox(
                    "Analysis method",
                    _METHODS,
                    help="Methods marked '(tree models)' or '(classification)' require the appropriate model/task.",
                )
                _needs_labels = method in (
                    "Permutation Importance", "Learning Curve",
                    "Confusion Matrix (classification)", "Classification Report (classification)",
                )
                _needs_shap = method in (
                    "SHAP Summary", "SHAP Dependence", "SHAP Waterfall",
                    "SHAP Interactions (tree models)",
                )
                top_n_xai = st.slider("Top-N features", 3, min(15, len(_model_feat_cols)), 6,
                                      key="xai_top_n",
                                      disabled=method in ("SHAP Interactions (tree models)",
                                                          "Decision Tree Visualisation",
                                                          "Confusion Matrix (classification)",
                                                          "Classification Report (classification)"))
                if method == "Decision Tree Visualisation":
                    st.session_state["tree_max_depth"] = int(
                        st.slider("Max tree depth to display", 1, 10, 4, key="xai_tree_depth")
                    )

                disabled_btn = (_needs_labels and y_xai is None)
                if disabled_btn:
                    st.warning("This method needs ground-truth labels — load data in the **Data** tab.")

                if st.button("▶  Compute", type="primary", disabled=disabled_btn):
                    config_TU_Ilmenau.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
                    config_TU_Ilmenau.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
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
                                    }).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True))
                                    st.session_state.update(
                                        shap_values=sv, shap_importance=imp_df,
                                        shap_X=X_xai, perm_importance=None,
                                    )

                            elif method == "SHAP Dependence":
                                xai.shap_dependence_plots(mdl_xai, X_xai, _model_feat_cols,
                                                          top_n=top_n_xai)
                                st.image(str(config_TU_Ilmenau.FIGURES_DIR / "shap_dependence.pdf"),
                                         caption="SHAP Dependence Plots")

                            elif method == "SHAP Waterfall":
                                xai.shap_waterfall_plots(mdl_xai, X_xai, _model_feat_cols)
                                for i in range(3):
                                    p = config_TU_Ilmenau.FIGURES_DIR / f"shap_waterfall_{i}.pdf"
                                    if p.exists():
                                        st.image(str(p))

                            elif method == "SHAP Interactions (tree models)":
                                df_inter = xai.shap_interaction_matrix(
                                    mdl_xai, X_xai, _model_feat_cols)
                                if not df_inter.empty:
                                    st.subheader("Top interaction pairs")
                                    st.dataframe(df_inter.head(20).round(6),
                                                 use_container_width=True)
                                    st.download_button(
                                        "⬇  shap_interaction_pairs.csv",
                                        df_inter.head(50).to_csv(index=False).encode(),
                                        "shap_interaction_pairs.csv", "text/csv",
                                    )

                            elif method == "ICE Plots":
                                fig_ice = xai.ice_plots(mdl_xai, X_xai, _model_feat_cols,
                                                        top_n=top_n_xai)
                                st.pyplot(fig_ice, clear_figure=True)

                            elif method == "ALE Plots":
                                fig_ale = xai.ale_plots(mdl_xai, X_xai, _model_feat_cols,
                                                        top_n=top_n_xai)
                                st.pyplot(fig_ale, clear_figure=True)

                            elif method == "PDP Plots":
                                fig_pdp = xai.pdp_plots(mdl_xai, X_xai, _model_feat_cols,
                                                        top_n=top_n_xai)
                                st.pyplot(fig_pdp, clear_figure=True)

                            elif method == "Permutation Importance":
                                imp_df = xai.permutation_importance(
                                    mdl_xai, X_xai, y_xai, _model_feat_cols)
                                st.session_state.perm_importance = imp_df
                                st.session_state.shap_importance = None

                            elif method == "Learning Curve":
                                from sklearn.base import clone
                                try:
                                    m_clone = clone(mdl_xai)
                                except Exception:
                                    m_clone = mdl_xai
                                fig_lc = xai.learning_curve_plot(
                                    m_clone, X_xai, y_xai, _model_feat_cols)
                                st.pyplot(fig_lc, clear_figure=True)

                            elif method == "Decision Tree Visualisation":
                                max_d = st.session_state.get("tree_max_depth", 4)
                                fig_tree = xai.show_tree(mdl_xai, _model_feat_cols,
                                                         max_depth=max_d)
                                st.pyplot(fig_tree, clear_figure=True)

                            elif method == "Confusion Matrix (classification)":
                                fig_cm = xai.confusion_matrix_plot(
                                    mdl_xai, X_xai, y_xai)
                                st.pyplot(fig_cm, clear_figure=True)

                            elif method == "Classification Report (classification)":
                                df_rep = xai.classification_report_plot(
                                    mdl_xai, X_xai, y_xai)
                                st.pyplot(plt.gcf(), clear_figure=True)
                                st.dataframe(df_rep.round(4), use_container_width=True)
                                st.download_button(
                                    "⬇  classification_report.csv",
                                    df_rep.to_csv().encode(),
                                    "classification_report.csv", "text/csv",
                                )

                        except Exception as exc:
                            st.error(f"{method} failed: {exc}")

                # ── SHAP summary results (persisted) ───────────────────────
                if st.session_state.shap_importance is not None and _HAS_SHAP:
                    imp    = st.session_state.shap_importance
                    sv     = st.session_state.shap_values
                    X_disp = st.session_state.shap_X if st.session_state.shap_X is not None else X_xai
                    X_df   = pd.DataFrame(X_disp, columns=_model_feat_cols)
                    h      = max(4.5, len(_model_feat_cols) * 0.32)

                    col_l, col_r = st.columns(2)
                    with col_l:
                        st.subheader("Mean |SHAP| – Bar")
                        plt.clf()
                        _shap.summary_plot(sv, X_df, plot_type="bar", show=False, plot_size=(6, h))
                        st.pyplot(plt.gcf(), clear_figure=True)
                    with col_r:
                        st.subheader("SHAP Beeswarm")
                        plt.clf()
                        _shap.summary_plot(sv, X_df, show=False, plot_size=(6, h))
                        st.pyplot(plt.gcf(), clear_figure=True)

                    st.dataframe(imp.round(5), use_container_width=True)
                    st.download_button(
                        "⬇  shap_importance.csv",
                        imp.to_csv(index=False).encode(),
                        "shap_importance.csv", "text/csv",
                    )

                # ── Permutation results (persisted) ───────────────────────
                if st.session_state.perm_importance is not None:
                    imp = st.session_state.perm_importance
                    fig_pi, ax_pi = plt.subplots(figsize=(9, max(4, len(imp) * 0.38)))
                    ax_pi.barh(
                        imp["feature"][::-1], imp["importance_mean"][::-1],
                        xerr=imp["importance_std"][::-1],
                        color="#4C72B0", ecolor="gray", alpha=0.85,
                    )
                    ax_pi.set_xlabel("Mean decrease in R²")
                    ax_pi.set_title("Permutation Importance")
                    plt.tight_layout()
                    st.pyplot(fig_pi, clear_figure=True)
                    st.dataframe(imp.round(4), use_container_width=True)
                    st.download_button(
                        "⬇  permutation_importance.csv",
                        imp.to_csv(index=False).encode(),
                        "permutation_importance.csv", "text/csv",
                    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — INVERSE ML
# Only uses saved models. Bounds come from FEATURE_REGISTRY — no data needed.
# ════════════════════════════════════════════════════════════════════════════
with tab_inv:
    st.header("Inverse ML")
    st.caption(
        "Find process parameters that produce a desired chromium layer thickness. "
        "No data upload needed — bounds come from the feature registry."
    )

    avail_inv = model_management.get_available_models(config_TU_Ilmenau.MODELS_DIR)
    if avail_inv.empty:
        st.info("No saved models found. Train and save a model first (→ **Model Training** tab).")
    else:
        sel_inv = st.selectbox(
            "Saved model",
            avail_inv["Name"].tolist(),
            key="inv_model_select",
        )
        mdl_inv, pre_inv, meta_inv = _load_saved(sel_inv)
        _inv_feat_cols = meta_inv.get("feature_cols", _fc()) if meta_inv else _fc()

        if mdl_inv is None:
            st.error(f"Could not load '{sel_inv}'")
        else:
            # ── Target ─────────────────────────────────────────────────────
            st.subheader("Target Thickness")
            t_min  = float(TARGET_META["data_min"])
            t_max  = float(TARGET_META["data_max"])
            t_mean = float(TARGET_META["data_mean"])

            col_sl, col_num = st.columns([3, 1])
            target_sl  = col_sl.slider("Target (µm)", t_min, t_max, t_mean, 0.005)
            target_val = col_num.number_input(
                "Exact (µm)", t_min, t_max, target_sl, 0.001, format="%.3f",
            )

            # ── Parameter editor ───────────────────────────────────────────
            st.divider()
            st.subheader("Parameter Configuration")
            st.caption(
                "**Fixed** = known value (not optimised). "
                "**Variable** = free parameter, optimised within the given bounds."
            )

            if "inv_param_df" not in st.session_state:
                st.session_state.inv_param_df = _build_param_df()

            edited_df = st.data_editor(
                st.session_state.inv_param_df,
                key="inv_param_editor",
                column_config={
                    "Feature":      st.column_config.TextColumn(disabled=True),
                    "Display Name": st.column_config.TextColumn(disabled=True),
                    "Unit":         st.column_config.TextColumn(disabled=True),
                    "Category":     st.column_config.TextColumn(disabled=True),
                    "Role": st.column_config.SelectboxColumn(
                        "Role", options=["Fixed", "Variable"], required=True,
                    ),
                    "Fixed Value": st.column_config.NumberColumn("Fixed Value", format="%.4f"),
                    "Min":         st.column_config.NumberColumn("Min",         format="%.4f"),
                    "Max":         st.column_config.NumberColumn("Max",         format="%.4f"),
                },
                hide_index=True,
                use_container_width=True,
                num_rows="fixed",
            )
            st.session_state.inv_param_df = edited_df

            feat_keys    = list(FEATURE_REGISTRY.keys())
            free_vars    = []
            fixed_params = {}
            bounds       = {}
            for i, row in edited_df.iterrows():
                feat = feat_keys[i]
                if row["Role"] == "Variable":
                    free_vars.append(feat)
                    bounds[feat] = (float(row["Min"]), float(row["Max"]))
                else:
                    fixed_params[feat] = float(row["Fixed Value"])

            if not free_vars:
                st.warning("Set at least one feature to **Variable** to enable optimisation.")
            else:
                st.caption(
                    f"**{len(free_vars)} variable(s):** "
                    + ", ".join(
                        f"`{FEATURE_REGISTRY[f]['display_name']}`" for f in free_vars
                    )
                )

            col_ta, col_tb = st.columns(2)
            n_trials    = int(col_ta.number_input("Trials (Bayesian)", 50, 2000, 300, 50))
            n_solutions = int(col_tb.number_input("Solutions to return",  1,   20,   5,  1))

            st.divider()
            if st.button("▶  Find Parameters", type="primary", disabled=not free_vars):
                with st.spinner(f"Bayesian search — {n_trials} trials…"):
                    try:
                        # Only pass features the model was actually trained on
                        _fixed_inv = {k: v for k, v in fixed_params.items() if k in _inv_feat_cols}
                        _free_inv  = [v for v in free_vars if v in _inv_feat_cols]
                        if not _free_inv:
                            st.error("None of the selected free variables are in this model's feature set.")
                            st.stop()
                        results = find_inputs(
                            model=mdl_inv,
                            preprocessor=pre_inv,
                            feature_names=_inv_feat_cols,
                            fixed_params=_fixed_inv,
                            free_vars=_free_inv,
                            target_thickness=float(target_val),
                            bounds={k: v for k, v in bounds.items() if k in _free_inv},
                            n_solutions=n_solutions,
                            method="bayesian",
                            n_trials=n_trials,
                        )
                        st.session_state.inverse_results = results
                        st.success(
                            f"Found {len(results)} solutions — "
                            f"best error: **{results['error'].iloc[0]:.4f} µm**"
                        )
                    except Exception as exc:
                        st.error(f"Optimisation failed: {exc}")

            if st.session_state.inverse_results is not None:
                res = st.session_state.inverse_results
                st.subheader("Top Solutions")
                display = res.copy()
                for col in display.columns:
                    if col not in ("predicted_thickness", "error"):
                        display[col] = display[col].round(4)
                display["predicted_thickness"] = display["predicted_thickness"].map("{:.4f} µm".format)
                display["error"] = display["error"].map("{:.5f} µm".format)
                st.dataframe(display, use_container_width=True)
                st.download_button(
                    "⬇  solutions.csv",
                    res.to_csv(index=False).encode(),
                    "inverse_ml_solutions.csv", "text/csv",
                )


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — FEATURES
# ════════════════════════════════════════════════════════════════════════════
with tab_feat:
    st.header("Feature Reference")
    st.caption("Overview of all 20 input features — properties and data-driven ranges.")

    rows = []
    for feat, meta in FEATURE_REGISTRY.items():
        rows.append({
            "Column name":  feat,
            "Display Name": meta["display_name"],
            "Unit":         meta["unit"],
            "Category":     meta["feature_type"],
            "Default Role": meta["default_role"],
            "Min":          meta["data_min"],
            "Max":          meta["data_max"],
            "Mean":         round(meta["data_mean"], 3),
            "Std":          round(meta["data_std"],  3),
            "Description":  meta["description"],
        })
    feat_df = pd.DataFrame(rows)

    _TYPE_COLORS = {
        "process":       "#d4e6f1",
        "bath":          "#d5f5e3",
        "contamination": "#fde8d8",
        "lifecycle":     "#f9ebea",
        "geometry":      "#f4f6f7",
    }

    categories = ["All"] + sorted(feat_df["Category"].unique().tolist())
    cat_filter = st.selectbox("Filter by category", categories)
    if cat_filter != "All":
        feat_df = feat_df[feat_df["Category"] == cat_filter]

    st.dataframe(
        feat_df.style.apply(
            lambda row: [
                f"background-color: {_TYPE_COLORS.get(row['Category'], '#ffffff')}"
                for _ in row
            ],
            axis=1,
        ),
        use_container_width=True,
        height=680,
    )

    # Legend
    st.divider()
    cols_leg = st.columns(len(_TYPE_COLORS))
    for col_leg, (cat, color) in zip(cols_leg, _TYPE_COLORS.items()):
        col_leg.markdown(
            f"<span style='background:{color};padding:3px 8px;"
            f"border-radius:4px;font-size:0.85em'>{cat}</span>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Target Variable")
    t = TARGET_META
    st.markdown(
        f"**{TARGET_COL}** — {t['description']}  \n"
        f"Unit: `{t['unit']}` · Range: {t['data_min']}–{t['data_max']} · "
        f"Mean: {t['data_mean']:.3f} · Std: {t['data_std']:.3f}"
    )
