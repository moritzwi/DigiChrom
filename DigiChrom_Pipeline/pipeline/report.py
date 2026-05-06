"""HTML/PDF-Bericht für die DigiChrom ML Pipeline.

Struktur:
  Deckblatt → Regression (CV | HPO | Ensembling | Finaler Prädiktor | SHAP/PDP | Inverse ML)
            → Klassifikation (CV | HPO | Ensembling | Finaler Prädiktor | SHAP/PDP | Inverse ML)

PDF-Konvertierung: weasyprint → pandoc → HTML-Fallback.
"""

from __future__ import annotations

import base64
import datetime
import io
import subprocess
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from .config import get_config


# ─── CSS ─────────────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.65;
    color: #222;
    padding: 36px 60px;
}
h1  { font-size: 22pt; color: #2c3e50; margin-bottom: 6px; }
h2  { font-size: 15pt; color: #4C72B0; border-bottom: 2px solid #4C72B0;
      padding-bottom: 4px; margin: 28px 0 12px; }
h2.clf { color: #DD8452; border-color: #DD8452; }
h3  { font-size: 11pt; color: #444; margin: 18px 0 7px; }
p   { margin-bottom: 8px; }

.cover { text-align: center; padding: 50px 0 40px; }
.cover h1 { font-size: 26pt; border-bottom: none; color: #2c3e50; }
.cover .subtitle { color: #555; font-size: 12pt; margin-top: 8px; }
.cover .datum    { color: #999; font-size: 9pt; margin-top: 4px; }

table { border-collapse: collapse; width: 100%; margin: 8px 0 16px; font-size: 9.5pt; }
th { padding: 6px 11px; color: white; text-align: left; }
td { padding: 5px 11px; border-bottom: 1px solid #e5e5e5; }
tr:nth-child(even) td { background: #f5f7fc; }
.t-reg th  { background: #4C72B0; }
.t-clf th  { background: #DD8452; }
.t-dark th { background: #2c3e50; }
.t-cover   { width: auto; margin: 24px auto 0; }
.t-cover td { border: none; padding: 4px 20px; }
.t-cover .lbl { color: #666; }
.t-cover .val { font-weight: bold; }
.t-winner td { background: #e8f4e8 !important; font-weight: bold; }

.info { background: #eef2ff; border-left: 4px solid #4C72B0;
        padding: 10px 14px; margin: 10px 0 14px; border-radius: 0 4px 4px 0; font-size: 9.5pt; }
.info.clf { background: #fff4ec; border-left-color: #DD8452; }
.info.dark { background: #f0f0f5; border-left-color: #2c3e50; }

img { max-width: 82%; display: block; margin: 10px auto 3px; }
.caption { text-align: center; color: #777; font-size: 8pt; margin-bottom: 14px; }

.break { page-break-after: always; height: 1px; }
hr { border: none; border-top: 1px solid #ddd; margin: 20px 0; }
.badge { display:inline-block; background:#2c3e50; color:white;
         padding:1px 7px; border-radius:3px; font-size:8.5pt; }
.badge.reg { background:#4C72B0; }
.badge.clf { background:#DD8452; }
"""


# ─── Hilfsfunktionen ─────────────────────────────────────────────────────────

def _b64(fig: plt.Figure, dpi: int = 150) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _img(fig: plt.Figure, caption: str = "", dpi: int = 150) -> str:
    b64 = _b64(fig, dpi)
    out = f'<img src="data:image/png;base64,{b64}" alt="{caption}">'
    if caption:
        out += f'\n<p class="caption">{caption}</p>'
    return out


def _df_to_html(df: pd.DataFrame, cls: str = "t-reg",
                winner_row: Optional[str] = None,
                winner_col: Optional[str] = None) -> str:
    """DataFrame → HTML-Tabelle. Hebt die Gewinner-Zeile grün hervor."""
    cols = list(df.columns)
    header = "".join(f"<th>{c}</th>" for c in cols)
    rows = ""
    for _, row in df.iterrows():
        cells = "".join(f"<td>{v}</td>" for v in row)
        row_cls = ""
        if winner_col and winner_row and str(row.get(winner_col, "")) == str(winner_row):
            row_cls = ' class="t-winner"'
        rows += f"<tr{row_cls}>{cells}</tr>\n"
    return (f'<table class="{cls}"><thead><tr>{header}</tr></thead>'
            f"<tbody>{rows}</tbody></table>")


def _info(text: str, style: str = "") -> str:
    cls = f"info {style}".strip()
    paras = "".join(f"<p>{p.strip()}</p>" for p in text.strip().split("\n\n") if p.strip())
    return f'<div class="{cls}">{paras}</div>'


def _kv_rows(pairs: list[tuple]) -> str:
    return "".join(
        f'<tr><td class="lbl">{k}</td><td class="val">{v}</td></tr>'
        for k, v in pairs
    )


# ─── Grafiken ────────────────────────────────────────────────────────────────

def _fig_scatter(y_test, preds, title) -> plt.Figure:
    r2 = float(1 - np.sum((np.array(y_test) - np.array(preds))**2) /
               np.sum((np.array(y_test) - np.array(y_test).mean())**2))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_test, preds, alpha=0.6, edgecolors="k", linewidths=0.3,
               s=40, color="#4C72B0")
    lo = min(float(np.min(y_test)), float(np.min(preds))) * 0.95
    hi = max(float(np.max(y_test)), float(np.max(preds))) * 1.05
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)
    ax.set_xlabel("Gemessen (µm)")
    ax.set_ylabel("Vorhergesagt (µm)")
    ax.set_title(f"{title}  R²={r2:.3f}")
    plt.tight_layout()
    return fig


def _fig_shap(importance_df, title, color="#4C72B0", shap_cv=None) -> plt.Figure:
    df = importance_df.head(15).copy()
    if shap_cv is not None and "std_abs_shap" in shap_cv.columns:
        df = df.merge(
            shap_cv[["feature", "std_abs_shap"]].rename(columns={"std_abs_shap": "std_cv"}),
            on="feature", how="left",
        )
    fig, ax = plt.subplots(figsize=(7, max(3, 0.38 * len(df) + 1.2)))
    xerr = df["std_cv"].fillna(0).values if "std_cv" in df.columns else None
    ax.barh(df["feature"], df["mean_abs_shap"], xerr=xerr,
            color=color, alpha=0.85, capsize=3, error_kw={"ecolor": "#555", "lw": 1})
    ax.invert_yaxis()
    ax.set_xlabel("Mittlerer |SHAP-Wert|")
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


def _fig_confusion(cm, clf_name) -> plt.Figure:
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Dünn", "Dick"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Konfusionsmatrix — {clf_name}", fontsize=11)
    plt.tight_layout()
    return fig


def _fig_roc(clf_model, X_scaled, y_true, label) -> plt.Figure:
    from sklearn.metrics import roc_curve, auc
    fig, ax = plt.subplots(figsize=(5, 5))
    if hasattr(clf_model, "predict_proba"):
        try:
            probs = clf_model.predict_proba(X_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_true, probs)
            auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, color="#DD8452",
                    label=f"{label}  (AUC={auc_val:.3f})")
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Falsch-Positiv-Rate (FPR)")
    ax.set_ylabel("Richtig-Positiv-Rate (TPR)")
    ax.set_title("ROC-Kurve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


def _fig_pdp(pdp_results, model_name, task="regression") -> plt.Figure:
    feats = list(pdp_results.keys())
    n = len(feats)
    if n == 0:
        fig, ax = plt.subplots(); ax.set_visible(False); return fig
    # ncols = min(2, n)
    # nrows = (n + ncols - 1) // ncols
    ncols = 1
    nrows = n
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 3.5))
    axes_flat = np.array(axes).flatten() if n > 1 else np.array([axes])
    color = "#4C72B0" if task == "regression" else "#DD8452"
    for ax, feat in zip(axes_flat, feats):
        grid, mean_pdp, std_pdp = pdp_results[feat]
        ax.plot(grid, mean_pdp, color=color, lw=2, label="Mittelwert")
        ax.fill_between(grid, mean_pdp - std_pdp, mean_pdp + std_pdp,
                        alpha=0.25, color=color, label="±1 Std.-Abw.")
        ax.set_xlabel(feat, fontsize=11)
        ax.set_ylabel("Part. Abhängigkeit", fontsize=11)
        ax.set_title(feat, fontsize=11)
        ax.legend(fontsize=7)
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    aufgabe = "Regression" if task == "regression" else "Klassifikation"
    fig.suptitle(f"PDP mit Faltenunsicherheit — {model_name} [{aufgabe}]", fontsize=11)
    plt.tight_layout()
    return fig

def _fig_pdp_single(feat, pdp_data, model_name, task="regression") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    grid, mean_pdp, std_pdp = pdp_data
    
    color = "#4C72B0" if task == "regression" else "#DD8452"
    ax.plot(grid, mean_pdp, color=color, lw=2, label="Mittelwert")
    ax.fill_between(grid, mean_pdp - std_pdp, mean_pdp + std_pdp, alpha=0.25, color=color)
    
    ax.set_title(f"{feat} — {model_name}", fontsize=12, fontweight='bold')
    ax.set_xlabel(feat, fontsize=10)
    ax.set_ylabel("Part. Abhängigkeit", fontsize=10)
    plt.tight_layout()
    return fig


# ─── PDF-Konvertierung ────────────────────────────────────────────────────────

def _html_to_pdf(html: str, pdf_path: Path) -> bool:
    try:
        from weasyprint import HTML
        HTML(string=html).write_pdf(str(pdf_path))
        return True
    except ImportError:
        pass
    except Exception as e:
        print(f"[report] weasyprint fehlgeschlagen: {e}")

    for engine in ["weasyprint", "wkhtmltopdf", ""]:
        try:
            html_tmp = pdf_path.with_suffix(".html")
            html_tmp.write_text(html, encoding="utf-8")
            cmd = ["pandoc", str(html_tmp), "-o", str(pdf_path), "--quiet"]
            if engine:
                cmd += [f"--pdf-engine={engine}"]
            subprocess.run(cmd, check=True, timeout=120)
            html_tmp.unlink(missing_ok=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue

    html_path = pdf_path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")
    print(f"[report] PDF-Konvertierung nicht verfügbar — HTML gespeichert: {html_path}")
    print("[report] Hinweis: 'pip install weasyprint' für direkte PDF-Ausgabe.")
    return False


# ─── Hauptfunktion ────────────────────────────────────────────────────────────

def generate_report(
    *,
    # ── Regression ───────────────────────────────────────────────────────────
    model_name_reg: str,                              # Bestes Einzelmodell
    final_model,                                       # Fitted bestes Einzelmodell
    final_pre,                                         # Preprocessor Einzelmodell
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_train: int,
    cv_results_reg: Optional[pd.DataFrame] = None,    # CV aller Modelle
    hpo_results_reg: Optional[dict] = None,           # best_rmse_reg: {model: rmse}
    ensemble_results_reg: Optional[pd.DataFrame] = None,  # ens_reg DataFrame
    metrics_reg: dict,                                 # Test-Metriken Einzelmodell
    best_predictor_reg_type: str,
    metrics_best_pred_reg: Optional[dict] = None,     # Test-Metriken best. Prädiktor
    shap_importance_reg: pd.DataFrame,
    shap_cv_reg: Optional[pd.DataFrame] = None,
    pdp_cv_reg: Optional[dict] = None,
    inverse_ml_solutions: Optional[pd.DataFrame] = None,
    inverse_ml_target: Optional[float] = None,
    # ── Klassifikation ───────────────────────────────────────────────────────
    clf_name: str,
    clf_model,
    clf_pre,
    y_test_cls: pd.Series,
    clf_cv: Optional[pd.DataFrame] = None,
    hpo_results_clf: Optional[dict] = None,           # best_f1_clf: {model: f1}
    ensemble_results_clf: Optional[pd.DataFrame] = None,
    clf_metrics: dict,
    best_predictor_clf_type: str,
    metrics_best_pred_clf: Optional[dict] = None,
    shap_importance_clf: pd.DataFrame,
    shap_cv_clf: Optional[pd.DataFrame] = None,
    pdp_cv_clf: Optional[dict] = None,
    # ── Gemeinsam ─────────────────────────────────────────────────────────────
    feature_names: list,
    median_threshold: float,
    save_path=None,
) -> Path:
    """Erstellt einen strukturierten HTML/PDF-Bericht."""

    save_path = Path(save_path or (get_config().REPORTS_DIR / "pipeline_report.pdf"))
    save_path.parent.mkdir(parents=True, exist_ok=True)

    jetzt    = datetime.datetime.now().strftime("%d.%m.%Y  %H:%M")
    n_total  = n_train + len(X_test)
    bp_reg   = metrics_best_pred_reg or metrics_reg
    bp_clf   = metrics_best_pred_clf or clf_metrics

    # Berechnungen
    y_pred_single = np.atleast_1d(final_model.predict(final_pre.transform(X_test.values)))

    from sklearn.metrics import confusion_matrix
    X_clf_s    = clf_pre.transform(X_test.values)
    y_pred_cls = clf_model.predict(X_clf_s)
    cm         = confusion_matrix(y_test_cls, y_pred_cls)

    parts: list[str] = []

    def add(html: str):
        parts.append(html)

    def pagebreak():
        add('<div class="break"></div>')

    # ═════════════════════════════════════════════════════════════════════════
    # DECKBLATT
    # ═════════════════════════════════════════════════════════════════════════
    add(f"""
<div class="cover">
  <h1>DigiChrom ML Pipeline</h1>
  <p class="subtitle">Vorhersage der galvanischen Chromschichtdicke</p>
  <p class="datum">Erstellt: {jetzt}</p>
  <hr style="margin: 20px auto; width: 55%; border-color: #ccc;">

  <table class="t-cover" style="margin: 0 auto; min-width: 520px;">
    <tbody>
      <tr><td colspan="4" style="font-weight:bold; font-size:10.5pt; padding-bottom:8px; color:#555;">
          Datensatz</td></tr>
      <tr>
        <td class="lbl">Gesamt</td><td class="val">{n_total}</td>
        <td class="lbl" style="padding-left:28px;">Training</td><td class="val">{n_train}</td>
      </tr>
      <tr>
        <td class="lbl">Testdaten</td><td class="val">{len(X_test)}</td>
        <td class="lbl" style="padding-left:28px;">Feature</td><td class="val">{len(feature_names)}</td>
      </tr>
      <tr>
        <td class="lbl">Zielwertbereich Testset</td>
        <td class="val" colspan="3">[{float(y_test.min()):.3f}, {float(y_test.max()):.3f}] µm</td>
      </tr>

      <tr><td colspan="4" style="padding-top:20px; font-weight:bold; font-size:10.5pt;
              color:#4C72B0;">Bestes Regressionsmodell
          <span class="badge reg">{best_predictor_reg_type}</span></td></tr>
      <tr>
        <td class="lbl">R²</td>
        <td class="val">{bp_reg.get('r2', float('nan')):.4f}</td>
        <td class="lbl" style="padding-left:28px;">RMSE</td>
        <td class="val">{bp_reg.get('rmse', float('nan')):.4f} µm</td>
      </tr>
      <tr>
        <td class="lbl">MAE</td>
        <td class="val">{bp_reg.get('mae', float('nan')):.4f} µm</td>
        <td colspan="2"></td>
      </tr>

      <tr><td colspan="4" style="padding-top:20px; font-weight:bold; font-size:10.5pt;
              color:#DD8452;">Bestes Klassifikationsmodell
          <span class="badge clf">{best_predictor_clf_type}</span></td></tr>
      <tr>
        <td class="lbl">Schwellenwert</td>
        <td class="val">{median_threshold:.4f} µm</td>
        <td class="lbl" style="padding-left:28px;">F1-Score</td>
        <td class="val">{bp_clf.get('f1', float('nan')):.4f}</td>
      </tr>
      <tr>
        <td class="lbl">AUC</td>
        <td class="val">{bp_clf.get('auc', float('nan')):.4f}</td>
        <td class="lbl" style="padding-left:28px;">Accuracy</td>
        <td class="val">{bp_clf.get('accuracy', float('nan')):.4f}</td>
      </tr>
    </tbody>
  </table>
</div>
""")
    pagebreak()

    # ═════════════════════════════════════════════════════════════════════════
    # REGRESSION
    # ═════════════════════════════════════════════════════════════════════════
    add('<h2>Regression — Schichtdickenvorhersage (µm)</h2>')
    add(_info(
        "Das Regressions-Modell sagt die Dicke der Chromschicht in µm vorher. "
        "Die Pipeline durchläuft Cross-Validation, Hyperparameter-Optimierung (HPO), "
        "Ensembling und wählt automatisch den besten Gesamtprädiktor aus."
    ))

    # ── 1. CV aller Modelle ──────────────────────────────────────────────────
    add('<h3>1 · Cross-Validation — Test aller Basismodelle</h3>')
    if cv_results_reg is not None and not cv_results_reg.empty:
        try:
            grp = (cv_results_reg.groupby("model")[["r2", "rmse", "mae"]]
                   .agg(["mean", "std"]).round(4))
            grp.columns = ["R² Mittel", "R² Std", "RMSE Mittel", "RMSE Std",
                           "MAE Mittel", "MAE Std"]
            grp = grp.sort_values("R² Mittel", ascending=False).reset_index()
            grp.rename(columns={"model": "Modell"}, inplace=True)
            add(_df_to_html(grp, cls="t-reg"))
        except Exception:
            add('<p style="color:#888;font-size:9pt;">CV-Ergebnisse nicht verfügbar.</p>')
    else:
        add('<p style="color:#888;font-size:9pt;">CV-Ergebnisse nicht übergeben.</p>')

    # ── 2. HPO-Ergebnisse ────────────────────────────────────────────────────
    add('<h3>2 · Ergebnisse nach Hyperparameter-Optimierung (HPO)</h3>')
    add(_info(
        "Bester RMSE je Modell aus der Hyperparameter-Optimierung (Optuna, 5-fach CV). "
        f"Das beste Einzelmodell ist <b>{model_name_reg}</b>."
    ))
    if hpo_results_reg:
        hpo_df = (pd.DataFrame(
            [{"Modell": k, "Bester RMSE (HPO)": f"{v:.4f} µm"}
             for k, v in sorted(hpo_results_reg.items(), key=lambda x: x[1])]
        ))
        add(_df_to_html(hpo_df, cls="t-reg", winner_row=model_name_reg, winner_col="Modell"))
    else:
        # Fallback: aus CV-Ergebnissen
        if cv_results_reg is not None and not cv_results_reg.empty:
            try:
                fb = (cv_results_reg.groupby("model")[["r2", "rmse", "mae"]]
                      .mean().round(4).sort_values("rmse").reset_index())
                fb.rename(columns={"model": "Modell", "r2": "R²",
                                   "rmse": "RMSE", "mae": "MAE"}, inplace=True)
                add(_df_to_html(fb, cls="t-reg",
                                winner_row=model_name_reg, winner_col="Modell"))
            except Exception:
                pass
        else:
            add('<p style="color:#888;font-size:9pt;">HPO-Ergebnisse nicht übergeben.</p>')

    # Test-Metriken bestes Einzelmodell (nach HPO) + Scatter
    add(f'<p style="margin-top:10px;font-size:9.5pt;">Testset-Metriken '
        f'<b>{model_name_reg}</b> (nach HPO, retrained): '
        f'R²&nbsp;=&nbsp;<b>{metrics_reg.get("r2", float("nan")):.4f}</b> &nbsp;|&nbsp; '
        f'RMSE&nbsp;=&nbsp;<b>{metrics_reg.get("rmse", float("nan")):.4f}&nbsp;µm</b> &nbsp;|&nbsp; '
        f'MAE&nbsp;=&nbsp;<b>{metrics_reg.get("mae", float("nan")):.4f}&nbsp;µm</b></p>')
    add(_img(
        _fig_scatter(y_test, y_pred_single,
                     f"Vorhersage vs. Messung — {model_name_reg}"),
        caption=f"Bestes Einzelmodell nach HPO: {model_name_reg}",
    ))

    # ── 3. Ensembling ────────────────────────────────────────────────────────
    add('<h3>3 · Ensembling — Alle Kombinationen</h3>')
    add(_info(
        "Die drei stärksten Einzelmodelle werden zu Averaging-, Weighted- und Stacking-Ensembles "
        "kombiniert. Die grün markierte Zeile ist das beste Ensemble."
    ))
    if ensemble_results_reg is not None and not ensemble_results_reg.empty:
        ens_show = ensemble_results_reg.copy().round(4)
        # Normalize column names
        ens_show.columns = [c.lower() for c in ens_show.columns]
        winner_ens = ens_show.loc[ens_show["rmse"].idxmin(), "ensemble"] \
            if "ensemble" in ens_show.columns and "rmse" in ens_show.columns else None
        ens_show.columns = [c.capitalize() for c in ens_show.columns]
        winner_col = "Ensemble" if "Ensemble" in ens_show.columns else None
        add(_df_to_html(ens_show, cls="t-reg",
                        winner_row=str(winner_ens).capitalize() if winner_ens else None,
                        winner_col=winner_col))
    else:
        add('<p style="color:#888;font-size:9pt;">Ensemble-Ergebnisse nicht übergeben.</p>')

    # ── 4. Finaler Prädiktor ─────────────────────────────────────────────────
    add('<h3>4 · Finaler Prädiktor</h3>')
    add(_info(
        f"Der finale Prädiktor ist <b>{best_predictor_reg_type}</b> "
        f"(gewählt nach Vergleich aller Einzelmodelle und Ensembles auf dem Testset)."
    ))
    add(f"""
<table class="t-reg" style="width:auto;">
  <thead><tr><th>Kennzahl</th><th>Wert</th></tr></thead>
  <tbody>
    <tr><td>R²</td><td><b>{bp_reg.get('r2', float('nan')):.4f}</b></td></tr>
    <tr><td>RMSE</td><td><b>{bp_reg.get('rmse', float('nan')):.4f} µm</b></td></tr>
    <tr><td>MAE</td><td><b>{bp_reg.get('mae', float('nan')):.4f} µm</b></td></tr>
    <tr><td>Modell</td><td>{best_predictor_reg_type}</td></tr>
  </tbody>
</table>
""")

    # ── 5. SHAP & PDP ────────────────────────────────────────────────────────
    add('<h3>5 · Featurewichtigkeit (SHAP) &amp; Partielle Abhängigkeitsplots (PDP)</h3>')
    _shap_who = (
        f"SHAP auf Testset: <b>{best_predictor_reg_type}</b>."
        + (f" Faltbasierte SHAP (±Std.-Abw.): <b>{model_name_reg}</b> (bestes Einzelmodell)."
           if shap_cv_reg is not None else "")
        + (f" PDP: <b>{model_name_reg}</b> über Cross-Validation-Folds."
           if pdp_cv_reg else "")
    )
    add(_info(
        "SHAP (SHapley Additive exPlanations) zeigt den Einfluss jedes Prozessparameters auf die "
        "Vorhersage. Fehlerbalken zeigen die Stabilität über Cross-Validation-Folds. "
        "PDP zeigt den durchschnittlichen Effekt bei Variation eines Parameters.\n\n"
        + _shap_who
    ))
    add(_img(
        _fig_shap(shap_importance_reg,
                  title=f"Featurewichtigkeit (SHAP) — {model_name_reg}",
                  color="#4C72B0", shap_cv=shap_cv_reg),
        caption="Mittlerer |SHAP-Wert| je Feature",
    ))
    # if pdp_cv_reg:
    #     add(_img(
    #         _fig_pdp(pdp_cv_reg, model_name_reg, task="regression"),
    #         caption="PDP mit Unsicherheitsband (±1 Std.-Abw. über Cross-Validation)",
    #     ))
    if pdp_cv_reg:
        for feat in pdp_cv_reg.keys():
            add(_img(
                _fig_pdp_single(feat, pdp_cv_reg[feat], model_name_reg, task="regression")
            ))

    # ── 6. Inverse ML ────────────────────────────────────────────────────────
    add('<h3>6 · Inverse ML — Parametervorschläge</h3>')
    if inverse_ml_solutions is not None and not inverse_ml_solutions.empty:
        target_str = f"{inverse_ml_target:.3f} µm" if inverse_ml_target else "—"
        add(_info(
            f"Bayesianische Suche nach Prozessparametern, die eine Zieldicke von "
            f"<b>{target_str}</b> erzeugen. Modell: <b>{best_predictor_reg_type}</b>."
        ))
        sol_show = inverse_ml_solutions.round(4)
        add(_df_to_html(sol_show, cls="t-reg"))
    else:
        add(_info("Inverse-ML wurde nicht ausgeführt oder Ergebnisse nicht übergeben.", style="dark"))

    pagebreak()

    # ═════════════════════════════════════════════════════════════════════════
    # KLASSIFIKATION
    # ═════════════════════════════════════════════════════════════════════════
    add('<h2 class="clf">Klassifikation — Dünn / Dick (Schwellenwert: '
        f'{median_threshold:.4f} µm)</h2>')
    add(_info(
        "Der binäre Klassifikator entscheidet, ob eine Schicht 'dünn' oder 'dick' ist. "
        "Schwellenwert = Median der Trainingsschichtdicken. "
        "Die Pipeline durchläuft dieselbe HPO- und Ensembling-Prozedur wie die Regression.",
        style="clf",
    ))

    # ── 1. CV ────────────────────────────────────────────────────────────────
    add('<h3>1 · Cross-Validation — Test aller Basisklassifikatoren</h3>')
    if clf_cv is not None and not clf_cv.empty:
        try:
            grp = (clf_cv.groupby("model")[["accuracy", "f1", "auc"]]
                   .agg(["mean", "std"]).round(4))
            grp.columns = ["Accuracy Mittel", "Accuracy Std",
                           "F1 Mittel", "F1 Std", "AUC Mittel", "AUC Std"]
            grp = grp.sort_values("F1 Mittel", ascending=False).reset_index()
            grp.rename(columns={"model": "Modell"}, inplace=True)
            add(_df_to_html(grp, cls="t-clf"))
        except Exception:
            add('<p style="color:#888;font-size:9pt;">CV-Ergebnisse nicht verfügbar.</p>')
    else:
        add('<p style="color:#888;font-size:9pt;">CV-Ergebnisse nicht übergeben.</p>')

    # ── 2. HPO ───────────────────────────────────────────────────────────────
    add('<h3>2 · Ergebnisse nach Hyperparameter-Optimierung (HPO)</h3>')
    add(_info(
        f"Bester F1-Score je Klassifikator aus der HPO. "
        f"Bestes Einzelmodell: <b>{clf_name}</b>.",
        style="clf",
    ))
    if hpo_results_clf:
        hpo_df_clf = (pd.DataFrame(
            [{"Modell": k, "Bester F1 (HPO)": f"{v:.4f}"}
             for k, v in sorted(hpo_results_clf.items(), key=lambda x: x[1], reverse=True)]
        ))
        add(_df_to_html(hpo_df_clf, cls="t-clf", winner_row=clf_name, winner_col="Modell"))
    else:
        if clf_cv is not None and not clf_cv.empty:
            try:
                fb = (clf_cv.groupby("model")[["accuracy", "f1", "auc"]]
                      .mean().round(4).sort_values("f1", ascending=False).reset_index())
                fb.rename(columns={"model": "Modell", "accuracy": "Accuracy",
                                   "f1": "F1", "auc": "AUC"}, inplace=True)
                add(_df_to_html(fb, cls="t-clf", winner_row=clf_name, winner_col="Modell"))
            except Exception:
                pass
        else:
            add('<p style="color:#888;font-size:9pt;">HPO-Ergebnisse nicht übergeben.</p>')

    add(f'<p style="margin-top:10px;font-size:9.5pt;">Testset-Metriken '
        f'<b>{clf_name}</b> (nach HPO, retrained): '
        f'F1&nbsp;=&nbsp;<b>{clf_metrics.get("f1", float("nan")):.4f}</b> &nbsp;|&nbsp; '
        f'AUC&nbsp;=&nbsp;<b>{clf_metrics.get("auc", float("nan")):.4f}</b> &nbsp;|&nbsp; '
        f'Accuracy&nbsp;=&nbsp;<b>{clf_metrics.get("accuracy", float("nan")):.4f}</b></p>')

    # Konfusionsmatrix + ROC nebeneinander
    cm_b64  = _b64(_fig_confusion(cm, clf_name))
    roc_b64 = _b64(_fig_roc(clf_model, X_clf_s, y_test_cls.values, clf_name))
    add(f"""
<table style="border:none; width:100%; margin:10px 0;">
  <tr>
    <td style="border:none; width:50%; text-align:center; vertical-align:top;">
      <img src="data:image/png;base64,{cm_b64}" style="max-width:95%;">
      <p class="caption">Konfusionsmatrix — {clf_name}</p>
    </td>
    <td style="border:none; width:50%; text-align:center; vertical-align:top;">
      <img src="data:image/png;base64,{roc_b64}" style="max-width:95%;">
      <p class="caption">ROC-Kurve — {clf_name}</p>
    </td>
  </tr>
</table>""")

    # ── 3. Ensembling ────────────────────────────────────────────────────────
    add('<h3>3 · Ensembling — Alle Kombinationen</h3>')
    add(_info(
        "Averaging-, Weighted- und Stacking-Ensembles der drei stärksten Klassifikatoren. "
        "Grün markiert: bestes Ensemble (nach F1).",
        style="clf",
    ))
    if ensemble_results_clf is not None and not ensemble_results_clf.empty:
        ens_clf_show = ensemble_results_clf.copy().round(4)
        ens_clf_show.columns = [c.lower() for c in ens_clf_show.columns]
        winner_ens_clf = ens_clf_show.loc[ens_clf_show["f1"].idxmax(), "ensemble"] \
            if "ensemble" in ens_clf_show.columns and "f1" in ens_clf_show.columns else None
        ens_clf_show.columns = [c.capitalize() for c in ens_clf_show.columns]
        add(_df_to_html(ens_clf_show, cls="t-clf",
                        winner_row=str(winner_ens_clf).capitalize() if winner_ens_clf else None,
                        winner_col="Ensemble" if "Ensemble" in ens_clf_show.columns else None))
    else:
        add('<p style="color:#888;font-size:9pt;">Ensemble-Ergebnisse nicht übergeben.</p>')

    # ── 4. Finaler Klassifikator ─────────────────────────────────────────────
    add('<h3>4 · Finaler Klassifikator</h3>')
    add(_info(
        f"Der finale Klassifikator ist <b>{best_predictor_clf_type}</b>.",
        style="clf",
    ))
    add(f"""
<table class="t-clf" style="width:auto;">
  <thead><tr><th>Kennzahl</th><th>Wert</th></tr></thead>
  <tbody>
    <tr><td>F1-Score</td><td><b>{bp_clf.get('f1', float('nan')):.4f}</b></td></tr>
    <tr><td>AUC</td><td><b>{bp_clf.get('auc', float('nan')):.4f}</b></td></tr>
    <tr><td>Accuracy</td><td><b>{bp_clf.get('accuracy', float('nan')):.4f}</b></td></tr>
    <tr><td>Modell</td><td>{best_predictor_clf_type}</td></tr>
  </tbody>
</table>
""")

    # ── 5. SHAP & PDP ────────────────────────────────────────────────────────
    add('<h3>5 · Featureswichtigkeit (SHAP) &amp; PDP</h3>')
    _shap_who_clf = (
        f"SHAP auf Testset: <b>{best_predictor_clf_type}</b>."
        + (f" Faltbasierte SHAP: <b>{clf_name}</b> (bestes Einzelmodell)."
           if shap_cv_clf is not None else "")
        + (f" PDP: <b>{clf_name}</b> über Cross-Validation-Folds."
           if pdp_cv_clf else "")
    )
    add(_info(
        "SHAP zeigt, welche Parameter die Dünn/Dick-Entscheidung am stärksten beeinflussen. "
        "PDP-Werte nahe 1,0 bedeuten: Modell erwartet Dickschicht; nahe 0,0: Dünnschicht.\n\n"
        + _shap_who_clf,
        style="clf",
    ))
    add(_img(
        _fig_shap(shap_importance_clf,
                  title=f"Featurewichtigkeit (SHAP) — {clf_name}",
                  color="#DD8452", shap_cv=shap_cv_clf),
        caption="Mittlerer |SHAP-Wert| je Feature — Klassifikation",
    ))
    # if pdp_cv_clf:
    #     add(_img(
    #         _fig_pdp(pdp_cv_clf, clf_name, task="classification"),
    #         caption="PDP mit Unsicherheitsband (±1 Std.-Abw.) — Klassifikation",
    #     ))
    if pdp_cv_reg:
        for feat in pdp_cv_reg.keys():
            add(_img(
                _fig_pdp_single(feat, pdp_cv_reg[feat], model_name_reg, task="classification")
            ))

    # ── 6. Inverse ML ────────────────────────────────────────────────────────
    add('<h3>6 · Inverse ML — Parametervorschläge</h3>')
    add(_info(
        "Inverse ML läuft auf dem Regressionsmodell — nicht auf dem Klassifikator. "
        "Die Parametervorschläge basieren auf dem finalen Regressorprädiktor.",
        style="clf",
    ))
    if inverse_ml_solutions is not None and not inverse_ml_solutions.empty:
        target_str = f"{inverse_ml_target:.3f} µm" if inverse_ml_target else "—"
        add(f'<p style="font-size:9.5pt;">Zieldicke: <b>{target_str}</b> &nbsp;|&nbsp; '
            f'Modell: <b>{best_predictor_reg_type}</b></p>')
        add(_df_to_html(inverse_ml_solutions.round(4), cls="t-clf"))
    else:
        add(_info("Inverse-ML wurde nicht ausgeführt oder Ergebnisse nicht übergeben.", style="dark"))

    # ── HTML zusammenbauen ────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <title>DigiChrom ML Pipeline Bericht</title>
  <style>{_CSS}</style>
</head>
<body>
{"".join(parts)}
<hr>
<p style="text-align:center; color:#aaa; font-size:8pt; margin-top:16px;">
  DigiChrom ML Pipeline &nbsp;|&nbsp; {jetzt}
</p>
</body>
</html>"""

    success  = _html_to_pdf(html, save_path)
    path_out = save_path if success else save_path.with_suffix(".html")
    print(f"[report] Bericht gespeichert → {path_out}")
    return path_out
