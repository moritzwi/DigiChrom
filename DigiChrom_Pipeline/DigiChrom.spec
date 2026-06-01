# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for DigiChrom — Streamlit desktop app."""

import sys
from pathlib import Path
import streamlit

ST_DIR  = Path(streamlit.__file__).parent
HERE    = Path(SPECPATH)

block_cipher = None

a = Analysis(
    [str(HERE / "launcher.py")],
    pathex=[str(HERE)],
    binaries=[],
    datas=[
        # ── Streamlit assets ───────────────────────────────────────────
        (str(ST_DIR / "static"),  "streamlit/static"),
        (str(ST_DIR / "runtime"), "streamlit/runtime"),
        (str(ST_DIR / "vendor"),  "streamlit/vendor"),
        # ── App source ────────────────────────────────────────────────
        (str(HERE / "app.py"),    "."),
        (str(HERE / "pipeline"),  "pipeline"),
    ],
    hiddenimports=[
        # streamlit internals
        "streamlit",
        "streamlit.web.cli",
        "streamlit.web.server",
        "streamlit.runtime.scriptrunner",
        "streamlit.components.v1",
        # ML libs
        "sklearn",
        "sklearn.utils._cython_blas",
        "sklearn.neighbors._partition_nodes",
        "sklearn.tree._utils",
        "xgboost",
        "catboost",
        "lightgbm",
        "shap",
        "optuna",
        "joblib",
        # data
        "pandas",
        "numpy",
        "matplotlib",
        "matplotlib.backends.backend_agg",
        "openpyxl",
        # stdlib / misc
        "pkg_resources.py2_compat",
        "importlib_metadata",
        "click",
        "altair",
        "pydeck",
        "pyarrow",
        "tzdata",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter", "PyQt5", "PyQt6", "wx",
        "IPython", "jupyter", "notebook",
        "pytest", "sphinx",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DigiChrom",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # no terminal window
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,              # add .icns path here for a custom icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="DigiChrom",
)

app = BUNDLE(
    coll,
    name="DigiChrom.app",
    icon=None,              # add .icns path here
    bundle_identifier="de.tum.digichrom",
    info_plist={
        "NSHighResolutionCapable": True,
        "CFBundleShortVersionString": "1.0.0",
    },
)
