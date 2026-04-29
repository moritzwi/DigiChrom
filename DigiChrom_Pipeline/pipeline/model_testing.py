import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor, GradientBoostingRegressor,
    HistGradientBoostingRegressor, RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeRegressor

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from .config import get_config
from .preprocessing import cross_val_splits, make_preprocessor

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
    print("[model_testing] Warning: xgboost not available; skipping XGBRegressor tests.")

try:
    from catboost import CatBoostRegressor
    _HAS_CAT = True
except Exception:
    _HAS_CAT = False
    print("[model_testing] Warning: catboost not available; skipping CatBoostRegressor tests.")

try:
    from lightgbm import LGBMRegressor
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False
    print("[model_testing] Warning: lightgbm not available; skipping LGBMRegressor tests.")

try:
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier,
        ExtraTreesClassifier, HistGradientBoostingClassifier,
        VotingClassifier,
    )
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.tree import DecisionTreeClassifier
    _HAS_CLF_IMPORTS = True
except Exception:
    _HAS_CLF_IMPORTS = False
    print("[model_testing] Warning: sklearn classifier imports failed; skipping deep learning classifier tests.")

try:
    from pytorch_tabnet.tab_model import TabNetRegressor as _TabNetRegressor

    _TABNET_FIT_PARAMS = {"max_epochs", "patience", "batch_size", "virtual_batch_size"}

    class TabNetRegressorWrapper(RegressorMixin, _TabNetRegressor):
        """Thin wrapper so TabNet fits the same sklearn-style API as the other models."""

        def __init__(self, **kwargs):
            self._fit_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in _TABNET_FIT_PARAMS}
            super().__init__(**kwargs)

        def fit(self, X, y):
            super().fit(
                X if isinstance(X, np.ndarray) else X.values,
                (y if isinstance(y, np.ndarray) else y.values).reshape(-1, 1),
                **self._fit_kwargs,
            )
            return self

        def predict(self, X):
            return super().predict(
                X if isinstance(X, np.ndarray) else X.values
            ).flatten()

        def __sklearn_tags__(self):
            try:
                tags = super().__sklearn_tags__()
            except AttributeError:
                from sklearn.utils._tags import Tags
                tags = Tags()
            tags.estimator_type = "regressor"
            return tags

    _HAS_TABNET = True
except Exception:
    _HAS_TABNET = False
    print("[model_testing] Warning: pytorch-tabnet not available; skipping TabNetRegressor tests.")

try:
    import torch
    import torch.nn as nn
    import torch.utils.data

    def _torch_device() -> "torch.device":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False
    print("[model_testing] Warning: torch not available; skipping all PyTorch-based model tests.")
    # Stub so nested `class _Net(nn.Module)` definitions don't crash at import time.
    # Actual torch operations are guarded by _HAS_TORCH checks inside fit().
    class _StubModule:
        pass
    class _StubNN:
        Module = _StubModule
        Linear = Sequential = ReLU = Dropout = Conv1d = _StubModule
        MultiheadAttention = TransformerEncoder = TransformerEncoderLayer = _StubModule
        Parameter = _StubModule
    nn = _StubNN


class TorchMLP(RegressorMixin, BaseEstimator):
    """Sklearn-compatible wrapper around a PyTorch feed-forward regression network.

    Architecture: Input → FC(hidden_sizes[0]) → ReLU → Dropout → ... → FC(1).
    """

    def __init__(
        self,
        hidden_sizes: list = None,
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 32,
    ) -> None:
        """Initialise hyperparameters.

        Args:
            hidden_sizes: List of hidden-layer widths, e.g. [128, 64].
            dropout: Dropout probability applied after each hidden layer.
            lr: Adam learning rate.
            epochs: Number of full training passes.
            batch_size: Mini-batch size.
        """
        self.hidden_sizes = hidden_sizes or [128, 64]
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_ = None
        self.input_size_ = None
        self.device_ = None

    def _build(self, n_in: int) -> "nn.Sequential":
        """Build the sequential PyTorch model.

        Args:
            n_in: Number of input features.

        Returns:
            Constructed nn.Sequential module.
        """
        layers = []
        prev = n_in
        for h in self.hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y) -> "TorchMLP":
        """Train the MLP on (X, y).

        Args:
            X: Training features as numpy array or DataFrame.
            y: Training targets as numpy array or Series.

        Returns:
            Self, for method chaining.

        Raises:
            ImportError: If torch is not installed.
        """
        if not _HAS_TORCH:
            raise ImportError("torch required for TorchMLP")
        X_t = torch.tensor(X if isinstance(X, np.ndarray) else X.values, dtype=torch.float32)
        y_t = torch.tensor(y if isinstance(y, np.ndarray) else y.values, dtype=torch.float32).unsqueeze(1)
        self.input_size_ = X_t.shape[1]
        self.device_ = _torch_device()
        X_t = X_t.to(self.device_)
        y_t = y_t.to(self.device_)
        torch.manual_seed(get_config().RANDOM_SEED)
        self.model_ = self._build(self.input_size_).to(self.device_)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss_fn(self.model_(xb), yb).backward()
                optimizer.step()
        self.model_.eval()
        return self

    def predict(self, X) -> np.ndarray:
        """Generate predictions for X.

        Args:
            X: Features as numpy array or DataFrame.

        Returns:
            1-D numpy array of predicted values.

        Raises:
            ImportError: If torch is not installed.
        """
        if not _HAS_TORCH:
            raise ImportError("torch required for TorchMLP")
        device = self.device_ or _torch_device()
        X_t = torch.tensor(X if isinstance(X, np.ndarray) else X.values, dtype=torch.float32).to(device)
        with torch.no_grad():
            return self.model_(X_t).squeeze().cpu().numpy()

    def get_params(self, deep: bool = True) -> dict:
        """Return hyperparameter dict (sklearn API).

        Args:
            deep: Ignored; present for sklearn compatibility.

        Returns:
            Dictionary of hyperparameter names to values.
        """
        return {
            "hidden_sizes": self.hidden_sizes,
            "dropout": self.dropout,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
        }

    def set_params(self, **params) -> "TorchMLP":
        """Set hyperparameters from a dict (sklearn API).

        Args:
            **params: Hyperparameter names and values.

        Returns:
            Self.
        """
        for k, v in params.items():
            setattr(self, k, v)
        return self


class TabCNNRegressor(RegressorMixin, BaseEstimator):
    """1D CNN for tabular regression.

    Each input row is treated as a length-n_features sequence with 1 channel.
    Conv1d layers capture local feature interactions; a dense head produces the output.
    """

    def __init__(
        self,
        n_filters: int = 64,
        kernel_size: int = 3,
        n_layers: int = 2,
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 32,
    ) -> None:
        self.n_filters   = n_filters
        self.kernel_size = kernel_size
        self.n_layers    = n_layers
        self.dropout     = dropout
        self.lr          = lr
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.model_      = None
        self.device_     = None

    class _Net(nn.Module):
        def __init__(self, n_feat, n_filters, kernel_size, n_layers, dropout):
            super().__init__()
            ks    = min(kernel_size, n_feat)
            in_ch = 1
            layers = []
            for _ in range(n_layers):
                layers += [nn.Conv1d(in_ch, n_filters, ks, padding=ks // 2), nn.ReLU(), nn.Dropout(dropout)]
                in_ch = n_filters
            self.convs = nn.Sequential(*layers)
            self.pool  = nn.AdaptiveAvgPool1d(1)   # (B, n_filters, F') → (B, n_filters, 1)
            self.fc    = nn.Linear(n_filters, 1)

        def forward(self, x):
            x = x.unsqueeze(1)           # (B, 1, F)
            x = self.convs(x)            # (B, n_filters, F')
            x = self.pool(x).squeeze(-1) # (B, n_filters)
            return self.fc(x)

    def fit(self, X, y) -> "TabCNNRegressor":
        if not _HAS_TORCH:
            raise ImportError("torch required for TabCNNRegressor")
        X_t = torch.tensor(X if isinstance(X, np.ndarray) else X.values, dtype=torch.float32)
        y_t = torch.tensor(y if isinstance(y, np.ndarray) else y.values, dtype=torch.float32).unsqueeze(1)
        self.device_ = _torch_device()
        X_t, y_t     = X_t.to(self.device_), y_t.to(self.device_)
        torch.manual_seed(get_config().RANDOM_SEED)
        self.model_ = self._Net(X_t.shape[1], self.n_filters, self.kernel_size,
                                self.n_layers, self.dropout).to(self.device_)
        opt  = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss = nn.MSELoss()
        ds   = torch.utils.data.TensorDataset(X_t, y_t)
        dl   = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                opt.zero_grad(); loss(self.model_(xb), yb).backward(); opt.step()
        self.model_.eval()
        return self

    def predict(self, X) -> np.ndarray:
        if not _HAS_TORCH:
            raise ImportError("torch required for TabCNNRegressor")
        dev = self.device_ or _torch_device()
        X_t = torch.tensor(X if isinstance(X, np.ndarray) else X.values, dtype=torch.float32).to(dev)
        with torch.no_grad():
            return self.model_(X_t).squeeze().cpu().numpy()

    def get_params(self, deep=True):
        return dict(n_filters=self.n_filters, kernel_size=self.kernel_size,
                    n_layers=self.n_layers, dropout=self.dropout,
                    lr=self.lr, epochs=self.epochs, batch_size=self.batch_size)

    def set_params(self, **p):
        for k, v in p.items(): setattr(self, k, v)
        return self


class FTTransformerRegressor(RegressorMixin, BaseEstimator):
    """Feature Tokenizer + Transformer for tabular regression (Gorishniy et al. 2021).

    Each numerical feature is projected to a d_token-dimensional embedding.
    A learnable [CLS] token is prepended. Transformer encoder processes all tokens.
    Prediction comes from the CLS token via a linear head.
    """

    def __init__(
        self,
        d_token:  int   = 64,
        n_heads:  int   = 8,
        n_layers: int   = 3,
        dropout:  float = 0.1,
        lr:       float = 1e-3,
        epochs:   int   = 200,
        batch_size: int = 32,
    ) -> None:
        self.d_token    = d_token
        self.n_heads    = n_heads
        self.n_layers   = n_layers
        self.dropout    = dropout
        self.lr         = lr
        self.epochs     = epochs
        self.batch_size = batch_size
        self.model_     = None
        self.device_    = None

    class _Net(nn.Module):
        def __init__(self, n_feat, d_token, n_heads, n_layers, dropout):
            super().__init__()
            # Per-feature linear tokenizer
            self.W   = nn.Parameter(torch.empty(n_feat, d_token))
            self.b   = nn.Parameter(torch.zeros(n_feat, d_token))
            nn.init.kaiming_uniform_(self.W, a=5 ** 0.5)
            self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
            nn.init.normal_(self.cls, std=0.02)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_token, nhead=n_heads, dropout=dropout,
                dim_feedforward=d_token * 4, batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            self.head    = nn.Sequential(nn.LayerNorm(d_token), nn.Linear(d_token, 1))

        def forward(self, x):
            tok = x.unsqueeze(-1) * self.W + self.b            # (B, F, d_token)
            cls = self.cls.expand(x.size(0), -1, -1)           # (B, 1, d_token)
            out = self.encoder(torch.cat([cls, tok], dim=1))   # (B, F+1, d_token)
            return self.head(out[:, 0, :])                      # CLS → scalar

    def fit(self, X, y) -> "FTTransformerRegressor":
        if not _HAS_TORCH:
            raise ImportError("torch required for FTTransformerRegressor")
        X_t = torch.tensor(X if isinstance(X, np.ndarray) else X.values, dtype=torch.float32)
        y_t = torch.tensor(y if isinstance(y, np.ndarray) else y.values, dtype=torch.float32).unsqueeze(1)
        self.device_ = _torch_device()
        X_t, y_t     = X_t.to(self.device_), y_t.to(self.device_)
        n_heads_eff  = max(1, min(self.n_heads, self.d_token))
        while self.d_token % n_heads_eff != 0:
            n_heads_eff -= 1
        torch.manual_seed(get_config().RANDOM_SEED)
        self.model_ = self._Net(X_t.shape[1], self.d_token, n_heads_eff,
                                self.n_layers, self.dropout).to(self.device_)
        opt  = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=1e-5)
        loss = nn.MSELoss()
        ds   = torch.utils.data.TensorDataset(X_t, y_t)
        dl   = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                opt.zero_grad(); loss(self.model_(xb), yb).backward(); opt.step()
        self.model_.eval()
        return self

    def predict(self, X) -> np.ndarray:
        if not _HAS_TORCH:
            raise ImportError("torch required for FTTransformerRegressor")
        dev = self.device_ or _torch_device()
        X_t = torch.tensor(X if isinstance(X, np.ndarray) else X.values, dtype=torch.float32).to(dev)
        with torch.no_grad():
            return self.model_(X_t).squeeze().cpu().numpy()

    def get_params(self, deep=True):
        return dict(d_token=self.d_token, n_heads=self.n_heads, n_layers=self.n_layers,
                    dropout=self.dropout, lr=self.lr, epochs=self.epochs, batch_size=self.batch_size)

    def set_params(self, **p):
        for k, v in p.items(): setattr(self, k, v)
        return self


class SAINTRegressor(RegressorMixin, BaseEstimator):
    """Simplified SAINT for tabular regression (Somepalli et al. 2021).

    Implements both self-attention (between features) and intersample attention
    (between rows in a batch). This is the key differentiator from FT-Transformer.
    """

    def __init__(
        self,
        d_token:    int   = 32,
        n_heads:    int   = 4,
        n_layers:   int   = 2,
        dropout:    float = 0.1,
        lr:         float = 1e-3,
        epochs:     int   = 200,
        batch_size: int   = 64,
    ) -> None:
        self.d_token    = d_token
        self.n_heads    = n_heads
        self.n_layers   = n_layers
        self.dropout    = dropout
        self.lr         = lr
        self.epochs     = epochs
        self.batch_size = batch_size
        self.model_     = None
        self.device_    = None

    class _Net(nn.Module):
        def __init__(self, n_feat, d_token, n_heads, n_layers, dropout):
            super().__init__()
            self.W   = nn.Parameter(torch.empty(n_feat, d_token))
            self.b   = nn.Parameter(torch.zeros(n_feat, d_token))
            nn.init.kaiming_uniform_(self.W, a=5 ** 0.5)
            self.feature_attn = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model=d_token, nhead=n_heads, dropout=dropout,
                                           dim_feedforward=d_token * 4,
                                           batch_first=True, norm_first=True)
                for _ in range(n_layers)
            ])
            self.inter_attn = nn.ModuleList([
                nn.MultiheadAttention(embed_dim=d_token, num_heads=n_heads,
                                      dropout=dropout, batch_first=True)
                for _ in range(n_layers)
            ])
            self.head = nn.Sequential(nn.LayerNorm(n_feat * d_token), nn.Linear(n_feat * d_token, 1))

        def forward(self, x):
            # x: (B, F)
            tok = x.unsqueeze(-1) * self.W + self.b       # (B, F, d_token)
            for feat_l, inter_l in zip(self.feature_attn, self.inter_attn):
                tok = feat_l(tok)                          # self-attention across features
                # intersample: transpose to (F, B, d_token), attend across batch
                t   = tok.permute(1, 0, 2)                # (F, B, d_token)
                t, _ = inter_l(t, t, t)
                tok = tok + t.permute(1, 0, 2)            # residual
            return self.head(tok.flatten(1))

    def fit(self, X, y) -> "SAINTRegressor":
        if not _HAS_TORCH:
            raise ImportError("torch required for SAINTRegressor")
        X_t = torch.tensor(X if isinstance(X, np.ndarray) else X.values, dtype=torch.float32)
        y_t = torch.tensor(y if isinstance(y, np.ndarray) else y.values, dtype=torch.float32).unsqueeze(1)
        self.device_ = _torch_device()
        X_t, y_t     = X_t.to(self.device_), y_t.to(self.device_)
        n_heads_eff  = max(1, min(self.n_heads, self.d_token))
        while self.d_token % n_heads_eff != 0:
            n_heads_eff -= 1
        torch.manual_seed(get_config().RANDOM_SEED)
        self.model_ = self._Net(X_t.shape[1], self.d_token, n_heads_eff,
                                self.n_layers, self.dropout).to(self.device_)
        opt  = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=1e-5)
        loss = nn.MSELoss()
        ds   = torch.utils.data.TensorDataset(X_t, y_t)
        dl   = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                opt.zero_grad(); loss(self.model_(xb), yb).backward(); opt.step()
        self.model_.eval()
        return self

    def predict(self, X) -> np.ndarray:
        if not _HAS_TORCH:
            raise ImportError("torch required for SAINTRegressor")
        dev = self.device_ or _torch_device()
        X_t = torch.tensor(X if isinstance(X, np.ndarray) else X.values, dtype=torch.float32).to(dev)
        with torch.no_grad():
            return self.model_(X_t).squeeze().cpu().numpy()

    def get_params(self, deep=True):
        return dict(d_token=self.d_token, n_heads=self.n_heads, n_layers=self.n_layers,
                    dropout=self.dropout, lr=self.lr, epochs=self.epochs, batch_size=self.batch_size)

    def set_params(self, **p):
        for k, v in p.items(): setattr(self, k, v)
        return self


class DeepGBMRegressor(RegressorMixin, BaseEstimator):
    """Deep GBM: gradient-boosted trees + neural residual correction.

    Stage 1: Train a GBM on (X, y) → get leaf-level embeddings.
    Stage 2: Train a small MLP on (X_scaled, GBM_prediction) → corrects residuals.
    Final prediction = GBM_pred + MLP_correction.
    """

    def __init__(
        self,
        n_estimators: int   = 200,
        max_depth:    int   = 4,
        hidden_size:  int   = 64,
        dropout:      float = 0.2,
        lr:           float = 1e-3,
        epochs:       int   = 100,
        batch_size:   int   = 32,
        random_state: int   = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self.hidden_size  = hidden_size
        self.dropout      = dropout
        self.lr           = lr
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.random_state = random_state
        self.gbm_         = None
        self.mlp_         = None
        self.device_      = None

    def fit(self, X, y) -> "DeepGBMRegressor":
        from sklearn.ensemble import HistGradientBoostingRegressor
        Xa = X if isinstance(X, np.ndarray) else X.values
        ya = y if isinstance(y, np.ndarray) else y.values
        # Stage 1: GBM
        self.gbm_ = HistGradientBoostingRegressor(
            max_iter=self.n_estimators, max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self.gbm_.fit(Xa, ya)
        gbm_preds = self.gbm_.predict(Xa)
        residuals = ya - gbm_preds
        # Stage 2: MLP corrects residuals using original features + GBM pred
        X_aug = np.column_stack([Xa, gbm_preds])
        if not _HAS_TORCH:
            raise ImportError("torch required for DeepGBMRegressor MLP stage")
        self.device_ = _torch_device()
        X_t = torch.tensor(X_aug, dtype=torch.float32).to(self.device_)
        y_t = torch.tensor(residuals, dtype=torch.float32).unsqueeze(1).to(self.device_)
        n_in = X_aug.shape[1]
        torch.manual_seed(self.random_state)
        self.mlp_ = nn.Sequential(
            nn.Linear(n_in, self.hidden_size), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2), nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1),
        ).to(self.device_)
        opt  = torch.optim.Adam(self.mlp_.parameters(), lr=self.lr)
        loss = nn.MSELoss()
        ds   = torch.utils.data.TensorDataset(X_t, y_t)
        dl   = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        self.mlp_.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                opt.zero_grad(); loss(self.mlp_(xb), yb).backward(); opt.step()
        self.mlp_.eval()
        return self

    def predict(self, X) -> np.ndarray:
        Xa = X if isinstance(X, np.ndarray) else X.values
        gbm_preds = self.gbm_.predict(Xa)
        X_aug = np.column_stack([Xa, gbm_preds])
        dev   = self.device_ or _torch_device()
        X_t   = torch.tensor(X_aug, dtype=torch.float32).to(dev)
        with torch.no_grad():
            mlp_corr = self.mlp_(X_t).squeeze().cpu().numpy()
        return gbm_preds + mlp_corr

    def get_params(self, deep=True):
        return dict(n_estimators=self.n_estimators, max_depth=self.max_depth,
                    hidden_size=self.hidden_size, dropout=self.dropout,
                    lr=self.lr, epochs=self.epochs, batch_size=self.batch_size,
                    random_state=self.random_state)

    def set_params(self, **p):
        for k, v in p.items(): setattr(self, k, v)
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Device detection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _has_cuda() -> bool:
    """Return True if a CUDA GPU is available via PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _device_kwargs(model_name: str) -> dict:
    """Return device-specific kwargs to inject into XGBoost / CatBoost / LightGBM.

    On CUDA machines this enables GPU training.  On Apple MPS / CPU machines
    these libraries do not support the accelerator, so an empty dict is returned.

    The mapping is:
        xgboost  → device='cuda'
        catboost → task_type='GPU'
        lightgbm → device='gpu'

    Args:
        model_name: Pipeline model key.

    Returns:
        Dict of extra kwargs to merge into the model constructor call.
    """
    if not _has_cuda():
        return {}
    _gpu_map = {
        "xgboost":  {"device": "cuda"},
        "catboost": {"task_type": "GPU"},
        "lightgbm": {"device": "gpu"},
    }
    return _gpu_map.get(model_name, {})


# ─────────────────────────────────────────────────────────────────────────────
# Multioutput wrapper
# ─────────────────────────────────────────────────────────────────────────────

def wrap_multioutput(model) -> MultiOutputRegressor:
    """Wrap a single-output regressor for multi-target regression.

    Args:
        model: Any fitted or unfitted sklearn-compatible regressor.

    Returns:
        MultiOutputRegressor wrapping the given model.
    """
    return MultiOutputRegressor(model)


# ─────────────────────────────────────────────────────────────────────────────
# CV CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_cv_csvs(df: pd.DataFrame, prefix: str = "reg") -> None:
    """Save per-fold results and summary to REPORTS_DIR."""
    get_config().REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_path    = get_config().REPORTS_DIR / f"cv_fold_results_{prefix}.csv"
    summary_path = get_config().REPORTS_DIR / f"cv_summary_metrics_{prefix}.csv"
    df.to_csv(fold_path, index=False)
    metric_cols = [c for c in df.columns if c not in ("model", "fold")]
    summary = df.groupby("model")[metric_cols].agg(["mean", "std"]).round(6)
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(summary_path)
    print(f"[model_testing] CV results → {fold_path}, {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Deep Learning Classifiers (parallel to regression DL models)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from pytorch_tabnet.tab_model import TabNetClassifier as _TabNetClassifier
    _TABNET_FIT_PARAMS_CLF = {"max_epochs", "patience", "batch_size", "virtual_batch_size"}

    class TabNetClassifierWrapper(_TabNetClassifier):
        """Thin sklearn-compatible wrapper around pytorch-tabnet TabNetClassifier."""

        def __init__(self, **kwargs):
            self._fit_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in _TABNET_FIT_PARAMS_CLF}
            super().__init__(**kwargs)

        def fit(self, X, y):
            super().fit(
                X if isinstance(X, np.ndarray) else X.values,
                (y if isinstance(y, np.ndarray) else y.values).astype(int).reshape(-1),
                **self._fit_kwargs,
            )
            return self

        def predict(self, X):
            return super().predict(X if isinstance(X, np.ndarray) else X.values)

        def predict_proba(self, X):
            return super().predict_proba(X if isinstance(X, np.ndarray) else X.values)

        def __sklearn_tags__(self):
            try:
                tags = super().__sklearn_tags__()
            except AttributeError:
                from sklearn.utils._tags import Tags
                tags = Tags()
            tags.estimator_type = "classifier"
            return tags

    _HAS_TABNET_CLF = True
except Exception:
    _HAS_TABNET_CLF = False


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """PyTorch MLP for binary classification (BCEWithLogitsLoss). Mirrors TorchMLP."""

    def __init__(self, hidden_sizes=None, dropout=0.2, lr=1e-3, epochs=200, batch_size=32):
        self.hidden_sizes = hidden_sizes or [128, 64]
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_ = None
        self.device_ = None

    def _build(self, n_in):
        layers, prev = [], n_in
        for h in self.hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        if not _HAS_TORCH:
            raise ImportError("torch required for TorchMLPClassifier")
        X_t = torch.tensor(X if isinstance(X, np.ndarray) else X.values, dtype=torch.float32)
        y_t = torch.tensor(y if isinstance(y, np.ndarray) else y.values, dtype=torch.float32).unsqueeze(1)
        self.device_ = _torch_device()
        X_t, y_t = X_t.to(self.device_), y_t.to(self.device_)
        torch.manual_seed(get_config().RANDOM_SEED)
        self.model_ = self._build(X_t.shape[1]).to(self.device_)
        opt     = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        ds = torch.utils.data.TensorDataset(X_t, y_t)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                opt.zero_grad(); loss_fn(self.model_(xb), yb).backward(); opt.step()
        self.model_.eval()
        return self

    def _logits(self, X):
        if not _HAS_TORCH:
            raise ImportError("torch required for TorchMLPClassifier")
        dev = self.device_ or _torch_device()
        X_t = torch.tensor(X if isinstance(X, np.ndarray) else X.values, dtype=torch.float32).to(dev)
        with torch.no_grad():
            return self.model_(X_t).squeeze().cpu()

    def predict_proba(self, X):
        p = torch.sigmoid(self._logits(X)).numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (torch.sigmoid(self._logits(X)).numpy() >= 0.5).astype(int)

    def get_params(self, deep=True):
        return dict(hidden_sizes=self.hidden_sizes, dropout=self.dropout,
                    lr=self.lr, epochs=self.epochs, batch_size=self.batch_size)

    def set_params(self, **p):
        for k, v in p.items(): setattr(self, k, v)
        return self


class TabCNNClassifier(ClassifierMixin, TabCNNRegressor):
    """1D CNN for binary classification. Shares _Net architecture with TabCNNRegressor."""

    def fit(self, X, y):
        if not _HAS_TORCH:
            raise ImportError("torch required for TabCNNClassifier")
        X_t = torch.tensor(X if isinstance(X, np.ndarray) else X.values, dtype=torch.float32)
        y_t = torch.tensor(y if isinstance(y, np.ndarray) else y.values, dtype=torch.float32).unsqueeze(1)
        self.device_ = _torch_device()
        X_t, y_t = X_t.to(self.device_), y_t.to(self.device_)
        torch.manual_seed(get_config().RANDOM_SEED)
        self.model_ = self._Net(X_t.shape[1], self.n_filters, self.kernel_size,
                                self.n_layers, self.dropout).to(self.device_)
        opt     = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        ds = torch.utils.data.TensorDataset(X_t, y_t)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                opt.zero_grad(); loss_fn(self.model_(xb), yb).backward(); opt.step()
        self.model_.eval()
        return self

    def predict_proba(self, X):
        if not _HAS_TORCH:
            raise ImportError("torch required for TabCNNClassifier")
        dev = self.device_ or _torch_device()
        X_t = torch.tensor(X if isinstance(X, np.ndarray) else X.values, dtype=torch.float32).to(dev)
        with torch.no_grad():
            p = torch.sigmoid(self.model_(X_t).squeeze()).cpu().numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class FTTransformerClassifier(ClassifierMixin, FTTransformerRegressor):
    """Feature Tokenizer + Transformer for binary classification."""

    def fit(self, X, y):
        if not _HAS_TORCH:
            raise ImportError("torch required for FTTransformerClassifier")
        X_t = torch.tensor(X if isinstance(X, np.ndarray) else X.values, dtype=torch.float32)
        y_t = torch.tensor(y if isinstance(y, np.ndarray) else y.values, dtype=torch.float32).unsqueeze(1)
        self.device_ = _torch_device()
        X_t, y_t = X_t.to(self.device_), y_t.to(self.device_)
        n_heads_eff = max(1, min(self.n_heads, self.d_token))
        while self.d_token % n_heads_eff != 0:
            n_heads_eff -= 1
        torch.manual_seed(get_config().RANDOM_SEED)
        self.model_ = self._Net(X_t.shape[1], self.d_token, n_heads_eff,
                                self.n_layers, self.dropout).to(self.device_)
        opt     = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=1e-5)
        loss_fn = nn.BCEWithLogitsLoss()
        ds = torch.utils.data.TensorDataset(X_t, y_t)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                opt.zero_grad(); loss_fn(self.model_(xb), yb).backward(); opt.step()
        self.model_.eval()
        return self

    def predict_proba(self, X):
        if not _HAS_TORCH:
            raise ImportError("torch required for FTTransformerClassifier")
        dev = self.device_ or _torch_device()
        X_t = torch.tensor(X if isinstance(X, np.ndarray) else X.values, dtype=torch.float32).to(dev)
        with torch.no_grad():
            p = torch.sigmoid(self.model_(X_t).squeeze()).cpu().numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class SAINTClassifier(ClassifierMixin, SAINTRegressor):
    """SAINT for binary classification."""

    def fit(self, X, y):
        if not _HAS_TORCH:
            raise ImportError("torch required for SAINTClassifier")
        X_t = torch.tensor(X if isinstance(X, np.ndarray) else X.values, dtype=torch.float32)
        y_t = torch.tensor(y if isinstance(y, np.ndarray) else y.values, dtype=torch.float32).unsqueeze(1)
        self.device_ = _torch_device()
        X_t, y_t = X_t.to(self.device_), y_t.to(self.device_)
        n_heads_eff = max(1, min(self.n_heads, self.d_token))
        while self.d_token % n_heads_eff != 0:
            n_heads_eff -= 1
        torch.manual_seed(get_config().RANDOM_SEED)
        self.model_ = self._Net(X_t.shape[1], self.d_token, n_heads_eff,
                                self.n_layers, self.dropout).to(self.device_)
        opt     = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=1e-5)
        loss_fn = nn.BCEWithLogitsLoss()
        ds = torch.utils.data.TensorDataset(X_t, y_t)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                opt.zero_grad(); loss_fn(self.model_(xb), yb).backward(); opt.step()
        self.model_.eval()
        return self

    def predict_proba(self, X):
        if not _HAS_TORCH:
            raise ImportError("torch required for SAINTClassifier")
        dev = self.device_ or _torch_device()
        X_t = torch.tensor(X if isinstance(X, np.ndarray) else X.values, dtype=torch.float32).to(dev)
        with torch.no_grad():
            p = torch.sigmoid(self.model_(X_t).squeeze()).cpu().numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class DeepGBMClassifier(ClassifierMixin, BaseEstimator):
    """DeepGBM for binary classification: GBM stage 1 + MLP stage 2 (BCEWithLogitsLoss)."""

    def __init__(self, n_estimators=200, max_depth=4, hidden_size=64, dropout=0.2,
                 lr=1e-3, epochs=100, batch_size=32, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self.hidden_size  = hidden_size
        self.dropout      = dropout
        self.lr           = lr
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.random_state = random_state
        self.gbm_         = None
        self.mlp_         = None
        self.device_      = None

    def fit(self, X, y):
        from sklearn.ensemble import HistGradientBoostingClassifier
        Xa = X if isinstance(X, np.ndarray) else X.values
        ya = y if isinstance(y, np.ndarray) else y.values
        self.gbm_ = HistGradientBoostingClassifier(
            max_iter=self.n_estimators, max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self.gbm_.fit(Xa, ya)
        gbm_proba = self.gbm_.predict_proba(Xa)[:, 1]
        X_aug     = np.column_stack([Xa, gbm_proba])
        if not _HAS_TORCH:
            raise ImportError("torch required for DeepGBMClassifier MLP stage")
        self.device_ = _torch_device()
        X_t = torch.tensor(X_aug, dtype=torch.float32).to(self.device_)
        y_t = torch.tensor(ya, dtype=torch.float32).unsqueeze(1).to(self.device_)
        n_in = X_aug.shape[1]
        torch.manual_seed(self.random_state)
        self.mlp_ = nn.Sequential(
            nn.Linear(n_in, self.hidden_size), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2), nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1),
        ).to(self.device_)
        opt     = torch.optim.Adam(self.mlp_.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        ds = torch.utils.data.TensorDataset(X_t, y_t)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        self.mlp_.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                opt.zero_grad(); loss_fn(self.mlp_(xb), yb).backward(); opt.step()
        self.mlp_.eval()
        return self

    def predict_proba(self, X):
        Xa        = X if isinstance(X, np.ndarray) else X.values
        gbm_proba = self.gbm_.predict_proba(Xa)[:, 1]
        X_aug     = np.column_stack([Xa, gbm_proba])
        dev       = self.device_ or _torch_device()
        X_t       = torch.tensor(X_aug, dtype=torch.float32).to(dev)
        with torch.no_grad():
            p = torch.sigmoid(self.mlp_(X_t).squeeze()).cpu().numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_params(self, deep=True):
        return dict(n_estimators=self.n_estimators, max_depth=self.max_depth,
                    hidden_size=self.hidden_size, dropout=self.dropout,
                    lr=self.lr, epochs=self.epochs, batch_size=self.batch_size,
                    random_state=self.random_state)

    def set_params(self, **p):
        for k, v in p.items(): setattr(self, k, v)
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Classification helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_classifiers() -> dict:
    """Return all available classifiers for binary classification.

    Keys match config.CLASSIFIER_DEFAULTS so that train_final_classifier()
    and tune_classifiers() can look them up without key translation.
    Includes deep learning classifiers when torch is available.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                                   ExtraTreesClassifier, HistGradientBoostingClassifier)
    from sklearn.tree import DecisionTreeClassifier
    d = get_config().CLASSIFIER_DEFAULTS
    clfs = {
        "logistic":               LogisticRegression(**d["logistic"]),
        "cart":                   DecisionTreeClassifier(**{k: v for k, v in d["cart"].items()
                                                           if k != "criterion"}),
        "c50":                    DecisionTreeClassifier(**d["c50"]),
        "gradient_boosting":      GradientBoostingClassifier(**d["gradient_boosting"]),
        "random_forest":          RandomForestClassifier(**d["random_forest"]),
        "extra_trees":            ExtraTreesClassifier(**d["extra_trees"]),
        "hist_gradient_boosting": HistGradientBoostingClassifier(**d["hist_gradient_boosting"]),
    }
    if _HAS_XGB:
        from xgboost import XGBClassifier
        clfs["xgboost"] = XGBClassifier(**d["xgboost"], eval_metric="logloss",
                                        **_device_kwargs("xgboost"))
    if _HAS_CAT:
        from catboost import CatBoostClassifier
        clfs["catboost"] = CatBoostClassifier(**d["catboost"],
                                              **_device_kwargs("catboost"))
    if _HAS_LGB:
        from lightgbm import LGBMClassifier
        clfs["lightgbm"] = LGBMClassifier(**d["lightgbm"],
                                          **_device_kwargs("lightgbm"))
    if _HAS_TORCH:
        clfs["mlp"]            = TorchMLPClassifier(**d.get("mlp", {}))
        clfs["tab_cnn"]        = TabCNNClassifier(**d.get("tab_cnn", {}))
        clfs["ft_transformer"] = FTTransformerClassifier(**d.get("ft_transformer", {}))
        clfs["saint"]          = SAINTClassifier(**d.get("saint", {}))
        clfs["deep_gbm"]       = DeepGBMClassifier(**d.get("deep_gbm", {}))
    if _HAS_TABNET_CLF:
        clfs["tabnet"] = TabNetClassifierWrapper(**d.get("tabnet", {}))
    return clfs


def _clf_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
    """Compute accuracy, F1 and (if probabilities given) AUC."""
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1":       float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        try:
            m["auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            m["auc"] = float("nan")
    return m


def evaluate_classifiers(
    X: pd.DataFrame,
    y: pd.Series,
    models: dict = None,
    cv_splits: list = None,
    per_fold_hpo: bool = False,
    n_hpo_trials: int = 50,
) -> pd.DataFrame:
    """Cross-validate all classifiers. y must be binary (0/1).

    Args:
        X: Feature DataFrame.
        y: Binary label Series (0/1).
        models: Dict of name → classifier. Defaults to get_classifiers().
        cv_splits: Pre-computed splits. Defaults to cross_val_splits(X, y).
        per_fold_hpo: If True, run a separate Optuna study per fold before
            fitting. Only applies to models supported by
            hp_tuning._clf_objective_factory.
        n_hpo_trials: Optuna trials per fold when per_fold_hpo=True.

    Returns:
        DataFrame with columns ['model', 'fold', 'accuracy', 'f1', 'auc'].
    """
    if models is None:
        models = get_classifiers()
    if cv_splits is None:
        cv_splits = cross_val_splits(X, y)
    pre = make_preprocessor()
    records = []
    X_arr, y_arr = X.values, y.values
    for name, model in models.items():
        print(f"[model_testing] Evaluating classifier {name}"
              f"{' (per-fold HPO)' if per_fold_hpo else ''}...")

        fold_params_list: list[dict] = [{}] * len(cv_splits)
        if per_fold_hpo:
            try:
                from hp_tuning import tune_per_fold_classifier
                fold_params_list = tune_per_fold_classifier(name, X, y, cv_splits,
                                                            n_trials=n_hpo_trials)
            except (ValueError, ImportError) as e:
                print(f"[model_testing] Skipping per-fold HPO for {name}: {e}")

        for fold_idx, (tr, val) in enumerate(cv_splits):
            Xtr_s = pre.fit_transform(X_arr[tr])
            Xv_s  = pre.transform(X_arr[val])

            fold_model = model
            if per_fold_hpo and fold_params_list[fold_idx]:
                try:
                    from sklearn.base import clone
                    fold_model = clone(model)
                    fold_model.set_params(**fold_params_list[fold_idx])
                except Exception:
                    fold_model = model

            fold_model.fit(Xtr_s, y_arr[tr])
            preds = fold_model.predict(Xv_s)
            probs = fold_model.predict_proba(Xv_s)[:, 1] if hasattr(fold_model, "predict_proba") else None
            records.append({"model": name, "fold": fold_idx + 1,
                            **_clf_metrics(y_arr[val], preds, probs)})
    df_results = pd.DataFrame(records)
    _save_cv_csvs(df_results, prefix="clf")
    return df_results


def get_models() -> dict:
    """Instantiate all available candidate models with default hyperparameters.

    Models are included only if their optional dependencies are importable
    (xgboost, catboost, torch, lightgbm). Linear, Ridge, RandomForest,
    ExtraTrees, ElasticNet and HistGradientBoosting are always present.

    Returns:
        Ordered dictionary mapping model name to an unfitted estimator instance.
    """
    models = {
        "linear": LinearRegression(),
        "ridge": Ridge(**get_config().MODEL_DEFAULTS["ridge"]),
        "elasticnet": ElasticNet(**get_config().MODEL_DEFAULTS["elasticnet"]),
        "cart": DecisionTreeRegressor(**get_config().MODEL_DEFAULTS["cart"]),
        "gradient_boosting": GradientBoostingRegressor(**get_config().MODEL_DEFAULTS["gradient_boosting"]),
        "random_forest": RandomForestRegressor(**get_config().MODEL_DEFAULTS["random_forest"]),
        "extra_trees": ExtraTreesRegressor(**get_config().MODEL_DEFAULTS["extra_trees"]),
        "hist_gradient_boosting": HistGradientBoostingRegressor(**get_config().MODEL_DEFAULTS["hist_gradient_boosting"]),
    }
    if _HAS_XGB:
        models["xgboost"] = XGBRegressor(**get_config().MODEL_DEFAULTS["xgboost"],
                                         **_device_kwargs("xgboost"))
    if _HAS_CAT:
        models["catboost"] = CatBoostRegressor(**get_config().MODEL_DEFAULTS["catboost"],
                                               **_device_kwargs("catboost"))
    if _HAS_LGB:
        models["lightgbm"] = LGBMRegressor(**get_config().MODEL_DEFAULTS["lightgbm"],
                                           **_device_kwargs("lightgbm"))
    if _HAS_TORCH:
        models["mlp"]            = TorchMLP(**get_config().MODEL_DEFAULTS["mlp"])
        models["tab_cnn"]        = TabCNNRegressor(**get_config().MODEL_DEFAULTS.get("tab_cnn", {}))
        models["ft_transformer"] = FTTransformerRegressor(**get_config().MODEL_DEFAULTS.get("ft_transformer", {}))
        models["saint"]          = SAINTRegressor(**get_config().MODEL_DEFAULTS.get("saint", {}))
        models["deep_gbm"]       = DeepGBMRegressor(**get_config().MODEL_DEFAULTS.get("deep_gbm", {}))
    if _HAS_TABNET:
        models["tabnet"] = TabNetRegressorWrapper(**get_config().MODEL_DEFAULTS["tabnet"])
    return models


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute R², RMSE and MAE between true and predicted values.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted target values.

    Returns:
        Dictionary with keys 'r2', 'rmse', 'mae'.
    """
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def evaluate_all(
    X: pd.DataFrame,
    y: pd.Series,
    models: dict = None,
    cv_splits: list = None,
    preprocessor: Pipeline = None,
    per_fold_hpo: bool = False,
    n_hpo_trials: int = 50,
) -> pd.DataFrame:
    """Cross-validate all models and collect per-fold metrics.

    A fresh copy of preprocessor is fit on each training fold so that no
    data leakage occurs.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        models: Dict of name → estimator. Defaults to get_models().
        cv_splits: List of (train_idx, val_idx) tuples. Defaults to
            cross_val_splits(X, y).
        preprocessor: Unfitted sklearn Pipeline. Defaults to make_preprocessor().
        per_fold_hpo: If True, run a separate Optuna HPO study for every fold
            before fitting that fold.  Only applies to models supported by
            hp_tuning._objective_factory (tree/linear models).  Deep-learning
            models are skipped silently.
        n_hpo_trials: Number of Optuna trials per fold when per_fold_hpo=True.

    Returns:
        DataFrame with columns ['model', 'fold', 'r2', 'rmse', 'mae'],
        one row per (model, fold) combination.
    """
    if models is None:
        models = get_models()
    if cv_splits is None:
        cv_splits = cross_val_splits(X, y)
    if preprocessor is None:
        preprocessor = make_preprocessor()

    records = []
    X_arr = X.values
    y_arr = y.values

    for name, model in models.items():
        print(f"[model_testing] Evaluating {name} ({len(cv_splits)} folds"
              f"{', per-fold HPO' if per_fold_hpo else ''})...")

        # Pre-compute per-fold best params when requested
        fold_params_list: list[dict] = [{}] * len(cv_splits)
        if per_fold_hpo:
            try:
                from hp_tuning import tune_per_fold
                fold_params_list = tune_per_fold(name, X, y, cv_splits,
                                                 n_trials=n_hpo_trials)
            except (ValueError, ImportError) as e:
                print(f"[model_testing] Skipping per-fold HPO for {name}: {e}")

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
            y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

            X_tr_scaled  = preprocessor.fit_transform(X_tr)
            X_val_scaled = preprocessor.transform(X_val)

            # Re-instantiate model with fold-specific params when available
            fold_model = model
            if per_fold_hpo and fold_params_list[fold_idx]:
                try:
                    from sklearn.base import clone
                    fold_model = clone(model)
                    fold_model.set_params(**fold_params_list[fold_idx])
                except Exception:
                    fold_model = model

            fold_model.fit(X_tr_scaled, y_tr)
            preds = fold_model.predict(X_val_scaled)
            records.append({"model": name, "fold": fold_idx + 1,
                            **_metrics(y_val, preds)})

    df_results = pd.DataFrame(records)
    _save_cv_csvs(df_results, prefix="reg")
    return df_results


def comparison_plot(results_df: pd.DataFrame, save_path=None) -> matplotlib.figure.Figure:
    """Create and save boxplot comparison of CV metrics across all models.

    Args:
        results_df: DataFrame as returned by evaluate_all.
        save_path: Output path for the PDF. Defaults to config.FIGURES_DIR /
            'model_comparison.pdf'.

    Returns:
        Matplotlib Figure object.
    """
    save_path = save_path or get_config().FIGURES_DIR / "model_comparison.pdf"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, ["r2", "rmse", "mae"]):
        order = results_df.groupby("model")[metric].median().sort_values(
            ascending=(metric != "r2")
        ).index
        data = [results_df[results_df["model"] == m][metric].values for m in order]
        ax.boxplot(data, labels=order, patch_artist=True)
        ax.set_title(metric.upper())
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Model Comparison (CV)", fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[model_testing] Saved comparison plot → {save_path}")
    return fig


if __name__ == "__main__":
    from .data_loading import load_clean
    from .preprocessing import split_xy
    df, _ = load_clean()
    X, y = split_xy(df)
    results = evaluate_all(X, y)
    print(results.groupby("model")[["r2", "rmse", "mae"]].mean().round(3))
