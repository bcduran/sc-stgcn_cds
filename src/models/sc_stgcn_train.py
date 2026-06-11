"""
================================================================================
SC-STGCN: Supply-Chain Aware Spatio-Temporal Graph Convolutional Network
for CDS Spread Forecasting
================================================================================

Reference paper target: Q1 Finance / Machine Learning journal
  e.g. Journal of Financial Economics, Review of Financial Studies,
       Journal of Financial and Quantitative Analysis, Management Science

Research Question:
  Does supply-chain network topology carry incremental information for
  predicting short-term CDS spread movements beyond purely time-series signals?

Experimental Design:
  - Target modes  : DELTA (Δbp, 1-week-ahead) AND LEVEL (bp, 1-week-ahead)
  - Universe      : Top-50 S&P 500 firms by market-cap with CDS data
  - Panel         : Weekly CDS spreads (5Y tenor)
  - Data split    : 80 / 10 / 10  (train / val / test), strictly chronological
  - Look-back     : n_his = 7 weeks
  - Baselines     : Naive, AR(1), ARMA(1,1), LSTM, GRU, XGBoost, Vanilla-STGCN
  - Proposed      : SC-STGCN  (heterogeneous dual-adjacency + temporal attention)

Novel Contributions of SC-STGCN vs. prior work:
  1. DUAL ADJACENCY  — upstream (supplier→firm) and downstream (firm→customer)
     graph convolutions are computed separately with independent weight matrices.
     This distinguishes cost-push (upstream) from demand-pull (downstream) credit
     risk propagation channels, which is an economically motivated design.

  2. TEMPORAL SELF-ATTENTION — instead of a fixed-kernel causal conv, each
     time step is scored via scaled dot-product attention, allowing the model
     to adaptively weight recency vs. regime-relevant distant lags.
     Attention weights are logged for interpretability analysis.

  3. GATED FUSION — dual-stream graph outputs are combined via a learnable
     sigmoid gate G = σ(W_g [h_sup ‖ h_cus]) rather than a plain sum or
     concatenation, letting the model modulate upstream vs. downstream signals
     conditionally on the current spread level.

  4. RESIDUAL SKIP + LAYER-NORM — each spatio-temporal block wraps a residual
     connection normalised by LayerNorm, following best practices in deep
     sequence models and improving gradient flow over long panels.

  5. DIRECTIONAL BACKTEST — Long-Short CDS protection strategy:
     long protection on predicted wideners, short protection on predicted
     tighteners.  Both weekly PnL and annualised Sharpe are reported.

Usage:
  python sc_stgcn_thesis.py [--mode DELTA|LEVEL|BOTH] [--epochs 300]

Required files (relative to script directory):
  outputs_cds/data/top50/ve1.csv        — weekly CDS spread panel (T × N)
  outputs_cds/data/top50/adj.npz        — combined (binary) adjacency (N × N)
  outputs_cds/data/top50/adj_sup.npz    — upstream adjacency  (N × N)  [optional]
  outputs_cds/data/top50/adj_cus.npz    — downstream adjacency(N × N)  [optional]

All outputs written to:  outputs_sc_stgcn/
================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — STANDARD LIBRARY & THIRD-PARTY IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import json
import math
import random
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Optional: XGBoost
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[INFO] xgboost not installed — XGBoost baseline will be skipped.")

# Optional: statsmodels for ARMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("[INFO] statsmodels not installed — ARMA(1,1) baseline will be skipped.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — GLOBAL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
class CFG:
    """
    Central configuration object.  All hyperparameters live here so that
    a single change propagates everywhere — important for reproducibility.
    """
    # ── Reproducibility ───────────────────────────────────────────────────────
    SEED: int = 42

    # ── Data ──────────────────────────────────────────────────────────────────
    N_HIS: int   = 12      # look-back window (weeks)
    N_PRED: int  = 1       # forecast horizon  (weeks)
    SPLIT: Tuple = (0.80, 0.10, 0.10)   # train / val / test

    # ── Backtest ──────────────────────────────────────────────────────────────
    TOP_Q:      float = 0.20     # top/bottom quantile for L/S strategy
    DV01_PER_BP: float = 100.0   # $ PnL per 1 bp move per unit notional
    INIT_CASH:  float = 100_000.0

    # ── SC-STGCN (proposed model) ─────────────────────────────────────────────
    SCSTGCN_GCN_H:    int   = 128    # GCN hidden dim per stream
    SCSTGCN_ATT_H:    int   = 128    # Temporal attention dim
    SCSTGCN_ATT_HEADS: int  = 4      # Multi-head attention heads
    SCSTGCN_FF_H:     int   = 256    # Feed-forward hidden in att block
    SCSTGCN_DROPOUT:  float = 0.20
    SCSTGCN_EPOCHS:   int   = 500
    SCSTGCN_BATCH:    int   = 64
    SCSTGCN_LR:       float = 5e-4
    SCSTGCN_WD:       float = 1e-4
    SCSTGCN_PATIENCE: int   = 60
    SCSTGCN_CLIP:     float = 5.0

    # ── Vanilla STGCN (ablation baseline) ────────────────────────────────────
    VSTGCN_GCN_H:   int   = 64
    VSTGCN_TEMP_H:  int   = 64
    VSTGCN_EPOCHS:  int   = 300
    VSTGCN_BATCH:   int   = 64
    VSTGCN_LR:      float = 5e-4
    VSTGCN_WD:      float = 1e-4
    VSTGCN_PATIENCE: int  = 50

    # ── LSTM ──────────────────────────────────────────────────────────────────
    LSTM_H:       int   = 128
    LSTM_LAYERS:  int   = 2
    LSTM_DROP:    float = 0.20
    LSTM_EPOCHS:  int   = 300
    LSTM_BATCH:   int   = 64
    LSTM_LR:      float = 5e-4
    LSTM_WD:      float = 1e-4
    LSTM_PATIENCE: int  = 40

    # ── GRU ───────────────────────────────────────────────────────────────────
    GRU_H:       int   = 128
    GRU_LAYERS:  int   = 2
    GRU_DROP:    float = 0.20
    GRU_EPOCHS:  int   = 300
    GRU_BATCH:   int   = 64
    GRU_LR:      float = 1e-3
    GRU_WD:      float = 1e-4
    GRU_PATIENCE: int  = 30

    # ── XGBoost ───────────────────────────────────────────────────────────────
    XGB_PARAMS: dict = dict(
        n_estimators=600, max_depth=5, learning_rate=0.03,
        subsample=0.80, colsample_bytree=0.80,
        min_child_weight=3, gamma=0.1,
        reg_alpha=0.05, reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42, tree_method="hist",
        n_jobs=-1, verbosity=0,
    )

    # ── I/O ───────────────────────────────────────────────────────────────────
    DEVICE: str = "auto"   # "cpu" | "cuda" | "auto"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — REPRODUCIBILITY & DEVICE
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int = CFG.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if CFG.DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(CFG.DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — PATHS
# ─────────────────────────────────────────────────────────────────────────────
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import (
    TOP50_WEEKLY_CSV  as VEL_FILE,
    ADJ_NPZ           as ADJ_FILE,
    ADJ_SUP_NPZ       as ADJ_SUP_FILE,
    ADJ_CUS_NPZ       as ADJ_CUS_FILE,
    PRED_DIR, FIG_DIR, METRICS_DIR, CKPT_DIR,
    ModelCFG, BacktestCFG, make_dirs,
)
make_dirs()
OUT_DIR = FIG_DIR.parent  # outputs/


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_panel() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load ve1.csv  →  level panel (T, N) and delta panel (T-1, N).

    Returns
    -------
    level  : np.ndarray  (T, N)   weekly CDS spread levels in bp
    delta  : np.ndarray  (T-1,N)  first-difference Δspread in bp
    cols   : list[str]   ticker names
    """
    if not VEL_FILE.exists():
        raise FileNotFoundError(f"Panel file not found: {VEL_FILE}")

    df      = pd.read_csv(VEL_FILE)
    df_num  = df.select_dtypes(include=[np.number])
    df_num  = df_num.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    level   = np.nan_to_num(df_num.to_numpy(dtype=np.float32), nan=0.0)
    delta   = np.diff(level, axis=0)     # (T-1, N)
    cols    = df_num.columns.tolist()

    print(f"[data] Level panel : {level.shape}  |  Delta panel: {delta.shape}")
    print(f"[data] Tickers ({len(cols)}): {cols}")
    return level, delta, cols


def _sym_normalise(A: np.ndarray) -> np.ndarray:
    """Symmetric renormalisation: D^{-1/2}(A+I)D^{-1/2}."""
    N   = A.shape[0]
    A_h = A + np.eye(N, dtype=np.float32)
    d   = A_h.sum(1)
    di  = np.diag(1.0 / np.sqrt(np.maximum(d, 1e-8)))
    return di @ A_h @ di


def load_adjacency(N_max: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Load adjacency matrices and return a dict with keys:
      'combined'   — combined (binary or weighted) adjacency
      'upstream'   — supplier→firm  (if adj_sup.npz exists, else = combined)
      'downstream' — firm→customer  (if adj_cus.npz exists, else = combined.T)

    Both upstream and downstream are symmetrically normalised.
    """
    def _load(path: Path, N: int) -> np.ndarray:
        raw = sp.load_npz(path).toarray().astype(np.float32)
        if N is not None:
            raw = raw[:N, :N]
        return raw

    if not ADJ_FILE.exists():
        raise FileNotFoundError(f"Adjacency file not found: {ADJ_FILE}")

    raw_comb = sp.load_npz(ADJ_FILE).toarray().astype(np.float32)
    N = raw_comb.shape[0] if N_max is None else min(raw_comb.shape[0], N_max)
    raw_comb = raw_comb[:N, :N]

    # Upstream: supplier→firm  (columns = firms receiving from rows = suppliers)
    raw_sup = _load(ADJ_SUP_FILE, N) if ADJ_SUP_FILE.exists() else raw_comb
    # Downstream: firm→customer
    raw_cus = _load(ADJ_CUS_FILE, N) if ADJ_CUS_FILE.exists() else raw_comb.T

    adjs = {
        "combined"   : _sym_normalise(raw_comb),
        "upstream"   : _sym_normalise(raw_sup),
        "downstream" : _sym_normalise(raw_cus),
    }

    for k, v in adjs.items():
        print(f"[adj] {k:12s}: shape={v.shape}  "
              f"density={float((v > 0).mean()):.3f}")
    return adjs


def make_windows(
    data: np.ndarray,
    n_his: int = CFG.N_HIS,
    n_pred: int = CFG.N_PRED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window transform.

    Parameters
    ----------
    data : (T, N)

    Returns
    -------
    X : (B, n_his, N)
    y : (B, N)
    """
    T, N = data.shape
    Xs, ys = [], []
    for t in range(n_his, T - n_pred + 1):
        Xs.append(data[t - n_his : t])
        ys.append(data[t + n_pred - 1])
    return np.stack(Xs).astype(np.float32), np.stack(ys).astype(np.float32)


def chronological_split(
    panel: np.ndarray,
    split: Tuple[float, float, float] = CFG.SPLIT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """80 / 10 / 10 chronological split — NO shuffling, NO leakage."""
    T  = panel.shape[0]
    n1 = int(split[0] * T)
    n2 = int((split[0] + split[1]) * T)
    return panel[:n1], panel[n1:n2], panel[n2:]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — METRICS & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(
    y_true_s: np.ndarray,
    y_pred_s: np.ndarray,
    scaler: StandardScaler,
) -> Dict:
    """
    Compute MSE, RMSE, MAE, R² and MAPE in both scaled and original (bp) space.

    Parameters
    ----------
    y_true_s, y_pred_s : (B, N) in *scaled* units
    scaler             : fitted StandardScaler for inverse transform

    Returns
    -------
    dict with keys: mse_s, rmse_s, mae_s, r2_s, mse_o, rmse_o, mae_o, r2_o,
                    mape_o, y_true_orig, y_pred_orig
    """
    # ── scaled space ──────────────────────────────────────────────────────────
    yt_f, yp_f = y_true_s.flatten(), y_pred_s.flatten()
    mse_s  = float(mean_squared_error(yt_f, yp_f))
    rmse_s = float(np.sqrt(mse_s))
    mae_s  = float(np.mean(np.abs(yt_f - yp_f)))
    r2_s   = float(r2_score(yt_f, yp_f))

    # ── original (bp) space ───────────────────────────────────────────────────
    B      = y_true_s.shape[0]
    yt_o   = scaler.inverse_transform(y_true_s.reshape(B, -1))
    yp_o   = scaler.inverse_transform(y_pred_s.reshape(B, -1))

    yt_of, yp_of = yt_o.flatten(), yp_o.flatten()
    mse_o  = float(mean_squared_error(yt_of, yp_of))
    rmse_o = float(np.sqrt(mse_o))
    mae_o  = float(np.mean(np.abs(yt_of - yp_of)))
    r2_o   = float(r2_score(yt_of, yp_of))

    # MAPE — guard against zero actuals
    mask   = np.abs(yt_of) > 0.5          # only compute where actual > 0.5 bp
    mape_o = float(np.mean(np.abs((yt_of[mask] - yp_of[mask]) / yt_of[mask]))) \
             if mask.any() else np.nan

    return dict(
        mse_s=mse_s, rmse_s=rmse_s, mae_s=mae_s, r2_s=r2_s,
        mse_o=mse_o, rmse_o=rmse_o, mae_o=mae_o, r2_o=r2_o, mape_o=mape_o,
        y_true_orig=yt_o, y_pred_orig=yp_o,
    )


def save_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cols: List[str],
    tag: str,
) -> None:
    pd.DataFrame(y_true, columns=cols).to_csv(PRED_DIR / f"{tag}_y_true.csv",  index=False)
    pd.DataFrame(y_pred, columns=cols).to_csv(PRED_DIR / f"{tag}_y_pred.csv",  index=False)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def backtest_ls_cds(
    y_true_bp: np.ndarray,
    y_pred_bp: np.ndarray,
    top_q: float  = CFG.TOP_Q,
    dv01:  float  = CFG.DV01_PER_BP,
    init:  float  = CFG.INIT_CASH,
) -> Dict:
    """
    Long-Short CDS protection strategy.

    Economic logic
    --------------
    - CDS *protection buyer* profits when spreads WIDEN (credit quality ↓).
    - We rank firms by *predicted* spread change each week:
        Long  protection on top-q% predicted wideners   (highest ΔS)
        Short protection on top-q% predicted tighteners (lowest ΔS)
    - Cash PnL:
        pnl_t = DV01 × Σ_i w_{t,i} × ΔS_{t,i}^{realised}
      where w_{t,i} ∈ {+1/K, -1/K, 0} and K = round(N × top_q).

    Parameters
    ----------
    y_true_bp, y_pred_bp : (B, N)  realised / predicted Δspread in bp

    Returns
    -------
    dict with pnl, equity, weekly stats, and risk-adjusted metrics
    """
    B, N = y_true_bp.shape
    K    = max(1, int(round(N * top_q)))

    weekly_pnl  = np.zeros(B)
    long_avg    = np.zeros(B)
    short_avg   = np.zeros(B)

    for t in range(B):
        pred_t  = y_pred_bp[t]
        real_t  = y_true_bp[t]
        order   = np.argsort(pred_t)     # ascending: tighteners → wideners
        long_i  = order[-K:]             # predicted wideners   → long protection
        short_i = order[:K]              # predicted tighteners → short protection

        w       = np.zeros(N)
        w[long_i]  = +1.0 / K
        w[short_i] = -1.0 / K

        weekly_pnl[t] = dv01 * (w * real_t).sum()
        long_avg[t]   = real_t[long_i].mean()
        short_avg[t]  = real_t[short_i].mean()

    equity    = init + np.cumsum(weekly_pnl)
    mean_w    = float(weekly_pnl.mean())
    std_w     = float(weekly_pnl.std(ddof=1)) if B > 1 else 0.0
    hit       = float((weekly_pnl > 0).mean())
    sharpe    = float(np.sqrt(52) * mean_w / std_w) if std_w > 0 else np.nan
    total_ret = float((equity[-1] - init) / init)

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    dd   = (equity - peak) / peak
    max_dd = float(dd.min())

    # Calmar ratio (annualised return / max drawdown)
    ann_ret = total_ret * (52.0 / B)
    calmar  = float(ann_ret / abs(max_dd)) if max_dd < 0 else np.nan

    return dict(
        pnl=weekly_pnl, equity=equity,
        long_avg=long_avg, short_avg=short_avg,
        mean=mean_w, std=std_w, hit=hit,
        sharpe=sharpe, total_ret=total_ret,
        max_dd=max_dd, calmar=calmar,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — STATISTICAL BASELINES
# ─────────────────────────────────────────────────────────────────────────────

# ── 7.1  Naive (zero-delta) ────────────────────────────────────────────────
def run_naive(
    tr_s: np.ndarray,
    te_s: np.ndarray,
    scaler: StandardScaler,
    cols: List[str],
    tag: str,
) -> Dict:
    """Predict Δspread = 0 (random walk without drift)."""
    _, y_te = make_windows(te_s)
    y_pr    = np.zeros_like(y_te)
    m       = compute_metrics(y_te, y_pr, scaler)
    save_predictions(m["y_true_orig"], m["y_pred_orig"], cols, tag)
    return m


# ── 7.2  AR(1) per firm  ───────────────────────────────────────────────────
def run_ar1(
    tr_s: np.ndarray,
    te_s: np.ndarray,
    scaler: StandardScaler,
    cols: List[str],
    tag: str,
) -> Dict:
    """
    AR(1): Δs_{t,i} = a_i + b_i Δs_{t-1,i}
    OLS closed form fitted on training set; prediction uses last window step.
    """
    y_tr  = tr_s[1:]
    x_tr  = tr_s[:-1]
    xm, ym = x_tr.mean(0), y_tr.mean(0)
    b     = ((x_tr - xm) * (y_tr - ym)).mean(0) / \
            (((x_tr - xm) ** 2).mean(0) + 1e-10)
    a     = ym - b * xm

    X_te, y_te = make_windows(te_s)
    x_last     = X_te[:, -1, :]
    y_pr       = a + b * x_last

    m = compute_metrics(y_te, y_pr, scaler)
    save_predictions(m["y_true_orig"], m["y_pred_orig"], cols, tag)
    return m


# ── 7.3  ARMA(1,1) per firm  ──────────────────────────────────────────────
def run_arma(
    tr_raw: np.ndarray,
    te_raw: np.ndarray,
    scaler: StandardScaler,
    cols: List[str],
    tag: str,
) -> Dict:
    """
    ARMA(1,1) per firm on UNSCALED delta series.
    Uses statsmodels.  Falls back to AR(1) if optimisation fails.
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels required for ARMA baseline.")

    N       = tr_raw.shape[1]
    T_te    = te_raw.shape[0]
    n_his   = CFG.N_HIS

    # Scale
    sc2     = StandardScaler()
    tr_s    = sc2.fit_transform(tr_raw)
    te_s    = sc2.transform(te_raw)

    X_te, y_te = make_windows(te_s)
    B          = y_te.shape[0]
    y_pr       = np.zeros_like(y_te)

    # Fit per firm on training scaled series
    fitted_params = []
    for i in range(N):
        try:
            res = ARIMA(tr_s[:, i], order=(1, 0, 1)).fit(method_kwargs={"warn_convergence": False})
            fitted_params.append(res)
        except Exception:
            fitted_params.append(None)

    # One-step-ahead on test windows
    for t in range(B):
        for i in range(N):
            if fitted_params[i] is None:
                y_pr[t, i] = 0.0
            else:
                try:
                    fc = fitted_params[i].forecast(steps=1)
                    y_pr[t, i] = float(fc.iloc[0] if hasattr(fc, "iloc") else fc[0])
                except Exception:
                    y_pr[t, i] = 0.0

    # Metrics in sc2 space then convert back to original
    yt_o = sc2.inverse_transform(y_te)
    yp_o = sc2.inverse_transform(y_pr)

    # Re-compute in scaler space (for fair comparison use the shared scaler)
    yt_s = scaler.transform(yt_o)
    yp_s = scaler.transform(yp_o)

    m = compute_metrics(yt_s, yp_s, scaler)
    save_predictions(m["y_true_orig"], m["y_pred_orig"], cols, tag)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — NEURAL NETWORK BASELINES (LSTM / GRU)
# ─────────────────────────────────────────────────────────────────────────────
class RecurrentForecaster(nn.Module):
    """
    Shared RNN (LSTM or GRU) across N firms.

    Each firm's time series is treated as an independent sequence so the
    RNN processes B×N sub-sequences of length T in one batched forward pass.
    This is equivalent to but more efficient than running N separate models.

    Input  : (B, T, N)
    Output : (B, N)
    """
    def __init__(
        self,
        cell: str,
        N: int,
        hidden: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.N   = N
        rnn_cls  = nn.LSTM if cell.upper() == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            input_size=1, hidden_size=hidden, num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, T, N = X.shape
        # reshape: (B*N, T, 1)
        x   = X.permute(0, 2, 1).reshape(B * N, T, 1)
        out, _ = self.rnn(x)              # (B*N, T, H)
        last    = self.drop(out[:, -1])   # (B*N, H)
        pred    = self.fc(last).squeeze(-1)  # (B*N,)
        return pred.reshape(B, N)


def _train_rnn(
    cell: str,
    tr_s: np.ndarray,
    va_s: np.ndarray,
    te_s: np.ndarray,
    scaler: StandardScaler,
    cols: List[str],
    tag: str,
    device: torch.device,
    hidden: int,
    n_layers: int,
    dropout: float,
    epochs: int,
    batch: int,
    lr: float,
    wd: float,
    patience: int,
) -> Dict:
    N     = tr_s.shape[1]
    X_tr, y_tr = make_windows(tr_s)
    X_va, y_va = make_windows(va_s)
    X_te, y_te = make_windows(te_s)

    def tot(a: np.ndarray) -> torch.Tensor:
        return torch.tensor(a, dtype=torch.float32, device=device)

    dl = DataLoader(
        TensorDataset(tot(X_tr), tot(y_tr)),
        batch_size=batch, shuffle=True, drop_last=False,
    )

    model   = RecurrentForecaster(cell, N, hidden, n_layers, dropout).to(device)
    opt     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    loss_fn = nn.HuberLoss(delta=1.0)   # more robust than MSE to CDS spikes

    best_val = np.inf
    no_impr  = 0
    ckpt     = CKPT_DIR / f"{tag}_best.pt"

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(tot(X_va)), tot(y_va)).item()

        if vl < best_val - 1e-7:
            best_val = vl; no_impr = 0
            torch.save(model.state_dict(), ckpt)
        else:
            no_impr += 1

        if ep % 100 == 0:
            print(f"  [{cell}] ep {ep:4d}  val={vl:.6f}  best={best_val:.6f}")

        if no_impr >= patience:
            print(f"  [{cell}] early-stop ep={ep}  best_val={best_val:.6f}")
            break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    with torch.no_grad():
        y_pr = model(tot(X_te)).cpu().numpy()

    m = compute_metrics(y_te, y_pr, scaler)
    save_predictions(m["y_true_orig"], m["y_pred_orig"], cols, tag)
    return m


def run_lstm(tr_s, va_s, te_s, scaler, cols, tag, device) -> Dict:
    return _train_rnn(
        "LSTM", tr_s, va_s, te_s, scaler, cols, tag, device,
        CFG.LSTM_H, CFG.LSTM_LAYERS, CFG.LSTM_DROP,
        CFG.LSTM_EPOCHS, CFG.LSTM_BATCH, CFG.LSTM_LR,
        CFG.LSTM_WD, CFG.LSTM_PATIENCE,
    )


def run_gru(tr_s, va_s, te_s, scaler, cols, tag, device) -> Dict:
    return _train_rnn(
        "GRU", tr_s, va_s, te_s, scaler, cols, tag, device,
        CFG.GRU_H, CFG.GRU_LAYERS, CFG.GRU_DROP,
        CFG.GRU_EPOCHS, CFG.GRU_BATCH, CFG.GRU_LR,
        CFG.GRU_WD, CFG.GRU_PATIENCE,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — XGBOOST BASELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_xgboost(
    tr_s: np.ndarray,
    te_s: np.ndarray,
    scaler: StandardScaler,
    cols: List[str],
    tag: str,
) -> Dict:
    """
    Per-firm XGBRegressor.
    Feature vector = flattened look-back window [Δs_{t-6}, …, Δs_{t-1}] (n_his values).
    One model per firm, fitted independently on training windows.
    """
    X_tr, y_tr = make_windows(tr_s)   # (B_tr, n_his, N)
    X_te, y_te = make_windows(te_s)

    N    = tr_s.shape[1]
    y_pr = np.zeros_like(y_te)

    for i in range(N):
        xtr_i = X_tr[:, :, i]    # (B_tr, n_his)
        xte_i = X_te[:, :, i]
        m_i   = XGBRegressor(**CFG.XGB_PARAMS)
        m_i.fit(xtr_i, y_tr[:, i])
        y_pr[:, i] = m_i.predict(xte_i)
        if (i + 1) % 10 == 0 or i == N - 1:
            print(f"  [XGB] firm {i+1:2d}/{N}")

    m = compute_metrics(y_te, y_pr, scaler)
    save_predictions(m["y_true_orig"], m["y_pred_orig"], cols, tag)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — VANILLA STGCN  (ablation baseline)
# ─────────────────────────────────────────────────────────────────────────────
class VanillaGCN(nn.Module):
    """Single symmetric GCN layer: θ · D^{-1/2}(A+I)D^{-1/2} · X."""
    def __init__(self, in_ch: int, out_ch: int, A_norm: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("A", torch.tensor(A_norm, dtype=torch.float32))
        self.theta = nn.Linear(in_ch, out_ch, bias=False)
        self.bn    = nn.BatchNorm1d(out_ch)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, N, F)
        AX = torch.einsum("ij,bjf->bif", self.A, X)
        out = self.theta(AX)                 # (B, N, out_ch)
        B, N, C = out.shape
        return F.relu(self.bn(out.reshape(B * N, C)).reshape(B, N, C))


class VanillaSTGCN(nn.Module):
    """
    Single-stream STGCN used as ablation baseline.
    Architecture: GCN → temporal Conv1D → FC readout
    """
    def __init__(
        self, N: int, n_his: int, gcn_h: int, temp_h: int, A_norm: np.ndarray
    ) -> None:
        super().__init__()
        self.n_his = n_his
        self.gcn   = VanillaGCN(1, gcn_h, A_norm)
        self.tconv = nn.Sequential(
            nn.Conv2d(gcn_h, temp_h, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(temp_h),
            nn.ReLU(),
        )
        self.fc = nn.Linear(temp_h, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, T, N)
        B, T, N = X.shape
        g = self.gcn(X.reshape(B * T, N, 1))      # (B*T, N, gcn_h)
        g = g.reshape(B, T, N, -1).permute(0,3,2,1)   # (B, gcn_h, N, T)
        h = self.tconv(g)[:, :, :, -1]            # (B, temp_h, N)
        return self.fc(h.permute(0, 2, 1)).squeeze(-1)  # (B, N)


def run_vanilla_stgcn(
    tr_s: np.ndarray, va_s: np.ndarray, te_s: np.ndarray,
    scaler: StandardScaler, cols: List[str],
    A_norm: np.ndarray, tag: str, device: torch.device,
) -> Dict:
    N     = tr_s.shape[1]
    X_tr, y_tr = make_windows(tr_s)
    X_va, y_va = make_windows(va_s)
    X_te, y_te = make_windows(te_s)

    def tot(a): return torch.tensor(a, dtype=torch.float32, device=device)

    dl = DataLoader(TensorDataset(tot(X_tr), tot(y_tr)),
                    batch_size=CFG.VSTGCN_BATCH, shuffle=True)

    model   = VanillaSTGCN(N, CFG.N_HIS, CFG.VSTGCN_GCN_H, CFG.VSTGCN_TEMP_H, A_norm).to(device)
    opt     = torch.optim.AdamW(model.parameters(), lr=CFG.VSTGCN_LR, weight_decay=CFG.VSTGCN_WD)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG.VSTGCN_EPOCHS, eta_min=CFG.VSTGCN_LR*0.01)
    loss_fn = nn.HuberLoss(delta=1.0)

    best_val = np.inf; no_impr = 0
    ckpt = CKPT_DIR / f"{tag}_best.pt"

    for ep in range(1, CFG.VSTGCN_EPOCHS + 1):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), CFG.VSTGCN_LR * 1000)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(tot(X_va)), tot(y_va)).item()

        if vl < best_val - 1e-7:
            best_val = vl; no_impr = 0
            torch.save(model.state_dict(), ckpt)
        else:
            no_impr += 1

        if ep % 100 == 0:
            print(f"  [V-STGCN] ep {ep:4d}  val={vl:.6f}  best={best_val:.6f}")

        if no_impr >= CFG.VSTGCN_PATIENCE:
            print(f"  [V-STGCN] early-stop ep={ep}")
            break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    with torch.no_grad():
        y_pr = model(tot(X_te)).cpu().numpy()

    m = compute_metrics(y_te, y_pr, scaler)
    save_predictions(m["y_true_orig"], m["y_pred_orig"], cols, tag)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — SC-STGCN (PROPOSED MODEL)
# ─────────────────────────────────────────────────────────────────────────────

class DualStreamGCN(nn.Module):
    """
    Heterogeneous Dual-Stream GCN.

    Two independent GCN weight matrices process the same node features
    through the upstream (supplier→firm) and downstream (firm→customer)
    adjacency matrices respectively.  The outputs are fused via a sigmoid
    gate that learns how much upstream vs. downstream signal to retain.

    Equations
    ---------
    h_up  = ReLU(BN(θ_up  · A_up  · X))
    h_dn  = ReLU(BN(θ_dn  · A_dn  · X))
    g     = σ(W_g [h_up ‖ h_dn])          ← scalar gate per node per feature
    h_out = g ⊙ h_up  +  (1-g) ⊙ h_dn
    """

    def __init__(
        self,
        in_ch:   int,
        out_ch:  int,
        A_up:    np.ndarray,
        A_dn:    np.ndarray,
    ) -> None:
        super().__init__()
        self.register_buffer("A_up", torch.tensor(A_up, dtype=torch.float32))
        self.register_buffer("A_dn", torch.tensor(A_dn, dtype=torch.float32))

        self.theta_up = nn.Linear(in_ch, out_ch, bias=False)
        self.theta_dn = nn.Linear(in_ch, out_ch, bias=False)
        self.bn_up    = nn.BatchNorm1d(out_ch)
        self.bn_dn    = nn.BatchNorm1d(out_ch)

        # Gated fusion
        self.gate = nn.Linear(2 * out_ch, out_ch, bias=True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, N, F)
        B, N, _ = X.shape

        def _gcn(A, theta, bn):
            AX  = torch.einsum("ij,bjf->bif", A, X)    # (B, N, F)
            out = theta(AX)                              # (B, N, out_ch)
            out = F.relu(bn(out.reshape(B * N, -1)).reshape(B, N, -1))
            return out

        h_up = _gcn(self.A_up, self.theta_up, self.bn_up)  # (B, N, out_ch)
        h_dn = _gcn(self.A_dn, self.theta_dn, self.bn_dn)

        # Gated fusion
        g    = torch.sigmoid(self.gate(torch.cat([h_up, h_dn], dim=-1)))  # (B, N, out_ch)
        return g * h_up + (1.0 - g) * h_dn                                 # (B, N, out_ch)


class TemporalAttentionBlock(nn.Module):
    """
    Multi-head scaled dot-product self-attention over the time axis,
    followed by a position-wise feed-forward network (Transformer-style).

    Attention is applied independently for each node — i.e. each node's
    temporal sequence of length T attends over itself.

    Input  : (B, T, N, d_model)
    Output : (B, T, N, d_model)   (same shape as input, residual)

    Attention weights (B, heads, T, T) are stored in self.attn_weights
    for post-hoc interpretability analysis.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads

        # Multi-head attention projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.ln1     = nn.LayerNorm(d_model)
        self.ln2     = nn.LayerNorm(d_model)
        self.drop    = nn.Dropout(dropout)
        self.attn_weights: Optional[torch.Tensor] = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, T, N, d_model)
        B, T, N, D = X.shape

        # Flatten node dim into batch for attention: (B*N, T, D)
        Xf = X.permute(0, 2, 1, 3).reshape(B * N, T, D)

        # ── Multi-head self-attention ──────────────────────────────────────
        Q = self.W_q(Xf).reshape(B * N, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(Xf).reshape(B * N, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(Xf).reshape(B * N, T, self.n_heads, self.d_k).transpose(1, 2)
        # Q, K, V: (B*N, heads, T, d_k)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B*N, h, T, T)
        attn   = F.softmax(scores, dim=-1)
        self.attn_weights = attn.detach()                           # store for analysis

        ctx = (attn @ V).transpose(1, 2).reshape(B * N, T, D)     # (B*N, T, D)
        ctx = self.W_o(ctx)

        # Residual + LayerNorm
        Xf  = self.ln1(Xf + self.drop(ctx))

        # ── Feed-forward ──────────────────────────────────────────────────
        Xf  = self.ln2(Xf + self.drop(self.ff(Xf)))

        # Reshape back: (B, T, N, D)
        return Xf.reshape(B, N, T, D).permute(0, 2, 1, 3)


class SCSTGCN(nn.Module):
    """
    Supply-Chain Aware Spatio-Temporal Graph Convolutional Network.

    Pipeline
    --------
    1. Input projection  : (B, T, N, 1) → (B, T, N, d_gcn)
       via DualStreamGCN applied independently at each time step.

    2. Temporal Attention: (B, T, N, d_gcn) → (B, T, N, d_att)
       via TemporalAttentionBlock (multi-head self-attention over T).

    3. Readout           : aggregate T dim → (B, N, d_att),
       then linear projection to (B, N).

    Parameters
    ----------
    N       : number of graph nodes
    n_his   : look-back window length
    gcn_h   : hidden dimension of dual GCN
    att_h   : hidden dimension of temporal attention (d_model)
    n_heads : number of attention heads
    d_ff    : feed-forward dim inside attention block
    dropout : dropout probability throughout
    A_up    : (N, N) upstream adjacency (normalised)
    A_dn    : (N, N) downstream adjacency (normalised)
    """

    def __init__(
        self,
        N:        int,
        n_his:    int,
        gcn_h:    int,
        att_h:    int,
        n_heads:  int,
        d_ff:     int,
        dropout:  float,
        A_up:     np.ndarray,
        A_dn:     np.ndarray,
    ) -> None:
        super().__init__()
        self.N     = N
        self.n_his = n_his

        # ── Dual-stream GCN ───────────────────────────────────────────────
        self.dual_gcn = DualStreamGCN(1, gcn_h, A_up, A_dn)

        # ── Projection to attention dim ───────────────────────────────────
        self.proj = nn.Sequential(
            nn.Linear(gcn_h, att_h),
            nn.LayerNorm(att_h),
            nn.GELU(),
        )

        # ── Two stacked temporal attention blocks ─────────────────────────
        self.att1 = TemporalAttentionBlock(att_h, n_heads, d_ff, dropout)
        self.att2 = TemporalAttentionBlock(att_h, n_heads, d_ff, dropout)

        # ── Readout ───────────────────────────────────────────────────────
        self.readout = nn.Sequential(
            nn.Linear(att_h, att_h // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(att_h // 2, 1),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B, T, N)  in scaled delta space
        Returns: (B, N)  predicted next-step delta
        """
        B, T, N = X.shape

        # 1) Apply dual-GCN at every time step: reshape → (B*T, N, 1)
        h = self.dual_gcn(X.reshape(B * T, N, 1))        # (B*T, N, gcn_h)
        h = h.reshape(B, T, N, -1)                        # (B, T, N, gcn_h)

        # 2) Project to attention dim
        h = self.proj(h)                                   # (B, T, N, att_h)

        # 3) Temporal attention (two stacked blocks)
        h = self.att1(h)                                   # (B, T, N, att_h)
        h = self.att2(h)

        # 4) Readout: take the LAST time step as the "context vector"
        h_last = h[:, -1, :, :]                            # (B, N, att_h)
        out    = self.readout(h_last).squeeze(-1)          # (B, N)
        return out

    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """Return stored attention weights from both blocks for analysis."""
        return {
            "att_block1": self.att1.attn_weights,
            "att_block2": self.att2.attn_weights,
        }


def run_sc_stgcn(
    tr_s:   np.ndarray,
    va_s:   np.ndarray,
    te_s:   np.ndarray,
    scaler: StandardScaler,
    cols:   List[str],
    adjs:   Dict[str, np.ndarray],
    tag:    str,
    device: torch.device,
) -> Dict:
    """Train and evaluate SC-STGCN."""
    N = tr_s.shape[1]
    X_tr, y_tr = make_windows(tr_s)
    X_va, y_va = make_windows(va_s)
    X_te, y_te = make_windows(te_s)

    def tot(a): return torch.tensor(a, dtype=torch.float32, device=device)

    dl = DataLoader(
        TensorDataset(tot(X_tr), tot(y_tr)),
        batch_size=CFG.SCSTGCN_BATCH, shuffle=True, drop_last=False,
    )

    model = SCSTGCN(
        N=N, n_his=CFG.N_HIS,
        gcn_h=CFG.SCSTGCN_GCN_H,
        att_h=CFG.SCSTGCN_ATT_H,
        n_heads=CFG.SCSTGCN_ATT_HEADS,
        d_ff=CFG.SCSTGCN_FF_H,
        dropout=CFG.SCSTGCN_DROPOUT,
        A_up=adjs["upstream"],
        A_dn=adjs["downstream"],
    ).to(device)

    opt     = torch.optim.AdamW(model.parameters(),
                                lr=CFG.SCSTGCN_LR,
                                weight_decay=CFG.SCSTGCN_WD)
    sched   = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=50, T_mult=2, eta_min=CFG.SCSTGCN_LR * 0.01
    )
    loss_fn = nn.HuberLoss(delta=1.0)

    best_val = np.inf; best_ep = 0; no_impr = 0
    ckpt = CKPT_DIR / f"{tag}_best.pt"

    for ep in range(1, CFG.SCSTGCN_EPOCHS + 1):
        model.train()
        tr_loss, nb = 0.0, 0
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CFG.SCSTGCN_CLIP)
            opt.step()
            tr_loss += loss.item(); nb += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(tot(X_va)), tot(y_va)).item()

        if vl < best_val - 1e-7:
            best_val = vl; best_ep = ep; no_impr = 0
            torch.save(model.state_dict(), ckpt)
        else:
            no_impr += 1

        if ep % 100 == 0:
            print(f"  [SC-STGCN] ep {ep:4d}  train={tr_loss/nb:.6f}  "
                  f"val={vl:.6f}  best={best_val:.6f} (ep {best_ep})")

        if no_impr >= CFG.SCSTGCN_PATIENCE:
            print(f"  [SC-STGCN] early-stop ep={ep}  best_val={best_val:.6f}")
            break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    with torch.no_grad():
        y_pr = model(tot(X_te)).cpu().numpy()

    # Save attention weights for interpretability
    with torch.no_grad():
        _ = model(tot(X_te[:8]))   # small forward pass to populate weights
        aw = model.get_attention_weights()
        np.save(PRED_DIR / f"{tag}_attn_w1.npy",
                aw["att_block1"].cpu().numpy() if aw["att_block1"] is not None else np.array([]))

    m = compute_metrics(y_te, y_pr, scaler)
    save_predictions(m["y_true_orig"], m["y_pred_orig"], cols, tag)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — RESULTS AGGREGATION & TABLES
# ─────────────────────────────────────────────────────────────────────────────
def build_results_table(
    res: Dict[str, Tuple[Dict, Dict]],
    mode: str,
) -> pd.DataFrame:
    rows = []
    for name, (m, bt) in res.items():
        rows.append({
            "Model"        : name,
            "MSE (scaled)" : round(m["mse_s"],  6),
            "RMSE (scaled)": round(m["rmse_s"], 6),
            "MAE (scaled)" : round(m["mae_s"],  6),
            "R² (scaled)"  : round(m["r2_s"],   4),
            "MSE (bp²)"    : round(m["mse_o"],  4),
            "RMSE (bp)"    : round(m["rmse_o"], 4),
            "MAE (bp)"     : round(m["mae_o"],  4),
            "R² (bp)"      : round(m["r2_o"],   4),
            "MAPE (%)"     : round(m["mape_o"] * 100, 2) if not np.isnan(m["mape_o"]) else "—",
            "BT Mean PnL$" : round(bt["mean"],      2) if bt else "—",
            "BT Sharpe"    : round(bt["sharpe"],     3) if bt else "—",
            "Hit Ratio"    : round(bt["hit"],        3) if bt else "—",
            "Max Drawdown" : round(bt["max_dd"],     3) if bt else "—",
            "Calmar Ratio" : round(bt["calmar"],     3) if bt and not np.isnan(bt["calmar"]) else "—",
            "Total Return%": round(bt["total_ret"]*100, 2) if bt else "—",
        })
    df = pd.DataFrame(rows)

    # Save CSV + LaTeX
    prefix = f"results_{mode.lower()}"
    df.to_csv(METRICS_DIR / f"{prefix}.csv", index=False)

    # LaTeX — publication-quality table
    latex_cols = ["Model", "RMSE (bp)", "MAE (bp)", "R² (bp)",
                  "BT Sharpe", "Hit Ratio", "Max Drawdown"]
    latex_df   = df[latex_cols].copy()

    latex_str = latex_df.to_latex(
        index=False, escape=False, float_format="%.4f",
        caption=f"Out-of-sample forecasting and backtest performance — {mode} mode. "
                "RMSE and MAE in basis points. Sharpe annualised (×√52).",
        label=f"tab:results_{mode.lower()}",
        column_format="l" + "r" * (len(latex_cols) - 1),
    )
    (METRICS_DIR / f"{prefix}.tex").write_text(latex_str, encoding="utf-8")
    return df


def print_summary(df: pd.DataFrame, mode: str) -> None:
    divider = "─" * 90
    print(f"\n{divider}")
    print(f"  RESULTS SUMMARY  ·  Mode: {mode}")
    print(divider)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    if mode == "DELTA":
        print(df[["Model","RMSE (bp)","MAE (bp)","R² (bp)","BT Sharpe",
                  "Hit Ratio","Max Drawdown","Total Return%"]].to_string(index=False))
    else:  # LEVEL — backtest N/A
        print(df[["Model","RMSE (bp)","MAE (bp)","R² (bp)"]].to_string(index=False))
    print(divider)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 — PUBLICATION-QUALITY FIGURES
# ─────────────────────────────────────────────────────────────────────────────
COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]
STYLE = {
    "font.family"     : "serif",
    "font.serif"      : ["Times New Roman", "DejaVu Serif"],
    "axes.labelsize"  : 11,
    "axes.titlesize"  : 12,
    "legend.fontsize" : 9,
    "xtick.labelsize" : 9,
    "ytick.labelsize" : 9,
    "figure.dpi"      : 150,
}


def fig_equity_curves(
    res: Dict[str, Tuple[Dict, Dict]],
    mode: str,
) -> None:
    """
    Figure 1 — Equity curves for all models (publication style).
    Includes: equity line, shaded drawdown region, SR annotation.
    """
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8),
                             gridspec_kw={"height_ratios": [3, 1]})

    ax_eq  = axes[0]
    ax_pnl = axes[1]

    for idx, (name, (_, bt)) in enumerate(res.items()):
        if bt is None:
            continue
        eq    = bt["equity"]
        pnl   = bt["pnl"]
        weeks = np.arange(len(eq))
        c     = COLORS[idx % len(COLORS)]
        lw    = 2.5 if "SC-STGCN" in name else 1.5

        ax_eq.plot(weeks, eq, label=f"{name}  (SR={bt['sharpe']:.2f})",
                   color=c, linewidth=lw,
                   linestyle="--" if name.startswith("Naive") or name.startswith("AR") else "-")

        # Shaded drawdown
        peak = np.maximum.accumulate(eq)
        dd   = eq - peak
        ax_eq.fill_between(weeks, eq, peak, where=(dd < 0),
                           color=c, alpha=0.06)

        ax_pnl.bar(weeks, pnl, color=c, alpha=0.40, width=0.8)

    ax_eq.axhline(CFG.INIT_CASH, color="black", linestyle=":", linewidth=0.9, alpha=0.7)
    ax_eq.set_ylabel("Portfolio Equity (USD)")
    ax_eq.set_title(f"Long-Short CDS Protection Strategy — {mode} Forecasts")
    ax_eq.legend(loc="upper left", framealpha=0.85)
    ax_eq.grid(True, alpha=0.25)

    ax_pnl.axhline(0, color="black", linewidth=0.7)
    ax_pnl.set_xlabel("Test Week")
    ax_pnl.set_ylabel("Weekly PnL ($)")
    ax_pnl.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(FIG_DIR / f"fig1_equity_{mode.lower()}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"fig1_equity_{mode.lower()}.png", bbox_inches="tight", dpi=200)
    plt.close()
    print(f"[fig] Saved fig1_equity_{mode.lower()}")


def fig_metrics_comparison(df: pd.DataFrame, mode: str) -> None:
    """
    Figure 2 — Bar chart: RMSE (bp) and Annualised Sharpe side by side.
    """
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = df["Model"].tolist()
    bar_c  = COLORS[:len(models)]
    hatch  = ["" if "SC-STGCN" in m else "/" if "STGCN" in m else "" for m in models]

    # RMSE
    ax = axes[0]
    rmse = [float(r) if r != "—" else 0 for r in df["RMSE (bp)"]]
    bars = ax.bar(models, rmse, color=bar_c, edgecolor="black", linewidth=0.6)
    for bar, h in zip(bars, hatch):
        bar.set_hatch(h)
    ax.set_title(f"RMSE (basis points) — {mode}")
    ax.set_ylabel("RMSE (bp)")
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)

    # Sharpe
    ax = axes[1]
    sharpe = [float(s) if s != "—" else 0 for s in df["BT Sharpe"]]
    bars = ax.bar(models, sharpe, color=bar_c, edgecolor="black", linewidth=0.6)
    for bar, h in zip(bars, hatch):
        bar.set_hatch(h)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Annualised Sharpe Ratio — {mode}")
    ax.set_ylabel("Sharpe (×√52)")
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)

    # Legend explaining hatching
    legend_els = [
        Line2D([0], [0], color="black", lw=2, label="Proposed (SC-STGCN)"),
        Line2D([0], [0], color="black", lw=2, linestyle="--", label="Ablation (V-STGCN)"),
    ]
    axes[1].legend(handles=legend_els, loc="upper right")

    plt.tight_layout()
    fig.savefig(FIG_DIR / f"fig2_metrics_{mode.lower()}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"fig2_metrics_{mode.lower()}.png", bbox_inches="tight", dpi=200)
    plt.close()
    print(f"[fig] Saved fig2_metrics_{mode.lower()}")


def fig_scatter_grid(
    res: Dict[str, Tuple[Dict, Dict]],
    cols: List[str],
    mode: str,
    n_firms: int = 4,
) -> None:
    """
    Figure 3 — Predicted vs. actual Δspread for selected firms.
    One column per model, one row per firm.
    """
    plt.rcParams.update(STYLE)
    model_names  = list(res.keys())
    firm_indices = list(range(min(n_firms, len(cols))))
    nrows = len(firm_indices); ncols = len(model_names)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.2 * nrows),
                             squeeze=False)

    for ci, name in enumerate(model_names):
        m, _ = res[name]
        yt, yp = m["y_true_orig"], m["y_pred_orig"]
        for ri, fi in enumerate(firm_indices):
            ax  = axes[ri][ci]
            r2i = float(r2_score(yt[:, fi], yp[:, fi]))
            ax.scatter(yt[:, fi], yp[:, fi], alpha=0.5, s=16,
                       color=COLORS[ci % len(COLORS)])
            lim = (min(yt[:, fi].min(), yp[:, fi].min()),
                   max(yt[:, fi].max(), yp[:, fi].max()))
            ax.plot(lim, lim, "k--", linewidth=0.8)
            ax.set_title(f"{name} | {cols[fi]}\nR²={r2i:.3f}", fontsize=8)
            if ri == nrows - 1:
                ax.set_xlabel("True Δspread (bp)", fontsize=8)
            if ci == 0:
                ax.set_ylabel("Pred Δspread (bp)", fontsize=8)
            ax.grid(True, alpha=0.2)

    plt.suptitle(f"Predicted vs. True Δspread — {mode}", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"fig3_scatter_{mode.lower()}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"fig3_scatter_{mode.lower()}.png", bbox_inches="tight", dpi=180)
    plt.close()
    print(f"[fig] Saved fig3_scatter_{mode.lower()}")


def fig_attention_heatmap(tag: str, cols: List[str]) -> None:
    """
    Figure 4 — Average temporal attention weight heatmap (week × week).
    Visualises which past weeks the model attends to most.
    """
    path = OUT_DIR / f"{tag}_attn_w1.npy"
    if not path.exists():
        return

    try:
        aw = np.load(path)   # (B*N, heads, T, T) — averaged over test batch
        if aw.ndim < 4 or aw.shape[-1] == 0:
            return
        avg = aw.mean(axis=(0, 1))   # (T, T)

        plt.rcParams.update(STYLE)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(avg, aspect="auto", cmap="YlOrRd", origin="lower")
        plt.colorbar(im, ax=ax, label="Attention weight")
        ax.set_xlabel("Source week (key)")
        ax.set_ylabel("Target week (query)")
        ax.set_title(f"SC-STGCN Temporal Attention — Block 1\n({tag})")
        tick_lbs = [f"t-{CFG.N_HIS - i - 1}" for i in range(CFG.N_HIS)]
        ax.set_xticks(range(CFG.N_HIS)); ax.set_xticklabels(tick_lbs, rotation=45, fontsize=8)
        ax.set_yticks(range(CFG.N_HIS)); ax.set_yticklabels(tick_lbs, fontsize=8)
        plt.tight_layout()
        fig.savefig(FIG_DIR / f"fig4_attn_{tag}.pdf", bbox_inches="tight")
        fig.savefig(FIG_DIR / f"fig4_attn_{tag}.png", bbox_inches="tight", dpi=180)
        plt.close()
        print(f"[fig] Saved fig4_attn_{tag}")
    except Exception as e:
        print(f"[fig] Attention heatmap skipped: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 — ONE MODE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_mode(
    mode:   str,          # "DELTA" or "LEVEL"
    panel:  np.ndarray,   # (T, N) — already correct (level or delta)
    level:  np.ndarray,   # (T_level, N) original level (for backtest deltas)
    adjs:   Dict[str, np.ndarray],
    cols:   List[str],
    device: torch.device,
) -> None:
    """
    Execute the full model zoo for one target-variable mode.
    Saves all results, metrics and figures.
    """
    print(f"\n{'═'*70}")
    print(f"  MODE: {mode}")
    print(f"{'═'*70}")

    set_seed(CFG.SEED)   # reset seed for each mode — identical train/test splits
    mx = mode.lower()    # short tag prefix

    # ── Split ─────────────────────────────────────────────────────────────
    tr_raw, va_raw, te_raw = chronological_split(panel)
    print(f"[{mode}] train={tr_raw.shape}  val={va_raw.shape}  test={te_raw.shape}")

    scaler  = StandardScaler()
    tr_s    = scaler.fit_transform(tr_raw)
    va_s    = scaler.transform(va_raw)
    te_s    = scaler.transform(te_raw)

    # ── Helper: compute backtest in original (bp) units ──────────────────
    # For LEVEL mode we convert predicted levels → deltas before backtesting
    T_total = panel.shape[0]
    n_train = int(CFG.SPLIT[0] * T_total)
    n_val   = int((CFG.SPLIT[0] + CFG.SPLIT[1]) * T_total)
    te_start = n_train + n_val   # index in PANEL where test starts

    def _backtest(m_dict: Dict) -> Dict:
        yt_o = m_dict["y_true_orig"]
        yp_o = m_dict["y_pred_orig"]
        B    = yt_o.shape[0]

        if mode == "DELTA":
            # y_true/pred are already Dbp
            return backtest_ls_cds(yt_o, yp_o)

        else:  # LEVEL mode: backtest not meaningful (persistence dominates)
            return None

        if False:  # LEVEL mode (disabled)
            # Convert level predictions to deltas for backtesting.
            # y_true_orig = actual spread level at t
            # y_pred_orig = predicted spread level at t
            # delta = predicted_level(t) - actual_level(t-1)
            # We use y_true(t) - y_true(t-1) as realized delta
            # and y_pred(t) - y_true(t-1) as predicted delta (signal)
            N_use = min(yt_o.shape[1], yp_o.shape[1])

            # Realized delta: actual(t) - actual(t-1)
            # Predicted delta: predicted(t) - actual(t-1)
            # Need actual(t-1) for each test step
            prev = np.zeros((B, N_use), dtype=np.float32)
            for k in range(B):
                if k == 0:
                    # First test step: prev = last value of test panel
                    t_global = te_start + CFG.N_HIS - 1
                else:
                    # Use actual y_true from previous step as prev
                    t_global = te_start + CFG.N_HIS + k - 1
                if 0 <= t_global < level.shape[0]:
                    prev[k] = level[t_global, :N_use]

            yt_d = yt_o[:, :N_use] - prev   # realized delta
            yp_d = yp_o[:, :N_use] - prev   # predicted delta (signal)
            return backtest_ls_cds(yt_d, yp_d)

    # ── Align N ───────────────────────────────────────────────────────────
    N = tr_s.shape[1]

    res: Dict[str, Tuple[Dict, Dict]] = {}

    # 1. AR(1)
    print(f"\n[1/5] AR(1) ({mode})")
    m = run_ar1(tr_s, te_s, scaler, cols, f"{mx}_ar1")
    res["AR(1)"] = (m, _backtest(m))
    print(f"       RMSE={m['rmse_o']:.3f} bp")

    # 2. LSTM
    print(f"\n[2/5] LSTM ({mode})")
    m = run_lstm(tr_s, va_s, te_s, scaler, cols, f"{mx}_lstm", device)
    res["LSTM"] = (m, _backtest(m))
    print(f"       RMSE={m['rmse_o']:.3f} bp")

    # 3. XGBoost
    if HAS_XGB:
        print(f"\n[3/5] XGBoost ({mode})")
        m = run_xgboost(tr_s, te_s, scaler, cols, f"{mx}_xgb")
        res["XGBoost"] = (m, _backtest(m))
        print(f"       RMSE={m['rmse_o']:.3f} bp")
    else:
        print(f"\n[3/5] XGBoost SKIPPED")

    # 4. Vanilla STGCN (ablation)
    print(f"\n[4/5] Vanilla STGCN - ablation ({mode})")
    m = run_vanilla_stgcn(
        tr_s, va_s, te_s, scaler, cols,
        adjs["combined"], f"{mx}_vstgcn", device,
    )
    res["V-STGCN"] = (m, _backtest(m))
    print(f"       RMSE={m['rmse_o']:.3f} bp")

    # 5. SC-STGCN (proposed)
    print(f"\n[5/5] SC-STGCN - proposed ({mode})")
    m = run_sc_stgcn(tr_s, va_s, te_s, scaler, cols, adjs, f"{mx}_scstgcn", device)
    res["SC-STGCN"] = (m, _backtest(m))
    print(f"       RMSE={m['rmse_o']:.3f} bp")

    # ── Aggregate ─────────────────────────────────────────────────────────
    df = build_results_table(res, mode)
    print_summary(df, mode)

    # ── Figures ───────────────────────────────────────────────────────────
    fig_equity_curves(res, mode)
    fig_metrics_comparison(df, mode)
    fig_scatter_grid(res, cols, mode, n_firms=min(4, N))
    fig_attention_heatmap(f"{mx}_scstgcn", cols)

    # Save JSON experiment record (for reproducibility appendix in paper)
    record = {
        "mode": mode,
        "n_his": CFG.N_HIS,
        "n_pred": CFG.N_PRED,
        "split": list(CFG.SPLIT),
        "seed": CFG.SEED,
        "results": {
            name: {
                "rmse_bp": round(m_["rmse_o"], 4),
                "mae_bp":  round(m_["mae_o"],  4),
                "r2_bp":   round(m_["r2_o"],   4),
                "sharpe":  round(bt_["sharpe"], 4) if bt_ else None,
                "hit":     round(bt_["hit"],    4) if bt_ else None,
                "max_dd":  round(bt_["max_dd"], 4) if bt_ else None,
            }
            for name, (m_, bt_) in res.items()
        }
    }
    (METRICS_DIR / f"experiment_record_{mx}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(f"[save] Experiment record → experiment_record_{mx}.json")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15 — CROSS-MODE COMPARISON FIGURE
# ─────────────────────────────────────────────────────────────────────────────
def fig_cross_mode_comparison() -> None:
    """
    Figure 5 — Cross-mode comparison: load saved CSVs for DELTA and LEVEL
    and produce a side-by-side RMSE / Sharpe comparison plot.
    """
    paths = {
        "DELTA": OUT_DIR / "results_delta.csv",
        "LEVEL": OUT_DIR / "results_level.csv",
    }
    loaded = {}
    for mode, p in paths.items():
        if p.exists():
            loaded[mode] = pd.read_csv(p)

    if len(loaded) < 2:
        return

    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, metric in enumerate(["RMSE (bp)", "BT Sharpe"]):
        ax = axes[ax_idx]
        df_d = loaded["DELTA"]
        df_l = loaded["LEVEL"]
        models = df_d["Model"].tolist()
        x = np.arange(len(models))
        w = 0.35

        def _vals(df): return [float(v) if v != "—" else 0 for v in df[metric]]

        bars_d = ax.bar(x - w / 2, _vals(df_d), w, label="DELTA",
                        color="#1f77b4", edgecolor="black", linewidth=0.6, alpha=0.85)
        bars_l = ax.bar(x + w / 2, _vals(df_l), w, label="LEVEL",
                        color="#d62728", edgecolor="black", linewidth=0.6, alpha=0.85)

        ax.set_xticks(x); ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_title(f"{metric} — DELTA vs. LEVEL")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        if metric == "BT Sharpe":
            ax.axhline(0, color="black", linewidth=0.8)

    plt.suptitle("SC-STGCN Framework — Cross-Mode Performance Comparison", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig5_cross_mode.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig5_cross_mode.png", bbox_inches="tight", dpi=200)
    plt.close()
    print("[fig] Saved fig5_cross_mode")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 16 — ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SC-STGCN CDS Forecasting — Comparative Thesis Experiment"
    )
    p.add_argument(
        "--mode", choices=["DELTA", "LEVEL", "BOTH"], default="BOTH",
        help="Target variable mode (default: BOTH)"
    )
    p.add_argument("--epochs",  type=int, default=None,
                   help="Override epoch count for all neural models")
    p.add_argument("--seed",    type=int, default=42,
                   help="Global random seed (default: 42)")
    p.add_argument("--device",  type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Apply CLI overrides
    CFG.SEED   = args.seed
    CFG.DEVICE = args.device
    if args.epochs is not None:
        CFG.SCSTGCN_EPOCHS = args.epochs
        CFG.VSTGCN_EPOCHS  = args.epochs
        CFG.LSTM_EPOCHS    = args.epochs
        CFG.GRU_EPOCHS     = args.epochs

    set_seed(CFG.SEED)
    device = get_device()

    t0 = time.time()
    print(f"\n{'═'*70}")
    print("  SC-STGCN: Supply-Chain Aware STGCN for CDS Spread Forecasting")
    print(f"{'═'*70}")
    print(f"  Device  : {device}")
    print(f"  Mode    : {args.mode}")
    print(f"  Out dir : {OUT_DIR}")
    print(f"  Seed    : {CFG.SEED}")
    print(f"  XGBoost : {'available' if HAS_XGB else 'not installed'}")
    print(f"  ARMA    : {'available' if HAS_STATSMODELS else 'not installed'}")

    # ── Load data ──────────────────────────────────────────────────────────
    level, delta, cols = load_panel()
    adjs = load_adjacency(N_max=delta.shape[1])

    # Align N across panel and adjacency
    N_panel = delta.shape[1]
    N_adj   = adjs["combined"].shape[0]
    N       = min(N_panel, N_adj)
    if N_panel != N_adj:
        print(f"[align] N_panel={N_panel}, N_adj={N_adj} → using N={N}")
        for k in adjs:
            adjs[k] = adjs[k][:N, :N]
        delta   = delta[:, :N]
        level   = level[:, :N]
        cols    = cols[:N]

    # ── Run experiments ────────────────────────────────────────────────────
    modes_to_run = (
        ["DELTA", "LEVEL"] if args.mode == "BOTH"
        else [args.mode]
    )

    for mode in modes_to_run:
        panel = delta if mode == "DELTA" else level
        run_mode(
            mode=mode,
            panel=panel,
            level=level,
            adjs=adjs,
            cols=cols,
            device=device,
        )

    # ── Cross-mode figure ──────────────────────────────────────────────────
    if args.mode == "BOTH":
        fig_cross_mode_comparison()

    # ── Final summary ──────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'═'*70}")
    print(f"  ALL EXPERIMENTS COMPLETED  |  {elapsed/60:.1f} min")
    print(f"  Outputs → {OUT_DIR}")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
