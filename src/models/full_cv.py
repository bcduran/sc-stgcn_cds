"""
src/models/full_cv.py
======================
Tam Karsilastirmali Walk-Forward CV

Modeller:
  Baseline   : AR(1), LSTM, XGBoost
  Transformer: Informer, Autoformer, TimesNet
  Graph      : V-STGCN, SC-STGCN

3 Tablo:
  1. Regression  : RMSE, MAE, R²
  2. Classification: Acc, Prec, Recall, F1
  3. Portfolio   : Sharpe, Mean Return (bp/wk), Std, Max Drawdown

Diebold-Mariano testi: SC-STGCN vs her model

Calistirmak:
  python src/models/full_cv.py --epochs 300 --device cpu
"""

from __future__ import annotations
import argparse, random, sys, time, math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import load_npz
from scipy import stats
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import (
    TOP50_WEEKLY_CSV,
    ADJ_NPZ,
    ADJ_SUP_NPZ,
    ADJ_CUS_NPZ,
    METRICS_DIR,
    FIG_DIR,
    make_dirs,
)

try:
    from xgboost import XGBRegressor

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# CONFIG
# =============================================================================
SEED = 42
N_HIS = 8
GCN_H = 64
ATT_H = 64
ATT_HEADS = 2
FF_H = 128
DROPOUT = 0.30
LR = 1e-4
WD = 1e-3
BATCH = 64
PATIENCE = 40
CLIP = 5.0
TOP_Q = 0.20
DV01 = 100.0
INIT_CASH = 100_000.0


def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True


def norm_adj(A):
    A = A + np.eye(A.shape[0], dtype=np.float32)
    d = A.sum(1) ** -0.5
    return torch.tensor(np.diag(d) @ A @ np.diag(d), dtype=torch.float32)


def make_windows(data, n_his):
    X, Y = [], []
    for t in range(n_his, len(data)):
        X.append(data[t - n_his : t])
        Y.append(data[t])
    return np.array(X, np.float32), np.array(Y, np.float32)


# =============================================================================
# METRICS
# =============================================================================
def calc_regression(yt, yp):
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae = float(np.mean(np.abs(yt - yp)))
    ss_r = np.sum((yt - yp) ** 2)
    ss_t = np.sum((yt - yt.mean()) ** 2)
    r2 = float(1 - ss_r / ss_t) if ss_t > 1e-10 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def calc_classification(yt, yp):
    yt_d = (yt.flatten() > 0).astype(int)
    yp_d = (yp.flatten() > 0).astype(int)
    tp = int(np.sum((yt_d == 1) & (yp_d == 1)))
    tn = int(np.sum((yt_d == 0) & (yp_d == 0)))
    fp = int(np.sum((yt_d == 0) & (yp_d == 1)))
    fn = int(np.sum((yt_d == 1) & (yp_d == 0)))
    acc = float((tp + tn) / (tp + tn + fp + fn + 1e-10))
    prec = float(tp / (tp + fp + 1e-10))
    rec = float(tp / (tp + fn + 1e-10))
    f1 = float(2 * prec * rec / (prec + rec + 1e-10))
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def calc_portfolio(yt, yp, q=TOP_Q, dv01=DV01, cash=INIT_CASH):
    T, N = yt.shape
    K = max(1, int(q * N))
    pnl = []
    for t in range(T):
        r = np.argsort(yp[t])
        w = np.zeros(N)
        w[r[-K:]] = +1 / K
        w[r[:K]] = -1 / K
        pnl.append(dv01 * (w * yt[t]).sum())
    pnl = np.array(pnl)
    eq = cash + np.cumsum(pnl)
    sharpe = float(np.sqrt(52) * pnl.mean() / (pnl.std() + 1e-10))
    mean_ret = float(pnl.mean())
    std_ret = float(pnl.std())
    mdd = float(np.min(eq / np.maximum.accumulate(eq) - 1))
    hit = float((pnl > 0).mean())
    return {
        "sharpe": sharpe,
        "mean_ret_bp": mean_ret,
        "std_ret_bp": std_ret,
        "max_drawdown": mdd,
        "hit_ratio": hit,
    }, pnl


def calc_all(yt, yp):
    port_metrics, pnl = calc_portfolio(yt, yp)
    return {
        **calc_regression(yt, yp),
        **calc_classification(yt, yp),
        **port_metrics,
    }, pnl


def sq_errors(yt, yp):
    """Per-observation squared errors for DM test."""
    return ((yt - yp) ** 2).flatten()


# =============================================================================
# DIEBOLD-MARIANO TEST
# =============================================================================
def dm_test(e1, e2, h=1):
    """
    Diebold-Mariano test: H0: equal predictive accuracy
    e1, e2: squared error arrays (flat)
    Returns: DM stat, p-value (two-sided)
    """
    d = e1 - e2  # loss differential: positive = e1 worse
    T = len(d)
    d_bar = d.mean()
    # HAC variance (Newey-West with h-1 lags)
    gamma0 = np.var(d, ddof=0)
    gamma = (
        sum(
            (1 - k / h) * np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
            for k in range(1, h)
        )
        if h > 1
        else 0.0
    )
    var_d = (gamma0 + 2 * gamma) / T
    dm_stat = d_bar / np.sqrt(max(var_d, 1e-10))
    p_val = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_val)


# =============================================================================
# BASELINE MODELS
# =============================================================================
class LSTMModel(nn.Module):
    def __init__(self, N, h=128, layers=2, drop=0.20):
        super().__init__()
        self.lstm = nn.LSTM(
            N, h, layers, batch_first=True, dropout=drop if layers > 1 else 0.0
        )
        self.head = nn.Linear(h, N)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


# =============================================================================
# TRANSFORMER MODELS
# =============================================================================


# ── Informer ──────────────────────────────────────────────────────────────────
class ProbSparseAttention(nn.Module):
    """Simplified ProbSparse attention (Informer)."""

    def __init__(self, d_model, n_heads, factor=5, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.factor = factor
        d_k = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        H = self.n_heads
        d_k = D // H
        Q = self.Wq(x).view(B, L, H, d_k).transpose(1, 2)
        K = self.Wk(x).view(B, L, H, d_k).transpose(1, 2)
        V = self.Wv(x).view(B, L, H, d_k).transpose(1, 2)
        # Simplified: use top-k queries
        U_part = max(1, int(self.factor * math.log(L + 1)))
        U_part = min(U_part, L)
        idx = torch.randperm(L, device=x.device)[:U_part]
        Q_s = Q[:, :, idx, :]
        scores = (Q_s @ K.transpose(-2, -1)) / math.sqrt(d_k)
        attn = self.drop(torch.softmax(scores, -1))
        context = attn @ V
        # Scatter back
        out = torch.zeros_like(Q)
        out[:, :, idx, :] = context
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out(out)


class InformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff, drop):
        super().__init__()
        self.attn = ProbSparseAttention(d_model, n_heads, dropout=drop)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff), nn.GELU(), nn.Dropout(drop), nn.Linear(ff, d_model)
        )
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.n1(x + self.attn(x))
        return self.n2(x + self.ff(x))


class Informer(nn.Module):
    def __init__(self, N, seq_len, d_model=64, n_heads=4, ff=128, n_layers=2, drop=0.1):
        super().__init__()
        self.emb = nn.Linear(N, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.layers = nn.ModuleList(
            [InformerBlock(d_model, n_heads, ff, drop) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, N)

    def forward(self, x):
        # x: (B, seq_len, N)
        h = self.emb(x) + self.pos
        for layer in self.layers:
            h = layer(h)
        return self.head(self.norm(h[:, -1, :]))


# ── Autoformer ────────────────────────────────────────────────────────────────
class SeriesDecomp(nn.Module):
    """Moving average decomposition."""

    def __init__(self, kernel_size=3):
        super().__init__()
        self.avg = nn.AvgPool1d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        # x: (B,L,N) -> trend via avg pool
        trend = self.avg(x.transpose(1, 2)).transpose(1, 2)
        # Handle length mismatch
        if trend.shape[1] != x.shape[1]:
            trend = trend[:, : x.shape[1], :]
        seasonal = x - trend
        return seasonal, trend


class AutoCorrelation(nn.Module):
    """Simplified auto-correlation attention."""

    def __init__(self, d_model, n_heads, factor=3):
        super().__init__()
        self.n_heads = n_heads
        self.factor = factor
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        # FFT-based autocorrelation
        q_fft = torch.fft.rfft(Q, dim=1)
        k_fft = torch.fft.rfft(K, dim=1)
        corr = torch.fft.irfft(q_fft * k_fft.conj(), n=L, dim=1)
        # Top-k lags
        top_k = max(1, int(self.factor * math.log(L + 1)))
        top_k = min(top_k, L)
        weights, delays = corr.topk(top_k, dim=1)
        weights = torch.softmax(weights, dim=1)
        # Aggregate
        out = torch.zeros_like(V)
        for i in range(top_k):
            delay = delays[:, i : i + 1, :]
            rolled = torch.roll(V, int(delay.float().mean().item()), dims=1)
            out = out + weights[:, i : i + 1, :] * rolled
        return self.out(out)


class AutoformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff, drop):
        super().__init__()
        self.decomp1 = SeriesDecomp()
        self.decomp2 = SeriesDecomp()
        self.attn = AutoCorrelation(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff), nn.GELU(), nn.Dropout(drop), nn.Linear(ff, d_model)
        )
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        s, _ = self.decomp1(x + self.drop(self.attn(x)))
        s = self.n1(s)
        s2, _ = self.decomp2(s + self.drop(self.ff(s)))
        return self.n2(s2)


class Autoformer(nn.Module):
    def __init__(self, N, seq_len, d_model=64, n_heads=4, ff=128, n_layers=2, drop=0.1):
        super().__init__()
        self.emb = nn.Linear(N, d_model)
        self.layers = nn.ModuleList(
            [AutoformerBlock(d_model, n_heads, ff, drop) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, N)

    def forward(self, x):
        h = self.emb(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(self.norm(h[:, -1, :]))


# ── TimesNet ──────────────────────────────────────────────────────────────────
class TimesBlock(nn.Module):
    """2D temporal convolution block (TimesNet simplified)."""

    def __init__(self, d_model, ff, top_k=3, drop=0.1):
        super().__init__()
        self.top_k = top_k
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, ff, kernel_size=(1, 3), padding=(0, 1)),
            nn.GELU(),
            nn.Conv2d(ff, d_model, kernel_size=(1, 3), padding=(0, 1)),
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # x: (B,L,D)
        B, L, D = x.shape
        # FFT to find dominant periods
        fft_val = torch.fft.rfft(x, dim=1)
        amp = fft_val.abs().mean(-1)[:, 1:]  # skip DC
        top_k = min(self.top_k, amp.shape[1])
        _, top_idx = amp.topk(top_k, dim=1)

        res = torch.zeros_like(x)
        for i in range(top_k):
            period = int(top_idx[:, i].float().mean().item()) + 1
            period = max(1, min(period, L))
            # Pad to multiple of period
            pad_len = math.ceil(L / period) * period - L
            x_pad = F.pad(x.transpose(1, 2), (0, pad_len)).transpose(1, 2)
            # Reshape to 2D
            B2, L2, D2 = x_pad.shape
            rows = L2 // period
            x2 = x_pad.reshape(B2, rows, period, D2)
            x2 = x2.permute(0, 3, 1, 2)  # (B,D,rows,period)
            x2 = self.conv(x2)
            x2 = x2.permute(0, 2, 3, 1).reshape(B2, L2, D2)
            res = res + x2[:, :L, :]

        return self.norm(x + self.drop(res / top_k))


class TimesNet(nn.Module):
    def __init__(self, N, seq_len, d_model=64, ff=128, n_layers=2, top_k=3, drop=0.1):
        super().__init__()
        self.emb = nn.Linear(N, d_model)
        self.layers = nn.ModuleList(
            [TimesBlock(d_model, ff, top_k, drop) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, N)

    def forward(self, x):
        h = self.emb(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(self.norm(h[:, -1, :]))


# =============================================================================
# GRAPH MODELS
# =============================================================================
class GCNLayer(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.W = nn.Linear(i, o, bias=False)

    def forward(self, x, A):
        return torch.relu(self.W(A @ x))


class TemporalAttn(nn.Module):
    def __init__(self, d, heads, ff, drop):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d, ff), nn.GELU(), nn.Dropout(drop), nn.Linear(ff, d)
        )
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)

    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x = self.n1(x + a)
        return self.n2(x + self.ff(x))


class SCSTGCN(nn.Module):
    def __init__(self, N, n_his, gcn_h, att_h, heads, ff_h, drop, A_up, A_dn):
        super().__init__()
        self.register_buffer("A_up", A_up)
        self.register_buffer("A_dn", A_dn)
        self.gcn_up = GCNLayer(n_his, gcn_h)
        self.gcn_dn = GCNLayer(n_his, gcn_h)
        self.gate = nn.Linear(gcn_h * 2, gcn_h)
        self.proj = nn.Linear(gcn_h, att_h)
        self.attn = TemporalAttn(att_h, heads, ff_h, drop)
        self.head = nn.Linear(att_h, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        xT = x.permute(0, 2, 1)
        h_up = self.gcn_up(xT, self.A_up)
        h_dn = self.gcn_dn(xT, self.A_dn)
        g = torch.sigmoid(self.gate(torch.cat([h_up, h_dn], -1)))
        h = g * h_up + (1 - g) * h_dn
        h = self.drop(self.proj(h))
        return self.head(self.attn(h)).squeeze(-1)


class VSTGCN(nn.Module):
    def __init__(self, N, n_his, gcn_h, A):
        super().__init__()
        self.register_buffer("A", A)
        self.gcn = GCNLayer(n_his, gcn_h)
        self.proj = nn.Linear(gcn_h, gcn_h)
        self.head = nn.Linear(gcn_h, 1)

    def forward(self, x):
        h = self.gcn(x.permute(0, 2, 1), self.A)
        return self.head(torch.relu(self.proj(h))).squeeze(-1)


# =============================================================================
# TRAINING
# =============================================================================
def train_nn(model, Xtr, Ytr, Xva, Yva, device, epochs=300):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=LR * 0.01
    )
    lfn = nn.HuberLoss(delta=1.0)
    Xt = torch.tensor(Xtr).to(device)
    Yt = torch.tensor(Ytr).to(device)
    Xv = torch.tensor(Xva).to(device)
    Yv = torch.tensor(Yva).to(device)
    best, no_imp, state = float("inf"), 0, None
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(len(Xt), device=device)
        for i in range(0, len(idx), BATCH):
            b = idx[i : i + BATCH]
            opt.zero_grad()
            lfn(model(Xt[b]), Yt[b]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = lfn(model(Xv), Yv).item()
        if vl < best - 1e-6:
            best, no_imp = vl, 0
            state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                break
    if state:
        model.load_state_dict(state)
    return model, ep + 1


def pred_nn(model, X, device):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X).to(device)).cpu().numpy()


# =============================================================================
# RUN FOLD
# =============================================================================
def run_fold(delta, N, Ac_t, Au_t, Ad_t, device, epochs, tr_end, va_end, te_end):
    sc = StandardScaler()
    tr_s = sc.fit_transform(delta[:tr_end])
    va_s = sc.transform(delta[tr_end:va_end])
    te_s = sc.transform(delta[va_end:te_end])
    Xtr, Ytr = make_windows(tr_s, N_HIS)
    Xva, Yva = make_windows(va_s, N_HIS)
    Xte, _ = make_windows(te_s, N_HIS)
    te_orig = delta[va_end:te_end][N_HIS:]
    results = {}
    sq_err = {}
    pnl_series = {}

    def register(name, pred):
        p = sc.inverse_transform(pred)
        metrics, pnl = calc_all(te_orig, p)
        results[name] = metrics
        sq_err[name] = sq_errors(te_orig, p)
        pnl_series[name] = pnl

    # AR(1)
    ps = np.zeros_like(te_s)
    for i in range(N):
        y = tr_s[:, i]
        phi = np.corrcoef(y[:-1], y[1:])[0, 1] if y.std() > 1e-8 else 0.0
        mu = y.mean() * (1 - phi)
        last = tr_s[-1, i]
        for t in range(len(te_s)):
            ps[t, i] = mu + phi * last
            last = te_s[t, i]
    register("AR(1)", ps[N_HIS:])

    # LSTM
    lstm = LSTMModel(N).to(device)
    lstm, ep = train_nn(lstm, Xtr, Ytr, Xva, Yva, device, epochs)
    register("LSTM", pred_nn(lstm, Xte, device))
    print(f"    LSTM ep={ep}")

    # XGBoost
    if HAS_XGB:
        Xtf = Xtr.reshape(len(Xtr), -1)
        Xtef = Xte.reshape(len(Xte), -1)
        ps_x = np.zeros((len(Xte), N))
        for i in range(N):
            m = XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=SEED,
                tree_method="hist",
                n_jobs=-1,
                verbosity=0,
            )
            m.fit(Xtf, Ytr[:, i])
            ps_x[:, i] = m.predict(Xtef)
        register("XGBoost", ps_x)

    # Informer
    inf_m = Informer(
        N, N_HIS, d_model=ATT_H, n_heads=ATT_HEADS, ff=FF_H, n_layers=2, drop=DROPOUT
    ).to(device)
    inf_m, ep = train_nn(inf_m, Xtr, Ytr, Xva, Yva, device, epochs)
    register("Informer", pred_nn(inf_m, Xte, device))
    print(f"    Informer ep={ep}")

    # Autoformer
    af_m = Autoformer(
        N, N_HIS, d_model=ATT_H, n_heads=ATT_HEADS, ff=FF_H, n_layers=2, drop=DROPOUT
    ).to(device)
    af_m, ep = train_nn(af_m, Xtr, Ytr, Xva, Yva, device, epochs)
    register("Autoformer", pred_nn(af_m, Xte, device))
    print(f"    Autoformer ep={ep}")

    # TimesNet
    tn_m = TimesNet(
        N, N_HIS, d_model=ATT_H, ff=FF_H, n_layers=2, top_k=3, drop=DROPOUT
    ).to(device)
    tn_m, ep = train_nn(tn_m, Xtr, Ytr, Xva, Yva, device, epochs)
    register("TimesNet", pred_nn(tn_m, Xte, device))
    print(f"    TimesNet ep={ep}")

    # V-STGCN
    vs = VSTGCN(N, N_HIS, GCN_H, Ac_t).to(device)
    vs, ep = train_nn(vs, Xtr, Ytr, Xva, Yva, device, epochs)
    register("V-STGCN", pred_nn(vs, Xte, device))
    print(f"    V-STGCN ep={ep}")

    # SC-STGCN
    sc_m = SCSTGCN(N, N_HIS, GCN_H, ATT_H, ATT_HEADS, FF_H, DROPOUT, Au_t, Ad_t).to(
        device
    )
    sc_m, ep = train_nn(sc_m, Xtr, Ytr, Xva, Yva, device, epochs)
    register("SC-STGCN", pred_nn(sc_m, Xte, device))
    print(f"    SC-STGCN ep={ep}")

    return results, sq_err, te_orig, pnl_series


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dataset",
        default="top50",
        choices=["top50", "top50_degree"],
        help="top50 = original | top50_degree = degree-ranked",
    )
    args = parser.parse_args()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    make_dirs()
    set_seed()

    SEP = "=" * 65
    print(SEP)
    print("  Full CV — AR/LSTM/XGB + Informer/Autoformer/TimesNet + V/SC-STGCN")
    print(f"  epochs={args.epochs} | device={device}")
    print(SEP)

    # Veri
    # Dataset secimi
    BASE = Path(__file__).resolve().parents[2]
    if args.dataset == "top50":
        csv_path = TOP50_WEEKLY_CSV
        adj_path = ADJ_NPZ
        sup_path = ADJ_SUP_NPZ
        cus_path = ADJ_CUS_NPZ
        label = "top50 (original weight-ranked)"
    else:
        csv_path = BASE / "data" / "top50_degree" / "ve1.csv"
        adj_path = BASE / "data" / "top50_degree" / "adj.npz"
        sup_path = BASE / "data" / "top50_degree" / "adj_sup.npz"
        cus_path = BASE / "data" / "top50_degree" / "adj_cus.npz"
        label = "top50_degree (degree-ranked, 50/50 connected)"

    print(f"\n  Dataset: {label}")

    # ve1.csv: ilk kolon tarih olabilir (top50_degree) ya da olmayabilir (top50)
    df = pd.read_csv(csv_path, index_col=None)
    # Ilk kolon sayisal degil ise tarih kolonudur, at
    try:
        df.iloc[:, 0].astype(np.float32)
    except (ValueError, TypeError):
        df = df.iloc[:, 1:]  # tarih kolonunu at
    level = df.values.astype(np.float32)
    delta = np.diff(level, axis=0)
    T, N = delta.shape
    dates = pd.date_range(start="2015-01-09", periods=T, freq="W-FRI")
    covid_week = int(np.searchsorted(dates, pd.Timestamp("2020-03-01")))
    delta_pre = delta[:covid_week]
    print(f"  Pre-COVID: {covid_week} hafta  N={N}")

    # Adjacency
    Ac = load_npz(str(adj_path)).toarray().astype(np.float32)
    Au = load_npz(str(sup_path)).toarray().astype(np.float32)
    Ad = load_npz(str(cus_path)).toarray().astype(np.float32)
    Ac_t = norm_adj(Ac).to(device)
    Au_t = norm_adj(Au).to(device)
    Ad_t = norm_adj(Ad).to(device)

    # 2 fold
    folds = [
        (int(covid_week * 0.65), int(covid_week * 0.78), int(covid_week * 0.92)),
        (int(covid_week * 0.78), int(covid_week * 0.92), covid_week),
    ]
    print(f"\n  Fold yapisi:")
    for i, (a, b, c) in enumerate(folds):
        print(
            f"    Fold {i+1}: train[0:{a}] val[{a}:{b}] test[{b}:{c}] "
            f"({c-b} hafta | {dates[b].date()} -> {dates[c-1].date()})"
        )

    MODEL_NAMES = [
        "AR(1)",
        "LSTM",
        "Informer",
        "Autoformer",
        "TimesNet",
        "V-STGCN",
        "SC-STGCN",
    ]
    if HAS_XGB:
        MODEL_NAMES.insert(2, "XGBoost")

    all_results = []
    all_sq_errs = []
    all_pnl_series = []
    for fi, (tr_end, va_end, te_end) in enumerate(folds):
        print(f"\n  --- Fold {fi+1}/2 ---")
        set_seed(SEED + fi)
        res, sq_err, _, pnl_s = run_fold(
            delta_pre, N, Ac_t, Au_t, Ad_t, device, args.epochs, tr_end, va_end, te_end
        )
        all_results.append(res)
        all_sq_errs.append(sq_err)
        all_pnl_series.append(pnl_s)

    # Aggregate
    reg_metrics = ["rmse", "mae", "r2"]
    clf_metrics = ["accuracy", "precision", "recall", "f1"]
    port_metrics = ["sharpe", "mean_ret_bp", "std_ret_bp", "max_drawdown", "hit_ratio"]
    all_metrics = reg_metrics + clf_metrics + port_metrics

    agg = {}
    for name in MODEL_NAMES:
        vals = {m: [] for m in all_metrics}
        for fr in all_results:
            if name in fr:
                for m in all_metrics:
                    v = fr[name].get(m, float("nan"))
                    if not np.isnan(v):
                        vals[m].append(v)
        agg[name] = {}
        for m in all_metrics:
            v = np.array(vals[m])
            agg[name][f"{m}_mean"] = float(np.mean(v)) if len(v) > 0 else float("nan")
            agg[name][f"{m}_std"] = float(np.std(v)) if len(v) > 0 else float("nan")

    # DM Test: SC-STGCN vs her model
    dm_results = {}
    sc_sq = np.concatenate([f["SC-STGCN"] for f in all_sq_errs])
    for name in MODEL_NAMES:
        if name == "SC-STGCN":
            continue
        other_sq = np.concatenate([f[name] for f in all_sq_errs])
        dm_stat, p_val = dm_test(other_sq, sc_sq)  # positive = other worse
        dm_results[name] = {"dm_stat": dm_stat, "p_val": p_val}

    # ── TABLO 1: Regression ──────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  TABLO 1 — REGRESSION RESULTS")
    print(
        f"  {'Model':<14} {'RMSE':>8} {'±':>6} {'MAE':>8} {'±':>6} {'R²':>8} {'±':>6}  DM-p"
    )
    print(f"  {'-'*75}")
    for name in MODEL_NAMES:
        a = agg[name]
        dm_p = dm_results[name]["p_val"] if name in dm_results else float("nan")
        star = (
            "***"
            if dm_p < 0.01
            else "**" if dm_p < 0.05 else "*" if dm_p < 0.10 else ""
        )
        print(
            f"  {name:<14} "
            f"{a['rmse_mean']:>8.4f} {a['rmse_std']:>6.4f} "
            f"{a['mae_mean']:>8.4f} {a['mae_std']:>6.4f} "
            f"{a['r2_mean']:>8.4f} {a['r2_std']:>6.4f}  "
            f"{dm_p:>6.3f}{star}"
        )

    # ── TABLO 2: Classification ───────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  TABLO 2 — CLASSIFICATION RESULTS")
    print(
        f"  {'Model':<14} {'Acc':>8} {'±':>6} {'Prec':>8} {'±':>6} "
        f"{'Recall':>8} {'±':>6} {'F1':>8} {'±':>6}"
    )
    print(f"  {'-'*80}")
    for name in MODEL_NAMES:
        a = agg[name]
        print(
            f"  {name:<14} "
            f"{a['accuracy_mean']:>8.4f} {a['accuracy_std']:>6.4f} "
            f"{a['precision_mean']:>8.4f} {a['precision_std']:>6.4f} "
            f"{a['recall_mean']:>8.4f} {a['recall_std']:>6.4f} "
            f"{a['f1_mean']:>8.4f} {a['f1_std']:>6.4f}"
        )

    # ── TABLO 3: Portfolio ────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  TABLO 3 — PORTFOLIO RESULTS")
    print(
        f"  {'Model':<14} {'Sharpe':>8} {'±':>6} {'MeanRet':>9} {'Std':>8} {'MaxDD':>8} {'Hit':>7}"
    )
    print(f"  {'-'*70}")
    for name in MODEL_NAMES:
        a = agg[name]
        print(
            f"  {name:<14} "
            f"{a['sharpe_mean']:>8.4f} {a['sharpe_std']:>6.4f} "
            f"{a['mean_ret_bp_mean']:>9.4f} "
            f"{a['std_ret_bp_mean']:>8.4f} "
            f"{a['max_drawdown_mean']:>8.4f} "
            f"{a['hit_ratio_mean']:>7.4f}"
        )

    # ── DM Test Ozet ─────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  DIEBOLD-MARIANO TEST (SC-STGCN vs baselines)")
    print("  H0: equal predictive accuracy | * p<0.10  ** p<0.05  *** p<0.01")
    print(f"  {'Model':<14} {'DM stat':>10} {'p-value':>10}  Result")
    print(f"  {'-'*50}")
    for name, dm in dm_results.items():
        star = (
            "***"
            if dm["p_val"] < 0.01
            else "**" if dm["p_val"] < 0.05 else "*" if dm["p_val"] < 0.10 else "n.s."
        )
        direction = "SC better" if dm["dm_stat"] > 0 else "SC worse"
        print(
            f"  {name:<14} {dm['dm_stat']:>10.4f} {dm['p_val']:>10.4f}  {star} {direction}"
        )

    # CSV kaydet
    rows = []
    for name in MODEL_NAMES:
        a = agg[name]
        dm = dm_results.get(name, {})
        row = {"Model": name}
        for m in all_metrics:
            row[f"{m}_mean"] = a[f"{m}_mean"]
            row[f"{m}_std"] = a[f"{m}_std"]
        row["dm_stat"] = dm.get("dm_stat", float("nan"))
        row["dm_pval"] = dm.get("p_val", float("nan"))
        rows.append(row)
    out_stem = f"full_cv_{args.dataset}"
    pd.DataFrame(rows).to_csv(METRICS_DIR / f"{out_stem}_results.csv", index=False)

    # PnL serileri kaydet (equity curve figürü için)
    pnl_rows = []
    for fi, pnl_s in enumerate(all_pnl_series):
        for name, pnl in pnl_s.items():
            for t, v in enumerate(pnl):
                pnl_rows.append({"fold": fi + 1, "model": name, "week": t, "pnl": v})
    pd.DataFrame(pnl_rows).to_csv(METRICS_DIR / f"{out_stem}_pnl.csv", index=False)
    print(f"  [OK] {out_stem}_pnl.csv")

    # LaTeX tabloları
    tex_path = METRICS_DIR / f"{out_stem}_tables.tex"
    with open(tex_path, "w") as f:
        # Tablo 1
        f.write("% TABLE 1 — REGRESSION\n")
        f.write("\\begin{table}[ht]\n\\centering\n")
        f.write("\\caption{Regression results — pre-COVID walk-forward CV (2 folds)}\n")
        f.write("\\label{tab:regression}\n")
        f.write("\\begin{tabular}{lccccccr}\n\\toprule\n")
        f.write(
            "Model & RMSE & $\\pm$ & MAE & $\\pm$ & $R^2$ & $\\pm$ & DM $p$ \\\\\n\\midrule\n"
        )
        for name in MODEL_NAMES:
            a = agg[name]
            dm = dm_results.get(name, {})
            dm_p = dm.get("p_val", float("nan"))
            star = (
                "$^{***}$"
                if dm_p < 0.01
                else "$^{**}$" if dm_p < 0.05 else "$^{*}$" if dm_p < 0.10 else ""
            )
            bold_open = "\\textbf{" if name == "SC-STGCN" else ""
            bold_cls = "}" if name == "SC-STGCN" else ""
            f.write(
                f"{bold_open}{name}{bold_cls} & "
                f"{a['rmse_mean']:.4f} & {a['rmse_std']:.4f} & "
                f"{a['mae_mean']:.4f} & {a['mae_std']:.4f} & "
                f"{a['r2_mean']:+.4f} & {a['r2_std']:.4f} & "
                f"{dm_p:.3f}{star} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

        # Tablo 2
        f.write("% TABLE 2 — CLASSIFICATION\n")
        f.write("\\begin{table}[ht]\n\\centering\n")
        f.write("\\caption{Classification results}\n")
        f.write("\\label{tab:classification}\n")
        f.write("\\begin{tabular}{lcccccccc}\n\\toprule\n")
        f.write(
            "Model & Acc & $\\pm$ & Prec & $\\pm$ & Recall & $\\pm$ & F1 & $\\pm$ \\\\\n\\midrule\n"
        )
        for name in MODEL_NAMES:
            a = agg[name]
            bold_open = "\\textbf{" if name == "SC-STGCN" else ""
            bold_cls = "}" if name == "SC-STGCN" else ""
            f.write(
                f"{bold_open}{name}{bold_cls} & "
                f"{a['accuracy_mean']:.4f} & {a['accuracy_std']:.4f} & "
                f"{a['precision_mean']:.4f} & {a['precision_std']:.4f} & "
                f"{a['recall_mean']:.4f} & {a['recall_std']:.4f} & "
                f"{a['f1_mean']:.4f} & {a['f1_std']:.4f} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

        # Tablo 3
        f.write("% TABLE 3 — PORTFOLIO\n")
        f.write("\\begin{table}[ht]\n\\centering\n")
        f.write("\\caption{Portfolio results (DV01=\\$100, top-20\\% long-short)}\n")
        f.write("\\label{tab:portfolio}\n")
        f.write("\\begin{tabular}{lcccccr}\n\\toprule\n")
        f.write(
            "Model & Sharpe & $\\pm$ & Mean Ret (bp) & Std (bp) & Max DD & Hit \\\\\n\\midrule\n"
        )
        for name in MODEL_NAMES:
            a = agg[name]
            bold_open = "\\textbf{" if name == "SC-STGCN" else ""
            bold_cls = "}" if name == "SC-STGCN" else ""
            f.write(
                f"{bold_open}{name}{bold_cls} & "
                f"{a['sharpe_mean']:.4f} & {a['sharpe_std']:.4f} & "
                f"{a['mean_ret_bp_mean']:.4f} & "
                f"{a['std_ret_bp_mean']:.4f} & "
                f"{a['max_drawdown_mean']:.4f} & "
                f"{a['hit_ratio_mean']:.4f} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"\n  [OK] full_cv_results.csv")
    print(f"  [OK] full_cv_tables.tex")
    print(f"\n{SEP}  TAMAMLANDI\n{SEP}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"  Total: {(time.time()-t0)/60:.1f} min")
