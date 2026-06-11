"""
src/models/sc_stgcn_full.py
============================
Amac:
  661 ticker (full universe) uzerinde SC-STGCN ve baseline modelleri calistir.
  Top-50 ile karsilastir, R² ve diger metrikleri raporla.

Girdi:
  data/full/ve1_full.csv
  data/full/adj_full.npz
  data/full/adj_sup_full.npz
  data/full/adj_cus_full.npz

Cikti:
  outputs/metrics/results_full_delta.csv
  outputs/figures/fig_full_*.png

Calistirmak:
  python src/models/sc_stgcn_full.py --epochs 300 --device cpu
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.sparse import load_npz
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import (
    METRICS_DIR, FIG_DIR, CKPT_DIR, PRED_DIR, make_dirs,
    VE1_FULL_CSV, ADJ_FULL_NPZ, ADJ_SUP_FULL_NPZ, ADJ_CUS_FULL_NPZ, FULL_DIR,
)

ADJ_FULL     = ADJ_FULL_NPZ
ADJ_SUP_FULL = ADJ_SUP_FULL_NPZ
ADJ_CUS_FULL = ADJ_CUS_FULL_NPZ

# Hyperparameters (grid search sonucu)
N_HIS     = 8
SPLIT     = (0.80, 0.10, 0.10)
GCN_H     = 64
ATT_H     = 64
FF_H      = 128
ATT_HEADS = 2
DROPOUT   = 0.30
LR        = 1e-4
WD        = 1e-3
BATCH     = 64
PATIENCE  = 60
SEED      = 42


# =============================================================================
# MODEL
# =============================================================================
def _norm_adj(A: np.ndarray) -> torch.Tensor:
    A = A + np.eye(A.shape[0], dtype=np.float32)
    d = A.sum(1) ** -0.5
    D = np.diag(d)
    return torch.tensor(D @ A @ D, dtype=torch.float32)


class GCNLayer(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.W = nn.Linear(in_f, out_f, bias=False)

    def forward(self, x, A):
        return torch.relu(self.W(A @ x))


class TemporalAttn(nn.Module):
    def __init__(self, d, heads, ff, drop):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(d, ff), nn.GELU(), nn.Dropout(drop), nn.Linear(ff, d))
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)

    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x = self.n1(x + a)
        return self.n2(x + self.ff(x))


class SCSTGCNFull(nn.Module):
    def __init__(self, N, n_his, gcn_h, att_h, att_heads, ff_h,
                 drop, A_up, A_dn):
        super().__init__()
        self.register_buffer("A_up", A_up)
        self.register_buffer("A_dn", A_dn)
        self.gcn_up = GCNLayer(n_his, gcn_h)
        self.gcn_dn = GCNLayer(n_his, gcn_h)
        self.gate   = nn.Linear(gcn_h * 2, gcn_h)
        self.proj   = nn.Linear(gcn_h, att_h)
        self.attn   = TemporalAttn(att_h, att_heads, ff_h, drop)
        self.head   = nn.Linear(att_h, 1)
        self.drop   = nn.Dropout(drop)

    def forward(self, x):
        B, T, N = x.shape
        xT   = x.permute(0, 2, 1)
        h_up = self.gcn_up(xT, self.A_up)
        h_dn = self.gcn_dn(xT, self.A_dn)
        g    = torch.sigmoid(self.gate(torch.cat([h_up, h_dn], -1)))
        h    = g * h_up + (1 - g) * h_dn
        h    = self.drop(self.proj(h))
        h    = self.attn(h)
        return self.head(h).squeeze(-1)


# =============================================================================
# YARDIMCILAR
# =============================================================================
def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_windows(data, n_his):
    X, Y = [], []
    for t in range(n_his, len(data)):
        X.append(data[t - n_his:t])
        Y.append(data[t])
    return np.array(X, np.float32), np.array(Y, np.float32)


def compute_metrics(y_true, y_pred, tag=""):
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    # Aggregate R²
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2_agg = float(1 - ss_res / ss_tot) if ss_tot > 1e-10 else float("nan")
    # Firm-by-firm R² mean
    r2_firms = []
    for i in range(y_true.shape[1]):
        yt, yp = y_true[:, i], y_pred[:, i]
        ss_r = np.sum((yt - yp)**2)
        ss_t = np.sum((yt - yt.mean())**2)
        if ss_t > 1e-10:
            r2_firms.append(1 - ss_r/ss_t)
    r2_mean = float(np.mean(r2_firms)) if r2_firms else float("nan")
    r2_pos  = sum(1 for r in r2_firms if r > 0)
    return {
        "rmse": rmse, "mae": mae,
        "r2_agg": r2_agg, "r2_mean_firm": r2_mean,
        "r2_positive_firms": r2_pos,
        "n_firms": y_true.shape[1]
    }


def train_model(model, X_tr, Y_tr, X_va, Y_va, device,
                epochs=300, batch=BATCH, lr=LR, wd=WD, patience=PATIENCE):
    opt     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr*0.01)
    loss_fn = nn.HuberLoss(delta=1.0)

    X_tr_t = torch.tensor(X_tr).to(device)
    Y_tr_t = torch.tensor(Y_tr).to(device)
    X_va_t = torch.tensor(X_va).to(device)
    Y_va_t = torch.tensor(Y_va).to(device)

    best_val = float("inf")
    no_impr  = 0
    best_state = None

    for ep in range(epochs):
        model.train()
        idx = torch.randperm(len(X_tr_t), device=device)
        for i in range(0, len(idx), batch):
            b = idx[i:i+batch]
            opt.zero_grad()
            loss_fn(model(X_tr_t[b]), Y_tr_t[b]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_va_t), Y_va_t).item()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            no_impr  = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_impr += 1
            if no_impr >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


# =============================================================================
# ANA AKIS
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",  type=int,   default=300)
    parser.add_argument("--device",  type=str,   default="cpu")
    args   = parser.parse_args()
    device = torch.device(args.device)

    make_dirs()
    set_seed()

    print("=" * 65)
    print("  SC-STGCN Full Universe (661 tickers)")
    print("=" * 65)

    # 1) Veri
    print("\n[1/4] Veri yukleniyor...")
    if not VE1_FULL_CSV.exists():
        print("  [ERROR] ve1_full.csv bulunamadi.")
        print("  Once calistir: python src/pipeline/08_full_universe_panel.py")
        return

    df    = pd.read_csv(VE1_FULL_CSV)
    level = df.values.astype(np.float32)
    delta = np.diff(level, axis=0)
    T, N  = delta.shape
    print(f"  Level: {level.shape}  Delta: {delta.shape}")
    print(f"  Tickers: {N}")

    # Split
    n_tr = int(SPLIT[0] * T)
    n_va = int(SPLIT[1] * T)
    tr_raw, va_raw, te_raw = delta[:n_tr], delta[n_tr:n_tr+n_va], delta[n_tr+n_va:]
    print(f"  Train={tr_raw.shape[0]} Val={va_raw.shape[0]} Test={te_raw.shape[0]}")

    sc = StandardScaler()
    tr_s = sc.fit_transform(tr_raw)
    va_s = sc.transform(va_raw)
    te_s = sc.transform(te_raw)

    X_tr, Y_tr = make_windows(tr_s, N_HIS)
    X_va, Y_va = make_windows(va_s, N_HIS)
    X_te, Y_te = make_windows(te_s, N_HIS)

    # 2) Adjacency
    print("\n[2/4] Adjacency matrisler yukleniyor...")
    if not ADJ_FULL.exists():
        print("  [ERROR] adj_full.npz bulunamadi.")
        print("  Once calistir: python src/pipeline/09_full_adjacency.py")
        return

    A_comb = load_npz(ADJ_FULL).toarray().astype(np.float32)
    A_up   = load_npz(ADJ_SUP_FULL).toarray().astype(np.float32)
    A_dn   = load_npz(ADJ_CUS_FULL).toarray().astype(np.float32)

    A_up_t   = _norm_adj(A_up).to(device)
    A_dn_t   = _norm_adj(A_dn).to(device)
    A_comb_t = _norm_adj(A_comb).to(device)

    density = (A_comb > 0).sum() / (N * N)
    print(f"  Shape: {A_comb.shape}  Density: {density:.4%}")
    connected = int(((A_comb.sum(1) > 0) | (A_comb.sum(0) > 0)).sum())
    print(f"  Baglantili: {connected}/{N}")

    results = {}

    # 3) AR(1)
    print("\n[3/4] Modeller egitiliyor...")
    print("\n  [1/3] AR(1)...")
    Y_pred_ar1 = np.zeros_like(Y_te)
    for i in range(N):
        x = tr_s[:, i]
        phi = np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 1 else 0
        mu  = x.mean() * (1 - phi)
        last = va_s[-1, i]
        for t in range(len(Y_te)):
            pred = mu + phi * last
            Y_pred_ar1[t, i] = pred
            last = te_s[t, i] if t < len(te_s) else pred

    Y_te_orig    = sc.inverse_transform(Y_te)
    Y_ar1_orig   = sc.inverse_transform(Y_pred_ar1)
    m = compute_metrics(Y_te_orig, Y_ar1_orig)
    results["AR(1)"] = m
    print(f"    RMSE={m['rmse']:.3f}  R²_agg={m['r2_agg']:+.4f}  "
          f"R²_mean={m['r2_mean_firm']:+.4f}  pos={m['r2_positive_firms']}/{N}")

    # 4) SC-STGCN
    print("\n  [2/3] SC-STGCN...")
    model = SCSTGCNFull(
        N=N, n_his=N_HIS, gcn_h=GCN_H, att_h=ATT_H,
        att_heads=ATT_HEADS, ff_h=FF_H, drop=DROPOUT,
        A_up=A_up_t, A_dn=A_dn_t
    ).to(device)

    model = train_model(model, X_tr, Y_tr, X_va, Y_va, device,
                        epochs=args.epochs)

    model.eval()
    with torch.no_grad():
        Y_pred_sc = model(torch.tensor(X_te).to(device)).cpu().numpy()

    Y_sc_orig = sc.inverse_transform(Y_pred_sc)
    m = compute_metrics(Y_te_orig, Y_sc_orig)
    results["SC-STGCN"] = m
    print(f"    RMSE={m['rmse']:.3f}  R²_agg={m['r2_agg']:+.4f}  "
          f"R²_mean={m['r2_mean_firm']:+.4f}  pos={m['r2_positive_firms']}/{N}")

    # 5) V-STGCN (ablation)
    print("\n  [3/3] V-STGCN...")

    class VSTGCN(nn.Module):
        def __init__(self, N, n_his, gcn_h, A):
            super().__init__()
            self.register_buffer("A", A)
            self.gcn  = GCNLayer(n_his, gcn_h)
            self.proj = nn.Linear(gcn_h, gcn_h)
            self.head = nn.Linear(gcn_h, 1)

        def forward(self, x):
            B, T, N = x.shape
            h = self.gcn(x.permute(0,2,1), self.A)
            h = torch.relu(self.proj(h))
            return self.head(h).squeeze(-1)

    vmodel = VSTGCN(N, N_HIS, GCN_H, A_comb_t).to(device)
    vmodel = train_model(vmodel, X_tr, Y_tr, X_va, Y_va, device,
                         epochs=args.epochs)

    vmodel.eval()
    with torch.no_grad():
        Y_pred_vs = vmodel(torch.tensor(X_te).to(device)).cpu().numpy()

    Y_vs_orig = sc.inverse_transform(Y_pred_vs)
    m = compute_metrics(Y_te_orig, Y_vs_orig)
    results["V-STGCN"] = m
    print(f"    RMSE={m['rmse']:.3f}  R²_agg={m['r2_agg']:+.4f}  "
          f"R²_mean={m['r2_mean_firm']:+.4f}  pos={m['r2_positive_firms']}/{N}")

    # Sonuclar
    print(f"\n{'='*65}")
    print("  FULL UNIVERSE RESULTS")
    print(f"{'='*65}")
    print(f"  {'Model':<12} {'RMSE':>8} {'MAE':>8} {'R²_agg':>9} "
          f"{'R²_mean':>9} {'R²_pos':>8}")
    print(f"  {'-'*60}")
    for name, m in results.items():
        print(f"  {name:<12} {m['rmse']:>8.4f} {m['mae']:>8.4f} "
              f"{m['r2_agg']:>+9.4f} {m['r2_mean_firm']:>+9.4f} "
              f"{m['r2_positive_firms']:>5}/{m['n_firms']}")

    # Kaydet
    rows = [{"Model": k, **v} for k, v in results.items()]
    out_csv = METRICS_DIR / "results_full_delta.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n  [OK] {out_csv}")
    print(f"\n[DONE] sc_stgcn_full.py tamamlandi.")


if __name__ == "__main__":
    main()
