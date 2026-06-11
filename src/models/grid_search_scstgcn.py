"""
src/models/grid_search_scstgcn.py
===================================
SC-STGCN icin tam kapsamli grid search.

Arama uzayi:
  N_HIS     : [8, 12, 16]
  GCN_H     : [64, 128, 256]
  ATT_H     : [64, 128]
  FF_H      : [128, 256]
  DROPOUT   : [0.10, 0.20, 0.30]
  LR        : [1e-3, 5e-4, 1e-4]
  WD        : [1e-4, 1e-3]
  ATT_HEADS : [2, 4]

Toplam: 3x3x2x2x3x3x2x2 = 648 kombinasyon
Her trial ~1-2 dk -> toplam ~10-20 saat (CPU)

Calistirmak:
  python src/models/grid_search_scstgcn.py --device cpu
  python src/models/grid_search_scstgcn.py --device cuda  (GPU varsa)

Sonuclar: outputs/metrics/grid_search_results.csv
          outputs/metrics/grid_search_best.json

Sonra en iyi parametreler otomatik sc_stgcn_train.py'e yazilir.
"""

from __future__ import annotations
import argparse
import json
import sys
import time
import itertools
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from scipy.sparse import load_npz

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import (
    TOP50_WEEKLY_CSV, ADJ_NPZ, ADJ_SUP_NPZ, ADJ_CUS_NPZ,
    METRICS_DIR, make_dirs,
)

# =============================================================================
# GRID TANIMI
# =============================================================================
GRID = {
    "n_his"    : [8, 12, 16],
    "gcn_h"    : [64, 128, 256],
    "att_h"    : [64, 128],
    "ff_h"     : [128, 256],
    "dropout"  : [0.10, 0.20, 0.30],
    "lr"       : [1e-3, 5e-4, 1e-4],
    "wd"       : [1e-4, 1e-3],
    "att_heads": [2, 4],
}

# Sabit egitim parametreleri
EPOCHS   = 200
BATCH    = 32
PATIENCE = 25

# =============================================================================
# MODEL
# =============================================================================
class GCNLayer(nn.Module):
    def __init__(self, in_f: int, out_f: int):
        super().__init__()
        self.W = nn.Linear(in_f, out_f, bias=False)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.W(A @ x))


class TemporalAttn(nn.Module):
    def __init__(self, d: int, heads: int, ff: int, drop: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(d, ff), nn.GELU(), nn.Dropout(drop), nn.Linear(ff, d)
        )
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(x, x, x)
        x = self.n1(x + a)
        return self.n2(x + self.ff(x))


class SCSTGCNTunable(nn.Module):
    def __init__(self, N, n_his, gcn_h, att_h, att_heads, ff_h, drop, A_up, A_dn):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
def norm_adj(path, N: int) -> torch.Tensor:
    A = load_npz(path).toarray().astype(np.float32)
    A = A + np.eye(N, dtype=np.float32)
    d = A.sum(1) ** -0.5
    D = np.diag(d)
    return torch.tensor(D @ A @ D, dtype=torch.float32)


def make_windows(data: np.ndarray, n_his: int):
    X, Y = [], []
    for t in range(n_his, len(data)):
        X.append(data[t - n_his:t])
        Y.append(data[t])
    return np.array(X, np.float32), np.array(Y, np.float32)


def load_data(n_his: int):
    df    = pd.read_csv(TOP50_WEEKLY_CSV)
    level = df.values.astype(np.float32)
    delta = np.diff(level, axis=0)
    T     = delta.shape[0]
    n_tr  = int(0.80 * T)
    n_va  = int(0.10 * T)
    sc    = StandardScaler()
    tr_s  = sc.fit_transform(delta[:n_tr])
    va_s  = sc.transform(delta[n_tr:n_tr + n_va])
    return make_windows(tr_s, n_his) + make_windows(va_s, n_his)


def load_adjs(device):
    A_comb = load_npz(ADJ_NPZ).toarray()
    N      = A_comb.shape[0]
    A_up   = norm_adj(ADJ_SUP_NPZ, N).to(device)
    A_dn   = norm_adj(ADJ_CUS_NPZ, N).to(device)
    return A_up, A_dn, N


def run_trial(params: dict, device: torch.device,
              A_up, A_dn, N) -> tuple[float, float]:
    """
    Tek bir hyperparameter seti icin egit ve val RMSE dondur.
    Donus: (val_rmse_scaled, egitim_suresi_sn)
    """
    n_his     = params["n_his"]
    gcn_h     = params["gcn_h"]
    att_h     = params["att_h"]
    ff_h      = params["ff_h"]
    dropout   = params["dropout"]
    lr        = params["lr"]
    wd        = params["wd"]
    att_heads = params["att_heads"]

    # att_h / att_heads bolunemiyorsa gecersiz
    if att_h % att_heads != 0:
        return float("inf"), 0.0

    # Veri
    X_tr, Y_tr, X_va, Y_va = load_data(n_his)
    X_tr_t = torch.tensor(X_tr).to(device)
    Y_tr_t = torch.tensor(Y_tr).to(device)
    X_va_t = torch.tensor(X_va).to(device)
    Y_va_t = torch.tensor(Y_va).to(device)

    # Model
    model = SCSTGCNTunable(
        N=N, n_his=n_his,
        gcn_h=gcn_h, att_h=att_h, att_heads=att_heads,
        ff_h=ff_h, drop=dropout,
        A_up=A_up, A_dn=A_dn,
    ).to(device)

    opt     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS, eta_min=lr * 0.01
    )
    loss_fn = nn.HuberLoss(delta=1.0)

    best_val = float("inf")
    no_impr  = 0
    t0       = time.time()

    for ep in range(EPOCHS):
        # Egitim
        model.train()
        idx = torch.randperm(len(X_tr_t), device=device)
        for i in range(0, len(idx), BATCH):
            b = idx[i:i + BATCH]
            opt.zero_grad()
            loss_fn(model(X_tr_t[b]), Y_tr_t[b]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

        # Validasyon
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_va_t), Y_va_t).item()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            no_impr  = 0
        else:
            no_impr += 1
            if no_impr >= PATIENCE:
                break

    elapsed = time.time() - t0

    # Val RMSE (scaled) hesapla
    model.eval()
    with torch.no_grad():
        pred = model(X_va_t).cpu().numpy()
    val_rmse = float(np.sqrt(np.mean((pred - Y_va) ** 2)))

    return val_rmse, elapsed


# =============================================================================
# ANA AKIS
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true",
                        help="Tamamlanan trial'lari atlayarak devam et")
    args   = parser.parse_args()
    device = torch.device(args.device)

    make_dirs()

    # Tum kombinasyonlari olustur
    keys   = list(GRID.keys())
    values = list(GRID.values())
    combos = list(itertools.product(*values))
    total  = len(combos)

    print("=" * 65)
    print("  SC-STGCN Grid Search")
    print("=" * 65)
    print(f"  Toplam kombinasyon : {total}")
    print(f"  Device             : {device}")
    print(f"  Her trial ~1-2 dk  -> toplam ~{total * 1}-{total * 2} dk")
    print(f"  Sonuc dosyasi      : grid_search_results.csv")
    print()

    # Devam modunda tamamlananlar
    results_path = METRICS_DIR / "grid_search_results.csv"
    done_params  = set()
    results      = []

    if args.resume and results_path.exists():
        df_done = pd.read_csv(results_path)
        results = df_done.to_dict("records")
        for r in results:
            key = tuple(r[k] for k in keys)
            done_params.add(key)
        print(f"  [RESUME] {len(done_params)} trial tamamlanmis, devam ediliyor...\n")

    # Adjacency matrislerini bir kez yukle
    A_up, A_dn, N = load_adjs(device)

    # Grid search
    for trial_idx, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        key    = combo

        if key in done_params:
            continue

        # att_h / att_heads gecerlilik
        if params["att_h"] % params["att_heads"] != 0:
            val_rmse = float("inf")
            elapsed  = 0.0
        else:
            val_rmse, elapsed = run_trial(params, device, A_up, A_dn, N)

        result = {**params, "val_rmse": round(val_rmse, 6),
                  "elapsed_s": round(elapsed, 1)}
        results.append(result)

        # Her trial sonrasi CSV'ye yaz (kesintiye karsi)
        pd.DataFrame(results).to_csv(results_path, index=False)

        # Progress
        completed = len(results)
        best_so_far = min(r["val_rmse"] for r in results if r["val_rmse"] != float("inf"))
        print(f"  [{completed:>3}/{total}] "
              f"n_his={params['n_his']:>2}  gcn_h={params['gcn_h']:>3}  "
              f"att_h={params['att_h']:>3}  ff_h={params['ff_h']:>3}  "
              f"drop={params['dropout']:.2f}  lr={params['lr']:.0e}  "
              f"wd={params['wd']:.0e}  heads={params['att_heads']}  "
              f"-> val_rmse={val_rmse:.5f}  "
              f"best={best_so_far:.5f}  ({elapsed:.0f}s)")

    # ==========================================================================
    # SONUCLAR
    # ==========================================================================
    df = pd.read_csv(results_path)
    df = df[df["val_rmse"] != float("inf")].sort_values("val_rmse")

    print(f"\n{'='*65}")
    print("  GRID SEARCH TAMAMLANDI")
    print(f"{'='*65}")
    print(f"\n  En iyi 10 kombinasyon:")
    print(f"  {'#':<4} {'val_rmse':>10}  {'n_his':>6} {'gcn_h':>6} "
          f"{'att_h':>6} {'ff_h':>5} {'drop':>5} {'lr':>8} {'wd':>8} {'heads':>6}")
    print(f"  {'-'*75}")
    for i, row in df.head(10).iterrows():
        print(f"  {df.index.get_loc(i)+1:<4} {row['val_rmse']:>10.5f}  "
              f"{int(row['n_his']):>6} {int(row['gcn_h']):>6} "
              f"{int(row['att_h']):>6} {int(row['ff_h']):>5} "
              f"{row['dropout']:>5.2f} {row['lr']:>8.1e} "
              f"{row['wd']:>8.1e} {int(row['att_heads']):>6}")

    best = df.iloc[0].to_dict()
    print(f"\n  EN IYI PARAMETRELER:")
    for k, v in best.items():
        if k not in ("val_rmse", "elapsed_s"):
            print(f"    {k:<12} = {v}")
    print(f"\n  Val RMSE: {best['val_rmse']:.5f}")

    # JSON kaydet
    best_json = {k: v for k, v in best.items() if k not in ("val_rmse","elapsed_s")}
    best_json["val_rmse"] = best["val_rmse"]
    best_path = METRICS_DIR / "grid_search_best.json"
    best_path.write_text(json.dumps(best_json, indent=2), encoding="utf-8")
    print(f"\n  [OK] {results_path.name}")
    print(f"  [OK] {best_path.name}")

    # sc_stgcn_train.py'i guncelle
    apply_best(best_json)


def apply_best(best: dict):
    """En iyi parametreleri sc_stgcn_train.py'e otomatik yazar."""
    model_path = Path(__file__).parent / "sc_stgcn_train.py"
    if not model_path.exists():
        print(f"  [WARN] {model_path} bulunamadi, manuel guncelleme gerekli.")
        return

    content = model_path.read_text(encoding="utf-8")

    param_map = {
        "n_his"    : ("N_HIS: int   =",           int(best["n_his"])),
        "gcn_h"    : ("SCSTGCN_GCN_H:    int   =", int(best["gcn_h"])),
        "att_h"    : ("SCSTGCN_ATT_H:    int   =", int(best["att_h"])),
        "ff_h"     : ("SCSTGCN_FF_H:     int   =", int(best["ff_h"])),
        "att_heads": ("SCSTGCN_ATT_HEADS: int  =", int(best["att_heads"])),
        "dropout"  : ("SCSTGCN_DROPOUT:  float =", float(best["dropout"])),
        "lr"       : ("SCSTGCN_LR:       float =", float(best["lr"])),
        "wd"       : ("SCSTGCN_WD:       float =", float(best["wd"])),
    }

    for key, (pattern, value) in param_map.items():
        content = re.sub(
            re.escape(pattern) + r"[^\n]+",
            f"{pattern} {value}  # grid-search tuned",
            content,
        )

    model_path.write_text(content, encoding="utf-8")
    print(f"\n  [OK] sc_stgcn_train.py guncellendi!")
    print(f"\n  Simdi calistir:")
    print(f"  python src/models/sc_stgcn_train.py --mode DELTA --epochs 500")


if __name__ == "__main__":
    main()
