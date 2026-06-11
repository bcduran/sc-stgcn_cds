"""
src/models/precovid_analysis.py
================================
Pre-COVID dönemi analizi (2015-01-01 → 2020-02-28)

COVID dönemini exclude ederek:
  1. Veri istatistikleri (kurtosis, ACF vs full sample)
  2. Walk-forward CV (2 fold — pre-COVID içinde)
  3. Event study (pre-COVID şokları)
  4. Full sample ile karşılaştırma tablosu

Calistirmak:
  python src/models/precovid_analysis.py --epochs 300 --device cpu
"""

from __future__ import annotations
import argparse, random, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from scipy.sparse import load_npz
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import (
    TOP50_WEEKLY_CSV, ADJ_NPZ, ADJ_SUP_NPZ, ADJ_CUS_NPZ,
    METRICS_DIR, FIG_DIR, make_dirs,
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
SEED      = 42
N_HIS     = 8
COVID_WEEK = None   # otomatik hesaplanacak (~265. hafta civarı)

GCN_H     = 64
ATT_H     = 64
ATT_HEADS = 2
FF_H      = 128
DROPOUT   = 0.30
LR        = 1e-4
WD        = 1e-3
BATCH     = 64
PATIENCE  = 40
CLIP      = 5.0
TOP_Q     = 0.20
DV01      = 100.0
INIT_CASH = 100_000.0


# =============================================================================
# UTILS
# =============================================================================
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def norm_adj(A):
    A = A + np.eye(A.shape[0], dtype=np.float32)
    d = A.sum(1) ** -0.5
    return torch.tensor(np.diag(d) @ A @ np.diag(d), dtype=torch.float32)

def make_windows(data, n_his):
    X, Y = [], []
    for t in range(n_his, len(data)):
        X.append(data[t-n_his:t]); Y.append(data[t])
    return np.array(X, np.float32), np.array(Y, np.float32)

def calc_metrics(yt, yp):
    rmse = float(np.sqrt(np.mean((yt-yp)**2)))
    mae  = float(np.mean(np.abs(yt-yp)))
    ss_r = np.sum((yt-yp)**2); ss_t = np.sum((yt-yt.mean())**2)
    r2   = float(1-ss_r/ss_t) if ss_t > 1e-10 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}

def backtest(yt, yp, q=TOP_Q, dv01=DV01, cash=INIT_CASH):
    T, N = yt.shape; K = max(1, int(q*N)); pnl = []
    for t in range(T):
        r = np.argsort(yp[t]); w = np.zeros(N)
        w[r[-K:]] = +1/K; w[r[:K]] = -1/K
        pnl.append(dv01*(w*yt[t]).sum())
    pnl = np.array(pnl); eq = cash + np.cumsum(pnl)
    return {
        "bt_sharpe":    float(np.sqrt(52)*pnl.mean()/(pnl.std()+1e-10)),
        "hit_ratio":    float((pnl>0).mean()),
        "max_drawdown": float(np.min(eq/np.maximum.accumulate(eq)-1)),
        "total_return_pct": float((eq[-1]-cash)/cash*100),
    }


# =============================================================================
# MODELS
# =============================================================================
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
        self.ff   = nn.Sequential(nn.Linear(d,ff), nn.GELU(), nn.Dropout(drop), nn.Linear(ff,d))
        self.n1   = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
    def forward(self, x):
        a, _ = self.attn(x,x,x); x = self.n1(x+a)
        return self.n2(x + self.ff(x))

class SCSTGCN(nn.Module):
    def __init__(self, N, n_his, gcn_h, att_h, att_heads, ff_h, drop, A_up, A_dn):
        super().__init__()
        self.register_buffer("A_up", A_up)
        self.register_buffer("A_dn", A_dn)
        self.gcn_up = GCNLayer(n_his, gcn_h)
        self.gcn_dn = GCNLayer(n_his, gcn_h)
        self.gate   = nn.Linear(gcn_h*2, gcn_h)
        self.proj   = nn.Linear(gcn_h, att_h)
        self.attn   = TemporalAttn(att_h, att_heads, ff_h, drop)
        self.head   = nn.Linear(att_h, 1)
        self.drop   = nn.Dropout(drop)
    def forward(self, x):
        xT   = x.permute(0,2,1)
        h_up = self.gcn_up(xT, self.A_up)
        h_dn = self.gcn_dn(xT, self.A_dn)
        g    = torch.sigmoid(self.gate(torch.cat([h_up,h_dn],-1)))
        h    = g*h_up + (1-g)*h_dn
        h    = self.drop(self.proj(h))
        return self.head(self.attn(h)).squeeze(-1)

class VSTGCN(nn.Module):
    def __init__(self, N, n_his, gcn_h, A):
        super().__init__()
        self.register_buffer("A", A)
        self.gcn  = GCNLayer(n_his, gcn_h)
        self.proj = nn.Linear(gcn_h, gcn_h)
        self.head = nn.Linear(gcn_h, 1)
    def forward(self, x):
        h = self.gcn(x.permute(0,2,1), self.A)
        return self.head(torch.relu(self.proj(h))).squeeze(-1)

class LSTMModel(nn.Module):
    def __init__(self, N, h=128, layers=2, drop=0.20):
        super().__init__()
        self.lstm = nn.LSTM(N, h, layers, batch_first=True,
                            dropout=drop if layers>1 else 0.0)
        self.head = nn.Linear(h, N)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:,-1,:])


# =============================================================================
# TRAINING
# =============================================================================
def train_nn(model, Xtr, Ytr, Xva, Yva, device,
             lr=LR, wd=WD, epochs=300, batch=BATCH,
             patience=PATIENCE, clip=CLIP):
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)
    lfn   = nn.HuberLoss(delta=1.0)
    Xt = torch.tensor(Xtr).to(device); Yt = torch.tensor(Ytr).to(device)
    Xv = torch.tensor(Xva).to(device); Yv = torch.tensor(Yva).to(device)
    best, no_imp, state = float("inf"), 0, None
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(len(Xt), device=device)
        for i in range(0, len(idx), batch):
            b = idx[i:i+batch]; opt.zero_grad()
            lfn(model(Xt[b]), Yt[b]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad(): vl = lfn(model(Xv), Yv).item()
        if vl < best-1e-6:
            best, no_imp = vl, 0
            state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= patience: break
    if state: model.load_state_dict(state)
    return model, ep+1

def pred_nn(model, X, device):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X).to(device)).cpu().numpy()


# =============================================================================
# RUN ONE SPLIT
# =============================================================================
def run_split(delta, n_firms, Ac_t, Au_t, Ad_t, device, epochs,
              train_end, val_end, test_end, label=""):
    tr_r = delta[:train_end]
    va_r = delta[train_end:val_end]
    te_r = delta[val_end:test_end]

    sc = StandardScaler()
    tr_s = sc.fit_transform(tr_r)
    va_s = sc.transform(va_r)
    te_s = sc.transform(te_r)

    Xtr, Ytr = make_windows(tr_s, N_HIS)
    Xva, Yva = make_windows(va_s, N_HIS)
    Xte, Yte = make_windows(te_s, N_HIS)
    te_orig  = te_r[N_HIS:]

    results = {}

    # AR(1)
    ps = np.zeros_like(te_s)
    for i in range(n_firms):
        y = tr_s[:,i]; phi = (np.corrcoef(y[:-1],y[1:])[0,1] if y.std()>1e-8 else 0.0)
        mu = y.mean()*(1-phi); last = tr_s[-1,i]
        for t in range(len(te_s)):
            ps[t,i] = mu+phi*last; last = te_s[t,i]
    ar1p = sc.inverse_transform(ps)[N_HIS:]
    results["AR(1)"] = {**calc_metrics(te_orig, ar1p), **backtest(te_orig, ar1p)}

    # LSTM
    lstm = LSTMModel(n_firms).to(device)
    lstm, ep = train_nn(lstm, Xtr, Ytr, Xva, Yva, device, epochs=epochs)
    lstmp = sc.inverse_transform(pred_nn(lstm, Xte, device))
    results["LSTM"] = {**calc_metrics(te_orig, lstmp), **backtest(te_orig, lstmp)}

    # XGBoost
    if HAS_XGB:
        Xtf = Xtr.reshape(len(Xtr),-1); Xtef = Xte.reshape(len(Xte),-1)
        ps_x = np.zeros((len(Xte), n_firms))
        for i in range(n_firms):
            m = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             objective="reg:squarederror", random_state=SEED,
                             tree_method="hist", n_jobs=-1, verbosity=0)
            m.fit(Xtf, Ytr[:,i]); ps_x[:,i] = m.predict(Xtef)
        xgbp = sc.inverse_transform(ps_x)
        results["XGBoost"] = {**calc_metrics(te_orig, xgbp), **backtest(te_orig, xgbp)}

    # V-STGCN
    vs = VSTGCN(n_firms, N_HIS, GCN_H, Ac_t).to(device)
    vs, ep = train_nn(vs, Xtr, Ytr, Xva, Yva, device, epochs=epochs)
    vsp = sc.inverse_transform(pred_nn(vs, Xte, device))
    results["V-STGCN"] = {**calc_metrics(te_orig, vsp), **backtest(te_orig, vsp)}

    # SC-STGCN
    sc_m = SCSTGCN(n_firms, N_HIS, GCN_H, ATT_H, ATT_HEADS, FF_H,
                   DROPOUT, Au_t, Ad_t).to(device)
    sc_m, ep = train_nn(sc_m, Xtr, Ytr, Xva, Yva, device, epochs=epochs)
    scp = sc.inverse_transform(pred_nn(sc_m, Xte, device))
    results["SC-STGCN"] = {**calc_metrics(te_orig, scp), **backtest(te_orig, scp)}

    return results, te_orig


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", default="auto")
    args   = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)

    make_dirs(); set_seed()

    SEP = "=" * 65
    print(SEP)
    print("  Pre-COVID vs Full Sample Analizi")
    print(SEP)

    # Veri yükle
    df    = pd.read_csv(TOP50_WEEKLY_CSV)
    level = df.values.astype(np.float32)
    delta = np.diff(level, axis=0)
    T, N  = delta.shape

    # Tarih index'i oluştur
    dates = pd.date_range(start="2015-01-09", periods=T, freq="W-FRI")
    covid_start = pd.Timestamp("2020-03-01")
    covid_week  = int(np.searchsorted(dates, covid_start))
    print(f"\n  Full delta: {T} hafta ({dates[0].date()} → {dates[-1].date()})")
    print(f"  COVID başlangıcı: ~hafta {covid_week} ({dates[covid_week].date()})")
    print(f"  Pre-COVID delta : {covid_week} hafta")

    # Adjacency
    Ac = load_npz(ADJ_NPZ).toarray().astype(np.float32)
    Au = load_npz(ADJ_SUP_NPZ).toarray().astype(np.float32)
    Ad = load_npz(ADJ_CUS_NPZ).toarray().astype(np.float32)
    Ac_t = norm_adj(Ac).to(device)
    Au_t = norm_adj(Au).to(device)
    Ad_t = norm_adj(Ad).to(device)

    # ------------------------------------------------------------------
    # 1) İSTATİSTİKSEL KARŞILAŞTIRMA
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("  [1] Dağılım İstatistikleri: Pre-COVID vs Full Sample")
    print(SEP)

    delta_pre  = delta[:covid_week]
    delta_full = delta

    for label, d in [("Pre-COVID", delta_pre), ("Full Sample", delta_full)]:
        flat = d.flatten()
        print(f"\n  {label} ({len(d)} hafta):")
        print(f"    Mean      : {np.nanmean(flat):+.4f} bp")
        print(f"    Std       : {np.nanstd(flat):.4f} bp")
        print(f"    Skewness  : {stats.skew(flat):.4f}")
        print(f"    Kurtosis  : {stats.kurtosis(flat):.4f}")
        print(f"    |Δ| p95   : {np.percentile(np.abs(flat), 95):.2f} bp")
        print(f"    |Δ| p99   : {np.percentile(np.abs(flat), 99):.2f} bp")

        # Naive R² üst sınırı
        te_start = int(0.9 * len(d))
        te = d[te_start:]
        ss_tot   = np.sum((te - te.mean())**2)
        ss_naive = np.sum(te**2)
        naive_r2 = float(1 - ss_naive/ss_tot) if ss_tot > 0 else float("nan")
        print(f"    Naive R²  : {naive_r2:+.4f}")

        # Lag-1 ACF
        acfs = []
        for i in range(N):
            s = d[:,i]; s = s[~np.isnan(s)]
            if len(s) > 10 and s.std() > 1e-8:
                try:
                    r = np.corrcoef(s[:-1], s[1:])[0,1]
                    if not np.isnan(r): acfs.append(r)
                except: pass
        print(f"    Lag-1 ACF : {np.mean(acfs):+.4f}")
        print(f"    Teorik R² üst sınırı (φ²): {np.mean(acfs)**2:.4f}")

    # ------------------------------------------------------------------
    # 2) PRE-COVID WALK-FORWARD CV
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print(f"  [2] Pre-COVID Walk-Forward CV")
    print(f"  Pre-COVID veri: {covid_week} hafta")
    print(SEP)

    # Pre-COVID içinde 2 fold
    # Fold 1: train[0:160] val[160:195] test[195:230]
    # Fold 2: train[0:195] val[195:230] test[230:covid_week]
    folds_pre = [
        (int(covid_week*0.65), int(covid_week*0.78), int(covid_week*0.92)),
        (int(covid_week*0.78), int(covid_week*0.92), covid_week),
    ]

    all_results_pre = []
    model_names = ["AR(1)", "LSTM", "V-STGCN", "SC-STGCN"]
    if HAS_XGB: model_names.insert(2, "XGBoost")

    for fi, (tr_end, va_end, te_end) in enumerate(folds_pre):
        print(f"\n  Fold {fi+1}/2: train[0:{tr_end}] val[{tr_end}:{va_end}] test[{va_end}:{te_end}]")
        print(f"  Test dönemi: {dates[va_end].date()} → {dates[te_end-1].date()}")
        set_seed(SEED + fi)
        res, te_orig = run_split(
            delta[:covid_week], N, Ac_t, Au_t, Ad_t,
            device, args.epochs, tr_end, va_end, te_end
        )
        all_results_pre.append(res)
        print(f"  {'Model':<12} {'RMSE':>7} {'MAE':>7} {'R²':>8} {'Sharpe':>8} {'Hit':>7}")
        print(f"  {'-'*55}")
        for name, r in res.items():
            print(f"  {name:<12} {r['rmse']:>7.4f} {r['mae']:>7.4f} "
                  f"{r['r2']:>8.4f} {r['bt_sharpe']:>8.4f} {r['hit_ratio']:>7.4f}")

    # Aggregate pre-COVID
    print(f"\n  Pre-COVID Ortalama Sonuçlar (2 fold):")
    print(f"  {'Model':<12} {'RMSE':>8} {'MAE':>8} {'R²':>9} {'Sharpe':>9} {'Hit':>8}")
    print(f"  {'-'*60}")
    pre_agg = {}
    for name in model_names:
        vals = {m: [] for m in ["rmse","mae","r2","bt_sharpe","hit_ratio"]}
        for fr in all_results_pre:
            if name in fr:
                for m in vals: vals[m].append(fr[name][m])
        pre_agg[name] = {m: np.mean(v) for m,v in vals.items() if v}
        a = pre_agg[name]
        print(f"  {name:<12} {a.get('rmse',float('nan')):>8.4f} "
              f"{a.get('mae',float('nan')):>8.4f} "
              f"{a.get('r2',float('nan')):>9.4f} "
              f"{a.get('bt_sharpe',float('nan')):>9.4f} "
              f"{a.get('hit_ratio',float('nan')):>8.4f}")

    # ------------------------------------------------------------------
    # 3) FULL SAMPLE SONUÇLARI (karşılaştırma için)
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("  [3] Karşılaştırma: Pre-COVID vs Full Sample")
    print(SEP)

    # Full sample walk-forward CV sonuçları (önceden hesaplandı)
    full_csv = METRICS_DIR / "walk_forward_cv.csv"
    if full_csv.exists():
        df_full = pd.read_csv(full_csv)
        print(f"\n  Full Sample (3 fold, önceki run):")
        print(f"  {'Model':<12} {'RMSE':>8} {'MAE':>8} {'R²':>9} {'Sharpe':>9} {'Hit':>8}")
        print(f"  {'-'*60}")
        for _, row in df_full.iterrows():
            print(f"  {row['Model']:<12} {row['RMSE_mean']:>8.4f} "
                  f"{row['MAE_mean']:>8.4f} "
                  f"{row['R2_mean']:>9.4f} "
                  f"{row['Sharpe_mean']:>9.4f} "
                  f"{row['Hit_mean']:>8.4f}")

    print(f"\n  Pre-COVID (2 fold, bu run):")
    print(f"  {'Model':<12} {'RMSE':>8} {'MAE':>8} {'R²':>9} {'Sharpe':>9} {'Hit':>8}")
    print(f"  {'-'*60}")
    for name in model_names:
        a = pre_agg.get(name, {})
        print(f"  {name:<12} {a.get('rmse',float('nan')):>8.4f} "
              f"{a.get('mae',float('nan')):>8.4f} "
              f"{a.get('r2',float('nan')):>9.4f} "
              f"{a.get('bt_sharpe',float('nan')):>9.4f} "
              f"{a.get('hit_ratio',float('nan')):>8.4f}")

    # ------------------------------------------------------------------
    # 4) GÖRSEL
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: Kurtosis karşılaştırması
    ax = axes[0]
    kurt_pre  = stats.kurtosis(delta[:covid_week].flatten())
    kurt_full = stats.kurtosis(delta.flatten())
    ax.bar(["Pre-COVID", "Full Sample"], [kurt_pre, kurt_full],
           color=["#2196F3", "#E53935"], alpha=0.8)
    ax.set_title("Excess Kurtosis\n(lower = more normal)", fontsize=11)
    ax.set_ylabel("Kurtosis")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate([kurt_pre, kurt_full]):
        ax.text(i, v+5, f"{v:.0f}", ha="center", fontsize=10, fontweight="bold")

    # Panel B: Sharpe karşılaştırması
    ax = axes[1]
    if full_csv.exists():
        df_full = pd.read_csv(full_csv)
        full_sharpe = {row["Model"]: row["Sharpe_mean"] for _, row in df_full.iterrows()}
    else:
        full_sharpe = {}

    pre_sharpe = {n: pre_agg[n].get("bt_sharpe", 0) for n in model_names}
    x = np.arange(len(model_names)); width = 0.35
    ax.bar(x - width/2, [full_sharpe.get(n, 0) for n in model_names],
           width, label="Full Sample", color="#E53935", alpha=0.7)
    ax.bar(x + width/2, [pre_sharpe.get(n, 0) for n in model_names],
           width, label="Pre-COVID",   color="#2196F3", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(model_names, rotation=20, fontsize=9)
    ax.set_title("Backtest Sharpe\nFull vs Pre-COVID", fontsize=11)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    # Panel C: R² karşılaştırması
    ax = axes[2]
    full_r2 = {row["Model"]: row["R2_mean"]
               for _, row in df_full.iterrows()} if full_csv.exists() else {}
    pre_r2  = {n: pre_agg[n].get("r2", 0) for n in model_names}
    ax.bar(x - width/2, [full_r2.get(n, 0) for n in model_names],
           width, label="Full Sample", color="#E53935", alpha=0.7)
    ax.bar(x + width/2, [pre_r2.get(n, 0) for n in model_names],
           width, label="Pre-COVID",   color="#2196F3", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(model_names, rotation=20, fontsize=9)
    ax.set_title("R² (Mean across folds)\nFull vs Pre-COVID", fontsize=11)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Pre-COVID vs Full Sample: Model Performance Comparison",
                 fontsize=13)
    plt.tight_layout()
    out = FIG_DIR / "fig_precovid_comparison.png"
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"\n[fig] {out.name}")

    # Kaydet
    rows = []
    for name in model_names:
        a = pre_agg.get(name, {})
        rows.append({"Model": name, "Sample": "Pre-COVID",
                     "RMSE": a.get("rmse"), "MAE": a.get("mae"),
                     "R2": a.get("r2"), "Sharpe": a.get("bt_sharpe"),
                     "Hit": a.get("hit_ratio")})
    pd.DataFrame(rows).to_csv(METRICS_DIR / "precovid_results.csv", index=False)

    print(f"\n[DONE] precovid_analysis.py tamamlandi.")
    print(f"  Pre-COVID kurtosis : {kurt_pre:.1f}")
    print(f"  Full kurtosis      : {kurt_full:.1f}")
    print(f"  Fark               : {kurt_full - kurt_pre:.1f} (COVID etkisi)")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"  Total: {(time.time()-t0)/60:.1f} min")
