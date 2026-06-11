"""
src/models/idiosyncratic_cv.py
================================
Idiosyncratic CDS Spread Tahmini

Yontem:
  1. Common factor cikar: epsilon_i(t) = Delta_s_i(t) - mean(Delta_s(t))
  2. Kalan idiosyncratic spread degisimini tahmin et
  3. Pre-COVID, 2-fold walk-forward CV

Motivasyon:
  - Market faktörü dominant olduğundan Δs tahmini zor
  - Supply chain sinyali idiosyncratic kanaldan geçiyor
  - Common factor çıkarıldıktan sonra SC-STGCN'in avantajı daha net görünmeli

Calistirmak:
  python src/models/idiosyncratic_cv.py --epochs 300 --device cpu
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

def remove_common_factor(delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Common factor cikar.
    common(t)    = cross-sectional mean of Delta_s(t)
    idio_i(t)    = Delta_s_i(t) - common(t)
    """
    common = delta.mean(axis=1, keepdims=True)   # (T, 1)
    idio   = delta - common                       # (T, N)
    return idio, common.flatten()

def calc_metrics(yt, yp):
    rmse = float(np.sqrt(np.mean((yt-yp)**2)))
    mae  = float(np.mean(np.abs(yt-yp)))
    ss_r = np.sum((yt-yp)**2); ss_t = np.sum((yt-yt.mean())**2)
    r2   = float(1-ss_r/ss_t) if ss_t > 1e-10 else float("nan")

    yt_dir = (yt.flatten() > 0).astype(int)
    yp_dir = (yp.flatten() > 0).astype(int)
    tp = int(np.sum((yt_dir==1)&(yp_dir==1)))
    tn = int(np.sum((yt_dir==0)&(yp_dir==0)))
    fp = int(np.sum((yt_dir==0)&(yp_dir==1)))
    fn = int(np.sum((yt_dir==1)&(yp_dir==0)))
    accuracy  = float((tp+tn)/(tp+tn+fp+fn+1e-10))
    precision = float(tp/(tp+fp+1e-10))
    recall    = float(tp/(tp+fn+1e-10))
    f1        = float(2*precision*recall/(precision+recall+1e-10))

    mask = np.abs(yt.flatten()) > 0.1
    mape = float(np.mean(np.abs(
        (yt.flatten()[mask]-yp.flatten()[mask])/yt.flatten()[mask]))*100) \
        if mask.sum() > 0 else float("nan")

    return {"rmse":rmse,"mae":mae,"r2":r2,"mape":mape,
            "accuracy":accuracy,"precision":precision,
            "recall":recall,"f1":f1}

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
# RUN FOLD
# =============================================================================
def run_fold(idio, n_firms, Ac_t, Au_t, Ad_t, device,
             epochs, tr_end, va_end, te_end):
    tr_r = idio[:tr_end]
    va_r = idio[tr_end:va_end]
    te_r = idio[va_end:te_end]

    sc = StandardScaler()
    tr_s = sc.fit_transform(tr_r)
    va_s = sc.transform(va_r)
    te_s = sc.transform(te_r)

    Xtr, Ytr = make_windows(tr_s, N_HIS)
    Xva, Yva = make_windows(va_s, N_HIS)
    Xte, Yte = make_windows(te_s, N_HIS)
    te_orig  = te_r[N_HIS:]   # bp scale idiosyncratic

    results = {}

    # AR(1) — idiosyncratic icin phi kucuk olur
    ps = np.zeros_like(te_s)
    for i in range(n_firms):
        y = tr_s[:,i]
        phi = (np.corrcoef(y[:-1],y[1:])[0,1] if y.std()>1e-8 else 0.0)
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
    print(f"    LSTM ep={ep}")

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
    print(f"    V-STGCN ep={ep}")

    # SC-STGCN
    sc_m = SCSTGCN(n_firms, N_HIS, GCN_H, ATT_H, ATT_HEADS, FF_H,
                   DROPOUT, Au_t, Ad_t).to(device)
    sc_m, ep = train_nn(sc_m, Xtr, Ytr, Xva, Yva, device, epochs=epochs)
    scp = sc.inverse_transform(pred_nn(sc_m, Xte, device))
    results["SC-STGCN"] = {**calc_metrics(te_orig, scp), **backtest(te_orig, scp)}
    print(f"    SC-STGCN ep={ep}")

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
    print("  Idiosyncratic CDS Spread CV — Pre-COVID Weekly (2015-2020)")
    print(f"  epochs={args.epochs} | device={device}")
    print(SEP)

    # Veri
    df    = pd.read_csv(TOP50_WEEKLY_CSV)
    level = df.values.astype(np.float32)
    delta = np.diff(level, axis=0)
    T, N  = delta.shape
    tickers = df.columns.tolist()

    # Pre-COVID
    dates = pd.date_range(start="2015-01-09", periods=T, freq="W-FRI")
    covid_week = int(np.searchsorted(dates, pd.Timestamp("2020-03-01")))
    delta_pre  = delta[:covid_week]

    # Common factor cikar
    idio, common = remove_common_factor(delta_pre)

    print(f"\n  Full delta   : {T} hafta")
    print(f"  Pre-COVID    : {covid_week} hafta")
    print(f"  Common factor statistikleri:")
    print(f"    Mean std   : {common.std():.4f} bp")
    print(f"    Range      : [{common.min():.2f}, {common.max():.2f}] bp")
    print(f"\n  Idiosyncratic statistikleri:")
    print(f"    Std        : {idio.std():.4f} bp  "
          f"(raw delta std: {delta_pre.std():.4f} bp)")
    print(f"    Kurtosis   : {stats.kurtosis(idio.flatten()):.1f}  "
          f"(raw: {stats.kurtosis(delta_pre.flatten()):.1f})")
    # ACF
    acfs_idio = []; acfs_raw = []
    for i in range(N):
        s = idio[:,i]; sr = delta_pre[:,i]
        if s.std()>1e-8:
            acfs_idio.append(np.corrcoef(s[:-1],s[1:])[0,1])
        if sr.std()>1e-8:
            acfs_raw.append(np.corrcoef(sr[:-1],sr[1:])[0,1])
    phi_idio = np.mean(acfs_idio); phi_raw = np.mean(acfs_raw)
    print(f"    Lag-1 ACF  : {phi_idio:+.4f}  (raw: {phi_raw:+.4f})")
    print(f"    Teorik R²  : {phi_idio**2:.4f}  (raw: {phi_raw**2:.4f})")

    # Adjacency
    Ac = load_npz(ADJ_NPZ).toarray().astype(np.float32)
    Au = load_npz(ADJ_SUP_NPZ).toarray().astype(np.float32)
    Ad = load_npz(ADJ_CUS_NPZ).toarray().astype(np.float32)
    Ac_t = norm_adj(Ac).to(device)
    Au_t = norm_adj(Au).to(device)
    Ad_t = norm_adj(Ad).to(device)

    # 2 fold — precovid_analysis ile ayni yapi
    folds = [
        (int(covid_week*0.65), int(covid_week*0.78), int(covid_week*0.92)),
        (int(covid_week*0.78), int(covid_week*0.92), covid_week),
    ]
    print(f"\n  Fold yapisi:")
    for i,(a,b,c) in enumerate(folds):
        print(f"    Fold {i+1}: train[0:{a}] val[{a}:{b}] test[{b}:{c}] "
              f"({c-b} hafta | {dates[b].date()} -> {dates[c-1].date()})")

    model_names = ["AR(1)", "LSTM", "V-STGCN", "SC-STGCN"]
    if HAS_XGB: model_names.insert(2, "XGBoost")

    all_results = []
    for fi, (tr_end, va_end, te_end) in enumerate(folds):
        print(f"\n  --- Fold {fi+1}/2 ---")
        set_seed(SEED + fi)
        res, te_orig = run_fold(
            idio, N, Ac_t, Au_t, Ad_t,
            device, args.epochs, tr_end, va_end, te_end
        )
        all_results.append(res)
        print(f"\n  Fold {fi+1} sonuclari (IDIOSYNCRATIC):")
        print(f"  {'Model':<12} {'RMSE':>7} {'MAE':>7} {'R2':>7} "
              f"{'Acc':>7} {'F1':>7} {'Sharpe':>8} {'Hit':>7}")
        print(f"  {'-'*70}")
        for name, r in res.items():
            print(f"  {name:<12} {r['rmse']:>7.4f} {r['mae']:>7.4f} "
                  f"{r['r2']:>7.4f} {r['accuracy']:>7.4f} {r['f1']:>7.4f} "
                  f"{r['bt_sharpe']:>8.4f} {r['hit_ratio']:>7.4f}")

    # Aggregate
    metrics_list = ["rmse","mae","r2","mape","accuracy","precision",
                    "recall","f1","bt_sharpe","hit_ratio",
                    "max_drawdown","total_return_pct"]
    agg = {}
    for name in model_names:
        vals = {m: [] for m in metrics_list}
        for fr in all_results:
            if name in fr:
                for m in metrics_list:
                    if m in fr[name]: vals[m].append(fr[name][m])
        agg[name] = {}
        for m in metrics_list:
            v = np.array([x for x in vals[m] if not np.isnan(x)])
            agg[name][f"{m}_mean"] = float(np.mean(v)) if len(v)>0 else float("nan")
            agg[name][f"{m}_std"]  = float(np.std(v))  if len(v)>0 else float("nan")

    print(f"\n{SEP}")
    print("  IDIOSYNCRATIC CV RESULTS (2 FOLDS)")
    print(SEP)
    print(f"\n  {'Model':<12} {'RMSE':>8} {'+-':>6} {'MAE':>8} {'+-':>6} "
          f"{'R2':>8} {'Acc':>7} {'F1':>7} {'Sharpe':>9} {'+-':>6} {'Hit':>7}")
    print(f"  {'-'*95}")
    for name in model_names:
        a = agg[name]
        print(f"  {name:<12} "
              f"{a['rmse_mean']:>8.4f} {a['rmse_std']:>6.4f} "
              f"{a['mae_mean']:>8.4f} {a['mae_std']:>6.4f} "
              f"{a['r2_mean']:>8.4f} "
              f"{a['accuracy_mean']:>7.4f} "
              f"{a['f1_mean']:>7.4f} "
              f"{a['bt_sharpe_mean']:>9.4f} {a['bt_sharpe_std']:>6.4f} "
              f"{a['hit_ratio_mean']:>7.4f}")

    # Ablation
    print(f"\n  Ablation: SC-STGCN vs V-STGCN (Idiosyncratic)")
    for m in ["rmse","mae","r2","accuracy","f1","bt_sharpe"]:
        sc_v = agg["SC-STGCN"][f"{m}_mean"]
        vs_v = agg["V-STGCN"][f"{m}_mean"]
        diff = sc_v - vs_v
        better = "SC better" if (m in ["r2","accuracy","f1","bt_sharpe"] and diff>0) or \
                                (m in ["rmse","mae"] and diff<0) else "V better"
        print(f"    {m:<12}: SC={sc_v:>+8.4f}  VS={vs_v:>+8.4f}  "
              f"diff={diff:>+8.4f}  [{better}]")

    # Karsilastirma: raw delta vs idiosyncratic
    print(f"\n{SEP}")
    print("  KARSILASTIRMA: Raw Delta vs Idiosyncratic")
    print(SEP)
    print(f"\n  Raw delta teorik R² : {phi_raw**2:.4f}")
    print(f"  Idio  teorik R²      : {phi_idio**2:.4f}")
    print(f"\n  SC-STGCN R² (raw)   : +0.040  (walk_forward_cv.py sonucu)")
    print(f"  SC-STGCN R² (idio)  : {agg['SC-STGCN']['r2_mean']:+.4f}")

    # Kaydet
    rows = []
    for name in model_names:
        a = agg[name]
        rows.append({
            "Model": name,
            "RMSE_mean": a["rmse_mean"], "RMSE_std": a["rmse_std"],
            "MAE_mean":  a["mae_mean"],  "MAE_std":  a["mae_std"],
            "R2_mean":   a["r2_mean"],   "R2_std":   a["r2_std"],
            "Accuracy_mean": a["accuracy_mean"],
            "F1_mean":   a["f1_mean"],
            "Sharpe_mean": a["bt_sharpe_mean"], "Sharpe_std": a["bt_sharpe_std"],
            "Hit_mean":  a["hit_ratio_mean"],
        })
    pd.DataFrame(rows).to_csv(
        METRICS_DIR / "idiosyncratic_cv_precovid.csv", index=False)

    # Gorsel
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cols = {"AR(1)":"#555","LSTM":"#2196F3","XGBoost":"#FF9800",
            "V-STGCN":"#9C27B0","SC-STGCN":"#E53935"}
    bar_cols = [cols.get(n,"gray") for n in model_names]

    for ax, (m, title) in zip(axes, [
        ("rmse",         "RMSE (bp) — lower is better"),
        ("r2",           "R² — higher is better"),
        ("bt_sharpe",    "Sharpe — higher is better"),
    ]):
        means = [agg[n][f"{m}_mean"] for n in model_names]
        stds  = [agg[n][f"{m}_std"]  for n in model_names]
        ax.bar(range(len(model_names)), means, color=bar_cols, alpha=0.85,
               yerr=stds, capsize=5)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=20, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.grid(axis="y", alpha=0.3)
        for i,(v,s) in enumerate(zip(means,stds)):
            offset = (s if not np.isnan(s) else 0) + 0.01*abs(v)
            ax.text(i, v+offset, f"{v:.3f}", ha="center", fontsize=8)

    plt.suptitle("Idiosyncratic CDS Spread CV (Pre-COVID, 2 folds)\n"
                 "Common factor removed: ε_i(t) = Δs_i(t) − mean(Δs(t))",
                 fontsize=12)
    plt.tight_layout()
    out = FIG_DIR / "fig_idiosyncratic_cv.png"
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"\n  [fig] {out.name}")
    print(f"\n[DONE] idiosyncratic_cv.py tamamlandi.")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"  Total: {(time.time()-t0)/60:.1f} min")
