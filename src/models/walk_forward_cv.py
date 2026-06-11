"""
src/models/walk_forward_cv.py
===============================
Walk-Forward Cross-Validation (2 fold, Pre-COVID weekly)

Pre-COVID donemi (2015-01-09 -> 2020-02-28) icin
2 fold expanding window CV.

Fold yapisi:
  Fold 1: train[0:covid*0.65] val[covid*0.65:covid*0.78] test[covid*0.78:covid*0.92]
  Fold 2: train[0:covid*0.78] val[covid*0.78:covid*0.92] test[covid*0.92:covid]

Calistirmak:
  python src/models/walk_forward_cv.py --epochs 300 --device cpu
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

def calc_metrics(yt, yp):
    # Regression metrics
    rmse = float(np.sqrt(np.mean((yt-yp)**2)))
    mae  = float(np.mean(np.abs(yt-yp)))
    ss_r = np.sum((yt-yp)**2); ss_t = np.sum((yt-yt.mean())**2)
    r2   = float(1-ss_r/ss_t) if ss_t > 1e-10 else float("nan")

    # Classification metrics — direction (widen=1, tighten=0)
    yt_dir = (yt.flatten() > 0).astype(int)
    yp_dir = (yp.flatten() > 0).astype(int)

    tp = int(np.sum((yt_dir == 1) & (yp_dir == 1)))
    tn = int(np.sum((yt_dir == 0) & (yp_dir == 0)))
    fp = int(np.sum((yt_dir == 0) & (yp_dir == 1)))
    fn = int(np.sum((yt_dir == 1) & (yp_dir == 0)))

    accuracy  = float((tp + tn) / (tp + tn + fp + fn + 1e-10))
    precision = float(tp / (tp + fp + 1e-10))
    recall    = float(tp / (tp + fn + 1e-10))
    f1        = float(2 * precision * recall / (precision + recall + 1e-10))

    # MAPE — sifira cok yakin gercek degerleri hariç tut (|yt| > 0.1 bp)
    mask = np.abs(yt.flatten()) > 0.1
    if mask.sum() > 0:
        mape = float(np.mean(np.abs(
            (yt.flatten()[mask] - yp.flatten()[mask]) /
            yt.flatten()[mask])) * 100)
    else:
        mape = float("nan")

    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape,
            "accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1}

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
        "pnl": pnl.tolist(),
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
def run_fold(delta, n_firms, Ac_t, Au_t, Ad_t, device,
             epochs, tr_end, va_end, te_end):
    tr_r = delta[:tr_end]
    va_r = delta[tr_end:va_end]
    te_r = delta[va_end:te_end]

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
    print("  Walk-Forward CV — Pre-COVID Weekly (2015-2020)")
    print(f"  epochs={args.epochs} | device={device}")
    print(SEP)

    # Veri yukle
    df    = pd.read_csv(TOP50_WEEKLY_CSV)
    level = df.values.astype(np.float32)
    delta = np.diff(level, axis=0)
    T, N  = delta.shape
    tickers = df.columns.tolist()

    # Pre-COVID siniri
    dates = pd.date_range(start="2015-01-09", periods=T, freq="W-FRI")
    covid_week = int(np.searchsorted(dates, pd.Timestamp("2020-03-01")))
    print(f"\n  Full delta  : {T} hafta")
    print(f"  Pre-COVID   : {covid_week} hafta ({dates[0].date()} -> 2020-02-28)")

    # Pre-COVID delta
    delta_pre = delta[:covid_week]

    # Adjacency
    Ac = load_npz(ADJ_NPZ).toarray().astype(np.float32)
    Au = load_npz(ADJ_SUP_NPZ).toarray().astype(np.float32)
    Ad = load_npz(ADJ_CUS_NPZ).toarray().astype(np.float32)
    Ac_t = norm_adj(Ac).to(device)
    Au_t = norm_adj(Au).to(device)
    Ad_t = norm_adj(Ad).to(device)

    # 2 fold — precovid_analysis.py ile birebir ayni
    folds = [
        (int(covid_week*0.65), int(covid_week*0.78), int(covid_week*0.92)),
        (int(covid_week*0.78), int(covid_week*0.92), covid_week),
    ]

    print(f"\n  Fold yapisi:")
    for i, (a,b,c) in enumerate(folds):
        print(f"    Fold {i+1}: train[0:{a}] val[{a}:{b}] test[{b}:{c}] "
              f"({c-b} hafta | {dates[b].date()} -> {dates[c-1].date()})")

    model_names = ["AR(1)", "LSTM", "V-STGCN", "SC-STGCN"]
    if HAS_XGB:
        model_names.insert(2, "XGBoost")

    all_results = []
    for fi, (tr_end, va_end, te_end) in enumerate(folds):
        print(f"\n  --- Fold {fi+1}/2 ---")
        set_seed(SEED + fi)
        res, te_orig = run_fold(
            delta_pre, N, Ac_t, Au_t, Ad_t,
            device, args.epochs, tr_end, va_end, te_end
        )
        all_results.append(res)
        print(f"\n  Fold {fi+1} sonuclari:")
        print(f"  {'Model':<12} {'RMSE':>7} {'MAE':>7} {'R2':>7} "
              f"{'MAPE%':>7} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} "
              f"{'Sharpe':>8} {'Hit':>7}")
        print(f"  {'-'*90}")
        for name, r in res.items():
            print(f"  {name:<12} {r['rmse']:>7.4f} {r['mae']:>7.4f} "
                  f"{r['r2']:>7.4f} {r['mape']:>7.2f} "
                  f"{r['accuracy']:>7.4f} {r['precision']:>7.4f} "
                  f"{r['recall']:>7.4f} {r['f1']:>7.4f} "
                  f"{r['bt_sharpe']:>8.4f} {r['hit_ratio']:>7.4f}")

    # Aggregate
    print(f"\n{SEP}")
    print("  WALK-FORWARD CV RESULTS (2 FOLDS)")
    print(SEP)

    agg = {}
    metrics_list = ["rmse","mae","r2","mape",
                    "accuracy","precision","recall","f1",
                    "bt_sharpe","hit_ratio",
                    "max_drawdown","total_return_pct"]
    for name in model_names:
        vals = {m: [] for m in metrics_list}
        for fr in all_results:
            if name in fr:
                for m in metrics_list:
                    if m in fr[name]: vals[m].append(fr[name][m])
        agg[name] = {}
        for m in metrics_list:
            v = np.array(vals[m])
            if len(v) > 0:
                agg[name][f"{m}_mean"] = float(np.mean(v))
                agg[name][f"{m}_std"]  = float(np.std(v))
            else:
                agg[name][f"{m}_mean"] = float("nan")
                agg[name][f"{m}_std"]  = float("nan")

    print(f"\n  {'Model':<12} {'RMSE':>8} {'+-':>6} {'MAE':>8} {'+-':>6} "
          f"{'R2':>7} {'MAPE%':>7} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} "
          f"{'Sharpe':>8} {'+-':>6} {'Hit':>7}")
    print(f"  {'-'*115}")
    for name in model_names:
        a = agg[name]
        print(f"  {name:<12} "
              f"{a['rmse_mean']:>8.4f} {a['rmse_std']:>6.4f} "
              f"{a['mae_mean']:>8.4f} {a['mae_std']:>6.4f} "
              f"{a['r2_mean']:>7.4f} "
              f"{a['mape_mean']:>7.2f} "
              f"{a['accuracy_mean']:>7.4f} "
              f"{a['precision_mean']:>7.4f} "
              f"{a['recall_mean']:>7.4f} "
              f"{a['f1_mean']:>7.4f} "
              f"{a['bt_sharpe_mean']:>8.4f} {a['bt_sharpe_std']:>6.4f} "
              f"{a['hit_ratio_mean']:>7.4f}")

    # Ablation
    print(f"\n  Ablation: SC-STGCN vs V-STGCN")
    for m in ["rmse","mae","r2","mape","accuracy","precision","recall","f1","bt_sharpe"]:
        sc_v = agg["SC-STGCN"][f"{m}_mean"]
        vs_v = agg["V-STGCN"][f"{m}_mean"]
        diff = sc_v - vs_v
        better = "SC better" if (m in ["r2","bt_sharpe"] and diff > 0) or \
                                (m in ["rmse","mae"] and diff < 0) else "V better"
        print(f"    {m:<12}: SC={sc_v:>+8.4f}  VS={vs_v:>+8.4f}  "
              f"diff={diff:>+8.4f}  [{better}]")

    # Kaydet — CSV
    rows = []
    for name in model_names:
        a = agg[name]
        rows.append({
            "Model": name,
            "RMSE_mean":      a["rmse_mean"],      "RMSE_std":      a["rmse_std"],
            "MAE_mean":       a["mae_mean"],        "MAE_std":       a["mae_std"],
            "R2_mean":        a["r2_mean"],         "R2_std":        a["r2_std"],
            "MAPE_mean":      a["mape_mean"],       "MAPE_std":      a["mape_std"],
            "Accuracy_mean":  a["accuracy_mean"],   "Accuracy_std":  a["accuracy_std"],
            "Precision_mean": a["precision_mean"],  "Precision_std": a["precision_std"],
            "Recall_mean":    a["recall_mean"],     "Recall_std":    a["recall_std"],
            "F1_mean":        a["f1_mean"],         "F1_std":        a["f1_std"],
            "Sharpe_mean":    a["bt_sharpe_mean"],  "Sharpe_std":    a["bt_sharpe_std"],
            "Hit_mean":       a["hit_ratio_mean"],  "Hit_std":       a["hit_ratio_std"],
        })
    df_out = pd.DataFrame(rows)
    csv_path = METRICS_DIR / "walk_forward_cv_precovid.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\n  [OK] {csv_path}")

    # LaTeX tablosu
    tex_path = METRICS_DIR / "walk_forward_cv_precovid.tex"
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[ht]\n\\vskip\\baselineskip\n")
        f.write("\\caption{Walk-forward cross-validation results "
                "(2 folds, pre-COVID weekly sample 2015--2020). "
                "Mean $\\pm$ standard deviation across folds.}\n")
        f.write("\\label{tab:wfcv}\n\\begin{center}\n")
        f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
        f.write("\\textbf{Model} & \\textbf{RMSE} & \\textbf{MAE} & "
                "\\textbf{$R^2$} & \\textbf{Sharpe} & "
                "\\textbf{Hit Ratio} \\\\\n\\midrule\n")
        for r in rows:
            rmse = f"${r['RMSE_mean']:.3f}" + " $\\pm$ " + f"{r['RMSE_std']:.3f}$"
            mae  = f"${r['MAE_mean']:.3f}"  + " $\\pm$ " + f"{r['MAE_std']:.3f}$"
            r2   = f"${r['R2_mean']:.3f}"   + " $\\pm$ " + f"{r['R2_std']:.3f}$"
            sh   = f"${r['Sharpe_mean']:.3f}" + " $\\pm$ " + f"{r['Sharpe_std']:.3f}$"
            hit  = f"${r['Hit_mean']:.3f}"  + " $\\pm$ " + f"{r['Hit_std']:.3f}$"
            f.write(f"{r['Model']} & {rmse} & {mae} & {r2} & {sh} & {hit} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{center}\n\\end{table}\n")
    print(f"  [OK] {tex_path.name}")

    # Fold bazinda kaydet
    fold_rows = []
    for fi, fr in enumerate(all_results):
        for name, r in fr.items():
            fold_rows.append({
                "fold": fi+1, "model": name,
                "rmse": r["rmse"], "mae": r["mae"], "r2": r["r2"],
                "bt_sharpe": r["bt_sharpe"],
                "hit_ratio": r["hit_ratio"],
            })
    pd.DataFrame(fold_rows).to_csv(
        METRICS_DIR / "walk_forward_by_fold_precovid.csv", index=False)

    # Gorsel
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    cols = {"AR(1)":"#555","LSTM":"#2196F3","XGBoost":"#FF9800",
            "V-STGCN":"#9C27B0","SC-STGCN":"#E53935"}
    bar_cols = [cols.get(n,"gray") for n in model_names]

    metrics_plot = [
        ("rmse",        "RMSE (bp)",       "lower is better"),
        ("mae",         "MAE (bp)",        "lower is better"),
        ("bt_sharpe",   "Backtest Sharpe", "higher is better"),
        ("hit_ratio",   "Hit Ratio",       "higher is better"),
    ]

    for ax, (m, title, note) in zip(axes.flatten(), metrics_plot):
        means = [agg[n][f"{m}_mean"] for n in model_names]
        stds  = [agg[n][f"{m}_std"]  for n in model_names]
        ax.bar(range(len(model_names)), means, color=bar_cols, alpha=0.85,
               yerr=stds, capsize=5, error_kw={"linewidth":1.5})
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=20, fontsize=9)
        ax.set_title(f"{title}\n({note})", fontsize=11)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.grid(axis="y", alpha=0.3)
        for i, (v, s) in enumerate(zip(means, stds)):
            offset = s + 0.03*abs(v) if not np.isnan(s) else 0.01
            ax.text(i, v + offset, f"{v:.3f}", ha="center", fontsize=8)

    plt.suptitle("Walk-Forward CV (2 folds) — Pre-COVID Weekly Sample\n"
                 "Error bars = std across folds", fontsize=12)
    plt.tight_layout()
    out1 = FIG_DIR / "fig_walk_forward_cv_precovid.png"
    fig.savefig(out1, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  [fig] {out1.name}")

    # Fold bazinda RMSE bar chart
    fig2, ax2 = plt.subplots(figsize=(11, 5))
    x = np.arange(len(model_names)); width = 0.4
    fold_cols = ["#1565C0", "#42A5F5"]
    for fi, fr in enumerate(all_results):
        vals = [fr[n]["rmse"] if n in fr else float("nan") for n in model_names]
        ax2.bar(x + fi*width - width/2, vals, width,
                label=f"Fold {fi+1}", color=fold_cols[fi], alpha=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(model_names, fontsize=10)
    ax2.set_ylabel("RMSE (bp)"); ax2.set_title("RMSE by Fold", fontsize=12)
    ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out2 = FIG_DIR / "fig_wf_rmse_by_fold_precovid.png"
    fig2.savefig(out2, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  [fig] {out2.name}")

    print(f"\n{'='*65}")
    print("  TAMAMLANDI")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"  Total: {(time.time()-t0)/60:.1f} min")
