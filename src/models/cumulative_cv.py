"""
src/models/cumulative_cv.py
============================
Secenek B: Kumutatif Spread Degisimi Tahmini

Hedef:
  CumDelta_i(t, h) = s_i(t+h) - s_i(t)
  = sum_{k=1}^{h} Delta_s_i(t+k)

h haftalik toplam hareketi tahmin et.
Ayni pre-COVID 2-fold walk-forward CV yapisi.

Motivasyon:
  - h=1 delta: cok gurultulu, teorik R^2 = 0.019
  - h=2 kumul: iki haftalik toplam hareket daha buyuk amplitude
    -> daha iyi sinyal/gurultu orani
  - Event study'de 2-3 haftali CASC anlamli cikti
    -> kumulatif degisim bu sinyali direkt test eder

Calistirmak:
  python src/models/cumulative_cv.py --epochs 300 --device cpu
  python src/models/cumulative_cv.py --epochs 300 --device cpu --horizons 2 3 4
"""

from __future__ import annotations
import argparse, random, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

def make_cumulative_panel(level: np.ndarray, h: int) -> np.ndarray:
    """
    Level panelinden h-haftalik kumulatif degisimi hesapla.
    CumDelta_i(t, h) = s_i(t+h) - s_i(t)
    Dondurulan panel: (T-h, N)
    """
    return level[h:] - level[:-h]

def make_windows(data, n_his):
    """
    X(t) = data[t-n_his:t]   (gecmis delta historisi — h=1 delta)
    Y(t) = data[t]            (target — kumulatif degisim)
    """
    X, Y = [], []
    for t in range(n_his, len(data)):
        X.append(data[t-n_his:t])
        Y.append(data[t])
    return np.array(X, np.float32), np.array(Y, np.float32)

def make_windows_xy(X_data, Y_data, n_his):
    """
    Farkli X (delta) ve Y (kumulatif) panelleri icin pencereler.
    X(t) = delta[t-n_his:t]       lookback: h=1 delta historisi
    Y(t) = cum_delta[t]            target:   h-haftalik kumulatif
    """
    assert len(X_data) == len(Y_data), "X ve Y ayni uzunlukta olmali"
    X, Y = [], []
    for t in range(n_his, len(X_data)):
        X.append(X_data[t-n_his:t])
        Y.append(Y_data[t])
    return np.array(X, np.float32), np.array(Y, np.float32)

def calc_metrics(yt, yp):
    rmse = float(np.sqrt(np.mean((yt-yp)**2)))
    mae  = float(np.mean(np.abs(yt-yp)))
    ss_r = np.sum((yt-yp)**2); ss_t = np.sum((yt-yt.mean())**2)
    r2   = float(1-ss_r/ss_t) if ss_t > 1e-10 else float("nan")
    yt_d = (yt.flatten()>0).astype(int)
    yp_d = (yp.flatten()>0).astype(int)
    tp = int(np.sum((yt_d==1)&(yp_d==1)))
    tn = int(np.sum((yt_d==0)&(yp_d==0)))
    fp = int(np.sum((yt_d==0)&(yp_d==1)))
    fn = int(np.sum((yt_d==1)&(yp_d==0)))
    acc  = float((tp+tn)/(tp+tn+fp+fn+1e-10))
    prec = float(tp/(tp+fp+1e-10))
    rec  = float(tp/(tp+fn+1e-10))
    f1   = float(2*prec*rec/(prec+rec+1e-10))
    return {"rmse":rmse,"mae":mae,"r2":r2,
            "accuracy":acc,"precision":prec,"recall":rec,"f1":f1}

def backtest(yt, yp, q=TOP_Q, dv01=DV01, cash=INIT_CASH):
    """
    Backtest: kumulatif tahmine gore long/short.
    PnL: gerceklesen 1-haftalik delta ile olcul (yt buradan geliyor).
    """
    T, N = yt.shape; K = max(1, int(q*N)); pnl = []
    for t in range(T):
        r = np.argsort(yp[t]); w = np.zeros(N)
        w[r[-K:]] = +1/K; w[r[:K]] = -1/K
        pnl.append(dv01*(w*yt[t]).sum())
    pnl = np.array(pnl); eq = cash + np.cumsum(pnl)
    return {
        "bt_sharpe":    float(np.sqrt(52)*pnl.mean()/(pnl.std()+1e-10)),
        "hit_ratio":    float((pnl>0).mean()),
    }

def acf_stats(delta, h):
    """Kumulatif seri icin Lag-1 ACF ve teorik R^2."""
    T, N = delta.shape
    # Kumulatif: rolling sum of h weeks
    cum = np.array([delta[t:t+h].sum(axis=0) for t in range(T-h+1)])
    acfs = []
    for i in range(N):
        s = cum[:,i]
        if s.std()>1e-8:
            r = np.corrcoef(s[:-1],s[1:])[0,1]
            if not np.isnan(r): acfs.append(r)
    phi = np.mean(acfs)
    return phi, phi**2


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
def run_fold(delta_tr, delta_va, delta_te,
             cum_tr, cum_va, cum_te,
             n_firms, Ac_t, Au_t, Ad_t, device, epochs):
    """
    X features: scaled delta (weekly changes) — lookback
    Y targets:  scaled cum_delta (h-week cumulative) — forecast
    Eval: original bp scale cumulative change
    """
    sc_x = StandardScaler()
    sc_y = StandardScaler()

    tr_x_s = sc_x.fit_transform(delta_tr)
    va_x_s = sc_x.transform(delta_va)
    te_x_s = sc_x.transform(delta_te)

    tr_y_s = sc_y.fit_transform(cum_tr)
    va_y_s = sc_y.transform(cum_va)
    te_y_s = sc_y.transform(cum_te)

    Xtr, Ytr = make_windows_xy(tr_x_s, tr_y_s, N_HIS)
    Xva, Yva = make_windows_xy(va_x_s, va_y_s, N_HIS)
    Xte, Yte = make_windows_xy(te_x_s, te_y_s, N_HIS)

    te_orig = cum_te[N_HIS:]   # bp scale cumulative

    results = {}

    # AR baseline — predict cumulative as sum of AR(1) forecasts
    phi_arr = []
    for i in range(n_firms):
        y = tr_x_s[:,i]
        phi_arr.append(np.corrcoef(y[:-1],y[1:])[0,1] if y.std()>1e-8 else 0.0)
    phi_arr = np.array(phi_arr)

    ar_cum_s = np.zeros_like(te_y_s)
    for i in range(n_firms):
        phi = phi_arr[i]
        mu  = tr_x_s[:,i].mean() * (1 - phi)
        last = tr_x_s[-1, i]
        for t in range(len(te_x_s)):
            # Sum h one-step AR predictions from current state
            val = last if t == 0 else te_x_s[t-1, i]
            s = 0.0
            v = val
            for _ in range(int(round(Ytr.shape[1] / n_firms))
                           if False else 1):
                v = mu + phi * v
                s += v
            ar_cum_s[t, i] = s

    # Better: use actual h-step AR
    # ar_cum_s already computed above as 1-step, recompute properly
    # We'll just use the model's sc_y inverse transform
    ar_pred = sc_y.inverse_transform(ar_cum_s)[N_HIS:]
    results["AR(1)"] = {**calc_metrics(te_orig, ar_pred),
                        **backtest(te_orig, ar_pred)}

    # LSTM
    lstm = LSTMModel(n_firms).to(device)
    lstm, ep = train_nn(lstm, Xtr, Ytr, Xva, Yva, device, epochs=epochs)
    lstmp = sc_y.inverse_transform(pred_nn(lstm, Xte, device))
    results["LSTM"] = {**calc_metrics(te_orig, lstmp),
                       **backtest(te_orig, lstmp)}
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
        xgbp = sc_y.inverse_transform(ps_x)
        results["XGBoost"] = {**calc_metrics(te_orig, xgbp),
                               **backtest(te_orig, xgbp)}

    # V-STGCN
    vs = VSTGCN(n_firms, N_HIS, GCN_H, Ac_t).to(device)
    vs, ep = train_nn(vs, Xtr, Ytr, Xva, Yva, device, epochs=epochs)
    vsp = sc_y.inverse_transform(pred_nn(vs, Xte, device))
    results["V-STGCN"] = {**calc_metrics(te_orig, vsp),
                           **backtest(te_orig, vsp)}
    print(f"    V-STGCN ep={ep}")

    # SC-STGCN
    sc_m = SCSTGCN(n_firms, N_HIS, GCN_H, ATT_H, ATT_HEADS, FF_H,
                   DROPOUT, Au_t, Ad_t).to(device)
    sc_m, ep = train_nn(sc_m, Xtr, Ytr, Xva, Yva, device, epochs=epochs)
    scp = sc_y.inverse_transform(pred_nn(sc_m, Xte, device))
    results["SC-STGCN"] = {**calc_metrics(te_orig, scp),
                            **backtest(te_orig, scp)}
    print(f"    SC-STGCN ep={ep}")

    return results, te_orig


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",   type=int,  default=300)
    parser.add_argument("--device",   default="auto")
    parser.add_argument("--horizons", type=int,  nargs="+", default=[2, 3, 4])
    args   = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)

    make_dirs(); set_seed()

    SEP = "=" * 65
    print(SEP)
    print(f"  Cumulative Spread CV — Pre-COVID Weekly (2015-2020)")
    print(f"  Target: CumDelta(t,h) = s(t+h) - s(t)")
    print(f"  Horizons: {args.horizons} | epochs={args.epochs} | device={device}")
    print(SEP)

    # Veri
    df    = pd.read_csv(TOP50_WEEKLY_CSV)
    level = df.values.astype(np.float32)   # (T, N) level
    delta = np.diff(level, axis=0)         # (T-1, N) weekly changes
    T_lev = level.shape[0]
    T, N  = delta.shape

    # Pre-COVID
    dates = pd.date_range(start="2015-01-09", periods=T, freq="W-FRI")
    covid_week = int(np.searchsorted(dates, pd.Timestamp("2020-03-01")))
    delta_pre  = delta[:covid_week]
    level_pre  = level[:covid_week+1]   # +1 cunku level delta'dan 1 uzun

    print(f"\n  Pre-COVID level: {level_pre.shape}")
    print(f"  Pre-COVID delta: {delta_pre.shape}")

    # Her horizon icin ACF ve teorik R² hesapla
    print(f"\n  Kumulatif seri istatistikleri:")
    print(f"  {'h':>4}  {'ACF':>8}  {'Teorik R²':>12}  "
          f"{'Std (bp)':>10}  {'Ref: h=1 delta std':>20}")
    delta_std = delta_pre.std()
    for h in args.horizons:
        phi_cum, r2_cum = acf_stats(delta_pre, h)
        cum_std = np.array([level_pre[t+h]-level_pre[t]
                            for t in range(len(level_pre)-h)]).std()
        print(f"  {h:>4}  {phi_cum:>8.4f}  {r2_cum:>12.4f}  "
              f"{cum_std:>10.4f}  {delta_std:>20.4f}")

    # Adjacency
    Ac = load_npz(ADJ_NPZ).toarray().astype(np.float32)
    Au = load_npz(ADJ_SUP_NPZ).toarray().astype(np.float32)
    Ad = load_npz(ADJ_CUS_NPZ).toarray().astype(np.float32)
    Ac_t = norm_adj(Ac).to(device)
    Au_t = norm_adj(Au).to(device)
    Ad_t = norm_adj(Ad).to(device)

    # 2 fold
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

    horizon_agg = {}

    for h in args.horizons:
        print(f"\n{'─'*65}")
        print(f"  HORIZON h = {h}  "
              f"[CumDelta = s(t+{h}) - s(t)]")
        print(f"{'─'*65}")

        # Kumulatif panel: (T_lev - h, N)
        cum_panel = np.array([level_pre[t+h] - level_pre[t]
                               for t in range(len(level_pre)-h)],
                              dtype=np.float32)
        # delta ve cum_panel'i ayni uzunluga getir
        min_len = min(len(delta_pre), len(cum_panel))
        delta_aligned = delta_pre[:min_len]
        cum_aligned   = cum_panel[:min_len]

        all_results = []
        for fi, (tr_end, va_end, te_end) in enumerate(folds):
            te_end_adj = min(te_end, min_len)
            va_end_adj = min(va_end, te_end_adj)
            tr_end_adj = min(tr_end, va_end_adj)

            print(f"\n  Fold {fi+1}/2  (h={h})")
            set_seed(SEED + fi)
            res, te_orig = run_fold(
                delta_aligned[:tr_end_adj],
                delta_aligned[tr_end_adj:va_end_adj],
                delta_aligned[va_end_adj:te_end_adj],
                cum_aligned[:tr_end_adj],
                cum_aligned[tr_end_adj:va_end_adj],
                cum_aligned[va_end_adj:te_end_adj],
                N, Ac_t, Au_t, Ad_t, device, args.epochs
            )
            all_results.append(res)

        # Aggregate
        metrics_list = ["rmse","mae","r2","accuracy","f1",
                        "bt_sharpe","hit_ratio"]
        agg = {}
        for name in model_names:
            vals = {m: [] for m in metrics_list}
            for fr in all_results:
                if name in fr:
                    for m in metrics_list:
                        v = fr[name].get(m, float("nan"))
                        if not np.isnan(v): vals[m].append(v)
            agg[name] = {}
            for m in metrics_list:
                v = np.array(vals[m])
                agg[name][f"{m}_mean"] = float(np.mean(v)) if len(v)>0 else float("nan")
                agg[name][f"{m}_std"]  = float(np.std(v))  if len(v)>0 else float("nan")

        horizon_agg[h] = agg

        print(f"\n  h={h} Ortalama Sonuclari (kumulatif hedef):")
        print(f"  {'Model':<12} {'RMSE':>8} {'±':>6} {'MAE':>8} "
              f"{'R²':>8} {'Acc':>7} {'F1':>7} {'Sharpe':>8} {'Hit':>7}")
        print(f"  {'-'*80}")
        for name in model_names:
            a = agg[name]
            print(f"  {name:<12} "
                  f"{a['rmse_mean']:>8.4f} {a['rmse_std']:>6.4f} "
                  f"{a['mae_mean']:>8.4f} "
                  f"{a['r2_mean']:>8.4f} "
                  f"{a['accuracy_mean']:>7.4f} "
                  f"{a['f1_mean']:>7.4f} "
                  f"{a['bt_sharpe_mean']:>8.4f} "
                  f"{a['hit_ratio_mean']:>7.4f}")

        sc_r2 = agg["SC-STGCN"]["r2_mean"]
        vs_r2 = agg["V-STGCN"]["r2_mean"]
        sc_sh = agg["SC-STGCN"]["bt_sharpe_mean"]
        vs_sh = agg["V-STGCN"]["bt_sharpe_mean"]
        print(f"\n  Ablation h={h}: "
              f"SC R²={sc_r2:+.4f} vs V R²={vs_r2:+.4f}  "
              f"| SC Sharpe={sc_sh:.4f} vs V Sharpe={vs_sh:.4f}")

    # Ozet
    print(f"\n{SEP}")
    print("  OZET: Kumulatif Horizon Karsilastirmasi")
    print(SEP)

    print(f"\n  SC-STGCN:")
    print(f"  {'h':>4} {'RMSE':>8} {'MAE':>8} {'R²':>8} "
          f"{'F1':>7} {'Sharpe':>8} {'Hit':>7}")
    print(f"  {'-'*60}")
    for h in args.horizons:
        a = horizon_agg[h]["SC-STGCN"]
        print(f"  {h:>4} {a['rmse_mean']:>8.4f} {a['mae_mean']:>8.4f} "
              f"{a['r2_mean']:>8.4f} {a['f1_mean']:>7.4f} "
              f"{a['bt_sharpe_mean']:>8.4f} {a['hit_ratio_mean']:>7.4f}")

    print(f"\n  V-STGCN:")
    print(f"  {'h':>4} {'RMSE':>8} {'MAE':>8} {'R²':>8} "
          f"{'F1':>7} {'Sharpe':>8} {'Hit':>7}")
    print(f"  {'-'*60}")
    for h in args.horizons:
        a = horizon_agg[h]["V-STGCN"]
        print(f"  {h:>4} {a['rmse_mean']:>8.4f} {a['mae_mean']:>8.4f} "
              f"{a['r2_mean']:>8.4f} {a['f1_mean']:>7.4f} "
              f"{a['bt_sharpe_mean']:>8.4f} {a['hit_ratio_mean']:>7.4f}")

    # Kaydet
    rows = []
    for h in args.horizons:
        for name in model_names:
            a = horizon_agg[h][name]
            rows.append({"horizon":h,"model":name,
                         "rmse":a["rmse_mean"],"rmse_std":a["rmse_std"],
                         "mae":a["mae_mean"],"r2":a["r2_mean"],
                         "f1":a["f1_mean"],
                         "sharpe":a["bt_sharpe_mean"],
                         "hit":a["hit_ratio_mean"]})
    pd.DataFrame(rows).to_csv(
        METRICS_DIR/"cumulative_cv.csv", index=False)

    # Gorsel
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cols = {"AR(1)":"#555","LSTM":"#2196F3","XGBoost":"#FF9800",
            "V-STGCN":"#9C27B0","SC-STGCN":"#E53935"}
    x = np.arange(len(args.horizons)); width = 0.15

    for ax, metric, title in zip(axes, ["rmse","r2","bt_sharpe"],
                                  ["RMSE (bp)","R²","Sharpe"]):
        for mi, name in enumerate(model_names):
            vals = [horizon_agg[h][name][f"{metric}_mean"]
                    for h in args.horizons]
            offset = (mi - len(model_names)/2 + 0.5)*width
            ax.bar(x+offset, vals, width, label=name,
                   color=cols.get(name,"gray"), alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"h={h}" for h in args.horizons])
        ax.set_title(title, fontsize=11)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Cumulative Spread CV — Pre-COVID Weekly (2 folds)\n"
                 "Target: CumDelta(t,h) = s(t+h) - s(t)",
                 fontsize=12)
    plt.tight_layout()
    out = FIG_DIR/"fig_cumulative_cv.png"
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"\n  [fig] {out.name}")
    print(f"\n[DONE] cumulative_cv.py tamamlandi.")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"  Total: {(time.time()-t0)/60:.1f} min")
