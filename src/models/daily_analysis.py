"""
src/models/daily_analysis.py
==============================
Weekly → Daily geçiş deneyi.

1. Daily CDS panel olustur (Top-50, 5Y tenor, PX5)
2. Istatistik karsilastirmasi (weekly vs daily)
3. Walk-forward CV (3 fold, daily)
4. Weekly sonuçlarla karşılaştırma

Calistirmak:
  python src/models/daily_analysis.py --epochs 300 --device cpu
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
    CDS_RAW_CSV, TOP50_WEEKLY_CSV,
    ADJ_NPZ, ADJ_SUP_NPZ, ADJ_CUS_NPZ,
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
N_HIS     = 10      # 10 gun lookback (haftalik 8'e denk)
GCN_H     = 64
ATT_H     = 64
ATT_HEADS = 2
FF_H      = 128
DROPOUT   = 0.30
LR        = 1e-4
WD        = 1e-3
BATCH     = 128     # Daily'de daha fazla sample var
PATIENCE  = 50
CLIP      = 5.0
TOP_Q     = 0.20
DV01      = 100.0
INIT_CASH = 100_000.0
ANNUALIZE = np.sqrt(252)  # Daily annualization


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

def backtest(yt, yp, q=TOP_Q, dv01=DV01, cash=INIT_CASH,
             ann=ANNUALIZE):
    T, N = yt.shape; K = max(1, int(q*N)); pnl = []
    for t in range(T):
        r = np.argsort(yp[t]); w = np.zeros(N)
        w[r[-K:]] = +1/K; w[r[:K]] = -1/K
        pnl.append(dv01*(w*yt[t]).sum())
    pnl = np.array(pnl); eq = cash + np.cumsum(pnl)
    return {
        "bt_sharpe":    float(ann*pnl.mean()/(pnl.std()+1e-10)),
        "hit_ratio":    float((pnl>0).mean()),
        "max_drawdown": float(np.min(eq/np.maximum.accumulate(eq)-1)),
        "total_return_pct": float((eq[-1]-cash)/cash*100),
    }


# =============================================================================
# DAILY PANEL OLUSTUR
# =============================================================================
def build_daily_panel(top50_tickers: list) -> pd.DataFrame:
    """
    Ham CDS verisinden Top-50 firmalar icin
    gunluk 5Y tenor (PX5) paneli olustur.
    """
    print("  Ham CDS verisi yukleniyor...")
    raw = pd.read_csv(CDS_RAW_CSV)
    cols = raw.columns.tolist()

    # Kolon isimlerini normalize et
    date_col   = cols[0]
    ticker_col = cols[1]
    # PX5 = 5Y tenor — genellikle 5. veya 6. kolon
    # Ham datada: Date, Ticker, Company, PX1, PX2, PX3, PX4, PX5, ...
    px5_col = None
    for c in cols:
        if "5" in str(c) and c != ticker_col and c != date_col:
            px5_col = c
            break
    if px5_col is None:
        # Sayısal kolonları bul
        num_cols = raw.select_dtypes(include=[np.number]).columns.tolist()
        px5_col  = num_cols[4] if len(num_cols) > 4 else num_cols[-1]

    print(f"  Kullanilan kolon: {px5_col} (5Y tenor)")
    print(f"  Top-50 tickerlar filtreniyor...")

    raw[ticker_col] = raw[ticker_col].astype(str).str.strip().str.upper()
    top50_upper = [t.upper() for t in top50_tickers]
    sub = raw[raw[ticker_col].isin(top50_upper)][[date_col, ticker_col, px5_col]].copy()
    sub.columns = ["Date", "Ticker", "PX5"]
    sub["Date"] = pd.to_datetime(sub["Date"], errors="coerce")
    sub = sub.dropna(subset=["Date"])

    print(f"  Filtrelenmiş satır: {len(sub):,}")

    # Pivot: Date x Ticker
    panel = sub.pivot_table(index="Date", columns="Ticker",
                             values="PX5", aggfunc="mean")
    panel = panel.sort_index()

    # Sadece top50'deki tickerlar
    available = [t for t in top50_upper if t in panel.columns]
    panel = panel[available]
    print(f"  Available tickers: {len(available)}/50")
    print(f"  Tarih aralığı: {panel.index[0].date()} → {panel.index[-1].date()}")
    print(f"  Toplam gün: {len(panel)}")

    # NaN temizle
    nan_before = panel.isna().sum().sum()
    panel = panel.ffill().bfill().fillna(panel.mean())
    nan_after = panel.isna().sum().sum()
    print(f"  NaN: {nan_before:,} → {nan_after}")

    return panel


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
             patience=PATIENCE, clip=CLIP, tag=""):
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
    if tag: print(f"    [{tag}] ep={ep+1}")
    return model

def pred_nn(model, X, device):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X).to(device)).cpu().numpy()


# =============================================================================
# RUN FOLD
# =============================================================================
def run_fold(delta, n_firms, Ac_t, Au_t, Ad_t, device,
             epochs, tr_end, va_end, te_end):
    tr_r = delta[:tr_end]; va_r = delta[tr_end:va_end]
    te_r = delta[va_end:te_end]

    sc = StandardScaler()
    tr_s = sc.fit_transform(tr_r); va_s = sc.transform(va_r)
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
    results["AR(1)"] = {**calc_metrics(te_orig, ar1p),
                         **backtest(te_orig, ar1p)}

    # LSTM
    lstm = LSTMModel(n_firms).to(device)
    lstm = train_nn(lstm, Xtr, Ytr, Xva, Yva, device,
                    epochs=epochs, tag="LSTM")
    lstmp = sc.inverse_transform(pred_nn(lstm, Xte, device))
    results["LSTM"] = {**calc_metrics(te_orig, lstmp),
                        **backtest(te_orig, lstmp)}

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
        results["XGBoost"] = {**calc_metrics(te_orig, xgbp),
                               **backtest(te_orig, xgbp)}

    # V-STGCN
    vs = VSTGCN(n_firms, N_HIS, GCN_H, Ac_t).to(device)
    vs = train_nn(vs, Xtr, Ytr, Xva, Yva, device,
                  epochs=epochs, tag="V-STGCN")
    vsp = sc.inverse_transform(pred_nn(vs, Xte, device))
    results["V-STGCN"] = {**calc_metrics(te_orig, vsp),
                           **backtest(te_orig, vsp)}

    # SC-STGCN
    sc_m = SCSTGCN(n_firms, N_HIS, GCN_H, ATT_H, ATT_HEADS, FF_H,
                   DROPOUT, Au_t, Ad_t).to(device)
    sc_m = train_nn(sc_m, Xtr, Ytr, Xva, Yva, device,
                    epochs=epochs, tag="SC-STGCN")
    scp = sc.inverse_transform(pred_nn(sc_m, Xte, device))
    results["SC-STGCN"] = {**calc_metrics(te_orig, scp),
                            **backtest(te_orig, scp)}

    return results, te_orig


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",   type=int,  default=300)
    parser.add_argument("--device",   default="auto")
    parser.add_argument("--precovid", action="store_true",
                        help="COVID oncesi dataya sinirla (2020-02-28)")
    args   = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)

    make_dirs(); set_seed()

    SEP = "=" * 65
    sample_label = "Pre-COVID" if args.precovid else "Full Sample"
    print(SEP)
    print(f"  Daily CDS Analizi ({sample_label})")
    print(SEP)

    # Top-50 ticker listesi
    top50_tickers = pd.read_csv(TOP50_WEEKLY_CSV, nrows=0).columns.tolist()
    print(f"  Top-50 tickers: {len(top50_tickers)}")

    # Daily panel oluştur
    print(f"\n[1/4] Daily panel olusturuluyor...")
    panel = build_daily_panel(top50_tickers)
    N     = len(panel.columns)

    # Pre-COVID filtresi
    if args.precovid:
        covid_start = pd.Timestamp("2020-03-01")
        panel = panel[panel.index < covid_start]
        print(f"  Pre-COVID filtresi: {panel.index[-1].date()}'ye kadar")

    # Level → delta
    level = panel.values.astype(np.float32)
    delta = np.diff(level, axis=0)
    T, n_firms = delta.shape
    dates = panel.index[1:]   # delta tarihleri
    sample_label = "Pre-COVID" if args.precovid else "Full Sample"

    print(f"\n  Daily delta ({sample_label}): {T} gün x {n_firms} firma")
    print(f"  Haftalık ile karşılaştırma: {T} gün ≈ {T//5} hafta")

    # Adjacency
    Ac = load_npz(ADJ_NPZ).toarray().astype(np.float32)
    Au = load_npz(ADJ_SUP_NPZ).toarray().astype(np.float32)
    Ad = load_npz(ADJ_CUS_NPZ).toarray().astype(np.float32)
    # Adjacency firmalar sırasına göre yeniden sırala
    tickers_daily = panel.columns.tolist()
    tickers_weekly = pd.read_csv(TOP50_WEEKLY_CSV, nrows=0).columns.tolist()
    # Index eşlemesi
    idx_map = []
    for t in tickers_daily:
        if t in tickers_weekly:
            idx_map.append(tickers_weekly.index(t))
        else:
            idx_map.append(None)
    valid = [i for i in idx_map if i is not None]
    # Sadece eşleşen satır/kolonları al
    Ac_sub = Ac[np.ix_(valid, valid)]
    Au_sub = Au[np.ix_(valid, valid)]
    Ad_sub = Ad[np.ix_(valid, valid)]
    n_firms = len(valid)
    delta   = delta[:, :n_firms]

    Ac_t = norm_adj(Ac_sub).to(device)
    Au_t = norm_adj(Au_sub).to(device)
    Ad_t = norm_adj(Ad_sub).to(device)

    # ------------------------------------------------------------------
    # 2) İSTATİSTİKSEL KARŞILAŞTIRMA
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("  [2] İstatistiksel Karşılaştırma: Daily vs Weekly")
    print(SEP)

    # Haftalık delta
    df_weekly = pd.read_csv(TOP50_WEEKLY_CSV)
    level_w   = df_weekly.values.astype(np.float32)
    delta_w   = np.diff(level_w, axis=0)

    for label, d, freq in [("Daily", delta, "gün"), ("Weekly", delta_w, "hafta")]:
        flat = d.flatten()
        flat_clean = flat[~np.isnan(flat)]
        print(f"\n  {label} ({len(d)} {freq} x {d.shape[1]} firma):")
        print(f"    Mean      : {np.nanmean(flat_clean):+.4f} bp")
        print(f"    Std       : {np.nanstd(flat_clean):.4f} bp")
        print(f"    Kurtosis  : {stats.kurtosis(flat_clean):.1f}")
        print(f"    |Δ| p95   : {np.percentile(np.abs(flat_clean), 95):.2f} bp")
        # Lag-1 ACF
        acfs = []
        for i in range(d.shape[1]):
            s = d[:,i]; s = s[~np.isnan(s)]
            if len(s) > 10 and s.std() > 1e-8:
                try:
                    r = np.corrcoef(s[:-1], s[1:])[0,1]
                    if not np.isnan(r): acfs.append(r)
                except: pass
        phi = np.mean(acfs)
        print(f"    Lag-1 ACF : {phi:+.4f}")
        print(f"    Teorik R² : {phi**2:.4f}")
        te_s = int(0.9*len(d))
        te = d[te_s:]
        ss_tot = np.sum((te - te.mean())**2)
        ss_naive = np.sum(te**2)
        print(f"    Naive R²  : {1 - ss_naive/ss_tot:+.4f}" if ss_tot > 0 else "    Naive R²: nan")

    # ------------------------------------------------------------------
    # 3) DAILY WALK-FORWARD CV
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print(f"  [3] Daily Walk-Forward CV (3 fold)")
    print(SEP)

    # 3 fold: 80/10/10 expanding window
    # T ~ 1746 gün
    folds = [
        (int(T*0.60), int(T*0.70), int(T*0.80)),
        (int(T*0.70), int(T*0.80), int(T*0.90)),
        (int(T*0.80), int(T*0.90), T),
    ]

    model_names = ["AR(1)", "LSTM", "V-STGCN", "SC-STGCN"]
    if HAS_XGB: model_names.insert(2, "XGBoost")

    all_results = []
    for fi, (tr_end, va_end, te_end) in enumerate(folds):
        te_days = te_end - va_end
        print(f"\n  Fold {fi+1}/3: train[0:{tr_end}] val[{tr_end}:{va_end}] "
              f"test[{va_end}:{te_end}] ({te_days} gün ≈ {te_days//5} hafta)")
        if te_days > 0 and va_end < len(dates):
            print(f"  Test: {dates[va_end].date()} → {dates[min(te_end-1, len(dates)-1)].date()}")
        set_seed(SEED + fi)
        res, te_orig = run_fold(
            delta, n_firms, Ac_t, Au_t, Ad_t,
            device, args.epochs, tr_end, va_end, te_end
        )
        all_results.append(res)
        print(f"  {'Model':<12} {'RMSE':>7} {'MAE':>7} {'R²':>8} {'Sharpe':>8} {'Hit':>7}")
        print(f"  {'-'*55}")
        for name, r in res.items():
            print(f"  {name:<12} {r['rmse']:>7.4f} {r['mae']:>7.4f} "
                  f"{r['r2']:>8.4f} {r['bt_sharpe']:>8.4f} "
                  f"{r['hit_ratio']:>7.4f}")

    # Aggregate
    print(f"\n  Daily CV Ortalama (3 fold):")
    print(f"  {'Model':<12} {'RMSE':>8} {'MAE':>8} {'R²':>9} "
          f"{'Sharpe':>9} {'±std':>7} {'Hit':>8}")
    print(f"  {'-'*65}")
    daily_agg = {}
    for name in model_names:
        vals = {m: [] for m in ["rmse","mae","r2","bt_sharpe","hit_ratio"]}
        for fr in all_results:
            if name in fr:
                for m in vals: vals[m].append(fr[name][m])
        daily_agg[name] = {m: np.mean(v) for m,v in vals.items() if v}
        daily_agg[name]["bt_sharpe_std"] = np.std([fr[name]["bt_sharpe"]
                                                    for fr in all_results
                                                    if name in fr])
        a = daily_agg[name]
        print(f"  {name:<12} {a.get('rmse',0):>8.4f} "
              f"{a.get('mae',0):>8.4f} "
              f"{a.get('r2',0):>9.4f} "
              f"{a.get('bt_sharpe',0):>9.4f} "
              f"{a.get('bt_sharpe_std',0):>7.4f} "
              f"{a.get('hit_ratio',0):>8.4f}")

    # ------------------------------------------------------------------
    # 4) KARŞILAŞTIRMA
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("  [4] SONUÇ: Daily vs Weekly Karşılaştırması")
    print(SEP)

    weekly_csv = METRICS_DIR / "walk_forward_cv.csv"
    if weekly_csv.exists():
        df_w = pd.read_csv(weekly_csv)
        weekly_agg = {row["Model"]: row for _, row in df_w.iterrows()}
        print(f"\n  {'Model':<12} {'W-RMSE':>8} {'D-RMSE':>8} "
              f"{'W-R²':>8} {'D-R²':>8} "
              f"{'W-Sharpe':>9} {'D-Sharpe':>9} "
              f"{'W-Hit':>7} {'D-Hit':>7}")
        print(f"  {'-'*80}")
        for name in model_names:
            w = weekly_agg.get(name, {})
            d = daily_agg.get(name, {})
            wr = w.get("RMSE_mean", float("nan"))
            dr = d.get("rmse", float("nan"))
            wr2 = w.get("R2_mean", float("nan"))
            dr2 = d.get("r2", float("nan"))
            ws = w.get("Sharpe_mean", float("nan"))
            ds = d.get("bt_sharpe", float("nan"))
            wh = w.get("Hit_mean", float("nan"))
            dh = d.get("hit_ratio", float("nan"))
            print(f"  {name:<12} {wr:>8.4f} {dr:>8.4f} "
                  f"{wr2:>8.4f} {dr2:>8.4f} "
                  f"{ws:>9.4f} {ds:>9.4f} "
                  f"{wh:>7.4f} {dh:>7.4f}")

    # Kaydet
    rows = [{"Model": n, "Freq": "Daily",
             "RMSE": daily_agg[n].get("rmse"),
             "MAE":  daily_agg[n].get("mae"),
             "R2":   daily_agg[n].get("r2"),
             "Sharpe": daily_agg[n].get("bt_sharpe"),
             "Hit":  daily_agg[n].get("hit_ratio")}
            for n in model_names if n in daily_agg]
    tag = "precovid" if args.precovid else "full"
    pd.DataFrame(rows).to_csv(METRICS_DIR / f"daily_cv_results_{tag}.csv", index=False)

    # Görsel
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(model_names)); width = 0.35

    if weekly_csv.exists():
        metrics_plot = [
            ("RMSE", [w.get("RMSE_mean",0) for w in [weekly_agg.get(n,{}) for n in model_names]],
                     [daily_agg.get(n,{}).get("rmse",0) for n in model_names]),
            ("R²",   [w.get("R2_mean",0) for w in [weekly_agg.get(n,{}) for n in model_names]],
                     [daily_agg.get(n,{}).get("r2",0) for n in model_names]),
            ("Sharpe",[w.get("Sharpe_mean",0) for w in [weekly_agg.get(n,{}) for n in model_names]],
                      [daily_agg.get(n,{}).get("bt_sharpe",0) for n in model_names]),
        ]
        for ax, (title, wvals, dvals) in zip(axes, metrics_plot):
            ax.bar(x-width/2, wvals, width, label="Weekly", color="#E53935", alpha=0.8)
            ax.bar(x+width/2, dvals, width, label="Daily",  color="#2196F3", alpha=0.8)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xticks(x); ax.set_xticklabels(model_names, rotation=20, fontsize=9)
            ax.set_title(title, fontsize=11)
            ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Daily vs Weekly: Model Performance Comparison\n"
                 "(Walk-Forward CV, 3 folds each)", fontsize=12)
    plt.tight_layout()
    out = FIG_DIR / f"fig_daily_vs_weekly_{tag}.png"
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"\n[fig] {out.name}")
    print(f"\n[DONE] daily_analysis.py tamamlandi.")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"  Total: {(time.time()-t0)/60:.1f} min")
