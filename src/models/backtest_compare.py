"""
src/models/backtest_compare.py
================================
Iki farkli backtest yontemi karsilastirmasi

Yontem A — Fixed-K:
  Her hafta en cok UP tahmin edilen K firma short,
  en cok DOWN tahmin edilen K firma long.
  STABLE tahminler goz ardi edilir.
  Toplam exposure her hafta sabit: 2K firma.

Yontem B — Proportional:
  Her hafta sadece UP/DOWN tahmin edilen firmalar pozisyon aliyor.
  Toplam exposure = (n_up + n_down) / N ile orantili.
  Az trade = az risk.

Her iki yontem icin:
  Sharpe, MeanRet, Std, MaxDD, Hit, AvgTrade%

Dataset: top50_degree
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
from configs.config import METRICS_DIR, FIG_DIR, make_dirs

BASE = Path(__file__).resolve().parents[2]
DEGREE_DIR = BASE / "data" / "top50_degree"

try:
    from xgboost import XGBRegressor

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def to_labels(delta, thr):
    labels = np.ones_like(delta, dtype=np.int64)
    labels[delta > thr] = 2
    labels[delta < -thr] = 0
    return labels


from sklearn.metrics import f1_score


def find_optimal_threshold(va_true, va_pred):
    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.arange(0.1, 3.0, 0.1):
        lt = to_labels(va_true.flatten(), thr)
        lp = to_labels(va_pred.flatten(), thr)
        if len(np.unique(lp)) < 2:
            continue
        f1 = f1_score(lt, lp, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr


# =============================================================================
# BACKTEST A — Fixed-K (her hafta sabit exposure)
# =============================================================================
def backtest_fixed_k(delta_te, pred_reg, thr, k_pct=0.20, dv01=DV01, cash=INIT_CASH):
    """
    Her hafta:
      - pred_reg siralamasinda en alt K firma: long (DOWN beklenti)
      - pred_reg siralamasinda en ust K firma: short (UP beklenti)
    Threshold'u gormezden gelir — her hafta tam K long K short.
    """
    T, N = delta_te.shape
    K = max(1, int(k_pct * N))
    pnl = []
    for t in range(T):
        r = np.argsort(pred_reg[t])
        w = np.zeros(N)
        w[r[:K]] = -1 / K  # en dusuk tahmin → long (DOWN)
        w[r[-K:]] = +1 / K  # en yuksek tahmin → short (UP)
        pnl.append(dv01 * (w * delta_te[t]).sum())
    pnl = np.array(pnl)
    eq = cash + np.cumsum(pnl)
    return _port_stats(pnl, eq, T)


# =============================================================================
# BACKTEST B — Proportional (threshold bazli, degisken exposure)
# =============================================================================
def backtest_proportional(delta_te, pred_reg, thr, dv01=DV01, cash=INIT_CASH):
    """
    Her hafta:
      - pred_reg > +thr → short (UP)
      - pred_reg < -thr → long (DOWN)
      - |pred_reg| <= thr → pozisyon yok (STABLE)
    Toplam exposure = aktif firma sayisi / N ile orantili.
    """
    T, N = delta_te.shape
    pnl = []
    active_pct = []
    for t in range(T):
        w = np.zeros(N)
        up_idx = np.where(pred_reg[t] > thr)[0]
        down_idx = np.where(pred_reg[t] < -thr)[0]
        n_active = len(up_idx) + len(down_idx)
        active_pct.append(n_active / N)
        if len(up_idx) > 0:
            w[up_idx] = +1 / len(up_idx)
        if len(down_idx) > 0:
            w[down_idx] = -1 / len(down_idx)
        # Exposure scaling: max 1.0 (tam pozisyon = tum firmalar aktif)
        scale = n_active / N
        pnl.append(dv01 * scale * (w * delta_te[t]).sum())
    pnl = np.array(pnl)
    eq = cash + np.cumsum(pnl)
    stats = _port_stats(pnl, eq, T)
    stats["avg_active_pct"] = float(np.mean(active_pct))
    stats["avg_trade_weeks"] = float(np.mean([p != 0 for p in pnl]))
    return stats


def _port_stats(pnl, eq, T):
    return {
        "sharpe": float(np.sqrt(52) * pnl.mean() / (pnl.std() + 1e-10)),
        "mean_ret": float(pnl.mean()),
        "std_ret": float(pnl.std()),
        "max_dd": float(np.min(eq / np.maximum.accumulate(eq) - 1)),
        "hit": float((pnl > 0).mean()),
        "avg_active_pct": float(1.0),
        "avg_trade_weeks": float((pnl != 0).mean()),
    }


# =============================================================================
# MODELS
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


class TransformerReg(nn.Module):
    def __init__(self, N, seq_len, d_model=64, n_heads=4, ff=128, n_layers=2, drop=0.1):
        super().__init__()
        self.emb = nn.Linear(N, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff,
            dropout=drop,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, N)

    def forward(self, x):
        h = self.emb(x)
        h = self.encoder(h)
        return self.head(self.norm(h[:, -1, :]))


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
    va_orig = delta[tr_end:va_end][N_HIS:]
    te_orig = delta[va_end:te_end][N_HIS:]

    reg_preds = {}

    def register(name, va_p, te_p):
        va_bp = sc.inverse_transform(va_p)
        te_bp = sc.inverse_transform(te_p)
        reg_preds[name] = (va_bp, te_bp)

    # AR(1)
    ps_va = np.zeros_like(va_s)
    ps_te = np.zeros_like(te_s)
    for i in range(N):
        y = tr_s[:, i]
        phi = np.corrcoef(y[:-1], y[1:])[0, 1] if y.std() > 1e-8 else 0.0
        mu = y.mean() * (1 - phi)
        last = tr_s[-1, i]
        for t in range(len(va_s)):
            ps_va[t, i] = mu + phi * last
            last = va_s[t, i]
        last = tr_s[-1, i]
        for t in range(len(te_s)):
            ps_te[t, i] = mu + phi * last
            last = te_s[t, i]
    register("AR(1)", ps_va[N_HIS:], ps_te[N_HIS:])

    # LSTM
    lstm = LSTMModel(N).to(device)
    lstm, ep = train_nn(lstm, Xtr, Ytr, Xva, Yva, device, epochs)
    Xva_w = make_windows(va_s, N_HIS)[0]
    register("LSTM", pred_nn(lstm, Xva_w, device), pred_nn(lstm, Xte, device))
    print(f"    LSTM ep={ep}")

    # XGBoost
    if HAS_XGB:
        Xtf = Xtr.reshape(len(Xtr), -1)
        Xvf = make_windows(va_s, N_HIS)[0].reshape(-1, N * N_HIS)
        Xtef = Xte.reshape(len(Xte), -1)
        pv = np.zeros((len(Xvf), N))
        pt = np.zeros((len(Xtef), N))
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
            pv[:, i] = m.predict(Xvf)
            pt[:, i] = m.predict(Xtef)
        register("XGBoost", pv, pt)

    # Transformers
    for tname in ["Informer", "Autoformer", "TimesNet"]:
        m = TransformerReg(
            N,
            N_HIS,
            d_model=ATT_H,
            n_heads=ATT_HEADS,
            ff=FF_H,
            n_layers=2,
            drop=DROPOUT,
        ).to(device)
        m, ep = train_nn(m, Xtr, Ytr, Xva, Yva, device, epochs)
        Xva_w = make_windows(va_s, N_HIS)[0]
        register(tname, pred_nn(m, Xva_w, device), pred_nn(m, Xte, device))
        print(f"    {tname} ep={ep}")

    # V-STGCN
    vs = VSTGCN(N, N_HIS, GCN_H, Ac_t).to(device)
    vs, ep = train_nn(vs, Xtr, Ytr, Xva, Yva, device, epochs)
    Xva_w = make_windows(va_s, N_HIS)[0]
    register("V-STGCN", pred_nn(vs, Xva_w, device), pred_nn(vs, Xte, device))
    print(f"    V-STGCN ep={ep}")

    # SC-STGCN
    sc_m = SCSTGCN(N, N_HIS, GCN_H, ATT_H, ATT_HEADS, FF_H, DROPOUT, Au_t, Ad_t).to(
        device
    )
    sc_m, ep = train_nn(sc_m, Xtr, Ytr, Xva, Yva, device, epochs)
    Xva_w = make_windows(va_s, N_HIS)[0]
    register("SC-STGCN", pred_nn(sc_m, Xva_w, device), pred_nn(sc_m, Xte, device))
    print(f"    SC-STGCN ep={ep}")

    # Optimal threshold (val Macro-F1)
    results_a = {}
    results_b = {}
    opt_thrs = {}
    for name, (va_bp, te_bp) in reg_preds.items():
        thr = find_optimal_threshold(va_orig, va_bp)
        opt_thrs[name] = thr
        results_a[name] = backtest_fixed_k(te_orig, te_bp, thr)
        results_b[name] = backtest_proportional(te_orig, te_bp, thr)

    return results_a, results_b, opt_thrs, te_orig


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", default="auto")
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
    print("  Backtest Karsilastirma: Fixed-K vs Proportional")
    print("  A) Fixed-K: her hafta sabit K long + K short")
    print("  B) Proportional: threshold bazli, degisken exposure")
    print(f"  epochs={args.epochs} | device={device}")
    print(SEP)

    df = pd.read_csv(DEGREE_DIR / "ve1.csv")
    try:
        df.iloc[:, 0].astype(float)
    except:
        df = df.iloc[:, 1:]
    level = df.values.astype(np.float32)
    delta = np.diff(level, axis=0)
    T, N = delta.shape
    dates = pd.date_range(start="2015-01-09", periods=T, freq="W-FRI")
    covid_week = int(np.searchsorted(dates, pd.Timestamp("2020-03-01")))
    delta_pre = delta[:covid_week]
    print(f"\n  Pre-COVID: {covid_week} hafta  N={N}")

    Ac_t = norm_adj(
        load_npz(str(DEGREE_DIR / "adj.npz")).toarray().astype(np.float32)
    ).to(device)
    Au_t = norm_adj(
        load_npz(str(DEGREE_DIR / "adj_sup.npz")).toarray().astype(np.float32)
    ).to(device)
    Ad_t = norm_adj(
        load_npz(str(DEGREE_DIR / "adj_cus.npz")).toarray().astype(np.float32)
    ).to(device)

    folds = [
        (int(covid_week * 0.65), int(covid_week * 0.78), int(covid_week * 0.92)),
        (int(covid_week * 0.78), int(covid_week * 0.92), covid_week),
    ]
    print(f"\n  Fold yapisi:")
    for i, (a, b, c) in enumerate(folds):
        print(
            f"    Fold {i+1}: test[{b}:{c}] ({c-b} hafta | {dates[b].date()} -> {dates[c-1].date()})"
        )

    model_names = [
        "AR(1)",
        "LSTM",
        "Informer",
        "Autoformer",
        "TimesNet",
        "V-STGCN",
        "SC-STGCN",
    ]
    if HAS_XGB:
        model_names.insert(2, "XGBoost")

    all_a = []
    all_b = []
    for fi, (tr_end, va_end, te_end) in enumerate(folds):
        print(f"\n  --- Fold {fi+1}/2 ---")
        set_seed(SEED + fi)
        ra, rb, opt_thrs, _ = run_fold(
            delta_pre, N, Ac_t, Au_t, Ad_t, device, args.epochs, tr_end, va_end, te_end
        )
        all_a.append(ra)
        all_b.append(rb)
        print(f"\n  Optimal thresholds: {opt_thrs}")

    # Aggregate
    prt_m = ["sharpe", "mean_ret", "std_ret", "max_dd", "hit", "avg_active_pct"]

    def agg_results(all_r):
        agg = {}
        for name in model_names:
            vals = {m: [] for m in prt_m}
            for fr in all_r:
                if name in fr:
                    for m in prt_m:
                        v = fr[name].get(m, float("nan"))
                        if not np.isnan(v):
                            vals[m].append(v)
            agg[name] = {}
            for m in prt_m:
                v = np.array(vals[m])
                agg[name][f"{m}_mean"] = (
                    float(np.mean(v)) if len(v) > 0 else float("nan")
                )
                agg[name][f"{m}_std"] = float(np.std(v)) if len(v) > 0 else float("nan")
        return agg

    agg_a = agg_results(all_a)
    agg_b = agg_results(all_b)

    def print_table(label, agg, note=""):
        print(f"\n{SEP}")
        print(f"  {label}")
        if note:
            print(f"  {note}")
        print(
            f"  {'Model':<14} {'Sharpe':>8} {'±':>6} {'MeanRet':>9} "
            f"{'Std':>8} {'MaxDD':>8} {'Hit':>7} {'Active%':>8}"
        )
        print(f"  {'-'*78}")
        for name in model_names:
            a = agg[name]
            print(
                f"  {name:<14} "
                f"{a['sharpe_mean']:>8.4f} {a['sharpe_std']:>6.4f} "
                f"{a['mean_ret_mean']:>9.4f} "
                f"{a['std_ret_mean']:>8.4f} "
                f"{a['max_dd_mean']:>8.4f} "
                f"{a['hit_mean']:>7.4f} "
                f"{a['avg_active_pct_mean']:>8.1%}"
            )

    print_table(
        "YONTEM A — Fixed-K (her hafta sabit K=%20 long + K short)",
        agg_a,
        "Not: Threshold sadece sinif etiketleri icin kullanilir, exposure sabit kalir.",
    )

    print_table(
        "YONTEM B — Proportional (degisken exposure, threshold bazli)",
        agg_b,
        "Not: STABLE tahminler neutral, exposure aktif firma oranina gore olceklenir.",
    )

    # Karsilastirma
    print(f"\n{SEP}")
    print("  SC-STGCN KARSILASTIRMA: A vs B")
    print(f"  {'Metrik':<14} {'A (Fixed-K)':>14} {'B (Prop)':>14} {'Fark':>10}")
    print(f"  {'-'*55}")
    for m in ["sharpe", "mean_ret", "std_ret", "max_dd", "hit"]:
        va = agg_a["SC-STGCN"][f"{m}_mean"]
        vb = agg_b["SC-STGCN"][f"{m}_mean"]
        print(f"  {m:<14} {va:>14.4f} {vb:>14.4f} {vb-va:>10.4f}")

    print(f"\n  Active% karsilastirmasi:")
    print(f"    A (Fixed-K)  : 100% (her hafta islem)")
    print(
        f"    B (Prop)     : {agg_b['SC-STGCN']['avg_active_pct_mean']:.1%} (ortalama aktif firma orani)"
    )

    # CSV
    rows = []
    for name in model_names:
        row = {"Model": name}
        for m in prt_m:
            row[f"A_{m}"] = agg_a[name][f"{m}_mean"]
            row[f"B_{m}"] = agg_b[name][f"{m}_mean"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(METRICS_DIR / "backtest_compare.csv", index=False)
    print(f"\n  [OK] backtest_compare.csv")
    print(f"\n{SEP}  TAMAMLANDI\n{SEP}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"  Total: {(time.time()-t0)/60:.1f} min")
