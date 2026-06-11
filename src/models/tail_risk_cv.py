"""
src/models/tail_risk_cv.py
============================
Tail Risk Siniflandirmasi

Arastirma sorusu:
  "Bu firma onumuzdeki hafta en yuksek %20 widener arasinda mi?"

Yontem:
  - Her hafta t'de, t+1 haftasinda en yuksek %20 spread widening
    yasayan firmalar pozitif sinif (1), digerleri negatif sinif (0)
  - Binary classification: supply chain bilgisi tail risk tahmininde
    yardimci oluyor mu?

Beklenti:
  - Supply chain bagli firmalar icin sinyal daha guclu olmali
  - SC-STGCN dual adjacency bu kanalda avantajli olmali

Modeller:
  1. AR(1) → thresholding ile classification
  2. LSTM classifier
  3. XGBoost classifier
  4. V-STGCN classifier
  5. SC-STGCN classifier

Metrikler:
  Accuracy, Precision, Recall, F1, AUC-ROC, Brier Score

Calistirmak:
  python src/models/tail_risk_cv.py --epochs 300 --device cpu --top_q 0.20
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
from sklearn.metrics import (roc_auc_score, brier_score_loss,
                              confusion_matrix)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import (
    TOP50_WEEKLY_CSV, ADJ_NPZ, ADJ_SUP_NPZ, ADJ_CUS_NPZ,
    METRICS_DIR, FIG_DIR, make_dirs,
)

try:
    from xgboost import XGBClassifier
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

def make_tail_labels(delta: np.ndarray, top_q: float = 0.20) -> np.ndarray:
    """
    Her hafta t'de, en yuksek top_q widener = 1, digerleri = 0.
    delta: (T, N)
    returns: labels (T, N) binary
    """
    T, N = delta.shape
    K = max(1, int(top_q * N))
    labels = np.zeros_like(delta, dtype=np.float32)
    for t in range(T):
        ranks = np.argsort(delta[t])
        labels[t, ranks[-K:]] = 1.0
    return labels

def calc_clf_metrics(yt_bin, yp_bin, yp_prob=None):
    """
    yt_bin: (T*N,) binary ground truth
    yp_bin: (T*N,) binary predictions
    yp_prob: (T*N,) probability scores (optional, for AUC)
    """
    yt = yt_bin.flatten().astype(int)
    yp = yp_bin.flatten().astype(int)

    tp = int(np.sum((yt==1)&(yp==1)))
    tn = int(np.sum((yt==0)&(yp==0)))
    fp = int(np.sum((yt==0)&(yp==1)))
    fn = int(np.sum((yt==1)&(yp==0)))

    accuracy  = float((tp+tn)/(tp+tn+fp+fn+1e-10))
    precision = float(tp/(tp+fp+1e-10))
    recall    = float(tp/(tp+fn+1e-10))
    f1        = float(2*precision*recall/(precision+recall+1e-10))

    auc = float("nan")
    brier = float("nan")
    if yp_prob is not None:
        try:
            auc   = float(roc_auc_score(yt, yp_prob.flatten()))
            brier = float(brier_score_loss(yt, yp_prob.flatten()))
        except Exception:
            pass

    return {"accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1,
            "auc": auc, "brier": brier}


# =============================================================================
# MODELS — classification head
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

class SCSTGCN_CLF(nn.Module):
    """SC-STGCN with sigmoid output for binary classification."""
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
        return torch.sigmoid(self.head(self.attn(h)).squeeze(-1))  # prob

class VSTGCN_CLF(nn.Module):
    def __init__(self, N, n_his, gcn_h, A):
        super().__init__()
        self.register_buffer("A", A)
        self.gcn  = GCNLayer(n_his, gcn_h)
        self.proj = nn.Linear(gcn_h, gcn_h)
        self.head = nn.Linear(gcn_h, 1)
    def forward(self, x):
        h = self.gcn(x.permute(0,2,1), self.A)
        return torch.sigmoid(self.head(torch.relu(self.proj(h))).squeeze(-1))

class LSTM_CLF(nn.Module):
    def __init__(self, N, h=128, layers=2, drop=0.20):
        super().__init__()
        self.lstm = nn.LSTM(N, h, layers, batch_first=True,
                            dropout=drop if layers>1 else 0.0)
        self.head = nn.Linear(h, N)
    def forward(self, x):
        out, _ = self.lstm(x)
        return torch.sigmoid(self.head(out[:,-1,:]))


# =============================================================================
# TRAINING
# =============================================================================
def train_clf(model, Xtr, Ytr, Xva, Yva, device,
              lr=LR, wd=WD, epochs=300, batch=BATCH,
              patience=PATIENCE, clip=CLIP):
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)
    lfn   = nn.BCELoss()
    Xt = torch.tensor(Xtr).to(device)
    Yt = torch.tensor(Ytr).to(device)
    Xv = torch.tensor(Xva).to(device)
    Yv = torch.tensor(Yva).to(device)
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

def pred_prob(model, X, device):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X).to(device)).cpu().numpy()


# =============================================================================
# RUN FOLD
# =============================================================================
def run_fold(delta, labels, n_firms, Ac_t, Au_t, Ad_t, device,
             epochs, tr_end, va_end, te_end, top_q):
    # Features: normalized delta
    sc = StandardScaler()
    tr_s = sc.fit_transform(delta[:tr_end])
    va_s = sc.transform(delta[tr_end:va_end])
    te_s = sc.transform(delta[va_end:te_end])

    # Labels
    lab_tr = labels[:tr_end]
    lab_va = labels[tr_end:va_end]
    lab_te = labels[va_end:te_end]

    Xtr, Ytr_bin = make_windows(tr_s, N_HIS)
    Xva, Yva_bin = make_windows(va_s, N_HIS)
    Xte, Yte_bin = make_windows(te_s, N_HIS)

    # Label windows
    _, Ltr = make_windows(lab_tr, N_HIS)
    _, Lva = make_windows(lab_va, N_HIS)
    _, Lte = make_windows(lab_te, N_HIS)

    results = {}

    # 1. AR(1) baseline — regression score → threshold → class
    ps = np.zeros_like(te_s)
    for i in range(n_firms):
        y = tr_s[:,i]
        phi = np.corrcoef(y[:-1],y[1:])[0,1] if y.std()>1e-8 else 0.0
        mu = y.mean()*(1-phi); last = tr_s[-1,i]
        for t in range(len(te_s)):
            ps[t,i] = mu+phi*last; last = te_s[t,i]
    # Threshold: top_q per week
    K = max(1, int(top_q * n_firms))
    ar1_prob = np.zeros_like(ps[N_HIS:])
    ar1_pred = np.zeros_like(ps[N_HIS:])
    for t in range(len(ar1_prob)):
        scores = ps[N_HIS+t]
        ar1_prob[t] = (scores - scores.min()) / (scores.max()-scores.min()+1e-8)
        top_k = np.argsort(scores)[-K:]
        ar1_pred[t, top_k] = 1.0
    results["AR(1)"] = calc_clf_metrics(Lte, ar1_pred, ar1_prob)

    # 2. LSTM classifier
    lstm = LSTM_CLF(n_firms).to(device)
    lstm, ep = train_clf(lstm, Xtr, Ltr, Xva, Lva, device, epochs=epochs)
    lstm_prob = pred_prob(lstm, Xte, device)
    lstm_pred = (lstm_prob > 0.5).astype(float)
    results["LSTM"] = calc_clf_metrics(Lte, lstm_pred, lstm_prob)
    print(f"    LSTM ep={ep}")

    # 3. XGBoost classifier
    if HAS_XGB:
        Xtf  = Xtr.reshape(len(Xtr),-1)
        Xtef = Xte.reshape(len(Xte),-1)
        xgb_prob = np.zeros((len(Xte), n_firms))
        xgb_pred = np.zeros((len(Xte), n_firms))
        for i in range(n_firms):
            m = XGBClassifier(n_estimators=300, max_depth=4,
                              learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8,
                              use_label_encoder=False,
                              eval_metric="logloss",
                              random_state=SEED, n_jobs=-1, verbosity=0)
            m.fit(Xtf, Ltr[:,i])
            xgb_prob[:,i] = m.predict_proba(Xtef)[:,1]
            xgb_pred[:,i] = (xgb_prob[:,i] > 0.5).astype(float)
        results["XGBoost"] = calc_clf_metrics(Lte, xgb_pred, xgb_prob)

    # 4. V-STGCN classifier
    vs = VSTGCN_CLF(n_firms, N_HIS, GCN_H, Ac_t).to(device)
    vs, ep = train_clf(vs, Xtr, Ltr, Xva, Lva, device, epochs=epochs)
    vs_prob = pred_prob(vs, Xte, device)
    vs_pred = (vs_prob > 0.5).astype(float)
    results["V-STGCN"] = calc_clf_metrics(Lte, vs_pred, vs_prob)
    print(f"    V-STGCN ep={ep}")

    # 5. SC-STGCN classifier
    sc_m = SCSTGCN_CLF(n_firms, N_HIS, GCN_H, ATT_H, ATT_HEADS, FF_H,
                        DROPOUT, Au_t, Ad_t).to(device)
    sc_m, ep = train_clf(sc_m, Xtr, Ltr, Xva, Lva, device, epochs=epochs)
    sc_prob = pred_prob(sc_m, Xte, device)
    sc_pred = (sc_prob > 0.5).astype(float)
    results["SC-STGCN"] = calc_clf_metrics(Lte, sc_pred, sc_prob)
    print(f"    SC-STGCN ep={ep}")

    return results


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top_q",  type=float, default=0.20,
                        help="Top widener fraction (default=0.20 = top 10 firms)")
    args   = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)

    make_dirs(); set_seed()

    SEP = "=" * 65
    print(SEP)
    print(f"  Tail Risk Classification — Pre-COVID Weekly (2015-2020)")
    print(f"  Top-{args.top_q*100:.0f}% widener each week = positive class")
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

    # Binary labels: top_q widener = 1
    labels = make_tail_labels(delta_pre, args.top_q)
    K = max(1, int(args.top_q * N))
    print(f"\n  Delta panel (pre-COVID): {delta_pre.shape}")
    print(f"  Positive class (top {args.top_q*100:.0f}%): {K} firms/week")
    print(f"  Class balance: {labels.mean()*100:.1f}% positive")

    # Label statistikleri
    print(f"\n  Label istatistikleri:")
    print(f"    Toplam pozitif ornekler: {int(labels.sum())}")
    print(f"    Toplam ornek          : {int(labels.size)}")
    print(f"    Beklenen dogruluk (random): {1-args.top_q:.1%}")

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

    all_results = []
    for fi, (tr_end, va_end, te_end) in enumerate(folds):
        print(f"\n  --- Fold {fi+1}/2 ---")
        set_seed(SEED + fi)
        res = run_fold(
            delta_pre, labels, N, Ac_t, Au_t, Ad_t,
            device, args.epochs, tr_end, va_end, te_end, args.top_q
        )
        all_results.append(res)
        print(f"\n  Fold {fi+1} sonuclari:")
        print(f"  {'Model':<12} {'Acc':>7} {'Prec':>7} {'Recall':>7} "
              f"{'F1':>7} {'AUC':>7} {'Brier':>7}")
        print(f"  {'-'*60}")
        for name, r in res.items():
            print(f"  {name:<12} {r['accuracy']:>7.4f} {r['precision']:>7.4f} "
                  f"{r['recall']:>7.4f} {r['f1']:>7.4f} "
                  f"{r['auc']:>7.4f} {r['brier']:>7.4f}")

    # Aggregate
    metrics_list = ["accuracy","precision","recall","f1","auc","brier"]
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

    print(f"\n{SEP}")
    print(f"  TAIL RISK CLASSIFICATION RESULTS (2 FOLDS, top-{args.top_q*100:.0f}%)")
    print(SEP)
    print(f"\n  Random baseline accuracy: {1-args.top_q:.1%}")
    print(f"\n  {'Model':<12} {'Acc':>8} {'+-':>6} {'Prec':>8} {'+-':>6} "
          f"{'Recall':>8} {'+-':>6} {'F1':>8} {'+-':>6} "
          f"{'AUC':>8} {'+-':>6} {'Brier':>8}")
    print(f"  {'-'*105}")
    for name in model_names:
        a = agg[name]
        print(f"  {name:<12} "
              f"{a['accuracy_mean']:>8.4f} {a['accuracy_std']:>6.4f} "
              f"{a['precision_mean']:>8.4f} {a['precision_std']:>6.4f} "
              f"{a['recall_mean']:>8.4f} {a['recall_std']:>6.4f} "
              f"{a['f1_mean']:>8.4f} {a['f1_std']:>6.4f} "
              f"{a['auc_mean']:>8.4f} {a['auc_std']:>6.4f} "
              f"{a['brier_mean']:>8.4f}")

    # Ablation
    print(f"\n  Ablation: SC-STGCN vs V-STGCN")
    for m in ["accuracy","precision","recall","f1","auc","brier"]:
        sc_v = agg["SC-STGCN"][f"{m}_mean"]
        vs_v = agg["V-STGCN"][f"{m}_mean"]
        diff = sc_v - vs_v
        better = "SC better" if (m != "brier" and diff > 0) or \
                                (m == "brier" and diff < 0) else "V better"
        print(f"    {m:<10}: SC={sc_v:>+7.4f}  VS={vs_v:>+7.4f}  "
              f"diff={diff:>+7.4f}  [{better}]")

    # Kaydet
    rows = []
    for name in model_names:
        a = agg[name]
        rows.append({"Model": name,
                     "Acc_mean": a["accuracy_mean"],   "Acc_std": a["accuracy_std"],
                     "Prec_mean": a["precision_mean"], "Prec_std": a["precision_std"],
                     "Rec_mean": a["recall_mean"],     "Rec_std": a["recall_std"],
                     "F1_mean": a["f1_mean"],          "F1_std": a["f1_std"],
                     "AUC_mean": a["auc_mean"],        "AUC_std": a["auc_std"],
                     "Brier_mean": a["brier_mean"]})
    pd.DataFrame(rows).to_csv(
        METRICS_DIR / f"tail_risk_cv_q{int(args.top_q*100)}.csv", index=False)

    # Gorsel
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cols_m = {"AR(1)":"#555","LSTM":"#2196F3","XGBoost":"#FF9800",
              "V-STGCN":"#9C27B0","SC-STGCN":"#E53935"}
    bar_cols = [cols_m.get(n,"gray") for n in model_names]

    for ax, (m, title) in zip(axes, [
        ("accuracy", f"Accuracy (random={1-args.top_q:.0%})"),
        ("f1",       "F1 Score"),
        ("auc",      "AUC-ROC"),
    ]):
        means = [agg[n][f"{m}_mean"] for n in model_names]
        stds  = [agg[n][f"{m}_std"]  for n in model_names]
        ax.bar(range(len(model_names)), means, color=bar_cols, alpha=0.85,
               yerr=stds, capsize=5)
        if m == "accuracy":
            ax.axhline(1-args.top_q, color="red", linewidth=1.2,
                       linestyle="--", label=f"Random ({1-args.top_q:.0%})")
            ax.legend(fontsize=9)
        if m == "auc":
            ax.axhline(0.5, color="red", linewidth=1.2, linestyle="--",
                       label="Random (0.50)")
            ax.legend(fontsize=9)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=20, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        for i,(v,s) in enumerate(zip(means,stds)):
            offset = (s if not np.isnan(s) else 0) + 0.002
            ax.text(i, v+offset, f"{v:.3f}", ha="center", fontsize=8)

    plt.suptitle(
        f"Tail Risk Classification — Pre-COVID Weekly (2 folds)\n"
        f"Positive class: top-{args.top_q*100:.0f}% wideners each week "
        f"(K={K} firms)",
        fontsize=12)
    plt.tight_layout()
    out = FIG_DIR / f"fig_tail_risk_cv_q{int(args.top_q*100)}.png"
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"\n  [fig] {out.name}")
    print(f"\n[DONE] tail_risk_cv.py tamamlandi.")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"  Total: {(time.time()-t0)/60:.1f} min")
