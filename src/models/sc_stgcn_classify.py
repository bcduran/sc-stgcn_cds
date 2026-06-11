"""
src/models/sc_stgcn_classify.py
=================================
Amac:
  CDS spread YONUNU tahmin et (classification):
    Class 0: STABLE   (|Δs| <= threshold)
    Class 1: WIDEN    (Δs  >  threshold)
    Class 2: TIGHTEN  (Δs  < -threshold)

  Metrikler: Accuracy, F1 (macro), AUC (OvR)
  R² sorunu yok — classification problemi.

Modeller:
  1. Majority Class Baseline
  2. Logistic Regression (per firm)
  3. LSTM Classifier
  4. XGBoost Classifier
  5. V-STGCN Classifier (ablation)
  6. SC-STGCN Classifier (proposed)

Calistirmak:
  python src/models/sc_stgcn_classify.py --threshold 0.5 --epochs 300 --device cpu
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    roc_auc_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from scipy.sparse import load_npz

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import (
    TOP50_WEEKLY_CSV, ADJ_NPZ, ADJ_SUP_NPZ, ADJ_CUS_NPZ,
    METRICS_DIR, FIG_DIR, PRED_DIR, CKPT_DIR, make_dirs,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
SPLIT = (0.80, 0.10, 0.10)
N_HIS = 8       # grid search sonucu
N_CLASSES = 3   # STABLE, WIDEN, TIGHTEN


def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_labels(delta: np.ndarray, threshold: float) -> np.ndarray:
    """
    Delta panel -> label panel
    0: STABLE, 1: WIDEN, 2: TIGHTEN
    """
    labels = np.zeros(delta.shape, dtype=np.int64)
    labels[delta >  threshold] = 1   # WIDEN
    labels[delta < -threshold] = 2   # TIGHTEN
    return labels


def make_windows_clf(data: np.ndarray, labels: np.ndarray, n_his: int):
    """Sliding window: X=(B, T, N), Y=(B, N) labels"""
    X, Y = [], []
    for t in range(n_his, len(data)):
        X.append(data[t - n_his:t])
        Y.append(labels[t])
    return np.array(X, np.float32), np.array(Y, np.int64)


def norm_adj(A: np.ndarray) -> torch.Tensor:
    A = A + np.eye(A.shape[0], dtype=np.float32)
    d = A.sum(1) ** -0.5
    D = np.diag(d)
    return torch.tensor(D @ A @ D, dtype=torch.float32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray = None) -> dict:
    """
    y_true, y_pred: (B*N,) flattened
    y_prob: (B*N, 3) probabilities for AUC
    """
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1w  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    auc  = float("nan")
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr",
                                average="macro")
        except Exception:
            pass
    return {"accuracy": acc, "f1_macro": f1, "f1_weighted": f1w, "auc_ovr": auc}


# =============================================================================
# MODELLER
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
        self.ff   = nn.Sequential(
            nn.Linear(d, ff), nn.GELU(), nn.Dropout(drop), nn.Linear(ff, d))
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)

    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x = self.n1(x + a)
        return self.n2(x + self.ff(x))


class SCSTGCNClassifier(nn.Module):
    """SC-STGCN: dual adjacency + gated fusion + attention -> N_CLASSES per node"""
    def __init__(self, N, n_his, gcn_h=64, att_h=64, att_heads=2,
                 ff_h=128, drop=0.30, A_up=None, A_dn=None):
        super().__init__()
        self.register_buffer("A_up", A_up)
        self.register_buffer("A_dn", A_dn)
        self.gcn_up = GCNLayer(n_his, gcn_h)
        self.gcn_dn = GCNLayer(n_his, gcn_h)
        self.gate   = nn.Linear(gcn_h * 2, gcn_h)
        self.proj   = nn.Linear(gcn_h, att_h)
        self.attn   = TemporalAttn(att_h, att_heads, ff_h, drop)
        self.head   = nn.Linear(att_h, N_CLASSES)
        self.drop   = nn.Dropout(drop)

    def forward(self, x):
        # x: (B, T, N)
        B, T, N = x.shape
        xT   = x.permute(0, 2, 1)          # (B, N, T)
        h_up = self.gcn_up(xT, self.A_up)  # (B, N, gcn_h)
        h_dn = self.gcn_dn(xT, self.A_dn)
        g    = torch.sigmoid(self.gate(torch.cat([h_up, h_dn], -1)))
        h    = g * h_up + (1 - g) * h_dn
        h    = self.drop(self.proj(h))
        h    = self.attn(h)
        return self.head(h)                 # (B, N, 3)


class VSTGCNClassifier(nn.Module):
    """Vanilla STGCN: single adjacency, no gated fusion"""
    def __init__(self, N, n_his, gcn_h=64, A=None):
        super().__init__()
        self.register_buffer("A", A)
        self.gcn  = GCNLayer(n_his, gcn_h)
        self.proj = nn.Linear(gcn_h, gcn_h)
        self.head = nn.Linear(gcn_h, N_CLASSES)

    def forward(self, x):
        B, T, N = x.shape
        h = self.gcn(x.permute(0, 2, 1), self.A)
        h = torch.relu(self.proj(h))
        return self.head(h)


class LSTMClassifier(nn.Module):
    """Per-firm LSTM -> classification"""
    def __init__(self, N, n_his, hidden=128, layers=2, drop=0.20):
        super().__init__()
        self.N = N
        self.lstm = nn.LSTM(N, hidden, layers, batch_first=True,
                            dropout=drop if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, N * N_CLASSES)

    def forward(self, x):
        # x: (B, T, N)
        out, _ = self.lstm(x)            # (B, T, hidden)
        out    = out[:, -1, :]           # (B, hidden)
        logits = self.head(out)          # (B, N*3)
        return logits.view(x.shape[0], self.N, N_CLASSES)


def train_nn(model, X_tr, Y_tr, X_va, Y_va, device,
             epochs=300, batch=64, lr=1e-4, patience=60):
    opt     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr*0.01)
    loss_fn = nn.CrossEntropyLoss()

    Xt = torch.tensor(X_tr).to(device)
    Yt = torch.tensor(Y_tr).to(device)   # (B, N)
    Xv = torch.tensor(X_va).to(device)
    Yv = torch.tensor(Y_va).to(device)

    best_val = float("inf")
    no_impr  = 0
    best_state = None

    for ep in range(epochs):
        model.train()
        idx = torch.randperm(len(Xt), device=device)
        for i in range(0, len(idx), batch):
            b = idx[i:i+batch]
            opt.zero_grad()
            logits = model(Xt[b])          # (B, N, 3)
            # reshape for CrossEntropyLoss: (B*N, 3) vs (B*N,)
            loss = loss_fn(
                logits.reshape(-1, N_CLASSES),
                Yt[b].reshape(-1)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vlogits = model(Xv)
            val_loss = loss_fn(
                vlogits.reshape(-1, N_CLASSES),
                Yv.reshape(-1)
            ).item()

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
    return model, ep + 1


def predict_nn(model, X_te, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_te).to(device))  # (B, N, 3)
        probs  = F.softmax(logits, dim=-1).cpu().numpy()
        preds  = logits.argmax(-1).cpu().numpy()        # (B, N)
    return preds, probs


# =============================================================================
# ANA AKIS
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="bp threshold for WIDEN/TIGHTEN (default: 0.5)")
    parser.add_argument("--epochs",   type=int,   default=300)
    parser.add_argument("--device",   type=str,   default="cpu")
    args   = parser.parse_args()
    device = torch.device(args.device)
    THR    = args.threshold

    make_dirs()
    set_seed()

    print("=" * 65)
    print(f"  SC-STGCN Classification  |  threshold = {THR} bp")
    print("=" * 65)

    # 1) Veri
    df    = pd.read_csv(TOP50_WEEKLY_CSV)
    level = df.values.astype(np.float32)
    delta = np.diff(level, axis=0)
    tickers = df.columns.tolist()
    N = len(tickers)
    T = delta.shape[0]

    # Labels
    labels = make_labels(delta, THR)

    # Split
    n_tr = int(SPLIT[0] * T)
    n_va = int(SPLIT[1] * T)
    tr_delta, va_delta, te_delta = delta[:n_tr], delta[n_tr:n_tr+n_va], delta[n_tr+n_va:]
    tr_labels, va_labels, te_labels = labels[:n_tr], labels[n_tr:n_tr+n_va], labels[n_tr+n_va:]

    # Class distribution
    print(f"\n  Label distribution (full):")
    for cls, name in [(0,"STABLE"), (1,"WIDEN"), (2,"TIGHTEN")]:
        pct = (labels == cls).mean()
        print(f"    {name}: {pct:.1%}")

    # Scale features (delta) — labels remain as is
    sc = StandardScaler()
    tr_s = sc.fit_transform(tr_delta)
    va_s = sc.transform(va_delta)
    te_s = sc.transform(te_delta)

    X_tr, Y_tr = make_windows_clf(tr_s, tr_labels, N_HIS)
    X_va, Y_va = make_windows_clf(va_s, va_labels, N_HIS)
    X_te, Y_te = make_windows_clf(te_s, te_labels, N_HIS)

    print(f"\n  Windows — Train:{X_tr.shape} Val:{X_va.shape} Test:{X_te.shape}")

    # Flattened test labels for metrics
    Y_te_flat = Y_te.flatten()

    # 2) Adjacency
    A_comb = load_npz(ADJ_NPZ).toarray().astype(np.float32)
    A_up   = load_npz(ADJ_SUP_NPZ).toarray().astype(np.float32)
    A_dn   = load_npz(ADJ_CUS_NPZ).toarray().astype(np.float32)
    A_up_t   = norm_adj(A_up).to(device)
    A_dn_t   = norm_adj(A_dn).to(device)
    A_comb_t = norm_adj(A_comb).to(device)

    results = {}

    # 3) Majority class baseline
    print(f"\n[1/5] Majority Class Baseline...")
    majority = np.bincount(Y_tr.flatten()).argmax()
    Y_maj = np.full_like(Y_te_flat, majority)
    m = compute_metrics(Y_te_flat, Y_maj)
    results["Majority"] = m
    print(f"  Acc={m['accuracy']:.4f}  F1={m['f1_macro']:.4f}")

    # 4) Logistic Regression (per firm, flattened features)
    print(f"\n[2/5] Logistic Regression...")
    # Flatten: (B, T, N) -> (B*N, T) per firm then stack
    X_tr_lr = X_tr.transpose(0,2,1).reshape(-1, N_HIS)  # (B*N, T)
    Y_tr_lr = Y_tr.flatten()
    X_te_lr = X_te.transpose(0,2,1).reshape(-1, N_HIS)

    lr_clf = LogisticRegression(max_iter=500, random_state=SEED, C=0.1)
    lr_clf.fit(X_tr_lr, Y_tr_lr)
    Y_lr_pred = lr_clf.predict(X_te_lr)
    Y_lr_prob = lr_clf.predict_proba(X_te_lr)
    m = compute_metrics(Y_te_flat, Y_lr_pred, Y_lr_prob)
    results["LogReg"] = m
    print(f"  Acc={m['accuracy']:.4f}  F1={m['f1_macro']:.4f}  AUC={m['auc_ovr']:.4f}")

    # 5) XGBoost
    if HAS_XGB:
        print(f"\n[3/5] XGBoost Classifier...")
        xgb = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", num_class=3,
            random_state=SEED, verbosity=0, n_jobs=-1,
            eval_metric="mlogloss",
        )
        xgb.fit(X_tr_lr, Y_tr_lr,
                eval_set=[(X_te_lr, Y_te_flat)],
                verbose=False)
        Y_xgb_pred = xgb.predict(X_te_lr)
        Y_xgb_prob = xgb.predict_proba(X_te_lr)
        m = compute_metrics(Y_te_flat, Y_xgb_pred, Y_xgb_prob)
        results["XGBoost"] = m
        print(f"  Acc={m['accuracy']:.4f}  F1={m['f1_macro']:.4f}  AUC={m['auc_ovr']:.4f}")

    # 6) LSTM Classifier
    print(f"\n[4/5] LSTM Classifier...")
    lstm_clf = LSTMClassifier(N, N_HIS).to(device)
    lstm_clf, ep = train_nn(lstm_clf, X_tr, Y_tr, X_va, Y_va, device,
                             epochs=args.epochs)
    Y_lstm_pred, Y_lstm_prob = predict_nn(lstm_clf, X_te, device)
    m = compute_metrics(Y_te_flat, Y_lstm_pred.flatten(),
                        Y_lstm_prob.reshape(-1, 3))
    results["LSTM"] = m
    print(f"  Acc={m['accuracy']:.4f}  F1={m['f1_macro']:.4f}  "
          f"AUC={m['auc_ovr']:.4f}  (ep={ep})")

    # 7) V-STGCN Classifier
    print(f"\n[5a/5] V-STGCN Classifier (ablation)...")
    vstgcn_clf = VSTGCNClassifier(N, N_HIS, A=A_comb_t).to(device)
    vstgcn_clf, ep = train_nn(vstgcn_clf, X_tr, Y_tr, X_va, Y_va, device,
                               epochs=args.epochs)
    Y_vs_pred, Y_vs_prob = predict_nn(vstgcn_clf, X_te, device)
    m = compute_metrics(Y_te_flat, Y_vs_pred.flatten(),
                        Y_vs_prob.reshape(-1, 3))
    results["V-STGCN"] = m
    print(f"  Acc={m['accuracy']:.4f}  F1={m['f1_macro']:.4f}  "
          f"AUC={m['auc_ovr']:.4f}  (ep={ep})")

    # 8) SC-STGCN Classifier
    print(f"\n[5b/5] SC-STGCN Classifier (proposed)...")
    scstgcn_clf = SCSTGCNClassifier(
        N, N_HIS, A_up=A_up_t, A_dn=A_dn_t).to(device)
    scstgcn_clf, ep = train_nn(scstgcn_clf, X_tr, Y_tr, X_va, Y_va, device,
                                epochs=args.epochs)
    Y_sc_pred, Y_sc_prob = predict_nn(scstgcn_clf, X_te, device)
    m = compute_metrics(Y_te_flat, Y_sc_pred.flatten(),
                        Y_sc_prob.reshape(-1, 3))
    results["SC-STGCN"] = m
    print(f"  Acc={m['accuracy']:.4f}  F1={m['f1_macro']:.4f}  "
          f"AUC={m['auc_ovr']:.4f}  (ep={ep})")

    # Sonuclar
    print(f"\n{'='*70}")
    print(f"  CLASSIFICATION RESULTS  |  threshold={THR} bp")
    print(f"{'='*70}")
    print(f"  {'Model':<12} {'Accuracy':>9} {'F1-Macro':>9} "
          f"{'F1-Weighted':>12} {'AUC-OvR':>9}")
    print(f"  {'-'*55}")
    for name, m in results.items():
        print(f"  {name:<12} {m['accuracy']:>9.4f} {m['f1_macro']:>9.4f} "
              f"{m['f1_weighted']:>12.4f} {m['auc_ovr']:>9.4f}")

    # Detay: SC-STGCN confusion matrix ve classification report
    print(f"\n  SC-STGCN Classification Report:")
    print(classification_report(Y_te_flat, Y_sc_pred.flatten(),
                                target_names=["STABLE","WIDEN","TIGHTEN"],
                                zero_division=0))

    # Kaydet
    tag = f"clf_{str(THR).replace('.','p')}"
    rows = [{"Model": k, **v} for k, v in results.items()]
    out_csv = METRICS_DIR / f"results_{tag}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"  [OK] {out_csv}")

    # Gorsel: F1 ve AUC bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = list(results.keys())
    f1s   = [results[n]["f1_macro"] for n in names]
    aucs  = [results[n]["auc_ovr"]  for n in names]
    colors = ["#4C72B0" if n == "SC-STGCN" else
              "#DD8452" if n == "V-STGCN" else "#aaaaaa"
              for n in names]

    axes[0].bar(names, f1s, color=colors, alpha=0.85)
    axes[0].set_title(f"F1-Macro (threshold={THR} bp)", fontsize=12)
    axes[0].set_ylabel("F1 Score")
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(f1s):
        axes[0].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)

    axes[1].bar(names, aucs, color=colors, alpha=0.85)
    axes[1].set_title(f"AUC-OvR (threshold={THR} bp)", fontsize=12)
    axes[1].set_ylabel("AUC")
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(aucs):
        if not np.isnan(v):
            axes[1].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)

    plt.suptitle("CDS Spread Direction Classification", fontsize=13)
    plt.tight_layout()
    out_fig = FIG_DIR / f"fig_classification_{tag}.png"
    fig.savefig(out_fig, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  [OK] {out_fig.name}")

    print(f"\n[DONE] sc_stgcn_classify.py tamamlandi.")


if __name__ == "__main__":
    main()
