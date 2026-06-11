"""
src/models/threeway_cv.py
==========================
3-Sinifli CDS Spread Yonu Tahmini

Siniflar:
  0 = DOWN   : Δs < -threshold
  1 = STABLE : |Δs| <= threshold
  2 = UP     : Δs > +threshold

3 Tablo:
  1. Classification : Acc, Macro-F1, Weighted-F1 + McNemar p
  2. Per-Class      : DOWN / STABLE / UP icin P, R, F1
  3. Portfolio      : UP->short, DOWN->long, STABLE->flat
                      Sharpe, MeanRet, Std, MaxDD, Hit

Dataset: top50_degree
Calistirmak:
  python src/models/threeway_cv.py --epochs 300 --device cpu --threshold 0.5
"""
from __future__ import annotations
import argparse, random, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import load_npz
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score,
                              precision_recall_fscore_support,
                              confusion_matrix)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import METRICS_DIR, FIG_DIR, make_dirs

BASE = Path(__file__).resolve().parents[2]
DEGREE_DIR = BASE / "data" / "top50_degree"

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED=42; N_HIS=8; GCN_H=64; ATT_H=64; ATT_HEADS=2; FF_H=128
DROPOUT=0.30; LR=1e-4; WD=1e-3; BATCH=64; PATIENCE=40; CLIP=5.0
DV01=100.0; INIT_CASH=100_000.0


def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True

def norm_adj(A):
    A=A+np.eye(A.shape[0],dtype=np.float32); d=A.sum(1)**-0.5
    return torch.tensor(np.diag(d)@A@np.diag(d),dtype=torch.float32)

def make_windows(data, n_his):
    X,Y=[],[]
    for t in range(n_his,len(data)):
        X.append(data[t-n_his:t]); Y.append(data[t])
    return np.array(X,np.float32), np.array(Y,np.float32)

def delta_to_labels(delta, threshold):
    labels = np.ones_like(delta, dtype=np.int64)
    labels[delta >  threshold] = 2
    labels[delta < -threshold] = 0
    return labels


# =============================================================================
# METRICS
# =============================================================================
def calc_clf(yt_flat, yp_flat):
    acc    = float(accuracy_score(yt_flat, yp_flat))
    mac_f1 = float(f1_score(yt_flat, yp_flat, average="macro",    zero_division=0))
    wgt_f1 = float(f1_score(yt_flat, yp_flat, average="weighted", zero_division=0))
    p,r,f,_= precision_recall_fscore_support(
        yt_flat, yp_flat, labels=[0,1,2], zero_division=0)
    cm = confusion_matrix(yt_flat, yp_flat, labels=[0,1,2])
    return {
        "accuracy":acc, "macro_f1":mac_f1, "weighted_f1":wgt_f1,
        "down_p":float(p[0]),"down_r":float(r[0]),"down_f1":float(f[0]),
        "stbl_p":float(p[1]),"stbl_r":float(r[1]),"stbl_f1":float(f[1]),
        "up_p"  :float(p[2]),"up_r"  :float(r[2]),"up_f1"  :float(f[2]),
        "cm":cm,
    }

def mcnemar_p(yt, yp_sc, yp_other):
    """McNemar test: SC-STGCN vs other model."""
    sc_correct    = (yp_sc    == yt)
    other_correct = (yp_other == yt)
    b = int(( sc_correct & ~other_correct).sum())   # SC right, other wrong
    c = int((~sc_correct &  other_correct).sum())   # SC wrong, other right
    n = b + c
    if n == 0:
        return 1.0
    stat = (abs(b-c)-1)**2 / n   # continuity correction
    return float(1 - chi2.cdf(stat, df=1))

def calc_portfolio(delta_te, pred_labels, dv01=DV01, cash=INIT_CASH):
    """
    UP (2)     -> short CDS (sell protection): profit when Δs > 0
    DOWN (0)   -> long  CDS (buy  protection): profit when Δs < 0
    STABLE (1) -> no position
    pnl_t = DV01 * sum_i [ w_i * Δs_i ]
      w_i = +1/K if UP, -1/K if DOWN, 0 if STABLE
    """
    T, N = delta_te.shape
    pnl = []
    for t in range(T):
        w = np.zeros(N)
        up_idx   = np.where(pred_labels[t] == 2)[0]
        down_idx = np.where(pred_labels[t] == 0)[0]
        K_up   = max(1, len(up_idx))
        K_down = max(1, len(down_idx))
        if len(up_idx)   > 0: w[up_idx]   = +1/K_up
        if len(down_idx) > 0: w[down_idx] = -1/K_down
        pnl.append(dv01 * (w * delta_te[t]).sum())
    pnl = np.array(pnl)
    eq  = cash + np.cumsum(pnl)
    sharpe   = float(np.sqrt(52)*pnl.mean()/(pnl.std()+1e-10))
    mean_ret = float(pnl.mean())
    std_ret  = float(pnl.std())
    mdd      = float(np.min(eq/np.maximum.accumulate(eq)-1))
    hit      = float((pnl>0).mean())
    return {"sharpe":sharpe,"mean_ret":mean_ret,
            "std_ret":std_ret,"max_dd":mdd,"hit":hit}


# =============================================================================
# MODELS
# =============================================================================
class GCNLayer(nn.Module):
    def __init__(self,i,o): super().__init__(); self.W=nn.Linear(i,o,bias=False)
    def forward(self,x,A): return torch.relu(self.W(A@x))

class TemporalAttn(nn.Module):
    def __init__(self,d,heads,ff,drop):
        super().__init__()
        self.attn=nn.MultiheadAttention(d,heads,dropout=drop,batch_first=True)
        self.ff=nn.Sequential(nn.Linear(d,ff),nn.GELU(),nn.Dropout(drop),nn.Linear(ff,d))
        self.n1=nn.LayerNorm(d); self.n2=nn.LayerNorm(d)
    def forward(self,x):
        a,_=self.attn(x,x,x); x=self.n1(x+a); return self.n2(x+self.ff(x))

class SCSTGCN_CLF(nn.Module):
    def __init__(self,N,n_his,gcn_h,att_h,heads,ff_h,drop,A_up,A_dn,nc=3):
        super().__init__()
        self.register_buffer("A_up",A_up); self.register_buffer("A_dn",A_dn)
        self.gcn_up=GCNLayer(n_his,gcn_h); self.gcn_dn=GCNLayer(n_his,gcn_h)
        self.gate=nn.Linear(gcn_h*2,gcn_h); self.proj=nn.Linear(gcn_h,att_h)
        self.attn=TemporalAttn(att_h,heads,ff_h,drop)
        self.head=nn.Linear(att_h,nc); self.drop=nn.Dropout(drop)
    def forward(self,x):
        xT=x.permute(0,2,1)
        h_up=self.gcn_up(xT,self.A_up); h_dn=self.gcn_dn(xT,self.A_dn)
        g=torch.sigmoid(self.gate(torch.cat([h_up,h_dn],-1)))
        h=g*h_up+(1-g)*h_dn; h=self.drop(self.proj(h))
        return self.head(self.attn(h))   # (B,N,3)

class VSTGCN_CLF(nn.Module):
    def __init__(self,N,n_his,gcn_h,A,nc=3):
        super().__init__(); self.register_buffer("A",A)
        self.gcn=GCNLayer(n_his,gcn_h); self.proj=nn.Linear(gcn_h,gcn_h)
        self.head=nn.Linear(gcn_h,nc)
    def forward(self,x):
        h=self.gcn(x.permute(0,2,1),self.A)
        return self.head(torch.relu(self.proj(h)))

class LSTM_CLF(nn.Module):
    def __init__(self,N,h=128,layers=2,drop=0.20,nc=3):
        super().__init__()
        self.lstm=nn.LSTM(N,h,layers,batch_first=True,
                          dropout=drop if layers>1 else 0.)
        self.head=nn.Linear(h,N*nc); self.N=N; self.nc=nc
    def forward(self,x):
        out,_=self.lstm(x)
        return self.head(out[:,-1,:]).view(-1,self.N,self.nc)

class Transformer_CLF(nn.Module):
    def __init__(self,N,seq_len,d_model=64,n_heads=4,ff=128,
                 n_layers=2,drop=0.1,nc=3):
        super().__init__()
        self.emb=nn.Linear(N,d_model)
        enc=nn.TransformerEncoderLayer(d_model=d_model,nhead=n_heads,
            dim_feedforward=ff,dropout=drop,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(enc,num_layers=n_layers)
        self.norm=nn.LayerNorm(d_model)
        self.head=nn.Linear(d_model,N*nc); self.N=N; self.nc=nc
    def forward(self,x):
        h=self.emb(x); h=self.encoder(h)
        return self.head(self.norm(h[:,-1,:])).view(-1,self.N,self.nc)


# =============================================================================
# TRAINING
# =============================================================================
def train_clf(model,Xtr,Ltr,Xva,Lva,device,epochs=300):
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=LR*0.01)
    lfn=nn.CrossEntropyLoss()
    Xt=torch.tensor(Xtr).to(device)
    Yt=torch.tensor(Ltr,dtype=torch.long).to(device)
    Xv=torch.tensor(Xva).to(device)
    Yv=torch.tensor(Lva,dtype=torch.long).to(device)
    best,no_imp,state=float("inf"),0,None
    for ep in range(epochs):
        model.train()
        idx=torch.randperm(len(Xt),device=device)
        for i in range(0,len(idx),BATCH):
            b=idx[i:i+BATCH]; opt.zero_grad()
            lg=model(Xt[b]); B,N,C=lg.shape
            lfn(lg.view(B*N,C),Yt[b].view(B*N)).backward()
            nn.utils.clip_grad_norm_(model.parameters(),CLIP); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            lv=model(Xv); Bv,Nv,Cv=lv.shape
            vl=lfn(lv.view(Bv*Nv,Cv),Yv.view(Bv*Nv)).item()
        if vl<best-1e-6: best,no_imp=vl,0; state={k:v.cpu().clone() for k,v in model.state_dict().items()}
        else:
            no_imp+=1
            if no_imp>=PATIENCE: break
    if state: model.load_state_dict(state)
    return model,ep+1

def pred_clf(model,X,device):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X).to(device)).argmax(-1).cpu().numpy()


# =============================================================================
# RUN FOLD
# =============================================================================
def run_fold(delta, labels, N, Ac_t, Au_t, Ad_t,
             device, epochs, tr_end, va_end, te_end, threshold):
    sc=StandardScaler()
    tr_s=sc.fit_transform(delta[:tr_end])
    va_s=sc.transform(delta[tr_end:va_end])
    te_s=sc.transform(delta[va_end:te_end])

    lab_tr=labels[:tr_end]; lab_va=labels[tr_end:va_end]; lab_te=labels[va_end:te_end]
    Xtr,Ytr=make_windows(tr_s,N_HIS)
    Xva,Yva=make_windows(va_s,N_HIS)
    Xte,_  =make_windows(te_s,N_HIS)
    _,Ltr=make_windows(lab_tr,N_HIS)
    _,Lva=make_windows(lab_va,N_HIS)
    _,Lte=make_windows(lab_te,N_HIS)

    te_delta = delta[va_end:te_end][N_HIS:]  # bp scale for portfolio

    clf_results  = {}
    port_results = {}
    pred_store   = {}

    def register(name, pred):
        pred_store[name] = pred
        clf_results[name]  = calc_clf(Lte.flatten(), pred.flatten())
        port_results[name] = calc_portfolio(te_delta, pred)

    # AR(1) — regresyon sonra threshold
    ps=np.zeros_like(te_s)
    for i in range(N):
        y=tr_s[:,i]; phi=np.corrcoef(y[:-1],y[1:])[0,1] if y.std()>1e-8 else 0.
        mu=y.mean()*(1-phi); last=tr_s[-1,i]
        for t in range(len(te_s)):
            ps[t,i]=mu+phi*last; last=te_s[t,i]
    ps_bp=sc.inverse_transform(ps)[N_HIS:]
    register("AR(1)", delta_to_labels(ps_bp, threshold))

    # LSTM
    lstm=LSTM_CLF(N).to(device)
    lstm,ep=train_clf(lstm,Xtr,Ltr,Xva,Lva,device,epochs)
    register("LSTM",pred_clf(lstm,Xte,device)); print(f"    LSTM ep={ep}")

    # XGBoost
    if HAS_XGB:
        Xtf=Xtr.reshape(len(Xtr),-1); Xtef=Xte.reshape(len(Xte),-1)
        ps_x=np.zeros((len(Xte),N),dtype=np.int64)
        for i in range(N):
            uniq=np.unique(Ltr[:,i])
            if len(uniq)<2:
                # Tek sinif: hepsini o sinif yap
                ps_x[:,i]=int(uniq[0])
                continue
            m=XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,
                            subsample=0.8,colsample_bytree=0.8,
                            eval_metric="mlogloss",random_state=SEED,
                            n_jobs=-1,verbosity=0)
            m.fit(Xtf,Ltr[:,i]); ps_x[:,i]=m.predict(Xtef)
        register("XGBoost",ps_x)

    # Transformers
    for tname in ["Informer","Autoformer","TimesNet"]:
        m=Transformer_CLF(N,N_HIS,d_model=ATT_H,n_heads=ATT_HEADS,
                          ff=FF_H,n_layers=2,drop=DROPOUT).to(device)
        m,ep=train_clf(m,Xtr,Ltr,Xva,Lva,device,epochs)
        register(tname,pred_clf(m,Xte,device)); print(f"    {tname} ep={ep}")

    # V-STGCN
    vs=VSTGCN_CLF(N,N_HIS,GCN_H,Ac_t).to(device)
    vs,ep=train_clf(vs,Xtr,Ltr,Xva,Lva,device,epochs)
    register("V-STGCN",pred_clf(vs,Xte,device)); print(f"    V-STGCN ep={ep}")

    # SC-STGCN
    sc_m=SCSTGCN_CLF(N,N_HIS,GCN_H,ATT_H,ATT_HEADS,FF_H,DROPOUT,Au_t,Ad_t).to(device)
    sc_m,ep=train_clf(sc_m,Xtr,Ltr,Xva,Lva,device,epochs)
    register("SC-STGCN",pred_clf(sc_m,Xte,device)); print(f"    SC-STGCN ep={ep}")

    # McNemar p-values (SC-STGCN vs others)
    mc_p={}
    yt_flat=Lte.flatten(); sc_flat=pred_store["SC-STGCN"].flatten()
    for name in pred_store:
        if name=="SC-STGCN": continue
        mc_p[name]=mcnemar_p(yt_flat,sc_flat,pred_store[name].flatten())

    return clf_results, port_results, mc_p, Lte, te_delta


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs",   type=int,   default=300)
    parser.add_argument("--device",   default="auto")
    parser.add_argument("--threshold",type=float, default=0.5)
    args=parser.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") \
           if args.device=="auto" else torch.device(args.device)
    make_dirs(); set_seed()

    SEP="="*65
    print(SEP)
    print(f"  3-Sinifli CDS Yonu Tahmini")
    print(f"  Dataset: top50_degree | Threshold: ±{args.threshold} bp")
    print(f"  DOWN(<-{args.threshold})  STABLE(±{args.threshold})  UP(>{args.threshold})")
    print(f"  epochs={args.epochs} | device={device}")
    print(SEP)

    # Veri
    df=pd.read_csv(DEGREE_DIR/"ve1.csv")
    try: df.iloc[:,0].astype(float)
    except: df=df.iloc[:,1:]
    level=df.values.astype(np.float32)
    delta=np.diff(level,axis=0); T,N=delta.shape
    dates=pd.date_range(start="2015-01-09",periods=T,freq="W-FRI")
    covid_week=int(np.searchsorted(dates,pd.Timestamp("2020-03-01")))
    delta_pre=delta[:covid_week]
    labels=delta_to_labels(delta_pre,args.threshold)

    lf=labels.flatten()
    print(f"\n  Pre-COVID: {covid_week} hafta  N={N}")
    print(f"  DOWN: {(lf==0).mean():.1%}  STABLE: {(lf==1).mean():.1%}  UP: {(lf==2).mean():.1%}")

    # Adjacency
    Ac_t=norm_adj(load_npz(str(DEGREE_DIR/"adj.npz")).toarray().astype(np.float32)).to(device)
    Au_t=norm_adj(load_npz(str(DEGREE_DIR/"adj_sup.npz")).toarray().astype(np.float32)).to(device)
    Ad_t=norm_adj(load_npz(str(DEGREE_DIR/"adj_cus.npz")).toarray().astype(np.float32)).to(device)

    # 2 fold
    folds=[(int(covid_week*0.65),int(covid_week*0.78),int(covid_week*0.92)),
           (int(covid_week*0.78),int(covid_week*0.92),covid_week)]
    print(f"\n  Fold yapisi:")
    for i,(a,b,c) in enumerate(folds):
        print(f"    Fold {i+1}: test[{b}:{c}] ({c-b} hafta | {dates[b].date()} -> {dates[c-1].date()})")

    model_names=["AR(1)","LSTM","Informer","Autoformer","TimesNet","V-STGCN","SC-STGCN"]
    if HAS_XGB: model_names.insert(2,"XGBoost")

    all_clf=[]; all_port=[]; all_mc=[]
    for fi,(tr_end,va_end,te_end) in enumerate(folds):
        print(f"\n  --- Fold {fi+1}/2 ---")
        set_seed(SEED+fi)
        cr,pr,mc,_,_=run_fold(delta_pre,labels,N,Ac_t,Au_t,Ad_t,
                               device,args.epochs,tr_end,va_end,te_end,
                               args.threshold)
        all_clf.append(cr); all_port.append(pr); all_mc.append(mc)

        print(f"\n  Fold {fi+1}:")
        print(f"  {'Model':<14} {'Acc':>7} {'MacF1':>7} {'D-F1':>7} {'S-F1':>7} {'U-F1':>7}  Sharpe")
        print(f"  {'-'*65}")
        for name in model_names:
            if name not in cr: continue
            c=cr[name]; p=pr[name]
            print(f"  {name:<14} {c['accuracy']:>7.4f} {c['macro_f1']:>7.4f} "
                  f"{c['down_f1']:>7.4f} {c['stbl_f1']:>7.4f} "
                  f"{c['up_f1']:>7.4f}  {p['sharpe']:>6.4f}")

    # Aggregate
    clf_m=["accuracy","macro_f1","weighted_f1",
           "down_p","down_r","down_f1",
           "stbl_p","stbl_r","stbl_f1",
           "up_p","up_r","up_f1"]
    prt_m=["sharpe","mean_ret","std_ret","max_dd","hit"]

    agg_c={}; agg_p={}; agg_mc={}
    for name in model_names:
        for agg,all_r,mlist in [(agg_c,all_clf,clf_m),(agg_p,all_port,prt_m)]:
            vals={m:[] for m in mlist}
            for fr in all_r:
                if name in fr:
                    for m in mlist:
                        v=fr[name].get(m,float("nan"))
                        if not np.isnan(v): vals[m].append(v)
            agg[name]={}
            for m in mlist:
                v=np.array(vals[m])
                agg[name][f"{m}_mean"]=float(np.mean(v)) if len(v)>0 else float("nan")
                agg[name][f"{m}_std"] =float(np.std(v))  if len(v)>0 else float("nan")
        # McNemar ortalama
        if name!="SC-STGCN":
            ps=[fr[name] for fr in all_mc if name in fr]
            agg_mc[name]=float(np.mean(ps)) if ps else float("nan")

    # ── TABLO 1: Classification ───────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  TABLO 1 — CLASSIFICATION  (threshold=±{args.threshold} bp)")
    print(f"  {'Model':<14} {'Acc':>8} {'±':>6} {'MacF1':>8} {'±':>6} "
          f"{'WgtF1':>8} {'±':>6}  McNemar-p")
    print(f"  {'-'*80}")
    for name in model_names:
        a=agg_c[name]; mc_p=agg_mc.get(name,float("nan"))
        star=("***" if mc_p<0.01 else "**" if mc_p<0.05
              else "*" if mc_p<0.10 else "n.s." if not np.isnan(mc_p) else "—")
        print(f"  {name:<14} "
              f"{a['accuracy_mean']:>8.4f} {a['accuracy_std']:>6.4f} "
              f"{a['macro_f1_mean']:>8.4f} {a['macro_f1_std']:>6.4f} "
              f"{a['weighted_f1_mean']:>8.4f} {a['weighted_f1_std']:>6.4f}  "
              f"{mc_p:>6.3f}{star}")

    # ── TABLO 2: Per-Class ────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  TABLO 2 — PER-CLASS PERFORMANCE")
    print(f"  {'Model':<14}   {'DOWN':^21}   {'STABLE':^21}   {'UP':^21}")
    print(f"  {'':14}   {'P':>6} {'R':>6} {'F1':>6}   {'P':>6} {'R':>6} {'F1':>6}   {'P':>6} {'R':>6} {'F1':>6}")
    print(f"  {'-'*80}")
    for name in model_names:
        a=agg_c[name]
        print(f"  {name:<14}   "
              f"{a['down_p_mean']:>6.4f} {a['down_r_mean']:>6.4f} {a['down_f1_mean']:>6.4f}   "
              f"{a['stbl_p_mean']:>6.4f} {a['stbl_r_mean']:>6.4f} {a['stbl_f1_mean']:>6.4f}   "
              f"{a['up_p_mean']:>6.4f} {a['up_r_mean']:>6.4f} {a['up_f1_mean']:>6.4f}")

    # ── TABLO 3: Portfolio ────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  TABLO 3 — PORTFOLIO  (UP->short, DOWN->long, STABLE->flat)")
    print(f"  {'Model':<14} {'Sharpe':>8} {'±':>6} {'MeanRet':>9} {'Std':>8} "
          f"{'MaxDD':>8} {'Hit':>7}")
    print(f"  {'-'*70}")
    for name in model_names:
        a=agg_p[name]
        print(f"  {name:<14} "
              f"{a['sharpe_mean']:>8.4f} {a['sharpe_std']:>6.4f} "
              f"{a['mean_ret_mean']:>9.4f} "
              f"{a['std_ret_mean']:>8.4f} "
              f"{a['max_dd_mean']:>8.4f} "
              f"{a['hit_mean']:>7.4f}")

    # ── Ablation ──────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  ABLATION: SC-STGCN vs V-STGCN")
    print(f"  {'Metrik':<16} {'SC':>8} {'V':>8} {'Diff':>8}")
    print(f"  {'-'*45}")
    for m,label in [("accuracy","Accuracy"),("macro_f1","MacroF1"),
                    ("down_f1","DOWN-F1"),("stbl_f1","STABLE-F1"),
                    ("up_f1","UP-F1"),("sharpe","Sharpe")]:
        if m in clf_m:
            sc_v=agg_c["SC-STGCN"][f"{m}_mean"]
            vs_v=agg_c["V-STGCN"][f"{m}_mean"]
        else:
            sc_v=agg_p["SC-STGCN"][f"{m}_mean"]
            vs_v=agg_p["V-STGCN"][f"{m}_mean"]
        diff=sc_v-vs_v
        better="SC ✓" if diff>0 else "V"
        print(f"  {label:<16} {sc_v:>+8.4f} {vs_v:>+8.4f} {diff:>+8.4f}  {better}")

    # CSV + LaTeX
    rows=[]
    for name in model_names:
        row={"Model":name}
        for m in clf_m: row[f"clf_{m}"]=agg_c[name][f"{m}_mean"]
        for m in prt_m: row[f"prt_{m}"]=agg_p[name][f"{m}_mean"]
        row["mcnemar_p"]=agg_mc.get(name,float("nan"))
        rows.append(row)
    out_csv=METRICS_DIR/f"threeway_cv_t{args.threshold}.csv"
    pd.DataFrame(rows).to_csv(out_csv,index=False)

    # LaTeX
    tex=METRICS_DIR/f"threeway_cv_t{args.threshold}_tables.tex"
    with open(tex,"w") as f:
        f.write(f"% Threshold = ±{args.threshold} bp\n\n")
        # Tablo 1
        f.write("% TABLE 1 - CLASSIFICATION\n")
        f.write("\\begin{table}[ht]\\centering\n")
        f.write(f"\\caption{{Three-way classification results (threshold $=\\pm{args.threshold}$~bp)}}\n")
        f.write("\\label{tab:clf3}\n")
        f.write("\\begin{tabular}{lccccccc}\\toprule\n")
        f.write("Model & Acc & $\\pm$ & Macro-F1 & $\\pm$ & Wtd-F1 & $\\pm$ & McNemar $p$ \\\\\\midrule\n")
        for name in model_names:
            a=agg_c[name]; mc_p=agg_mc.get(name,float("nan"))
            star=("$^{***}$" if mc_p<0.01 else "$^{**}$" if mc_p<0.05
                  else "$^{*}$" if mc_p<0.10 else "")
            bold="\\textbf{" if name=="SC-STGCN" else ""
            bolde="}" if name=="SC-STGCN" else ""
            f.write(f"{bold}{name}{bolde} & {a['accuracy_mean']:.4f} & {a['accuracy_std']:.4f} & "
                    f"{a['macro_f1_mean']:.4f} & {a['macro_f1_std']:.4f} & "
                    f"{a['weighted_f1_mean']:.4f} & {a['weighted_f1_std']:.4f} & "
                    f"{mc_p:.3f}{star} \\\\\n")
        f.write("\\bottomrule\\end{tabular}\\end{table}\n\n")
        # Tablo 2
        f.write("% TABLE 2 - PER-CLASS\n")
        f.write("\\begin{table}[ht]\\centering\n")
        f.write("\\caption{Per-class precision, recall, and F1}\n\\label{tab:perclass}\n")
        f.write("\\begin{tabular}{l ccc ccc ccc}\\toprule\n")
        f.write(" & \\multicolumn{3}{c}{DOWN} & \\multicolumn{3}{c}{STABLE} & \\multicolumn{3}{c}{UP} \\\\\n")
        f.write("\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n")
        f.write("Model & P & R & F1 & P & R & F1 & P & R & F1 \\\\\\midrule\n")
        for name in model_names:
            a=agg_c[name]
            bold="\\textbf{" if name=="SC-STGCN" else ""
            bolde="}" if name=="SC-STGCN" else ""
            f.write(f"{bold}{name}{bolde} & "
                    f"{a['down_p_mean']:.3f} & {a['down_r_mean']:.3f} & {a['down_f1_mean']:.3f} & "
                    f"{a['stbl_p_mean']:.3f} & {a['stbl_r_mean']:.3f} & {a['stbl_f1_mean']:.3f} & "
                    f"{a['up_p_mean']:.3f} & {a['up_r_mean']:.3f} & {a['up_f1_mean']:.3f} \\\\\n")
        f.write("\\bottomrule\\end{tabular}\\end{table}\n\n")
        # Tablo 3
        f.write("% TABLE 3 - PORTFOLIO\n")
        f.write("\\begin{table}[ht]\\centering\n")
        f.write("\\caption{Portfolio results: UP$\\to$short, DOWN$\\to$long, STABLE$\\to$flat}\n")
        f.write("\\label{tab:port3}\n")
        f.write("\\begin{tabular}{lcccccc}\\toprule\n")
        f.write("Model & Sharpe & $\\pm$ & Mean Ret (bp) & Std (bp) & Max DD & Hit \\\\\\midrule\n")
        for name in model_names:
            a=agg_p[name]
            bold="\\textbf{" if name=="SC-STGCN" else ""
            bolde="}" if name=="SC-STGCN" else ""
            f.write(f"{bold}{name}{bolde} & {a['sharpe_mean']:.4f} & {a['sharpe_std']:.4f} & "
                    f"{a['mean_ret_mean']:.4f} & {a['std_ret_mean']:.4f} & "
                    f"{a['max_dd_mean']:.4f} & {a['hit_mean']:.4f} \\\\\n")
        f.write("\\bottomrule\\end{tabular}\\end{table}\n")

    print(f"\n  [OK] {out_csv.name}")
    print(f"  [OK] {tex.name}")
    print(f"\n{SEP}  TAMAMLANDI\n{SEP}")


if __name__=="__main__":
    t0=time.time()
    main()
    print(f"  Total: {(time.time()-t0)/60:.1f} min")
