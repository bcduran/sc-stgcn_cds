"""
src/models/threeway_cv2.py
===========================
3-Sinifli CDS Spread Yonu Tahmini
Yontem: Regresyon -> Validation-Optimal Threshold -> 3 Sinif

Her fold icin:
  1. Her model icin regression tahminleri al
  2. Validation set'te optimal DOWN/UP threshold bul
     (Macro-F1'i maksimize eder)
  3. Test set'te bu threshold'u uygula

Avantaj:
  - SC-STGCN regresyonda zaten R2=+0.040 uretiyor
  - Model yeniden egitmeye gerek yok
  - Her model kendi optimal threshold'unu buluyor

3 Tablo:
  1. Classification : Acc, Macro-F1, Weighted-F1, McNemar-p
  2. Per-Class      : DOWN / STABLE / UP P, R, F1
  3. Portfolio      : UP->short, DOWN->long, STABLE->flat

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
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

import matplotlib; matplotlib.use("Agg")

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

def make_windows(data,n_his):
    X,Y=[],[]
    for t in range(n_his,len(data)):
        X.append(data[t-n_his:t]); Y.append(data[t])
    return np.array(X,np.float32),np.array(Y,np.float32)

def to_labels(delta, thr_down, thr_up):
    """Regresyon tahmini -> 3 sinif."""
    labels = np.ones_like(delta, dtype=np.int64)   # STABLE
    labels[delta >  thr_up]   = 2   # UP
    labels[delta < -thr_down] = 0   # DOWN
    return labels

def find_optimal_threshold(y_true_delta, y_pred_delta,
                           thr_grid=None):
    """
    Validation set'te Macro-F1'i maksimize eden threshold'u bul.
    Asimetrik: thr_down ve thr_up ayri ayri optimize edilebilir.
    Basitlik icin simetrik: thr_down = thr_up = thr
    """
    if thr_grid is None:
        thr_grid = np.arange(0.1, 3.0, 0.1)

    best_thr = 0.5
    best_f1  = -1.
    y_flat   = y_true_delta.flatten()
    p_flat   = y_pred_delta.flatten()

    for thr in thr_grid:
        lbl_true = to_labels(y_flat,  thr, thr)
        lbl_pred = to_labels(p_flat,  thr, thr)
        # En az 2 sinif olmali
        if len(np.unique(lbl_pred)) < 2:
            continue
        f1 = f1_score(lbl_true, lbl_pred,
                      average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1  = f1
            best_thr = thr

    return best_thr, best_f1


# =============================================================================
# METRICS
# =============================================================================
def calc_clf(yt_flat, yp_flat):
    acc    = float(accuracy_score(yt_flat, yp_flat))
    mac_f1 = float(f1_score(yt_flat, yp_flat, average="macro",    zero_division=0))
    wgt_f1 = float(f1_score(yt_flat, yp_flat, average="weighted", zero_division=0))
    p,r,f,_= precision_recall_fscore_support(
        yt_flat, yp_flat, labels=[0,1,2], zero_division=0)
    return {
        "accuracy":acc,"macro_f1":mac_f1,"weighted_f1":wgt_f1,
        "down_p":float(p[0]),"down_r":float(r[0]),"down_f1":float(f[0]),
        "stbl_p":float(p[1]),"stbl_r":float(r[1]),"stbl_f1":float(f[1]),
        "up_p"  :float(p[2]),"up_r"  :float(r[2]),"up_f1"  :float(f[2]),
    }

def mcnemar_p(yt, yp_sc, yp_other):
    sc_ok    = (yp_sc    == yt)
    other_ok = (yp_other == yt)
    b = int(( sc_ok & ~other_ok).sum())
    c = int((~sc_ok &  other_ok).sum())
    n = b + c
    if n == 0: return 1.0
    stat = (abs(b-c)-1)**2 / n
    return float(1 - chi2.cdf(stat, df=1))

def calc_portfolio(delta_te, pred_labels, dv01=DV01, cash=INIT_CASH):
    T,N = delta_te.shape; pnl=[]
    for t in range(T):
        w = np.zeros(N)
        up_idx   = np.where(pred_labels[t]==2)[0]
        down_idx = np.where(pred_labels[t]==0)[0]
        if len(up_idx)>0:   w[up_idx]   = +1/len(up_idx)
        if len(down_idx)>0: w[down_idx] = -1/len(down_idx)
        pnl.append(dv01*(w*delta_te[t]).sum())
    pnl=np.array(pnl); eq=cash+np.cumsum(pnl)
    return {
        "sharpe"  : float(np.sqrt(52)*pnl.mean()/(pnl.std()+1e-10)),
        "mean_ret": float(pnl.mean()),
        "std_ret" : float(pnl.std()),
        "max_dd"  : float(np.min(eq/np.maximum.accumulate(eq)-1)),
        "hit"     : float((pnl>0).mean()),
    }


# =============================================================================
# MODELS (Regression — aynı full_cv.py'deki)
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

class SCSTGCN(nn.Module):
    def __init__(self,N,n_his,gcn_h,att_h,heads,ff_h,drop,A_up,A_dn):
        super().__init__()
        self.register_buffer("A_up",A_up); self.register_buffer("A_dn",A_dn)
        self.gcn_up=GCNLayer(n_his,gcn_h); self.gcn_dn=GCNLayer(n_his,gcn_h)
        self.gate=nn.Linear(gcn_h*2,gcn_h); self.proj=nn.Linear(gcn_h,att_h)
        self.attn=TemporalAttn(att_h,heads,ff_h,drop)
        self.head=nn.Linear(att_h,1); self.drop=nn.Dropout(drop)
    def forward(self,x):
        xT=x.permute(0,2,1)
        h_up=self.gcn_up(xT,self.A_up); h_dn=self.gcn_dn(xT,self.A_dn)
        g=torch.sigmoid(self.gate(torch.cat([h_up,h_dn],-1)))
        h=g*h_up+(1-g)*h_dn; h=self.drop(self.proj(h))
        return self.head(self.attn(h)).squeeze(-1)

class VSTGCN(nn.Module):
    def __init__(self,N,n_his,gcn_h,A):
        super().__init__(); self.register_buffer("A",A)
        self.gcn=GCNLayer(n_his,gcn_h); self.proj=nn.Linear(gcn_h,gcn_h)
        self.head=nn.Linear(gcn_h,1)
    def forward(self,x):
        h=self.gcn(x.permute(0,2,1),self.A)
        return self.head(torch.relu(self.proj(h))).squeeze(-1)

class LSTMModel(nn.Module):
    def __init__(self,N,h=128,layers=2,drop=0.20):
        super().__init__()
        self.lstm=nn.LSTM(N,h,layers,batch_first=True,
                          dropout=drop if layers>1 else 0.)
        self.head=nn.Linear(h,N)
    def forward(self,x):
        out,_=self.lstm(x); return self.head(out[:,-1,:])

class TransformerReg(nn.Module):
    def __init__(self,N,seq_len,d_model=64,n_heads=4,ff=128,n_layers=2,drop=0.1):
        super().__init__()
        self.emb=nn.Linear(N,d_model)
        enc=nn.TransformerEncoderLayer(d_model=d_model,nhead=n_heads,
            dim_feedforward=ff,dropout=drop,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(enc,num_layers=n_layers)
        self.norm=nn.LayerNorm(d_model)
        self.head=nn.Linear(d_model,N)
    def forward(self,x):
        h=self.emb(x); h=self.encoder(h)
        return self.head(self.norm(h[:,-1,:]))


def train_nn(model,Xtr,Ytr,Xva,Yva,device,epochs=300):
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=LR*0.01)
    lfn=nn.HuberLoss(delta=1.0)
    Xt=torch.tensor(Xtr).to(device); Yt=torch.tensor(Ytr).to(device)
    Xv=torch.tensor(Xva).to(device); Yv=torch.tensor(Yva).to(device)
    best,no_imp,state=float("inf"),0,None
    for ep in range(epochs):
        model.train()
        idx=torch.randperm(len(Xt),device=device)
        for i in range(0,len(idx),BATCH):
            b=idx[i:i+BATCH]; opt.zero_grad()
            lfn(model(Xt[b]),Yt[b]).backward()
            nn.utils.clip_grad_norm_(model.parameters(),CLIP); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad(): vl=lfn(model(Xv),Yv).item()
        if vl<best-1e-6: best,no_imp=vl,0; state={k:v.cpu().clone() for k,v in model.state_dict().items()}
        else:
            no_imp+=1
            if no_imp>=PATIENCE: break
    if state: model.load_state_dict(state)
    return model,ep+1

def pred_nn(model,X,device):
    model.eval()
    with torch.no_grad(): return model(torch.tensor(X).to(device)).cpu().numpy()


# =============================================================================
# RUN FOLD
# =============================================================================
def run_fold(delta,N,Ac_t,Au_t,Ad_t,device,epochs,
             tr_end,va_end,te_end):
    sc=StandardScaler()
    tr_s=sc.fit_transform(delta[:tr_end])
    va_s=sc.transform(delta[tr_end:va_end])
    te_s=sc.transform(delta[va_end:te_end])

    Xtr,Ytr=make_windows(tr_s,N_HIS)
    Xva,Yva=make_windows(va_s,N_HIS)
    Xte,_  =make_windows(te_s,N_HIS)

    # Gercek bp cinsinden
    va_orig = delta[tr_end:va_end][N_HIS:]
    te_orig = delta[va_end:te_end][N_HIS:]

    reg_preds = {}   # model -> (va_pred_bp, te_pred_bp)

    def register(name, va_p_scaled, te_p_scaled):
        va_bp = sc.inverse_transform(va_p_scaled)
        te_bp = sc.inverse_transform(te_p_scaled)
        reg_preds[name] = (va_bp, te_bp)

    # AR(1)
    ps_va=np.zeros_like(va_s); ps_te=np.zeros_like(te_s)
    for i in range(N):
        y=tr_s[:,i]; phi=np.corrcoef(y[:-1],y[1:])[0,1] if y.std()>1e-8 else 0.
        mu=y.mean()*(1-phi); last=tr_s[-1,i]
        for t in range(len(va_s)):
            ps_va[t,i]=mu+phi*last; last=va_s[t,i]
        last=tr_s[-1,i]
        for t in range(len(te_s)):
            ps_te[t,i]=mu+phi*last; last=te_s[t,i]
    register("AR(1)", ps_va[N_HIS:], ps_te[N_HIS:])

    # LSTM
    lstm=LSTMModel(N).to(device)
    lstm,ep=train_nn(lstm,Xtr,Ytr,Xva,Yva,device,epochs)
    register("LSTM", pred_nn(lstm,make_windows(va_s,N_HIS)[0],device),
             pred_nn(lstm,Xte,device)); print(f"    LSTM ep={ep}")

    # XGBoost
    if HAS_XGB:
        Xtf=Xtr.reshape(len(Xtr),-1)
        Xvf=make_windows(va_s,N_HIS)[0].reshape(-1,N*N_HIS)
        Xtef=Xte.reshape(len(Xte),-1)
        pv=np.zeros((len(Xvf),N)); pt=np.zeros((len(Xtef),N))
        for i in range(N):
            m=XGBRegressor(n_estimators=300,max_depth=4,learning_rate=0.05,
                           subsample=0.8,colsample_bytree=0.8,
                           objective="reg:squarederror",random_state=SEED,
                           tree_method="hist",n_jobs=-1,verbosity=0)
            m.fit(Xtf,Ytr[:,i]); pv[:,i]=m.predict(Xvf); pt[:,i]=m.predict(Xtef)
        register("XGBoost", pv, pt)

    # Transformers
    for tname in ["Informer","Autoformer","TimesNet"]:
        m=TransformerReg(N,N_HIS,d_model=ATT_H,n_heads=ATT_HEADS,
                         ff=FF_H,n_layers=2,drop=DROPOUT).to(device)
        m,ep=train_nn(m,Xtr,Ytr,Xva,Yva,device,epochs)
        Xva_w=make_windows(va_s,N_HIS)[0]
        register(tname,pred_nn(m,Xva_w,device),
                 pred_nn(m,Xte,device))
        print(f"    {tname} ep={ep}")

    # V-STGCN
    vs=VSTGCN(N,N_HIS,GCN_H,Ac_t).to(device)
    vs,ep=train_nn(vs,Xtr,Ytr,Xva,Yva,device,epochs)
    Xva_w=make_windows(va_s,N_HIS)[0]
    register("V-STGCN",pred_nn(vs,Xva_w,device),
             pred_nn(vs,Xte,device)); print(f"    V-STGCN ep={ep}")

    # SC-STGCN
    sc_m=SCSTGCN(N,N_HIS,GCN_H,ATT_H,ATT_HEADS,FF_H,DROPOUT,Au_t,Ad_t).to(device)
    sc_m,ep=train_nn(sc_m,Xtr,Ytr,Xva,Yva,device,epochs)
    Xva_w=make_windows(va_s,N_HIS)[0]
    register("SC-STGCN",pred_nn(sc_m,Xva_w,device),
             pred_nn(sc_m,Xte,device)); print(f"    SC-STGCN ep={ep}")

    # Threshold optimizasyonu + 3-sinif metrikler
    clf_results={}; port_results={}; opt_thrs={}; pred_labels={}

    for name,(va_bp,te_bp) in reg_preds.items():
        # Validation'da optimal threshold bul
        opt_thr,val_f1 = find_optimal_threshold(va_orig,va_bp)
        opt_thrs[name] = opt_thr

        # Test'te uygula
        lbl_true = to_labels(te_orig.flatten(), opt_thr, opt_thr)
        lbl_pred = to_labels(te_bp.flatten(),   opt_thr, opt_thr)
        pred_labels[name] = to_labels(te_bp, opt_thr, opt_thr)

        clf_results[name]  = calc_clf(lbl_true, lbl_pred)
        port_results[name] = calc_portfolio(te_orig, pred_labels[name])

    # McNemar
    mc_p={}
    sc_lbl = pred_labels["SC-STGCN"].flatten()
    true_lbl = to_labels(te_orig.flatten(),
                         opt_thrs["SC-STGCN"],opt_thrs["SC-STGCN"])
    for name in reg_preds:
        if name=="SC-STGCN": continue
        mc_p[name] = mcnemar_p(true_lbl, sc_lbl,
                                pred_labels[name].flatten())

    print(f"\n  Optimal thresholds (val Macro-F1):")
    for name,thr in opt_thrs.items():
        print(f"    {name:<14}: {thr:.1f} bp")

    return clf_results, port_results, mc_p, te_orig


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=300)
    parser.add_argument("--device",default="auto")
    args=parser.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") \
           if args.device=="auto" else torch.device(args.device)
    make_dirs(); set_seed()

    SEP="="*65
    print(SEP)
    print("  3-Sinifli CDS Yonu: Regresyon + Val-Optimal Threshold")
    print("  Dataset: top50_degree (50/50 baglantili)")
    print("  Threshold: Validation Macro-F1 maksimizasyonu")
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
    print(f"\n  Pre-COVID: {covid_week} hafta  N={N}")

    # Adjacency
    Ac_t=norm_adj(load_npz(str(DEGREE_DIR/"adj.npz")).toarray().astype(np.float32)).to(device)
    Au_t=norm_adj(load_npz(str(DEGREE_DIR/"adj_sup.npz")).toarray().astype(np.float32)).to(device)
    Ad_t=norm_adj(load_npz(str(DEGREE_DIR/"adj_cus.npz")).toarray().astype(np.float32)).to(device)

    # 2 fold
    folds=[(int(covid_week*0.65),int(covid_week*0.78),int(covid_week*0.92)),
           (int(covid_week*0.78),int(covid_week*0.92),covid_week)]
    print(f"\n  Fold yapisi:")
    for i,(a,b,c) in enumerate(folds):
        print(f"    Fold {i+1}: train[0:{a}] val[{a}:{b}] test[{b}:{c}] "
              f"({c-b} hafta | {dates[b].date()} -> {dates[c-1].date()})")

    model_names=["AR(1)","LSTM","Informer","Autoformer","TimesNet",
                 "V-STGCN","SC-STGCN"]
    if HAS_XGB: model_names.insert(2,"XGBoost")

    all_clf=[]; all_port=[]; all_mc=[]
    for fi,(tr_end,va_end,te_end) in enumerate(folds):
        print(f"\n  --- Fold {fi+1}/2 ---")
        set_seed(SEED+fi)
        cr,pr,mc,_=run_fold(delta_pre,N,Ac_t,Au_t,Ad_t,
                             device,args.epochs,tr_end,va_end,te_end)
        all_clf.append(cr); all_port.append(pr); all_mc.append(mc)

        print(f"\n  Fold {fi+1}:")
        print(f"  {'Model':<14} {'Acc':>7} {'MacF1':>7} "
              f"{'D-F1':>7} {'S-F1':>7} {'U-F1':>7}  Sharpe")
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
        if name!="SC-STGCN":
            ps=[fr[name] for fr in all_mc if name in fr]
            agg_mc[name]=float(np.mean(ps)) if ps else float("nan")

    # TABLO 1
    print(f"\n{SEP}")
    print(f"  TABLO 1 — CLASSIFICATION")
    print(f"  (Regresyon + Val-Optimal Threshold -> 3 Sinif)")
    print(f"  {'Model':<14} {'Acc':>8} {'±':>6} {'MacF1':>8} {'±':>6} "
          f"{'WgtF1':>8} {'±':>6}  McNemar-p")
    print(f"  {'-'*82}")
    for name in model_names:
        a=agg_c[name]; mc_p=agg_mc.get(name,float("nan"))
        star=("***" if mc_p<0.01 else "**" if mc_p<0.05
              else "*" if mc_p<0.10 else "n.s." if not np.isnan(mc_p) else "—")
        print(f"  {name:<14} "
              f"{a['accuracy_mean']:>8.4f} {a['accuracy_std']:>6.4f} "
              f"{a['macro_f1_mean']:>8.4f} {a['macro_f1_std']:>6.4f} "
              f"{a['weighted_f1_mean']:>8.4f} {a['weighted_f1_std']:>6.4f}  "
              f"{mc_p:>6.3f}{star}")

    # TABLO 2
    print(f"\n{SEP}")
    print(f"  TABLO 2 — PER-CLASS PERFORMANCE")
    print(f"  {'Model':<14}   {'DOWN':^21}   {'STABLE':^21}   {'UP':^21}")
    print(f"  {'':14}   {'P':>6} {'R':>6} {'F1':>6}   {'P':>6} {'R':>6} {'F1':>6}   {'P':>6} {'R':>6} {'F1':>6}")
    print(f"  {'-'*90}")
    for name in model_names:
        a=agg_c[name]
        print(f"  {name:<14}   "
              f"{a['down_p_mean']:>6.4f} {a['down_r_mean']:>6.4f} {a['down_f1_mean']:>6.4f}   "
              f"{a['stbl_p_mean']:>6.4f} {a['stbl_r_mean']:>6.4f} {a['stbl_f1_mean']:>6.4f}   "
              f"{a['up_p_mean']:>6.4f} {a['up_r_mean']:>6.4f} {a['up_f1_mean']:>6.4f}")

    # TABLO 3
    print(f"\n{SEP}")
    print(f"  TABLO 3 — PORTFOLIO (UP->short, DOWN->long, STABLE->flat)")
    print(f"  {'Model':<14} {'Sharpe':>8} {'±':>6} {'MeanRet':>9} "
          f"{'Std':>8} {'MaxDD':>8} {'Hit':>7}")
    print(f"  {'-'*72}")
    for name in model_names:
        a=agg_p[name]
        print(f"  {name:<14} "
              f"{a['sharpe_mean']:>8.4f} {a['sharpe_std']:>6.4f} "
              f"{a['mean_ret_mean']:>9.4f} "
              f"{a['std_ret_mean']:>8.4f} "
              f"{a['max_dd_mean']:>8.4f} "
              f"{a['hit_mean']:>7.4f}")

    # ABLATION
    print(f"\n{SEP}")
    print(f"  ABLATION: SC-STGCN vs V-STGCN")
    print(f"  {'Metrik':<16} {'SC':>8} {'V':>8} {'Diff':>8}")
    print(f"  {'-'*45}")
    for m,label in [("accuracy","Accuracy"),("macro_f1","MacroF1"),
                    ("down_f1","DOWN-F1"),("stbl_f1","STABLE-F1"),
                    ("up_f1","UP-F1"),("sharpe","Sharpe")]:
        src = agg_c if m in clf_m else agg_p
        sc_v=src["SC-STGCN"][f"{m}_mean"]
        vs_v=src["V-STGCN"][f"{m}_mean"]
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
    out_csv=METRICS_DIR/"threeway_cv2_results.csv"
    pd.DataFrame(rows).to_csv(out_csv,index=False)

    # LaTeX
    tex=METRICS_DIR/"threeway_cv2_tables.tex"
    with open(tex,"w") as f:
        f.write("% 3-Class: Regression + Val-Optimal Threshold\n\n")
        # T1
        f.write("% TABLE 1 - CLASSIFICATION\n\\begin{table}[ht]\\centering\n")
        f.write("\\caption{Three-way classification: regression with validation-optimal threshold}\n")
        f.write("\\label{tab:clf3opt}\n")
        f.write("\\begin{tabular}{lccccccc}\\toprule\n")
        f.write("Model & Acc & $\\pm$ & Macro-F1 & $\\pm$ & Wtd-F1 & $\\pm$ & McNemar $p$ \\\\\\midrule\n")
        for name in model_names:
            a=agg_c[name]; mc_p=agg_mc.get(name,float("nan"))
            star=("$^{***}$" if mc_p<0.01 else "$^{**}$" if mc_p<0.05
                  else "$^{*}$" if mc_p<0.10 else "")
            bo="\\textbf{" if name=="SC-STGCN" else ""
            bc="}" if name=="SC-STGCN" else ""
            f.write(f"{bo}{name}{bc} & {a['accuracy_mean']:.4f} & {a['accuracy_std']:.4f} & "
                    f"{a['macro_f1_mean']:.4f} & {a['macro_f1_std']:.4f} & "
                    f"{a['weighted_f1_mean']:.4f} & {a['weighted_f1_std']:.4f} & "
                    f"{mc_p:.3f}{star} \\\\\n")
        f.write("\\bottomrule\\end{tabular}\\end{table}\n\n")
        # T2
        f.write("% TABLE 2 - PER-CLASS\n\\begin{table}[ht]\\centering\n")
        f.write("\\caption{Per-class precision, recall, F1}\n\\label{tab:perclass3}\n")
        f.write("\\begin{tabular}{l ccc ccc ccc}\\toprule\n")
        f.write(" & \\multicolumn{3}{c}{DOWN} & \\multicolumn{3}{c}{STABLE} & \\multicolumn{3}{c}{UP} \\\\\n")
        f.write("\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n")
        f.write("Model & P & R & F1 & P & R & F1 & P & R & F1 \\\\\\midrule\n")
        for name in model_names:
            a=agg_c[name]
            bo="\\textbf{" if name=="SC-STGCN" else ""; bc="}" if name=="SC-STGCN" else ""
            f.write(f"{bo}{name}{bc} & "
                    f"{a['down_p_mean']:.3f} & {a['down_r_mean']:.3f} & {a['down_f1_mean']:.3f} & "
                    f"{a['stbl_p_mean']:.3f} & {a['stbl_r_mean']:.3f} & {a['stbl_f1_mean']:.3f} & "
                    f"{a['up_p_mean']:.3f} & {a['up_r_mean']:.3f} & {a['up_f1_mean']:.3f} \\\\\n")
        f.write("\\bottomrule\\end{tabular}\\end{table}\n\n")
        # T3
        f.write("% TABLE 3 - PORTFOLIO\n\\begin{table}[ht]\\centering\n")
        f.write("\\caption{Portfolio: UP$\\to$short, DOWN$\\to$long, STABLE$\\to$flat}\n")
        f.write("\\label{tab:port3opt}\n")
        f.write("\\begin{tabular}{lcccccc}\\toprule\n")
        f.write("Model & Sharpe & $\\pm$ & Mean Ret (bp) & Std (bp) & Max DD & Hit \\\\\\midrule\n")
        for name in model_names:
            a=agg_p[name]
            bo="\\textbf{" if name=="SC-STGCN" else ""; bc="}" if name=="SC-STGCN" else ""
            f.write(f"{bo}{name}{bc} & {a['sharpe_mean']:.4f} & {a['sharpe_std']:.4f} & "
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
