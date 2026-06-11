"""
src/models/degree_weighted_cv.py
==================================
Degree-50 + Weighted Adjacency CV

Fark: binary 0/1 yerine agirlikli adjacency
  A[i,j] = normalize(kac kez disclosed) / row_max
  Daha sik iliski = daha yuksek agirlik = daha guclu mesaj
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
from configs.config import (METRICS_DIR, FIG_DIR, make_dirs,
                             TOP50_DEGREE_CSV, DEGREE_ADJ_NPZ)

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
TOP_Q=0.20; DV01=100.0; INIT_CASH=100_000.0


def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True

def norm_adj(A):
    """Spektral normalizasyon — agirlikli matris icin de calısır."""
    A = A + np.eye(A.shape[0], dtype=np.float32)
    d = A.sum(1) ** -0.5
    d[np.isinf(d)] = 0.
    return torch.tensor(np.diag(d) @ A @ np.diag(d), dtype=torch.float32)

def make_windows(data,n_his):
    X,Y=[],[]
    for t in range(n_his,len(data)):
        X.append(data[t-n_his:t]); Y.append(data[t])
    return np.array(X,np.float32),np.array(Y,np.float32)

def calc_metrics(yt,yp):
    rmse=float(np.sqrt(np.mean((yt-yp)**2)))
    mae=float(np.mean(np.abs(yt-yp)))
    ss_r=np.sum((yt-yp)**2); ss_t=np.sum((yt-yt.mean())**2)
    r2=float(1-ss_r/ss_t) if ss_t>1e-10 else float("nan")
    yt_d=(yt.flatten()>0).astype(int); yp_d=(yp.flatten()>0).astype(int)
    tp=int(np.sum((yt_d==1)&(yp_d==1))); tn=int(np.sum((yt_d==0)&(yp_d==0)))
    fp=int(np.sum((yt_d==0)&(yp_d==1))); fn=int(np.sum((yt_d==1)&(yp_d==0)))
    acc=float((tp+tn)/(tp+tn+fp+fn+1e-10))
    prec=float(tp/(tp+fp+1e-10)); rec=float(tp/(tp+fn+1e-10))
    f1=float(2*prec*rec/(prec+rec+1e-10))
    return {"rmse":rmse,"mae":mae,"r2":r2,
            "accuracy":acc,"precision":prec,"recall":rec,"f1":f1}

def backtest(yt,yp,q=TOP_Q,dv01=DV01,cash=INIT_CASH):
    T,N=yt.shape; K=max(1,int(q*N)); pnl=[]
    for t in range(T):
        r=np.argsort(yp[t]); w=np.zeros(N)
        w[r[-K:]]=+1/K; w[r[:K]]=-1/K
        pnl.append(dv01*(w*yt[t]).sum())
    pnl=np.array(pnl); eq=cash+np.cumsum(pnl)
    return {"bt_sharpe":float(np.sqrt(52)*pnl.mean()/(pnl.std()+1e-10)),
            "hit_ratio":float((pnl>0).mean())}


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
        self.lstm=nn.LSTM(N,h,layers,batch_first=True,dropout=drop if layers>1 else 0.)
        self.head=nn.Linear(h,N)
    def forward(self,x):
        out,_=self.lstm(x); return self.head(out[:,-1,:])


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


def run_fold(delta,N,Ac_t,Au_w_t,Ad_w_t,Au_b_t,device,epochs,
             tr_end,va_end,te_end):
    sc=StandardScaler()
    tr_s=sc.fit_transform(delta[:tr_end])
    va_s=sc.transform(delta[tr_end:va_end])
    te_s=sc.transform(delta[va_end:te_end])
    Xtr,Ytr=make_windows(tr_s,N_HIS)
    Xva,Yva=make_windows(va_s,N_HIS)
    Xte,_  =make_windows(te_s,N_HIS)
    te_orig=delta[va_end:te_end][N_HIS:]
    results={}

    # AR(1)
    ps=np.zeros_like(te_s)
    for i in range(N):
        y=tr_s[:,i]; phi=np.corrcoef(y[:-1],y[1:])[0,1] if y.std()>1e-8 else 0.
        mu=y.mean()*(1-phi); last=tr_s[-1,i]
        for t in range(len(te_s)):
            ps[t,i]=mu+phi*last; last=te_s[t,i]
    ar1p=sc.inverse_transform(ps)[N_HIS:]
    results["AR(1)"]={**calc_metrics(te_orig,ar1p),**backtest(te_orig,ar1p)}

    # LSTM
    lstm=LSTMModel(N).to(device)
    lstm,ep=train_nn(lstm,Xtr,Ytr,Xva,Yva,device,epochs)
    lstmp=sc.inverse_transform(pred_nn(lstm,Xte,device))
    results["LSTM"]={**calc_metrics(te_orig,lstmp),**backtest(te_orig,lstmp)}
    print(f"    LSTM ep={ep}")

    # XGBoost
    if HAS_XGB:
        Xtf=Xtr.reshape(len(Xtr),-1); Xtef=Xte.reshape(len(Xte),-1)
        ps_x=np.zeros((len(Xte),N))
        for i in range(N):
            m=XGBRegressor(n_estimators=300,max_depth=4,learning_rate=0.05,
                           subsample=0.8,colsample_bytree=0.8,
                           objective="reg:squarederror",random_state=SEED,
                           tree_method="hist",n_jobs=-1,verbosity=0)
            m.fit(Xtf,Ytr[:,i]); ps_x[:,i]=m.predict(Xtef)
        xgbp=sc.inverse_transform(ps_x)
        results["XGBoost"]={**calc_metrics(te_orig,xgbp),**backtest(te_orig,xgbp)}

    # V-STGCN (binary adj — baseline)
    vs=VSTGCN(N,N_HIS,GCN_H,Ac_t).to(device)
    vs,ep=train_nn(vs,Xtr,Ytr,Xva,Yva,device,epochs)
    vsp=sc.inverse_transform(pred_nn(vs,Xte,device))
    results["V-STGCN (binary)"]={**calc_metrics(te_orig,vsp),**backtest(te_orig,vsp)}
    print(f"    V-STGCN(bin) ep={ep}")

    # SC-STGCN (binary adj — degree-50 baseline)
    sc_bin=SCSTGCN(N,N_HIS,GCN_H,ATT_H,ATT_HEADS,FF_H,DROPOUT,Au_b_t,Au_b_t.T.contiguous()).to(device)
    sc_bin,ep=train_nn(sc_bin,Xtr,Ytr,Xva,Yva,device,epochs)
    scbp=sc.inverse_transform(pred_nn(sc_bin,Xte,device))
    results["SC-STGCN (binary)"]={**calc_metrics(te_orig,scbp),**backtest(te_orig,scbp)}
    print(f"    SC-STGCN(bin) ep={ep}")

    # SC-STGCN (WEIGHTED adj — yeni)
    sc_w=SCSTGCN(N,N_HIS,GCN_H,ATT_H,ATT_HEADS,FF_H,DROPOUT,Au_w_t,Ad_w_t).to(device)
    sc_w,ep=train_nn(sc_w,Xtr,Ytr,Xva,Yva,device,epochs)
    scwp=sc.inverse_transform(pred_nn(sc_w,Xte,device))
    results["SC-STGCN (weighted)"]={**calc_metrics(te_orig,scwp),**backtest(te_orig,scwp)}
    print(f"    SC-STGCN(wgt) ep={ep}")

    return results, te_orig


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
    print("  Degree-50 Weighted Adjacency CV")
    print("  Binary vs Weighted adj karsilastirmasi")
    print(f"  epochs={args.epochs} | device={device}")
    print(SEP)

    # Veri
    df    = pd.read_csv(TOP50_DEGREE_CSV, index_col=0)
    level = df.values.astype(np.float32)
    delta = np.diff(level, axis=0)
    T, N  = delta.shape

    # Pre-COVID
    dates = pd.date_range(start="2015-01-09", periods=T, freq="W-FRI")
    covid_week = int(np.searchsorted(dates, pd.Timestamp("2020-03-01")))
    delta_pre  = delta[:covid_week]
    print(f"\n  Pre-COVID: {covid_week} hafta  N={N}")

    # Binary adj (mevcut)
    Ac_bin = load_npz(str(DEGREE_DIR/"adj.npz")).toarray().astype(np.float32)
    Au_bin = load_npz(str(DEGREE_DIR/"adj_sup.npz")).toarray().astype(np.float32)
    np.fill_diagonal(Ac_bin,0); np.fill_diagonal(Au_bin,0)

    # Weighted adj (yeni)
    Au_w = load_npz(str(DEGREE_DIR/"adj_sup_weighted.npz")).toarray().astype(np.float32)
    Ad_w = load_npz(str(DEGREE_DIR/"adj_cus_weighted.npz")).toarray().astype(np.float32)
    np.fill_diagonal(Au_w,0); np.fill_diagonal(Ad_w,0)

    print(f"  Binary adj density   : {Ac_bin.mean():.4f}")
    print(f"  Weighted Au max      : {Au_w.max():.4f}")
    print(f"  Weighted Au mean(nz) : {Au_w[Au_w>0].mean():.4f}")

    Ac_t  = norm_adj(Ac_bin).to(device)
    Au_b_t = norm_adj(Au_bin).to(device)
    Au_w_t = norm_adj(Au_w).to(device)
    Ad_w_t = norm_adj(Ad_w).to(device)

    # 2 fold
    folds=[
        (int(covid_week*0.65),int(covid_week*0.78),int(covid_week*0.92)),
        (int(covid_week*0.78),int(covid_week*0.92),covid_week),
    ]
    print(f"\n  Fold yapisi:")
    for i,(a,b,c) in enumerate(folds):
        print(f"    Fold {i+1}: train[0:{a}] val[{a}:{b}] test[{b}:{c}] "
              f"({c-b} hafta | {dates[b].date()} -> {dates[c-1].date()})")

    model_names=["AR(1)","LSTM","V-STGCN (binary)",
                 "SC-STGCN (binary)","SC-STGCN (weighted)"]
    if HAS_XGB: model_names.insert(2,"XGBoost")

    all_results=[]
    for fi,(tr_end,va_end,te_end) in enumerate(folds):
        print(f"\n  --- Fold {fi+1}/2 ---")
        set_seed(SEED+fi)
        res,te_orig=run_fold(delta_pre,N,Ac_t,Au_w_t,Ad_w_t,Au_b_t,
                             device,args.epochs,tr_end,va_end,te_end)
        all_results.append(res)
        print(f"\n  Fold {fi+1} sonuclari:")
        print(f"  {'Model':<24} {'RMSE':>8} {'R2':>8} {'Sharpe':>8} {'F1':>7}")
        print(f"  {'-'*60}")
        for name,r in res.items():
            print(f"  {name:<24} {r['rmse']:>8.4f} {r['r2']:>8.4f} "
                  f"{r['bt_sharpe']:>8.4f} {r['f1']:>7.4f}")

    # Aggregate
    metrics_list=["rmse","mae","r2","accuracy","f1","bt_sharpe","hit_ratio"]
    agg={}
    for name in model_names:
        vals={m:[] for m in metrics_list}
        for fr in all_results:
            if name in fr:
                for m in metrics_list:
                    v=fr[name].get(m,float("nan"))
                    if not np.isnan(v): vals[m].append(v)
        agg[name]={}
        for m in metrics_list:
            v=np.array(vals[m])
            agg[name][f"{m}_mean"]=float(np.mean(v)) if len(v)>0 else float("nan")
            agg[name][f"{m}_std"] =float(np.std(v))  if len(v)>0 else float("nan")

    print(f"\n{SEP}")
    print("  WEIGHTED ADJ CV RESULTS (2 FOLDS)")
    print(SEP)
    print(f"\n  {'Model':<24} {'RMSE':>8} {'±':>6} {'R2':>8} "
          f"{'Acc':>7} {'F1':>7} {'Sharpe':>9} {'±':>6}")
    print(f"  {'-'*80}")
    for name in model_names:
        a=agg[name]
        print(f"  {name:<24} "
              f"{a['rmse_mean']:>8.4f} {a['rmse_std']:>6.4f} "
              f"{a['r2_mean']:>8.4f} "
              f"{a['accuracy_mean']:>7.4f} "
              f"{a['f1_mean']:>7.4f} "
              f"{a['bt_sharpe_mean']:>9.4f} {a['bt_sharpe_std']:>6.4f}")

    # Ana karsilastirma: Binary vs Weighted SC-STGCN
    print(f"\n  ANA KARSILASTIRMA: SC-STGCN Binary vs Weighted")
    print(f"  {'-'*55}")
    for m in ["rmse","mae","r2","accuracy","f1","bt_sharpe"]:
        w_v  = agg["SC-STGCN (weighted)"][f"{m}_mean"]
        b_v  = agg["SC-STGCN (binary)"][f"{m}_mean"]
        diff = w_v - b_v
        better = "Weighted better" if (m in ["r2","accuracy","f1","bt_sharpe"] and diff>0) or \
                                       (m in ["rmse","mae"] and diff<0) else "Binary better"
        print(f"    {m:<12}: W={w_v:>+8.4f}  B={b_v:>+8.4f}  diff={diff:>+8.4f}  [{better}]")

    # Tum ozet
    print(f"\n  {'Analiz':<30} {'SC R²':>8} {'V R²':>8} {'SC Sharpe':>10}")
    print(f"  {'-'*60}")
    refs=[
        ("Raw delta (top50)",     0.0401, 0.0122, 2.4325),
        ("Degree-50 raw",         0.0551, 0.0062, 2.7344),
        ("Degree-50 binary SC",   agg["SC-STGCN (binary)"]["r2_mean"],
                                   agg["V-STGCN (binary)"]["r2_mean"],
                                   agg["SC-STGCN (binary)"]["bt_sharpe_mean"]),
        ("Degree-50 WEIGHTED SC", agg["SC-STGCN (weighted)"]["r2_mean"],
                                   agg["V-STGCN (binary)"]["r2_mean"],
                                   agg["SC-STGCN (weighted)"]["bt_sharpe_mean"]),
    ]
    for label, sc_r2, vs_r2, sc_sh in refs:
        diff = sc_r2 - vs_r2
        print(f"  {label:<30} {sc_r2:>+8.4f} {vs_r2:>+8.4f} {sc_sh:>10.4f}")

    # CSV
    rows=[]
    for name in model_names:
        a=agg[name]
        rows.append({"Model":name,
                     "RMSE":a["rmse_mean"],"RMSE_std":a["rmse_std"],
                     "R2":a["r2_mean"],"F1":a["f1_mean"],
                     "Sharpe":a["bt_sharpe_mean"]})
    pd.DataFrame(rows).to_csv(METRICS_DIR/"degree_weighted_cv.csv",index=False)
    print(f"\n  [OK] degree_weighted_cv.csv")
    print(f"\n{SEP}  TAMAMLANDI\n{SEP}")


if __name__=="__main__":
    t0=time.time()
    main()
    print(f"  Total: {(time.time()-t0)/60:.1f} min")
