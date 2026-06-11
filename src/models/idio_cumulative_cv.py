"""
src/models/idio_cumulative_cv.py
==================================
Secenek C: Idiosyncratic Kumulatif Spread Degisimi

Hedef:
  Idiocum_i(t, h) = [s_i(t+h) - s_i(t)] - mean_j[s_j(t+h) - s_j(t)]
  = h-haftalik kumulatif degisim MINUS cross-sectional ortalama

Motivasyon:
  - Kumulatif serideki ortak faktor cikarilinca
    supply chain sinyali daha net gorumeli
  - Event study: downstream 2-3 hafta oncesinde -3.5 bp NET (common factor cikarilmis)
  - Bu tam olarak bu hedefin test ettigi sey

Calistirmak:
  python src/models/idio_cumulative_cv.py --epochs 300 --device cpu
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

SEED=42; N_HIS=8; GCN_H=64; ATT_H=64; ATT_HEADS=2; FF_H=128
DROPOUT=0.30; LR=1e-4; WD=1e-3; BATCH=64; PATIENCE=40; CLIP=5.0
TOP_Q=0.20; DV01=100.0; INIT_CASH=100_000.0


def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True

def norm_adj(A):
    A=A+np.eye(A.shape[0],dtype=np.float32); d=A.sum(1)**-0.5
    return torch.tensor(np.diag(d)@A@np.diag(d),dtype=torch.float32)

def make_windows_xy(X_data,Y_data,n_his):
    X,Y=[],[]
    for t in range(n_his,len(X_data)):
        X.append(X_data[t-n_his:t]); Y.append(Y_data[t])
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
    return {"rmse":rmse,"mae":mae,"r2":r2,"accuracy":acc,"f1":f1}

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
    def __init__(self,N,n_his,gcn_h,att_h,att_heads,ff_h,drop,A_up,A_dn):
        super().__init__()
        self.register_buffer("A_up",A_up); self.register_buffer("A_dn",A_dn)
        self.gcn_up=GCNLayer(n_his,gcn_h); self.gcn_dn=GCNLayer(n_his,gcn_h)
        self.gate=nn.Linear(gcn_h*2,gcn_h); self.proj=nn.Linear(gcn_h,att_h)
        self.attn=TemporalAttn(att_h,att_heads,ff_h,drop)
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


def train_nn(model,Xtr,Ytr,Xva,Yva,device,
             lr=LR,wd=WD,epochs=300,batch=BATCH,patience=PATIENCE,clip=CLIP):
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=lr*0.01)
    lfn=nn.HuberLoss(delta=1.0)
    Xt=torch.tensor(Xtr).to(device); Yt=torch.tensor(Ytr).to(device)
    Xv=torch.tensor(Xva).to(device); Yv=torch.tensor(Yva).to(device)
    best,no_imp,state=float("inf"),0,None
    for ep in range(epochs):
        model.train()
        idx=torch.randperm(len(Xt),device=device)
        for i in range(0,len(idx),batch):
            b=idx[i:i+batch]; opt.zero_grad()
            lfn(model(Xt[b]),Yt[b]).backward()
            nn.utils.clip_grad_norm_(model.parameters(),clip); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad(): vl=lfn(model(Xv),Yv).item()
        if vl<best-1e-6: best,no_imp=vl,0; state={k:v.cpu().clone() for k,v in model.state_dict().items()}
        else:
            no_imp+=1
            if no_imp>=patience: break
    if state: model.load_state_dict(state)
    return model,ep+1

def pred_nn(model,X,device):
    model.eval()
    with torch.no_grad(): return model(torch.tensor(X).to(device)).cpu().numpy()


def run_fold(idio_x,idio_y,n_firms,Ac_t,Au_t,Ad_t,device,epochs,
             tr_end,va_end,te_end):
    sc_x=StandardScaler(); sc_y=StandardScaler()
    tr_xs=sc_x.fit_transform(idio_x[:tr_end])
    va_xs=sc_x.transform(idio_x[tr_end:va_end])
    te_xs=sc_x.transform(idio_x[va_end:te_end])
    tr_ys=sc_y.fit_transform(idio_y[:tr_end])
    va_ys=sc_y.transform(idio_y[tr_end:va_end])
    te_ys=sc_y.transform(idio_y[va_end:te_end])

    Xtr,Ytr=make_windows_xy(tr_xs,tr_ys,N_HIS)
    Xva,Yva=make_windows_xy(va_xs,va_ys,N_HIS)
    Xte,Yte=make_windows_xy(te_xs,te_ys,N_HIS)
    te_orig=idio_y[va_end:te_end][N_HIS:]

    results={}

    # AR(1) on idiosyncratic
    ps=np.zeros_like(te_xs)
    for i in range(n_firms):
        y=tr_xs[:,i]; phi=np.corrcoef(y[:-1],y[1:])[0,1] if y.std()>1e-8 else 0.
        mu=y.mean()*(1-phi); last=tr_xs[-1,i]
        for t in range(len(te_xs)):
            ps[t,i]=mu+phi*last; last=te_xs[t,i]
    ar_pred=sc_y.inverse_transform(ps)[N_HIS:]
    results["AR(1)"]={**calc_metrics(te_orig,ar_pred),**backtest(te_orig,ar_pred)}

    lstm=LSTMModel(n_firms).to(device)
    lstm,ep=train_nn(lstm,Xtr,Ytr,Xva,Yva,device,epochs=epochs)
    lstmp=sc_y.inverse_transform(pred_nn(lstm,Xte,device))
    results["LSTM"]={**calc_metrics(te_orig,lstmp),**backtest(te_orig,lstmp)}
    print(f"    LSTM ep={ep}")

    if HAS_XGB:
        Xtf=Xtr.reshape(len(Xtr),-1); Xtef=Xte.reshape(len(Xte),-1)
        ps_x=np.zeros((len(Xte),n_firms))
        for i in range(n_firms):
            m=XGBRegressor(n_estimators=300,max_depth=4,learning_rate=0.05,
                           subsample=0.8,colsample_bytree=0.8,
                           objective="reg:squarederror",random_state=SEED,
                           tree_method="hist",n_jobs=-1,verbosity=0)
            m.fit(Xtf,Ytr[:,i]); ps_x[:,i]=m.predict(Xtef)
        xgbp=sc_y.inverse_transform(ps_x)
        results["XGBoost"]={**calc_metrics(te_orig,xgbp),**backtest(te_orig,xgbp)}

    vs=VSTGCN(n_firms,N_HIS,GCN_H,Ac_t).to(device)
    vs,ep=train_nn(vs,Xtr,Ytr,Xva,Yva,device,epochs=epochs)
    vsp=sc_y.inverse_transform(pred_nn(vs,Xte,device))
    results["V-STGCN"]={**calc_metrics(te_orig,vsp),**backtest(te_orig,vsp)}
    print(f"    V-STGCN ep={ep}")

    sc_m=SCSTGCN(n_firms,N_HIS,GCN_H,ATT_H,ATT_HEADS,FF_H,
                 DROPOUT,Au_t,Ad_t).to(device)
    sc_m,ep=train_nn(sc_m,Xtr,Ytr,Xva,Yva,device,epochs=epochs)
    scp=sc_y.inverse_transform(pred_nn(sc_m,Xte,device))
    results["SC-STGCN"]={**calc_metrics(te_orig,scp),**backtest(te_orig,scp)}
    print(f"    SC-STGCN ep={ep}")

    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=300)
    parser.add_argument("--device",default="auto")
    parser.add_argument("--horizons",type=int,nargs="+",default=[2,3])
    args=parser.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") \
           if args.device=="auto" else torch.device(args.device)

    make_dirs(); set_seed()
    SEP="="*65
    print(SEP)
    print(f"  Idiosyncratic Cumulative CV — Pre-COVID (2015-2020)")
    print(f"  Target: Idiocum(t,h) = [s(t+h)-s(t)] - mean_j[s_j(t+h)-s_j(t)]")
    print(f"  Horizons: {args.horizons} | epochs={args.epochs}")
    print(SEP)

    df=pd.read_csv(TOP50_WEEKLY_CSV)
    level=df.values.astype(np.float32)
    delta=np.diff(level,axis=0)
    T,N=delta.shape
    dates=pd.date_range(start="2015-01-09",periods=T,freq="W-FRI")
    covid_week=int(np.searchsorted(dates,pd.Timestamp("2020-03-01")))
    delta_pre=delta[:covid_week]
    level_pre=level[:covid_week+1]

    # Idiosyncratic delta (X features)
    common_delta=delta_pre.mean(axis=1,keepdims=True)
    idio_delta=delta_pre-common_delta

    print(f"\n  Idiosyncratic delta std : {idio_delta.std():.4f} bp")
    print(f"  Raw delta std           : {delta_pre.std():.4f} bp")

    Ac=load_npz(ADJ_NPZ).toarray().astype(np.float32)
    Au=load_npz(ADJ_SUP_NPZ).toarray().astype(np.float32)
    Ad=load_npz(ADJ_CUS_NPZ).toarray().astype(np.float32)
    Ac_t=norm_adj(Ac).to(device)
    Au_t=norm_adj(Au).to(device)
    Ad_t=norm_adj(Ad).to(device)

    folds=[
        (int(covid_week*0.65),int(covid_week*0.78),int(covid_week*0.92)),
        (int(covid_week*0.78),int(covid_week*0.92),covid_week),
    ]
    model_names=["AR(1)","LSTM","V-STGCN","SC-STGCN"]
    if HAS_XGB: model_names.insert(2,"XGBoost")

    horizon_agg={}

    for h in args.horizons:
        print(f"\n{'─'*65}")
        print(f"  h={h}: Idiocum = [s(t+{h})-s(t)] - cross_mean")
        print(f"{'─'*65}")

        # Idiosyncratic cumulative target
        cum_raw=np.array([level_pre[t+h]-level_pre[t]
                          for t in range(len(level_pre)-h)],dtype=np.float32)
        common_cum=cum_raw.mean(axis=1,keepdims=True)
        idio_cum=cum_raw-common_cum

        # ACF
        acfs=[]
        for i in range(N):
            s=idio_cum[:,i]
            if s.std()>1e-8:
                r=np.corrcoef(s[:-1],s[1:])[0,1]
                if not np.isnan(r): acfs.append(r)
        phi=np.mean(acfs)
        print(f"  Idiocum h={h}: ACF={phi:.4f}  teorik R²={phi**2:.4f}  "
              f"std={idio_cum.std():.4f} bp")

        min_len=min(len(idio_delta),len(idio_cum))
        idio_x=idio_delta[:min_len]
        idio_y=idio_cum[:min_len]

        all_results=[]
        for fi,(tr_end,va_end,te_end) in enumerate(folds):
            te_adj=min(te_end,min_len)
            va_adj=min(va_end,te_adj)
            tr_adj=min(tr_end,va_adj)
            print(f"\n  Fold {fi+1}/2  (h={h})")
            set_seed(SEED+fi)
            res=run_fold(idio_x,idio_y,N,Ac_t,Au_t,Ad_t,
                         device,args.epochs,tr_adj,va_adj,te_adj)
            all_results.append(res)

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

        horizon_agg[h]=agg

        print(f"\n  h={h} Sonuclari (idiosyncratic cumulative):")
        print(f"  {'Model':<12} {'RMSE':>8} {'±':>6} {'MAE':>8} "
              f"{'R²':>8} {'Acc':>7} {'F1':>7} {'Sharpe':>8}")
        print(f"  {'-'*75}")
        for name in model_names:
            a=agg[name]
            print(f"  {name:<12} "
                  f"{a['rmse_mean']:>8.4f} {a['rmse_std']:>6.4f} "
                  f"{a['mae_mean']:>8.4f} {a['r2_mean']:>8.4f} "
                  f"{a['accuracy_mean']:>7.4f} {a['f1_mean']:>7.4f} "
                  f"{a['bt_sharpe_mean']:>8.4f}")

        sc_r2=agg["SC-STGCN"]["r2_mean"]; vs_r2=agg["V-STGCN"]["r2_mean"]
        sc_sh=agg["SC-STGCN"]["bt_sharpe_mean"]; vs_sh=agg["V-STGCN"]["bt_sharpe_mean"]
        print(f"\n  Ablation h={h}: SC R²={sc_r2:+.4f} vs V R²={vs_r2:+.4f}  "
              f"| SC Sharpe={sc_sh:.4f} vs V Sharpe={vs_sh:.4f}")

    # Ozet karsilastirma tablosu
    print(f"\n{SEP}")
    print("  TUM ANALIZLER OZET — SC-STGCN vs V-STGCN ABLATION")
    print(SEP)
    print(f"\n  {'Analiz':<28} {'SC R²':>8} {'V R²':>8} {'Diff':>8} "
          f"{'SC Sharpe':>10} {'V Sharpe':>10}")
    print(f"  {'-'*75}")
    ref_results=[
        ("Raw delta h=1",       0.0401, 0.0122, 2.4325, 4.0542),
        ("Idiosyncratic h=1",   0.0207, 0.0156, 3.9202, 3.3337),
        ("Raw delta h=2",       0.0115,-0.0182, 1.7169,-1.0746),
    ]
    for label,sc_r2,vs_r2,sc_sh,vs_sh in ref_results:
        diff=sc_r2-vs_r2
        print(f"  {label:<28} {sc_r2:>+8.4f} {vs_r2:>+8.4f} {diff:>+8.4f} "
              f"{sc_sh:>10.4f} {vs_sh:>10.4f}")
    for h in args.horizons:
        label=f"Idio cumulative h={h}"
        sc_r2=horizon_agg[h]["SC-STGCN"]["r2_mean"]
        vs_r2=horizon_agg[h]["V-STGCN"]["r2_mean"]
        sc_sh=horizon_agg[h]["SC-STGCN"]["bt_sharpe_mean"]
        vs_sh=horizon_agg[h]["V-STGCN"]["bt_sharpe_mean"]
        diff=sc_r2-vs_r2
        print(f"  {label:<28} {sc_r2:>+8.4f} {vs_r2:>+8.4f} {diff:>+8.4f} "
              f"{sc_sh:>10.4f} {vs_sh:>10.4f}")

    # Kaydet
    rows=[]
    for h in args.horizons:
        for name in model_names:
            a=horizon_agg[h][name]
            rows.append({"horizon":h,"model":name,
                         "rmse":a["rmse_mean"],"r2":a["r2_mean"],
                         "f1":a["f1_mean"],"sharpe":a["bt_sharpe_mean"]})
    pd.DataFrame(rows).to_csv(METRICS_DIR/"idio_cumulative_cv.csv",index=False)
    print(f"\n[DONE] idio_cumulative_cv.py tamamlandi.")


if __name__=="__main__":
    t0=time.time()
    main()
    print(f"  Total: {(time.time()-t0)/60:.1f} min")
