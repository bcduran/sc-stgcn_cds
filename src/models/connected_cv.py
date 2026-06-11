"""
src/models/connected_cv.py
============================
Secenek 5: Sadece Baglantili Firmalar

Motivasyon:
  50 firmadan 11'i izole (hic supply chain baglantisi yok).
  Bu 11 firma model icin gurultu olusturuyor:
  - Graph convolution hic bilgi tasimiyor bu firmalar icin
  - Ama genel metrics'e dahil ediliyor
  - Supply chain sinyali sadece 39 baglantili firmada var

Yontem:
  - Ayni walk_forward_cv yapisi
  - Sadece 39 baglantili firma kullaniliyor
  - SC-STGCN vs V-STGCN ablation
  - Full 50 firma sonuclariyla karsilastirma

Beklenti:
  - SC-STGCN avantaji daha net gorumeli (izole firmalar cikinca)
  - R² daha yuksek olabilir
  - Sharpe farki degisebilir

Calistirmak:
  python src/models/connected_cv.py --epochs 300 --device cpu
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
        self.lstm=nn.LSTM(N,h,layers,batch_first=True,
                          dropout=drop if layers>1 else 0.)
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


def run_fold(delta,n_firms,Ac_t,Au_t,Ad_t,device,epochs,
             tr_end,va_end,te_end):
    sc=StandardScaler()
    tr_s=sc.fit_transform(delta[:tr_end])
    va_s=sc.transform(delta[tr_end:va_end])
    te_s=sc.transform(delta[va_end:te_end])
    Xtr,Ytr=make_windows(tr_s,N_HIS)
    Xva,Yva=make_windows(va_s,N_HIS)
    Xte,Yte=make_windows(te_s,N_HIS)
    te_orig=delta[va_end:te_end][N_HIS:]

    results={}

    # AR(1)
    ps=np.zeros_like(te_s)
    for i in range(n_firms):
        y=tr_s[:,i]; phi=np.corrcoef(y[:-1],y[1:])[0,1] if y.std()>1e-8 else 0.
        mu=y.mean()*(1-phi); last=tr_s[-1,i]
        for t in range(len(te_s)):
            ps[t,i]=mu+phi*last; last=te_s[t,i]
    ar1p=sc.inverse_transform(ps)[N_HIS:]
    results["AR(1)"]={**calc_metrics(te_orig,ar1p),**backtest(te_orig,ar1p)}

    lstm=LSTMModel(n_firms).to(device)
    lstm,ep=train_nn(lstm,Xtr,Ytr,Xva,Yva,device,epochs=epochs)
    lstmp=sc.inverse_transform(pred_nn(lstm,Xte,device))
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
        xgbp=sc.inverse_transform(ps_x)
        results["XGBoost"]={**calc_metrics(te_orig,xgbp),**backtest(te_orig,xgbp)}

    vs=VSTGCN(n_firms,N_HIS,GCN_H,Ac_t).to(device)
    vs,ep=train_nn(vs,Xtr,Ytr,Xva,Yva,device,epochs=epochs)
    vsp=sc.inverse_transform(pred_nn(vs,Xte,device))
    results["V-STGCN"]={**calc_metrics(te_orig,vsp),**backtest(te_orig,vsp)}
    print(f"    V-STGCN ep={ep}")

    sc_m=SCSTGCN(n_firms,N_HIS,GCN_H,ATT_H,ATT_HEADS,FF_H,
                 DROPOUT,Au_t,Ad_t).to(device)
    sc_m,ep=train_nn(sc_m,Xtr,Ytr,Xva,Yva,device,epochs=epochs)
    scp=sc.inverse_transform(pred_nn(sc_m,Xte,device))
    results["SC-STGCN"]={**calc_metrics(te_orig,scp),**backtest(te_orig,scp)}
    print(f"    SC-STGCN ep={ep}")

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
    print("  Connected-Only CV — Pre-COVID Weekly (2015-2020)")
    print(f"  epochs={args.epochs} | device={device}")
    print(SEP)

    # Veri
    df=pd.read_csv(TOP50_WEEKLY_CSV)
    level=df.values.astype(np.float32)
    delta=np.diff(level,axis=0)
    T,N=delta.shape
    tickers=df.columns.tolist()

    # Pre-COVID
    dates=pd.date_range(start="2015-01-09",periods=T,freq="W-FRI")
    covid_week=int(np.searchsorted(dates,pd.Timestamp("2020-03-01")))
    delta_pre=delta[:covid_week]

    # Adjacency — full 50
    Ac_full=load_npz(ADJ_NPZ).toarray().astype(np.float32)
    Au_full=load_npz(ADJ_SUP_NPZ).toarray().astype(np.float32)
    Ad_full=load_npz(ADJ_CUS_NPZ).toarray().astype(np.float32)

    # Baglantili firma indeksleri
    # Bir firma baglantili: A_comb satirinda veya sutununda en az 1 non-zero var
    A_comb=Ac_full.copy(); np.fill_diagonal(A_comb,0)
    connected_mask=(A_comb.sum(axis=1)+A_comb.sum(axis=0))>0
    connected_idx=np.where(connected_mask)[0]
    isolated_idx =np.where(~connected_mask)[0]
    N_conn=len(connected_idx)

    print(f"\n  Full sample    : {N} firma")
    print(f"  Baglantili     : {N_conn} firma")
    print(f"  Izole          : {len(isolated_idx)} firma")
    print(f"\n  Baglantili firmalar:")
    for i in connected_idx:
        print(f"    [{i:>2}] {tickers[i]}")
    print(f"\n  Izole firmalar:")
    for i in isolated_idx:
        print(f"    [{i:>2}] {tickers[i]}")

    # Sadece baglantili firmalar icin delta ve adjacency
    delta_conn=delta_pre[:,connected_idx]
    Ac_conn=Ac_full[np.ix_(connected_idx,connected_idx)]
    Au_conn=Au_full[np.ix_(connected_idx,connected_idx)]
    Ad_conn=Ad_full[np.ix_(connected_idx,connected_idx)]

    print(f"\n  Connected-only delta: {delta_conn.shape}")
    print(f"  Connected adj density: {(Ac_conn>0).mean():.3f}")

    Ac_t=norm_adj(Ac_conn).to(device)
    Au_t=norm_adj(Au_conn).to(device)
    Ad_t=norm_adj(Ad_conn).to(device)

    # ACF karsilastirma
    acfs_conn=[]; acfs_full=[]
    for i in range(N_conn):
        s=delta_conn[:,i]
        if s.std()>1e-8: acfs_conn.append(np.corrcoef(s[:-1],s[1:])[0,1])
    for i in range(delta_pre.shape[1]):
        s=delta_pre[:,i]
        if s.std()>1e-8: acfs_full.append(np.corrcoef(s[:-1],s[1:])[0,1])
    print(f"\n  Lag-1 ACF (connected): {np.mean(acfs_conn):+.4f}  "
          f"teorik R²={np.mean(acfs_conn)**2:.4f}")
    print(f"  Lag-1 ACF (full 50) : {np.mean(acfs_full):+.4f}  "
          f"teorik R²={np.mean(acfs_full)**2:.4f}")

    # 2 fold
    folds=[
        (int(covid_week*0.65),int(covid_week*0.78),int(covid_week*0.92)),
        (int(covid_week*0.78),int(covid_week*0.92),covid_week),
    ]
    print(f"\n  Fold yapisi:")
    for i,(a,b,c) in enumerate(folds):
        print(f"    Fold {i+1}: train[0:{a}] val[{a}:{b}] test[{b}:{c}] "
              f"({c-b} hafta | {dates[b].date()} -> {dates[c-1].date()})")

    model_names=["AR(1)","LSTM","V-STGCN","SC-STGCN"]
    if HAS_XGB: model_names.insert(2,"XGBoost")

    all_results=[]
    for fi,(tr_end,va_end,te_end) in enumerate(folds):
        print(f"\n  --- Fold {fi+1}/2 (connected only, N={N_conn}) ---")
        set_seed(SEED+fi)
        res,te_orig=run_fold(
            delta_conn,N_conn,Ac_t,Au_t,Ad_t,
            device,args.epochs,tr_end,va_end,te_end
        )
        all_results.append(res)
        print(f"\n  Fold {fi+1} sonuclari (N={N_conn} baglantili firma):")
        print(f"  {'Model':<12} {'RMSE':>7} {'MAE':>7} {'R2':>8} "
              f"{'Acc':>7} {'F1':>7} {'Sharpe':>8} {'Hit':>7}")
        print(f"  {'-'*70}")
        for name,r in res.items():
            print(f"  {name:<12} {r['rmse']:>7.4f} {r['mae']:>7.4f} "
                  f"{r['r2']:>8.4f} {r['accuracy']:>7.4f} {r['f1']:>7.4f} "
                  f"{r['bt_sharpe']:>8.4f} {r['hit_ratio']:>7.4f}")

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
    print(f"  CONNECTED-ONLY RESULTS (N={N_conn}, 2 folds)")
    print(SEP)
    print(f"\n  {'Model':<12} {'RMSE':>8} {'±':>6} {'MAE':>8} {'±':>6} "
          f"{'R²':>8} {'Acc':>7} {'F1':>7} {'Sharpe':>9} {'±':>6} {'Hit':>7}")
    print(f"  {'-'*95}")
    for name in model_names:
        a=agg[name]
        print(f"  {name:<12} "
              f"{a['rmse_mean']:>8.4f} {a['rmse_std']:>6.4f} "
              f"{a['mae_mean']:>8.4f} {a['mae_std']:>6.4f} "
              f"{a['r2_mean']:>8.4f} "
              f"{a['accuracy_mean']:>7.4f} "
              f"{a['f1_mean']:>7.4f} "
              f"{a['bt_sharpe_mean']:>9.4f} {a['bt_sharpe_std']:>6.4f} "
              f"{a['hit_ratio_mean']:>7.4f}")

    # Ablation
    print(f"\n  Ablation: SC-STGCN vs V-STGCN (N={N_conn})")
    for m in ["rmse","mae","r2","accuracy","f1","bt_sharpe"]:
        sc_v=agg["SC-STGCN"][f"{m}_mean"]
        vs_v=agg["V-STGCN"][f"{m}_mean"]
        diff=sc_v-vs_v
        better="SC better" if (m in ["r2","accuracy","f1","bt_sharpe"] and diff>0) or \
                               (m in ["rmse","mae"] and diff<0) else "V better"
        print(f"    {m:<12}: SC={sc_v:>+8.4f}  VS={vs_v:>+8.4f}  "
              f"diff={diff:>+8.4f}  [{better}]")

    # Full 50 vs Connected 39 karsilastirmasi
    print(f"\n{SEP}")
    print("  KARSILASTIRMA: Full 50 vs Connected-Only 39")
    print(SEP)
    print(f"\n  {'Metrik':<12} {'Full-50 SC':>12} {'Full-50 V':>12} "
          f"{'Conn-39 SC':>12} {'Conn-39 V':>12}")
    print(f"  {'-'*60}")
    full_vals={
        "R²":     (0.0401, 0.0122),
        "Sharpe": (2.4325, 4.0542),
        "Hit":    (0.6357, 0.7405),
        "F1":     (0.4995, 0.4297),
    }
    for metric,(full_sc,full_v) in full_vals.items():
        m_key={"R²":"r2","Sharpe":"bt_sharpe","Hit":"hit_ratio","F1":"f1"}[metric]
        conn_sc=agg["SC-STGCN"][f"{m_key}_mean"]
        conn_v =agg["V-STGCN"][f"{m_key}_mean"]
        print(f"  {metric:<12} {full_sc:>+12.4f} {full_v:>+12.4f} "
              f"{conn_sc:>+12.4f} {conn_v:>+12.4f}")

    # Kaydet
    rows=[{"model":name,
           "rmse":agg[name]["rmse_mean"],"r2":agg[name]["r2_mean"],
           "f1":agg[name]["f1_mean"],"sharpe":agg[name]["bt_sharpe_mean"],
           "hit":agg[name]["hit_ratio_mean"]}
          for name in model_names]
    pd.DataFrame(rows).to_csv(METRICS_DIR/"connected_cv.csv",index=False)
    print(f"\n[DONE] connected_cv.py tamamlandi.")


if __name__=="__main__":
    t0=time.time()
    main()
    print(f"  Total: {(time.time()-t0)/60:.1f} min")
