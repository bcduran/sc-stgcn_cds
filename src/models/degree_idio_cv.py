"""
src/models/degree_idio_cv.py
==============================
Degree-50 + Idiosyncratic Spread CV

Hedef: En yoğun supply chain bağlantılı 50 firmada,
       market-wide faktör çıkarıldıktan sonra kalan
       firm-specific spread değişimini tahmin et.

target_i(t) = Delta_s_i(t) - mean_j[Delta_s_j(t)]

Calistirmak:
  python src/models/degree_idio_cv.py --epochs 300 --device cpu
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
                             TOP50_DEGREE_CSV, DEGREE_ADJ_NPZ,
                             DEGREE_ADJ_SUP_NPZ, DEGREE_ADJ_CUS_NPZ)

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

def remove_common_factor(delta):
    common = delta.mean(axis=1, keepdims=True)
    idio   = delta - common
    return idio, common.flatten()

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


def run_fold(idio, N, Ac_t, Au_t, Ad_t, device, epochs,
             tr_end, va_end, te_end):
    sc_x = StandardScaler()
    sc_y = StandardScaler()

    # X features: idiosyncratic delta (lookback)
    tr_x = sc_x.fit_transform(idio[:tr_end])
    va_x = sc_x.transform(idio[tr_end:va_end])
    te_x = sc_x.transform(idio[va_end:te_end])

    # Y target: idiosyncratic delta (next step) — ayni seri
    tr_y = sc_y.fit_transform(idio[:tr_end])
    va_y = sc_y.transform(idio[tr_end:va_end])
    te_y = sc_y.transform(idio[va_end:te_end])

    Xtr,Ytr = make_windows(tr_x, N_HIS)
    Xva,Yva = make_windows(va_x, N_HIS)
    Xte,_   = make_windows(te_x, N_HIS)

    # Gercek bp scale idiosyncratic
    te_orig = idio[va_end:te_end][N_HIS:]

    results = {}

    # AR(1) on idiosyncratic
    ps = np.zeros_like(te_x)
    for i in range(N):
        y=tr_x[:,i]; phi=np.corrcoef(y[:-1],y[1:])[0,1] if y.std()>1e-8 else 0.
        mu=y.mean()*(1-phi); last=tr_x[-1,i]
        for t in range(len(te_x)):
            ps[t,i]=mu+phi*last; last=te_x[t,i]
    ar1p = sc_y.inverse_transform(ps)[N_HIS:]
    results["AR(1)"] = {**calc_metrics(te_orig,ar1p), **backtest(te_orig,ar1p)}

    # LSTM
    lstm=LSTMModel(N).to(device)
    lstm,ep=train_nn(lstm,Xtr,Ytr,Xva,Yva,device,epochs)
    lstmp=sc_y.inverse_transform(pred_nn(lstm,Xte,device))
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
        xgbp=sc_y.inverse_transform(ps_x)
        results["XGBoost"]={**calc_metrics(te_orig,xgbp),**backtest(te_orig,xgbp)}

    # V-STGCN
    vs=VSTGCN(N,N_HIS,GCN_H,Ac_t).to(device)
    vs,ep=train_nn(vs,Xtr,Ytr,Xva,Yva,device,epochs)
    vsp=sc_y.inverse_transform(pred_nn(vs,Xte,device))
    results["V-STGCN"]={**calc_metrics(te_orig,vsp),**backtest(te_orig,vsp)}
    print(f"    V-STGCN ep={ep}")

    # SC-STGCN
    sc_m=SCSTGCN(N,N_HIS,GCN_H,ATT_H,ATT_HEADS,FF_H,DROPOUT,Au_t,Ad_t).to(device)
    sc_m,ep=train_nn(sc_m,Xtr,Ytr,Xva,Yva,device,epochs)
    scp=sc_y.inverse_transform(pred_nn(sc_m,Xte,device))
    results["SC-STGCN"]={**calc_metrics(te_orig,scp),**backtest(te_orig,scp)}
    print(f"    SC-STGCN ep={ep}")

    return results


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
    print("  Degree-50 + Idiosyncratic CV")
    print("  Hedef: epsilon_i(t) = Delta_s_i(t) - mean_j[Delta_s_j(t)]")
    print("  Firma: 50/50 baglantili | density=0.159 | kurt<100")
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

    # Common factor cikar
    idio, common = remove_common_factor(delta_pre)

    from scipy.stats import kurtosis
    print(f"\n  Pre-COVID hafta  : {covid_week}")
    print(f"  Common factor std: {common.std():.4f} bp")
    print(f"  Idio std         : {idio.std():.4f} bp  (raw: {delta_pre.std():.4f})")
    print(f"  Idio kurtosis    : {kurtosis(idio.flatten()):.1f}  (raw: {kurtosis(delta_pre.flatten()):.1f})")

    # Lag-1 ACF ve teorik R²
    acfs=[]
    for i in range(N):
        s=idio[:,i]
        if s.std()>1e-8: acfs.append(np.corrcoef(s[:-1],s[1:])[0,1])
    phi=np.mean(acfs)
    print(f"  Lag-1 ACF        : {phi:+.4f}  Teorik R²: {phi**2:.4f}")

    # Adjacency
    Ac_np=load_npz(str(DEGREE_ADJ_NPZ)).toarray().astype(np.float32)
    Au_np=load_npz(str(DEGREE_ADJ_SUP_NPZ)).toarray().astype(np.float32)
    Ad_np=load_npz(str(DEGREE_ADJ_CUS_NPZ)).toarray().astype(np.float32)
    np.fill_diagonal(Ac_np,0)
    deg=Ac_np.sum(1)+Ac_np.sum(0)
    print(f"  Density          : {Ac_np.mean():.4f}")
    print(f"  Baglantili       : {(deg>0).sum()}/50")

    Ac_t=norm_adj(Ac_np).to(device)
    Au_t=norm_adj(Au_np).to(device)
    Ad_t=norm_adj(Ad_np).to(device)

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
        print(f"\n  --- Fold {fi+1}/2 ---")
        set_seed(SEED+fi)
        res=run_fold(idio,N,Ac_t,Au_t,Ad_t,
                     device,args.epochs,tr_end,va_end,te_end)
        all_results.append(res)
        print(f"\n  Fold {fi+1} sonuclari (IDIOSYNCRATIC):")
        print(f"  {'Model':<12} {'RMSE':>8} {'MAE':>8} {'R2':>8} "
              f"{'Acc':>7} {'F1':>7} {'Sharpe':>8} {'Hit':>7}")
        print(f"  {'-'*70}")
        for name,r in res.items():
            print(f"  {name:<12} {r['rmse']:>8.4f} {r['mae']:>8.4f} "
                  f"{r['r2']:>8.4f} {r['accuracy']:>7.4f} {r['f1']:>7.4f} "
                  f"{r['bt_sharpe']:>8.4f} {r['hit_ratio']:>7.4f}")

    # Aggregate
    metrics_list=["rmse","mae","r2","accuracy","precision","recall","f1",
                  "bt_sharpe","hit_ratio"]
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
    print("  DEGREE-50 + IDIOSYNCRATIC CV RESULTS (2 FOLDS)")
    print(SEP)
    print(f"\n  {'Model':<12} {'RMSE':>8} {'±':>6} {'MAE':>8} {'±':>6} "
          f"{'R2':>8} {'Acc':>7} {'F1':>7} {'Sharpe':>9} {'±':>6} {'Hit':>7}")
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
    print(f"\n  Ablation: SC-STGCN vs V-STGCN")
    for m in ["rmse","mae","r2","accuracy","f1","bt_sharpe"]:
        sc_v=agg["SC-STGCN"][f"{m}_mean"]
        vs_v=agg["V-STGCN"][f"{m}_mean"]
        diff=sc_v-vs_v
        better="SC better" if (m in ["r2","accuracy","f1","bt_sharpe"] and diff>0) or \
                               (m in ["rmse","mae"] and diff<0) else "V better"
        print(f"    {m:<12}: SC={sc_v:>+8.4f}  VS={vs_v:>+8.4f}  diff={diff:>+8.4f}  [{better}]")

    # Tum analizler ozet
    print(f"\n{SEP}")
    print("  TUM ANALIZLER OZET — SC-STGCN vs V-STGCN")
    print(SEP)
    print(f"\n  {'Analiz':<30} {'SC R²':>8} {'V R²':>8} {'Diff':>8} "
          f"{'SC Sharpe':>10} {'V Sharpe':>10}")
    print(f"  {'-'*78}")
    ref=[
        ("Raw delta h=1 (top50)",      0.0401, 0.0122, 2.4325, 4.0542),
        ("Idio h=1 (top50)",           0.0207, 0.0156, 3.9202, 3.3337),
        ("Degree-50 raw",              0.0551, 0.0062, 2.7344, 1.5633),
    ]
    for label,sc_r2,vs_r2,sc_sh,vs_sh in ref:
        diff=sc_r2-vs_r2
        print(f"  {label:<30} {sc_r2:>+8.4f} {vs_r2:>+8.4f} {diff:>+8.4f} "
              f"{sc_sh:>10.4f} {vs_sh:>10.4f}")
    # Yeni analiz
    sc_r2=agg["SC-STGCN"]["r2_mean"]
    vs_r2=agg["V-STGCN"]["r2_mean"]
    sc_sh=agg["SC-STGCN"]["bt_sharpe_mean"]
    vs_sh=agg["V-STGCN"]["bt_sharpe_mean"]
    diff=sc_r2-vs_r2
    print(f"  {'Degree-50 + Idio (YENI)':<30} {sc_r2:>+8.4f} {vs_r2:>+8.4f} "
          f"{diff:>+8.4f} {sc_sh:>10.4f} {vs_sh:>10.4f}")

    # CSV
    rows=[]
    for name in model_names:
        a=agg[name]
        rows.append({"Model":name,
                     "RMSE_mean":a["rmse_mean"],"RMSE_std":a["rmse_std"],
                     "MAE_mean":a["mae_mean"],"R2_mean":a["r2_mean"],
                     "Acc_mean":a["accuracy_mean"],"F1_mean":a["f1_mean"],
                     "Sharpe_mean":a["bt_sharpe_mean"],"Sharpe_std":a["bt_sharpe_std"],
                     "Hit_mean":a["hit_ratio_mean"]})
    pd.DataFrame(rows).to_csv(METRICS_DIR/"degree_idio_cv.csv",index=False)
    print(f"\n  [OK] degree_idio_cv.csv")
    print(f"\n{SEP}  TAMAMLANDI\n{SEP}")


if __name__=="__main__":
    t0=time.time()
    main()
    print(f"  Total: {(time.time()-t0)/60:.1f} min")
