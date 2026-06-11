"""
Bu scripti çalıştır:
  python save_confusion.py --epochs 300 --dataset top50_degree --device cpu

Çıktı: outputs/metrics/full_cv_top50_degree_confusion.csv
"""
import argparse, sys, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.sparse import load_npz
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

SEED=42; N_HIS=8
def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)
def norm_adj(A):
    A=A+np.eye(A.shape[0],dtype=np.float32); d=A.sum(1)**-0.5
    return torch.tensor(np.diag(d)@A@np.diag(d),dtype=torch.float32)
def make_windows(data,n):
    X,Y=[],[]
    for t in range(n,len(data)): X.append(data[t-n:t]); Y.append(data[t])
    return np.array(X,np.float32),np.array(Y,np.float32)
def to_labels(delta,thr):
    l=np.ones_like(delta,dtype=np.int64)
    l[delta>thr]=2; l[delta<-thr]=0
    return l
def find_thr(va_true,va_pred):
    best,bf=0.5,-1
    for thr in np.arange(0.1,3.0,0.1):
        lt=to_labels(va_true.flatten(),thr)
        lp=to_labels(va_pred.flatten(),thr)
        if len(np.unique(lp))<2: continue
        f=f1_score(lt,lp,average='macro',zero_division=0)
        if f>bf: bf=f; best=thr
    return best

# Minimal SC-STGCN re-import from full_cv.py
import importlib.util, os
fc_path = Path(__file__).parent / 'src' / 'models' / 'full_cv.py'
spec = importlib.util.spec_from_file_location('full_cv', fc_path)
fc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fc)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=300)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--dataset',default='top50_degree')
    args=parser.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
           if args.device=='auto' else torch.device(args.device)

    BASE = Path(__file__).resolve().parent
    csv_path = BASE/'data'/'top50_degree'/'ve1.csv'
    sup_path = BASE/'data'/'top50_degree'/'adj_sup.npz'
    cus_path = BASE/'data'/'top50_degree'/'adj_cus.npz'
    adj_path = BASE/'data'/'top50_degree'/'adj.npz'

    df = pd.read_csv(csv_path, index_col=None)
    try: df.iloc[:,0].astype(np.float32)
    except: df = df.iloc[:,1:]
    level = df.values.astype(np.float32)
    delta = np.diff(level,axis=0)
    T, N = delta.shape
    dates = pd.date_range('2015-01-09',periods=T,freq='W-FRI')
    covid_week = int(np.searchsorted(dates,pd.Timestamp('2020-03-01')))
    delta_pre = delta[:covid_week]

    Ac_t=fc.norm_adj(load_npz(str(adj_path)).toarray().astype(np.float32)).to(device)
    Au_t=fc.norm_adj(load_npz(str(sup_path)).toarray().astype(np.float32)).to(device)
    Ad_t=fc.norm_adj(load_npz(str(cus_path)).toarray().astype(np.float32)).to(device)

    folds=[
        (int(covid_week*0.65),int(covid_week*0.78),int(covid_week*0.92)),
        (int(covid_week*0.78),int(covid_week*0.92),covid_week),
    ]

    rows=[]
    for fi,(tr_end,va_end,te_end) in enumerate(folds):
        set_seed(SEED+fi)
        sc=StandardScaler()
        tr_s=sc.fit_transform(delta_pre[:tr_end])
        va_s=sc.transform(delta_pre[tr_end:va_end])
        te_s=sc.transform(delta_pre[va_end:te_end])
        Xtr,Ytr=make_windows(tr_s,N_HIS)
        Xva,Yva=make_windows(va_s,N_HIS)
        Xte,_  =make_windows(te_s,N_HIS)
        va_orig=delta_pre[tr_end:va_end][N_HIS:]
        te_orig=delta_pre[va_end:te_end][N_HIS:]

        # SC-STGCN only
        m=fc.SCSTGCN(N,N_HIS,64,64,2,128,0.30,Au_t,Ad_t).to(device)
        m,_=fc.train_nn(m,Xtr,Ytr,Xva,Yva,device,args.epochs)
        va_pred=sc.inverse_transform(fc.pred_nn(m,make_windows(va_s,N_HIS)[0],device))
        te_pred=sc.inverse_transform(fc.pred_nn(m,Xte,device))
        thr=find_thr(va_orig,va_pred)
        y_true=to_labels(te_orig.flatten(),0.5)
        y_pred=to_labels(te_pred.flatten(),thr)
        for yt,yp in zip(y_true,y_pred):
            rows.append({'fold':fi+1,'true':yt,'pred':yp})
        print(f"  Fold {fi+1} done, thr={thr:.2f}")

    out=pd.DataFrame(rows)
    outpath = BASE/'outputs'/'metrics'/'full_cv_top50_degree_confusion.csv'
    out.to_csv(outpath,index=False)
    print(f"Saved: {outpath}")

if __name__=='__main__': main()
