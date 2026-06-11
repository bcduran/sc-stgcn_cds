"""
SP500 weight sirasinda: kendi aralarinda baglantili + CDS OK -> ilk 50
"""
import pandas as pd, numpy as np, sys
from pathlib import Path
import scipy.sparse as sp_sparse
from scipy.stats import kurtosis

SC = Path("C:/Users/burha/OneDrive/Masaüstü/GNN/first_phase_and_performance/Supply Chain Data S&P500")
BASE    = Path("C:/Users/burha/OneDrive/Masaüstü/GNN_Thesis")
OUT_DIR = BASE / "data" / "top50_connected"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE))
from configs.config import CDS_RAW_CSV, make_dirs

N_TOP=50; MIN_COV=0.70; START="2015-01-01"; END="2020-02-28"; FREQ="W-FRI"

# 1. Ticker map
sup = pd.read_excel(SC/"a_Supplier_Tickers.xlsx", dtype=str)
cus = pd.read_excel(SC/"a_Customer_Tickers.xlsx", dtype=str)
n2t = {}
for df in [sup, cus]:
    for _, row in df.iterrows():
        nm = str(row.iloc[0]).strip(); tk = str(row.iloc[1]).strip().upper()
        if nm and tk and nm != "nan" and tk != "nan":
            n2t[nm] = tk

# 2. SP500 sirasi
sp500_order = (pd.read_csv(BASE/"data"/"sp500_companies.csv")["Symbol"]
               .dropna().astype(str).str.strip().str.upper().tolist())
sp500_set = set(sp500_order)
print(f"SP500: {len(sp500_order)} ticker")

# 3. CDS coverage
df_cds = pd.read_csv(CDS_RAW_CSV, parse_dates=["Date"])
df_cds = df_cds[(df_cds["Date"]>=START)&(df_cds["Date"]<=END)]
tot = df_cds.groupby("Ticker")["Date"].count()
val = df_cds.groupby("Ticker")["PX5"].apply(lambda x: x.notna().sum())
cov = (val/tot).to_dict()
print(f"CDS OK: {sum(1 for v in cov.values() if v>=MIN_COV)} ticker")

# 4. SP500 icindeki kenarlar
re = pd.read_csv(BASE/"data"/"processed"/"raw_edges.csv", dtype=str)
re.columns = re.columns.str.strip().str.upper()
re["S"] = re["SUPPLIER"].str.strip().map(n2t)
re["C"] = re["CUSTOMER"].str.strip().map(n2t)
mask = (re["S"].notna() & re["C"].notna() &
        re["S"].isin(sp500_set) & re["C"].isin(sp500_set) &
        re["S"] != re["C"])
edges = re[mask][["S","C"]].drop_duplicates()
print(f"SP500 icindeki kenar: {len(edges)}")

# 5. SP500 weight sirasinda: CDS OK + en az 1 baglantisi var -> ilk 50
# "Baglantili" = secilen diger firmalarla degil, SP500 genelinde
sp500_connected = set(edges["S"]) | set(edges["C"])
print(f"SP500 icinde baglantili: {len(sp500_connected)} firma")

selected = []
no_cds = []; no_conn = []
for t in sp500_order:
    if len(selected) >= N_TOP: break
    if cov.get(t, 0.) < MIN_COV: no_cds.append(t); continue
    if t not in sp500_connected: no_conn.append(t); continue
    selected.append(t)

print(f"\nCDS yetersiz  : {len(no_cds)}")
print(f"Izole atlanan : {len(no_conn)}: {no_conn[:10]}")
print(f"Secilen       : {len(selected)}")

if len(selected) < N_TOP:
    print(f"WARN: {N_TOP} doldurulamadi!")

print(f"\n  # Ticker   Coverage")
print(f"  {'-'*28}")
for i, t in enumerate(selected, 1):
    print(f"  {i:<3} {t:<8} {cov.get(t,0.):.1%}")

# Eski vs yeni
try:
    old = set(pd.read_csv(str(BASE/"data"/"top50"/"ve1.csv"), nrows=0).columns)
    new = set(selected)
    print(f"\nEklenenler  ({len(new-old)}): {sorted(new-old)}")
    print(f"Cikarilanlar({len(old-new)}): {sorted(old-new)}")
except: pass

# 6. Adjacency — secilen 50 icindeki gercek kenarlar
idx = {t: i for i, t in enumerate(selected)}
N = len(selected)
Au = np.zeros((N,N), dtype=np.float32)
Ad = np.zeros((N,N), dtype=np.float32)
cnt = 0
for _, row in edges.iterrows():
    s, c = row["S"], row["C"]
    if s in idx and c in idx:
        if Au[idx[c], idx[s]] == 0:
            Au[idx[c], idx[s]] = 1.
            Ad[idx[s], idx[c]] = 1.
            cnt += 1
Ac = ((Au+Ad)>0).astype(np.float32)
np.fill_diagonal(Au,0); np.fill_diagonal(Ad,0); np.fill_diagonal(Ac,0)
deg = Ac.sum(1)+Ac.sum(0)
print(f"\nAdj: {cnt} kenar  Baglantili:{(deg>0).sum()}  Izole:{(deg==0).sum()}  Density:{Ac.mean():.4f}")

sp_sparse.save_npz(str(OUT_DIR/"adj.npz"),     sp_sparse.csr_matrix(Ac))
sp_sparse.save_npz(str(OUT_DIR/"adj_sup.npz"), sp_sparse.csr_matrix(Au))
sp_sparse.save_npz(str(OUT_DIR/"adj_cus.npz"), sp_sparse.csr_matrix(Ad))

pd.DataFrame({
    "Rank":range(1,len(selected)+1), "Ticker":selected,
    "Coverage":[cov.get(t,0.) for t in selected],
    "Degree":[int(deg[i]) for i,t in enumerate(selected)],
}).to_csv(OUT_DIR/"top50_connected_firms.csv", index=False)

# 7. CDS Panel
df_cds2 = pd.read_csv(CDS_RAW_CSV, parse_dates=["Date"])
df_cds2 = df_cds2[(df_cds2["Date"]>=START)&(df_cds2["Date"]<=END)&df_cds2["Ticker"].isin(selected)]
series = [df_cds2[df_cds2["Ticker"]==t].set_index("Date")["PX5"].resample(FREQ).mean().rename(t) for t in selected]
panel = pd.concat(series,axis=1).sort_index().ffill().bfill().fillna(0)
delta = panel.diff().dropna()
acfs = [np.corrcoef(delta[c].dropna()[:-1],delta[c].dropna()[1:])[0,1] for c in delta.columns if delta[c].std()>1e-8]
phi = float(np.mean(acfs)) if acfs else 0.
print(f"Panel:{panel.shape}  ACF:{phi:+.4f}  R2:{phi**2:.4f}  Kurt:{kurtosis(delta.values.flatten()):.1f}")
panel.to_csv(OUT_DIR/"ve1.csv")
print(f"\nTAMAMLANDI: {len(selected)} firma -> {OUT_DIR}")
