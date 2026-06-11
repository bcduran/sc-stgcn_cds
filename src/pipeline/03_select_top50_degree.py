"""
src/pipeline/03_select_top50_degree.py
========================================
Kriter 1: Supply chain degree (SP500 icindeki baglanti sayisi) en yuksek
Kriter 2: Hem upstream (supplier) hem downstream (customer) baglantisi olan firmalar
Kriter 3: CDS coverage >= 70% (pre-COVID 2015-2020)

Secim:
  500 firma icinden → CDS OK + hem sup hem cus + degree sirasi → ilk 50

Cikti:
  data/top50_degree/ve1.csv
  data/top50_degree/adj.npz
  data/top50_degree/adj_sup.npz
  data/top50_degree/adj_cus.npz
  data/top50_degree/top50_degree_firms.csv
"""
import pandas as pd, numpy as np, sys
from pathlib import Path
import scipy.sparse as sp_sparse
from scipy.stats import kurtosis
from collections import Counter

SC   = Path("C:/Users/burha/OneDrive/Masaüstü/GNN/first_phase_and_performance/Supply Chain Data S&P500")
BASE = Path(__file__).resolve().parents[2]
OUT_DIR = BASE / "data" / "top50_degree"  # max_kurt=100
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BASE))
from configs.config import CDS_RAW_CSV, make_dirs

MIN_COV=0.70; START="2015-01-01"; END="2020-02-28"; FREQ="W-FRI"; N_TOP=50
make_dirs()
SEP="="*65
print(SEP)
print("  Top-50 Degree + Upstream&Downstream Firma Secimi")
print("  Kriter: CDS OK + hem sup hem cus + degree sirasi")
print(SEP)

# 1. Ticker map
sup = pd.read_excel(SC/"a_Supplier_Tickers.xlsx", dtype=str)
cus = pd.read_excel(SC/"a_Customer_Tickers.xlsx", dtype=str)
n2t = {}
for df in [sup, cus]:
    for _, row in df.iterrows():
        nm=str(row.iloc[0]).strip(); tk=str(row.iloc[1]).strip().upper()
        if nm and tk and nm!="nan" and tk!="nan": n2t[nm]=tk

# 2. SP500 ticker seti (tum 500)
sp500_all = (pd.read_csv(BASE/"data"/"sp500_companies.csv")["Symbol"]
             .dropna().astype(str).str.strip().str.upper().tolist())
sp500_set = set(sp500_all)
print(f"\nSP500 toplam: {len(sp500_all)} firma")

# 3. CDS coverage
df_cds = pd.read_csv(CDS_RAW_CSV, parse_dates=["Date"])
df_cds = df_cds[(df_cds["Date"]>=START)&(df_cds["Date"]<=END)]
tot  = df_cds.groupby("Ticker")["Date"].count()
val  = df_cds.groupby("Ticker")["PX5"].apply(lambda x: x.notna().sum())
cov  = (val/tot).to_dict()
mean = df_cds.groupby("Ticker")["PX5"].mean().to_dict()
print(f"CDS OK (>={MIN_COV:.0%}): {sum(1 for v in cov.values() if v>=MIN_COV)} ticker")

# 4. SP500 icindeki kenarlar
re = pd.read_csv(BASE/"data"/"processed"/"raw_edges.csv", dtype=str)
re.columns = re.columns.str.strip().str.upper()
re["S"] = re["SUPPLIER"].str.strip().map(n2t)
re["C"] = re["CUSTOMER"].str.strip().map(n2t)
re["S"] = re["S"].where(re["S"].notna(), None)
re["C"] = re["C"].where(re["C"].notna(), None)
mask = (re["S"].notna() & re["C"].notna() &
        re["S"].isin(sp500_set) & re["C"].isin(sp500_set) &
        (re["S"] != re["C"]))
edges = re[mask][["S","C"]].drop_duplicates()
print(f"SP500 icindeki kenar: {len(edges)}")

# 5. Her firma icin:
#    - total_degree: toplam kenar sayisi
#    - out_degree:   kac firmaya supplier (upstream baglanti)
#    - in_degree:    kac firmadan customer (downstream baglanti)
out_deg = Counter(edges["S"])  # supplier olarak gecme sayisi
in_deg  = Counter(edges["C"])  # customer olarak gecme sayisi
all_firms = set(out_deg.keys()) | set(in_deg.keys())

# Her firmanin kurtosis'ini hesapla (pre-COVID delta)
from scipy.stats import kurtosis as spkurt
firm_kurt = {}
for t in all_firms:
    sub = df_cds[df_cds["Ticker"] == t].set_index("Date")["PX5"]
    sub_w = sub.resample(FREQ).mean().ffill().bfill()
    d = sub_w.diff().dropna()
    if len(d) > 10:
        firm_kurt[t] = float(spkurt(d.values))

MAX_KURT = 100.0

firm_stats = []
for t in all_firms:
    od = out_deg.get(t, 0)
    id_ = in_deg.get(t, 0)
    firm_kurt_val = firm_kurt.get(t, 9999.)
    firm_stats.append({
        "Ticker": t,
        "Out_Degree": od,
        "In_Degree":  id_,
        "Total_Degree": od + id_,
        "Both": od > 0 and id_ > 0,
        "Coverage": cov.get(t, 0.),
        "CDS_OK": cov.get(t, 0.) >= MIN_COV,
        "Kurt": firm_kurt_val,
        "Kurt_OK": firm_kurt_val < MAX_KURT,
    })

df_firms = pd.DataFrame(firm_stats).sort_values(
    "Total_Degree", ascending=False).reset_index(drop=True)

print(f"\nSP500 icinde baglantili: {len(df_firms)} firma")
print(f"Hem sup hem cus       : {df_firms['Both'].sum()} firma")
print(f"CDS OK + hem sup hem cus: {((df_firms['Both'])&(df_firms['CDS_OK'])).sum()} firma")
print(f"CDS OK + hem sup hem cus + Kurt<{MAX_KURT:.0f}: {((df_firms['Both'])&(df_firms['CDS_OK'])&(df_firms['Kurt_OK'])).sum()} firma")

# 6. Top-50 sec: CDS OK + hem sup hem cus + degree sirasi
eligible = df_firms[(df_firms["Both"]) & (df_firms["CDS_OK"]) & (df_firms["Kurt_OK"])].copy()
eligible = eligible.sort_values("Total_Degree", ascending=False).reset_index(drop=True)
print(f"\nEligible (sirali): {len(eligible)} firma")

# Tam tablo goster
print(f"\n  {'#':<4} {'Ticker':<8} {'Out':>6} {'In':>6} {'Total':>7} {'Cov':>8}")
print(f"  {'-'*45}")
for i, row in eligible.head(60).iterrows():
    marker = " ← SEC" if i < N_TOP else ""
    print(f"  {i+1:<4} {row['Ticker']:<8} {row['Out_Degree']:>6} "
          f"{row['In_Degree']:>6} {row['Total_Degree']:>7} "
          f"{row['Coverage']:>8.1%}{marker}")

# Iteratif: izole kalanlari at, sonrakini ekle
pool = eligible["Ticker"].tolist()
selected = pool[:N_TOP]
pool_idx = N_TOP

def get_isolated(sel, edges):
    sel_set = set(sel)
    sub = edges[edges["S"].isin(sel_set) & edges["C"].isin(sel_set)]
    conn = set(sub["S"]) | set(sub["C"])
    return [t for t in sel if t not in conn]

for iteration in range(20):
    isolated = get_isolated(selected, edges)
    if not isolated:
        print(f"  Iterasyon {iteration+1}: Izole yok!")
        break
    print(f"  Iterasyon {iteration+1}: {len(isolated)} izole: {isolated}")
    for iso in isolated:
        selected.remove(iso)
        while pool_idx < len(pool):
            cand = pool[pool_idx]; pool_idx += 1
            if cand not in selected:
                selected.append(cand); break

selected = selected[:N_TOP]
print(f"\nSecilen {N_TOP} firma: {selected}")

# 7. Adjacency
sel_set = set(selected)
sub50   = edges[edges["S"].isin(sel_set) & edges["C"].isin(sel_set)]
idx = {t:i for i,t in enumerate(selected)}
N   = len(selected)
Au  = np.zeros((N,N),dtype=np.float32)
Ad  = np.zeros((N,N),dtype=np.float32)
cnt = 0
for _,row in sub50.iterrows():
    s,c=row["S"],row["C"]
    if s in idx and c in idx and Au[idx[c],idx[s]]==0:
        Au[idx[c],idx[s]]=1.; Ad[idx[s],idx[c]]=1.; cnt+=1
Ac=((Au+Ad)>0).astype(np.float32)
np.fill_diagonal(Au,0); np.fill_diagonal(Ad,0); np.fill_diagonal(Ac,0)
deg=Ac.sum(1)+Ac.sum(0)
print(f"\nAdj: {cnt} kenar  Baglantili:{(deg>0).sum()}  "
      f"Izole:{(deg==0).sum()}  Density:{Ac.mean():.4f}")

sp_sparse.save_npz(str(OUT_DIR/"adj.npz"),     sp_sparse.csr_matrix(Ac))
sp_sparse.save_npz(str(OUT_DIR/"adj_sup.npz"), sp_sparse.csr_matrix(Au))
sp_sparse.save_npz(str(OUT_DIR/"adj_cus.npz"), sp_sparse.csr_matrix(Ad))

# Firma bilgileri
sel_df = eligible.head(N_TOP).copy()
sel_df["Rank"] = range(1, N_TOP+1)
sel_df["PX5_mean_bp"] = sel_df["Ticker"].map(mean)
sel_df.to_csv(OUT_DIR/"top50_degree_firms.csv", index=False)

# Eski top-50 ile karsilastir
try:
    old = set(pd.read_csv(str(BASE/"data"/"top50"/"ve1.csv"),nrows=0).columns)
    new = set(selected)
    print(f"\nEklenenler  ({len(new-old)}): {sorted(new-old)}")
    print(f"Cikarilanlar({len(old-new)}): {sorted(old-new)}")
except: pass

# 8. CDS Panel
df2 = pd.read_csv(CDS_RAW_CSV, parse_dates=["Date"])
df2 = df2[(df2["Date"]>=START)&(df2["Date"]<=END)&df2["Ticker"].isin(selected)]
series = [df2[df2["Ticker"]==t].set_index("Date")["PX5"]
            .resample(FREQ).mean().rename(t) for t in selected]
panel = pd.concat(series,axis=1).sort_index().ffill().bfill().fillna(0)
delta = panel.diff().dropna()
acfs  = [np.corrcoef(delta[c].dropna()[:-1],delta[c].dropna()[1:])[0,1]
         for c in delta.columns if delta[c].std()>1e-8]
phi   = float(np.mean(acfs)) if acfs else 0.
print(f"Panel:{panel.shape}  ACF:{phi:+.4f}  R2:{phi**2:.4f}  "
      f"Kurt:{kurtosis(delta.values.flatten()):.1f}")
panel.to_csv(OUT_DIR/"ve1.csv")
print(f"\n[OK] -> {OUT_DIR}")
print(f"\n{SEP}")
print(f"  TAMAMLANDI: {len(selected)} firma")
print(f"  Density: {Ac.mean():.4f}  Baglantili:{int((deg>0).sum())}/50")
print(SEP)
