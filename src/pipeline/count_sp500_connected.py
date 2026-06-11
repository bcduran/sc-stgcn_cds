import pandas as pd, sys
from pathlib import Path

SC   = Path("C:/Users/burha/OneDrive/Masaüstü/GNN/first_phase_and_performance/Supply Chain Data S&P500")
BASE = Path("C:/Users/burha/OneDrive/Masaüstü/GNN_Thesis")
sys.path.insert(0, str(BASE))
from configs.config import CDS_RAW_CSV

# Ticker map
sup = pd.read_excel(SC/"a_Supplier_Tickers.xlsx", dtype=str)
cus = pd.read_excel(SC/"a_Customer_Tickers.xlsx", dtype=str)
n2t = {}
for df in [sup, cus]:
    for _, row in df.iterrows():
        nm=str(row.iloc[0]).strip(); tk=str(row.iloc[1]).strip().upper()
        if nm and tk and nm!="nan" and tk!="nan": n2t[nm]=tk

# SP500
sp500_order = (pd.read_csv(BASE/"data"/"sp500_companies.csv")["Symbol"]
               .dropna().astype(str).str.strip().str.upper().tolist())
sp500_set = set(sp500_order)

# SP500 icindeki kenarlar
re = pd.read_csv(BASE/"data"/"processed"/"raw_edges.csv", dtype=str)
re.columns = re.columns.str.strip().str.upper()
re["S"] = re["SUPPLIER"].str.strip().map(n2t)
re["C"] = re["CUSTOMER"].str.strip().map(n2t)
mask = (re["S"].notna() & re["C"].notna() &
        re["S"].isin(sp500_set) & re["C"].isin(sp500_set) &
        re["S"] != re["C"])
edges = re[mask][["S","C"]].drop_duplicates()
connected = set(edges["S"]) | set(edges["C"])

# CDS coverage
df_cds = pd.read_csv(CDS_RAW_CSV, parse_dates=["Date"])
df_cds = df_cds[(df_cds["Date"]>="2015-01-01")&(df_cds["Date"]<="2020-02-28")]
tot = df_cds.groupby("Ticker")["Date"].count()
val = df_cds.groupby("Ticker")["PX5"].apply(lambda x: x.notna().sum())
cov = (val/tot).to_dict()

print(f"SP500 toplam firma    : {len(sp500_order)}")
print(f"SP500 icindeki kenar  : {len(edges)}")
print(f"SP500 icinde baglantili: {len(connected)}")
print(f"CDS OK (>=70%)        : {sum(1 for v in cov.values() if v>=0.70)}")

# CDS OK + baglantili
both = [t for t in sp500_order if cov.get(t,0.)>=0.70 and t in connected]
print(f"\nCDS OK + SP500 baglantili: {len(both)} firma")
print(f"Bunlardan top-N subgraph baglantiligi:")
for N in [50, 75, 100, 150, 200]:
    sel = both[:N]
    sel_set = set(sel)
    sub = edges[edges["S"].isin(sel_set) & edges["C"].isin(sel_set)]
    conn_in_sub = set(sub["S"]) | set(sub["C"])
    print(f"  Top-{N:<4}: {len(conn_in_sub):>4} / {N} kendi aralarinda baglantili "
          f"({len(conn_in_sub)/N*100:.0f}%)")
