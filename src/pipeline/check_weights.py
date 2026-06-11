import pandas as pd, sys
sys.path.insert(0, '.')
from pathlib import Path

SC = Path("C:/Users/burha/OneDrive/Masaüstü/GNN/first_phase_and_performance/Supply Chain Data S&P500")
sup = pd.read_excel(SC/"a_Supplier_Tickers.xlsx", dtype=str)
cus = pd.read_excel(SC/"a_Customer_Tickers.xlsx", dtype=str)
n2t = {}
for df in [sup, cus]:
    for _, row in df.iterrows():
        nm=str(row.iloc[0]).strip(); tk=str(row.iloc[1]).strip().upper()
        if nm and tk and nm!="nan" and tk!="nan": n2t[nm]=tk

firms = pd.read_csv("data/top50_degree/top50_degree_firms.csv")
sel = set(firms["Ticker"].tolist())

re = pd.read_csv("data/processed/raw_edges.csv", dtype=str)
re.columns = re.columns.str.strip().str.upper()
re["S"] = re["SUPPLIER"].str.strip().map(n2t)
re["C"] = re["CUSTOMER"].str.strip().map(n2t)
re["S"] = re["S"].where(re["S"].notna(), None)
re["C"] = re["C"].where(re["C"].notna(), None)
mask = (re["S"].notna() & re["C"].notna() &
        re["S"].isin(sel) & re["C"].isin(sel) &
        (re["S"] != re["C"]))
edges = re[mask][["S","C"]]   # duplar dahil

counts = edges.groupby(["S","C"]).size().reset_index(name="Count")
print("Toplam kenar (duplar dahil):", len(edges))
print("Benzersiz cift             :", len(counts))
print("Max count                  :", counts["Count"].max())
print("Count > 1 olan             :", (counts["Count"]>1).sum())
print("\nEn sik 10 kenar:")
print(counts.sort_values("Count", ascending=False).head(10).to_string())
