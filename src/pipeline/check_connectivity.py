import pandas as pd
from pathlib import Path

SC = Path("C:/Users/burha/OneDrive/Masaüstü/GNN/first_phase_and_performance/Supply Chain Data S&P500")
sup = pd.read_excel(SC/"a_Supplier_Tickers.xlsx", dtype=str)
cus = pd.read_excel(SC/"a_Customer_Tickers.xlsx", dtype=str)

n2t = {}
for df in [sup, cus]:
    for _, row in df.iterrows():
        nm = str(row.iloc[0]).strip()
        tk = str(row.iloc[1]).strip().upper()
        if nm and tk and nm != "nan" and tk != "nan":
            n2t[nm] = tk

t2n = {v: k for k, v in n2t.items()}

top50   = pd.read_csv("data/top50_connected/top50_connected_firms.csv")
tickers = set(top50["Ticker"].tolist())

re = pd.read_csv("data/processed/raw_edges.csv", dtype=str)
re.columns = re.columns.str.strip().str.upper()
re["SUP_TICK"] = re["SUPPLIER"].str.strip().map(n2t)
re["CUS_TICK"] = re["CUSTOMER"].str.strip().map(n2t)

print("Top-50 firmalarinin raw_edges baglantilari:\n")
print("  Ticker     Sup_olarak   Cus_olarak    Toplam  Durum")
print("  " + "-"*55)
for tk in sorted(tickers):
    as_sup = int((re["SUP_TICK"] == tk).sum())
    as_cus = int((re["CUS_TICK"] == tk).sum())
    total  = as_sup + as_cus
    status = "BAGLANTILI" if total > 0 else "IZOLE"
    print(f"  {tk:<8}   {as_sup:>10,}   {as_cus:>10,}   {total:>7,}  {status}")

print()
izole = [tk for tk in sorted(tickers)
         if (re["SUP_TICK"]==tk).sum()+(re["CUS_TICK"]==tk).sum()==0]
print(f"Izole firmalar ({len(izole)}): {izole}")
