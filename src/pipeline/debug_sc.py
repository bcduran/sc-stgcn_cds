import re, sys
from pathlib import Path
import pandas as pd

SC_BASE  = Path("C:/Users/burha/OneDrive/Masaüstü/GNN/first_phase_and_performance/Supply Chain Data S&P500")
CUST_DIR = SC_BASE / "Customer Data S&P500"
SUP_XL   = SC_BASE / "a_Supplier_Tickers.xlsx"
CUST_XL  = SC_BASE / "a_Customer_Tickers.xlsx"
TICKER_PAT = re.compile(r'^[A-Z]{1,5}$')

print("=== SUPPLIER TICKER MAP (ilk 20 satir) ===")
df_s = pd.read_excel(SUP_XL, header=None, dtype=str)
print(f"Shape: {df_s.shape}")
print(df_s.head(20).to_string())

print("\n=== CUSTOMER TICKER MAP (ilk 10 satir) ===")
df_c = pd.read_excel(CUST_XL, header=None, dtype=str)
print(f"Shape: {df_c.shape}")
print(df_c.head(10).to_string())

print("\n=== CUSTOMER EXCEL DOSYA ADLARI (ilk 20) ===")
files = sorted(list(CUST_DIR.glob("*.xlsx")) + list(CUST_DIR.glob("*.xls")))
for f in files[:20]:
    print(f"  {f.stem}")

print("\n=== ESLESME KONTROLU ===")
name2tick = {}
for _, row in df_s.iterrows():
    cells = [str(c).strip() for c in row if pd.notna(c) and str(c).strip() not in ("","nan")]
    tks = [c.upper() for c in cells if TICKER_PAT.match(c.upper())]
    nms = [c for c in cells if not TICKER_PAT.match(c.upper())]
    if len(tks)==1 and nms:
        for nm in nms:
            name2tick[nm.lower()] = tks[0]

print(f"Map isim sayisi: {len(name2tick)}")
print("Ornek map (ilk 5):")
for k,v in list(name2tick.items())[:5]:
    print(f"  '{k}' -> {v}")

print("\nIlk 10 dosya eslesme:")
for f in files[:10]:
    stem = f.stem.strip()
    match = name2tick.get(stem.lower(), "YOK")
    print(f"  '{stem}' -> {match}")
