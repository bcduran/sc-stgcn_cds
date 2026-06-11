"""
src/pipeline/build_weighted_adj.py
====================================
Degree-50 icin agirlikli adjacency olustur.

Agirlik = raw_edges.csv'de (supplier, customer) cifti kac kez geciyor
Yani firma cifti ne kadar cok disclose edilmisse o kadar guclu iliski.

Cikti:
  data/top50_degree/adj_weighted.npz
  data/top50_degree/adj_sup_weighted.npz
  data/top50_degree/adj_cus_weighted.npz
"""
import pandas as pd, numpy as np, sys
from pathlib import Path
import scipy.sparse as sp_sparse

SC   = Path("C:/Users/burha/OneDrive/Masaüstü/GNN/first_phase_and_performance/Supply Chain Data S&P500")
BASE = Path(__file__).resolve().parents[2]
OUT_DIR = BASE / "data" / "top50_degree"
sys.path.insert(0, str(BASE))
from configs.config import make_dirs
make_dirs()

SEP="="*65
print(SEP)
print("  Weighted Adjacency — Degree-50")
print("  Agirlik = (sup,cust) cifti kac kez raw_edges'de geciyor")
print(SEP)

# Ticker map
sup_xl = pd.read_excel(SC/"a_Supplier_Tickers.xlsx", dtype=str)
cus_xl = pd.read_excel(SC/"a_Customer_Tickers.xlsx", dtype=str)
n2t = {}
for df in [sup_xl, cus_xl]:
    for _, row in df.iterrows():
        nm=str(row.iloc[0]).strip(); tk=str(row.iloc[1]).strip().upper()
        if nm and tk and nm!="nan" and tk!="nan": n2t[nm]=tk

# Secilen 50 firma
firms_df = pd.read_csv(OUT_DIR/"top50_degree_firms.csv")
tickers  = firms_df["Ticker"].tolist()
sel_set  = set(tickers)
idx      = {t:i for i,t in enumerate(tickers)}
N        = len(tickers)
print(f"\nDegree-50 firma sayisi: {N}")
print(f"Tickers: {tickers}")

# raw_edges'den agirlikli kenarlar
re = pd.read_csv(BASE/"data"/"processed"/"raw_edges.csv", dtype=str)
re.columns = re.columns.str.strip().str.upper()
re["S"] = re["SUPPLIER"].str.strip().map(n2t)
re["C"] = re["CUSTOMER"].str.strip().map(n2t)
re["S"] = re["S"].where(re["S"].notna(), None)
re["C"] = re["C"].where(re["C"].notna(), None)

# Sadece top50_degree firmalar arasi kenarlar
mask = (re["S"].notna() & re["C"].notna() &
        re["S"].isin(sel_set) & re["C"].isin(sel_set) &
        (re["S"] != re["C"]))
edges_all = re[mask][["S","C"]]

# Her (S, C) ciftinin kac kez geçtigi
edge_counts = edges_all.groupby(["S","C"]).size().reset_index(name="Count")
print(f"\nBenzersiz kenar: {len(edge_counts)}")
print(f"Toplam disclosure: {edge_counts['Count'].sum()}")
print(f"\nEn sik 10 kenar:")
print(edge_counts.sort_values("Count", ascending=False).head(10).to_string())

# Agirlikli matris olustur (normalize edilmemis ham sayimlar)
Au_raw = np.zeros((N,N), dtype=np.float32)  # supplier->firm
Ad_raw = np.zeros((N,N), dtype=np.float32)  # firm->customer

for _, row in edge_counts.iterrows():
    s, c, cnt = row["S"], row["C"], row["Count"]
    if s in idx and c in idx:
        ic, is_ = idx[c], idx[s]
        Au_raw[ic, is_] = float(cnt)   # j supplier of i
        Ad_raw[is_, ic] = float(cnt)   # i supplier of j

np.fill_diagonal(Au_raw, 0)
np.fill_diagonal(Ad_raw, 0)

# Normalize et — her satiri max ile normalize (0-1 arasi)
def normalize_rows(A):
    """Her satiri max degerine gore normalize et."""
    row_max = A.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1.  # sifir satirlari icin
    return A / row_max

Au_norm = normalize_rows(Au_raw.copy())
Ad_norm = normalize_rows(Ad_raw.copy())
Ac_norm = ((Au_norm + Ad_norm) > 0).astype(np.float32) * \
          np.maximum(Au_norm, Ad_norm)  # max weight for combined

# Binary adj ile karsilastir
Au_bin = load_bin = (Au_raw > 0).astype(np.float32)
np.fill_diagonal(Au_bin, 0)

print(f"\nAgirlik istatistikleri:")
print(f"  Max agirlik (Au_raw): {Au_raw.max():.0f}")
print(f"  Mean agirlik (Au_raw, nonzero): {Au_raw[Au_raw>0].mean():.2f}")
print(f"  Au_norm nonzero mean: {Au_norm[Au_norm>0].mean():.4f}")

deg = ((Au_norm + Ad_norm) > 0).astype(float)
np.fill_diagonal(deg, 0)
deg = deg.sum(1) + deg.sum(0)
print(f"  Baglantili node: {(deg>0).sum()}/50")

# Kaydet
sp_sparse.save_npz(str(OUT_DIR/"adj_sup_weighted.npz"),
                   sp_sparse.csr_matrix(Au_norm))
sp_sparse.save_npz(str(OUT_DIR/"adj_cus_weighted.npz"),
                   sp_sparse.csr_matrix(Ad_norm))
# Combined weighted
Ac_w = ((Au_norm + Ad_norm) > 0).astype(np.float32)
np.fill_diagonal(Ac_w, 0)
sp_sparse.save_npz(str(OUT_DIR/"adj_weighted.npz"),
                   sp_sparse.csr_matrix(Ac_w))

print(f"\n[OK] adj_sup_weighted.npz")
print(f"[OK] adj_cus_weighted.npz")
print(f"[OK] adj_weighted.npz")
print(f"\n{SEP}  TAMAMLANDI\n{SEP}")
