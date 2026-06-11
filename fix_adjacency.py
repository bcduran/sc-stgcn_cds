"""
fix_adjacency.py
Upstream ve downstream adjacency matrislerini dogru sekilde yeniden uretir.
Calistirmak: python fix_adjacency.py
"""
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs.config import (
    TOP50_SUPPLIER_XLSX, TOP50_CUSTOMER_XLSX,
    TOP50_WEEKLY_CSV, ADJ_NPZ, ADJ_SUP_NPZ, ADJ_CUS_NPZ,
)

# Ticker sirasi
tickers = pd.read_csv(TOP50_WEEKLY_CSV, nrows=0).columns.tolist()
idx = {t: i for i, t in enumerate(tickers)}
N   = len(tickers)
print(f"N={N} ticker")

sup_df = pd.read_excel(TOP50_SUPPLIER_XLSX, header=0)
cus_df = pd.read_excel(TOP50_CUSTOMER_XLSX, header=0)

# --- Upstream: supplier(nb) -> focal_firm
up_rows, up_cols = [], []
for _, row in sup_df.iterrows():
    focal = str(row.iloc[0]).strip().upper()
    if focal not in idx: continue
    fi = idx[focal]
    for v in row.iloc[1:].dropna():
        nb = str(v).strip().upper()
        if nb in idx:
            up_rows.append(idx[nb])  # source = supplier
            up_cols.append(fi)       # target = focal firm

# --- Downstream: focal_firm -> customer(nb)
dn_rows, dn_cols = [], []
for _, row in cus_df.iterrows():
    focal = str(row.iloc[0]).strip().upper()
    if focal not in idx: continue
    fi = idx[focal]
    for v in row.iloc[1:].dropna():
        nb = str(v).strip().upper()
        if nb in idx:
            dn_rows.append(fi)       # source = focal firm
            dn_cols.append(idx[nb])  # target = customer

A_up = csr_matrix((np.ones(len(up_rows), dtype=np.float32),
                   (up_rows, up_cols)), shape=(N, N))
A_dn = csr_matrix((np.ones(len(dn_rows), dtype=np.float32),
                   (dn_rows, dn_cols)), shape=(N, N))

# Downstream = transpose of upstream (ekonomik yorum:
# upstream A[i,j]=1 => supplier i -> firm j
# downstream A[j,i]=1 => firm j -> customer i  (ters yon)
A_dn = A_up.T.tocsr()

print(f"A_up nnz: {A_up.nnz}")
print(f"A_dn nnz: {A_dn.nnz}")
print(f"Same?      {np.allclose(A_up.toarray(), A_dn.toarray())}")
print(f"Transpose? {np.allclose(A_up.toarray(), A_dn.toarray().T)}")

# Combined (binary)
A_comb = (A_up + A_dn)
A_comb.data = np.ones_like(A_comb.data)
print(f"Combined nnz: {A_comb.nnz}")

save_npz(ADJ_SUP_NPZ, A_up)
save_npz(ADJ_CUS_NPZ, A_dn)
save_npz(ADJ_NPZ,     A_comb)

print(f"\n[OK] adj_sup.npz -> {ADJ_SUP_NPZ}")
print(f"[OK] adj_cus.npz -> {ADJ_CUS_NPZ}")
print(f"[OK] adj.npz     -> {ADJ_NPZ}")
