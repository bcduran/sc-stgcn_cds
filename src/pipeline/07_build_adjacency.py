"""
src/pipeline/07_build_adjacency.py
=====================================
Amac:
  Top-50 supply chain iliskilerinden SC-STGCN icin 3 adjacency matrisi uret:

    adj.npz      — Birlesik (binary): hem upstream hem downstream
    adj_sup.npz  — Upstream only: supplier -> firm  (A_up)
    adj_cus.npz  — Downstream only: firm -> customer (A_dn)

  SC-STGCN'in DualStreamGCN katmani A_up ve A_dn'yi ayri ayri kullanir.
  Bu ekonomik anlam tasiyor:
    - A_up: maliyet baskisi kanali (tedarikci kredi riski firmaya yayilir)
    - A_dn: talep cekme kanali (musteri kredi riski firmadan geriye yayilir)

Girdi:
  data/processed/e_top_50_supplier_info.xlsx
  data/processed/e_top_50_customer_info.xlsx
  data/processed/e_top50_companies.xlsx       (ticker sirasi icin)

Cikti:
  data/top50/adj.npz       (birlesik)
  data/top50/adj_sup.npz   (upstream)
  data/top50/adj_cus.npz   (downstream)
  data/top50/adj_node_order.txt  (ticker sirasi — ve1.csv ile ayni olmali)

Calistirmak:
  python src/pipeline/07_build_adjacency.py
"""

from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import (
    TOP50_XLSX,
    TOP50_WEEKLY_CSV,
    TOP50_SUPPLIER_XLSX,
    TOP50_CUSTOMER_XLSX,
    ADJ_NPZ,
    ADJ_SUP_NPZ,
    ADJ_CUS_NPZ,
    TOP50_DIR,
    make_dirs,
)


def load_ticker_order() -> list[str]:
    """
    ve1.csv'den ticker sirasini al.
    Adjacency matrisi bu sirayla eslesmelidir!
    """
    if TOP50_WEEKLY_CSV.exists():
        df = pd.read_csv(TOP50_WEEKLY_CSV, nrows=0)
        tickers = [str(c).strip().upper() for c in df.columns]
        print(f"  Ticker sirasi: ve1.csv'den alindi ({len(tickers)} ticker)")
        return tickers

    # Fallback: Top-50 xlsx
    df = pd.read_excel(TOP50_XLSX)
    tickers = df["Ticker"].astype(str).str.strip().str.upper().tolist()
    print(f"  Ticker sirasi: e_top50_companies.xlsx'den alindi ({len(tickers)} ticker)")
    return tickers


def load_relation_table(path: Path) -> pd.DataFrame:
    """
    Supplier veya Customer Excel'ini yukle.
    Format:
      Sutun A : Ticker (focal firma)
      Sutun B+: Iliskili tickerlar
    """
    df = pd.read_excel(path, header=0)
    return df


def build_adjacency(
    relation_df: pd.DataFrame,
    ticker_order: list[str],
    direction: str,    # "upstream" veya "downstream"
) -> csr_matrix:
    """
    Iliski tablosundan adjacency matrisi uret.

    direction="upstream" (supplier -> firm):
      - Satir: focal firma
      - B+ sutunlar: o firmanin tedarikci'leri
      - Edge: tedarikci -> focal  yani A[sup_idx, firm_idx] = 1

    direction="downstream" (firm -> customer):
      - Satir: focal firma
      - B+ sutunlar: o firmanin musterileri
      - Edge: focal -> musteri  yani A[firm_idx, cus_idx] = 1
    """
    idx = {t: i for i, t in enumerate(ticker_order)}
    N   = len(ticker_order)

    rows_list, cols_list = [], []

    for _, row in relation_df.iterrows():
        focal = str(row.iloc[0]).strip().upper()
        if focal not in idx:
            continue
        fi = idx[focal]

        neighbours = [
            str(v).strip().upper()
            for v in row.iloc[1:].dropna()
            if str(v).strip().upper() not in ("NAN", "NONE", "")
        ]

        for nb in neighbours:
            if nb not in idx:
                continue
            ni = idx[nb]

            if direction == "upstream":
                # supplier (nb) -> firm (focal)
                rows_list.append(ni)
                cols_list.append(fi)
            else:
                # firm (focal) -> customer (nb)
                rows_list.append(fi)
                cols_list.append(ni)

    data = np.ones(len(rows_list), dtype=np.float32)
    A    = csr_matrix((data, (rows_list, cols_list)), shape=(N, N))
    return A


def report(name: str, A: csr_matrix, tickers: list[str]) -> None:
    """Adjacency istatistiklerini yazdir."""
    N       = A.shape[0]
    nnz     = A.nnz
    density = nnz / (N * N)
    deg_out = np.array(A.sum(axis=1)).flatten()
    deg_in  = np.array(A.sum(axis=0)).flatten()

    print(f"\n  [{name}]")
    print(f"    Shape      : {A.shape}")
    print(f"    Non-zeros  : {nnz}")
    print(f"    Density    : {density:.4f} ({density*100:.2f}%)")
    print(f"    Out-degree : mean={deg_out.mean():.2f}  max={deg_out.max():.0f}")
    print(f"    In-degree  : mean={deg_in.mean():.2f}   max={deg_in.max():.0f}")

    # Baglantili dugumler
    connected = int(((deg_out > 0) | (deg_in > 0)).sum())
    print(f"    Connected  : {connected}/{N} dugum")

    # En baglantili 5 dugum
    total_deg = deg_out + deg_in
    top5_idx  = np.argsort(total_deg)[::-1][:5]
    top5      = [(tickers[i], int(total_deg[i])) for i in top5_idx if total_deg[i] > 0]
    if top5:
        print(f"    Top-5 hub  : {top5}")


def main() -> None:
    print("=" * 55)
    print("  07_build_adjacency.py")
    print("=" * 55)

    make_dirs()

    # 1) Ticker sirasi
    print("\n[1/5] Ticker sirasi belirleniyor...")
    tickers = load_ticker_order()
    print(f"  {tickers[:8]} ...")

    # Ticker sirasini kaydet
    node_order_path = TOP50_DIR / "adj_node_order.txt"
    node_order_path.write_text("\n".join(tickers))
    print(f"  [OK] {node_order_path.name}")

    # 2) Iliski tablolarini yukle
    print("\n[2/5] Iliski tablolari yukleniyor...")
    if not TOP50_SUPPLIER_XLSX.exists() or not TOP50_CUSTOMER_XLSX.exists():
        raise FileNotFoundError(
            "Supplier/Customer dosyalari bulunamadi.\n"
            "Once 04_build_sc_relations.py calistirin."
        )
    sup_df = load_relation_table(TOP50_SUPPLIER_XLSX)
    cus_df = load_relation_table(TOP50_CUSTOMER_XLSX)
    print(f"  Supplier tablosu: {sup_df.shape}")
    print(f"  Customer tablosu: {cus_df.shape}")

    # 3) Upstream adjacency (supplier -> firm)
    print("\n[3/5] Upstream adjacency (A_up) olusturuluyor...")
    A_up = build_adjacency(sup_df, tickers, direction="upstream")
    report("adj_sup / A_up", A_up, tickers)
    save_npz(ADJ_SUP_NPZ, A_up)
    print(f"  [OK] {ADJ_SUP_NPZ.name}")

    # 4) Downstream adjacency (firm -> customer)
    print("\n[4/5] Downstream adjacency (A_dn) olusturuluyor...")
    A_dn = build_adjacency(cus_df, tickers, direction="downstream")
    report("adj_cus / A_dn", A_dn, tickers)
    save_npz(ADJ_CUS_NPZ, A_dn)
    print(f"  [OK] {ADJ_CUS_NPZ.name}")

    # 5) Birlesik adjacency (A_up + A_dn, binary)
    print("\n[5/5] Birlesik adjacency (adj.npz) olusturuluyor...")
    A_combined = (A_up + A_dn)
    A_combined.data = np.ones_like(A_combined.data)   # binary yap
    report("adj / combined", A_combined, tickers)
    save_npz(ADJ_NPZ, A_combined)
    print(f"  [OK] {ADJ_NPZ.name}")

    # Kontrol: ve1.csv ile hizalama
    print("\n  Hizalama kontrolu:")
    if TOP50_WEEKLY_CSV.exists():
        vel_cols = pd.read_csv(TOP50_WEEKLY_CSV, nrows=0).columns.tolist()
        vel_cols = [c.strip().upper() for c in vel_cols]
        if vel_cols == tickers:
            print("  [OK] adj.npz ve ve1.csv ticker sirasi AYNI — model hazir!")
        else:
            mismatch = [(i, v, t) for i, (v, t) in enumerate(zip(vel_cols, tickers))
                       if v != t][:5]
            print(f"  [WARN] Siralama farkliligi var: {mismatch}")

    print(f"\n[DONE] 07_build_adjacency.py tamamlandi.")
    print(f"\n  Uretilen dosyalar:")
    print(f"    {ADJ_NPZ.name}      — birlesik adjacency")
    print(f"    {ADJ_SUP_NPZ.name}  — upstream (A_up)")
    print(f"    {ADJ_CUS_NPZ.name}  — downstream (A_dn)")


if __name__ == "__main__":
    main()
