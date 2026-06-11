"""
src/pipeline/09_full_adjacency.py
===================================
Amac:
  Full universe (661 ticker) icin supply chain adjacency matrisleri olustur.
  CDS ticker adlari -> raw_edges.csv sirket adlari fuzzy matching.

Girdi:
  data/full/tickers_full.txt
  Data/raw/cds.csv             (ticker -> sirket adi icin)
  data/processed/raw_edges.csv

Cikti:
  data/full/adj_full.npz       <- birlesik adjacency
  data/full/adj_sup_full.npz   <- upstream (supplier->firm)
  data/full/adj_cus_full.npz   <- downstream (firm->customer)
  data/full/adj_match_log.csv  <- esleme raporu

Calistirmak:
  python src/pipeline/09_full_adjacency.py
"""

from __future__ import annotations
from pathlib import Path
import sys
import re
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import (
    CDS_RAW_CSV,
    RAW_EDGES_CSV,
    TICKERS_FULL_TXT,
    ADJ_FULL_NPZ,
    ADJ_SUP_FULL_NPZ,
    ADJ_CUS_FULL_NPZ,
    MATCH_LOG_CSV,
    FULL_DIR,
    make_dirs,
)

THRESHOLD = 0.72


def _normalize(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"[&.,'\-\(\)]", " ", s)
    stopwords = {"inc","corp","co","ltd","llc","plc","the","group",
                 "company","companies","holdings","international",
                 "corporation","incorporated","limited"}
    return " ".join(t for t in s.split() if t and t not in stopwords)


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def main() -> None:
    print("=" * 60)
    print("  09_full_adjacency.py")
    print("=" * 60)

    make_dirs()

    # 1) Ticker listesi
    print(f"\n[1/4] Ticker listesi yukleniyor...")
    print(f"  Kaynak: {TICKERS_FULL_TXT}")
    if not TICKERS_FULL_TXT.exists():
        print("  [ERROR] tickers_full.txt bulunamadi.")
        print("  Once calistir: python src/pipeline/08_full_universe_panel.py")
        return
    tickers = TICKERS_FULL_TXT.read_text().strip().splitlines()
    N = len(tickers)
    idx = {t: i for i, t in enumerate(tickers)}
    print(f"  {N} ticker yuklendi.")

    # 2) CDS ticker -> sirket adi
    print(f"\n[2/4] CDS sirket adlari yukleniyor...")
    cds = pd.read_csv(CDS_RAW_CSV, usecols=["Ticker","Company"]).drop_duplicates()
    cds["Ticker"] = cds["Ticker"].astype(str).str.strip().str.upper()
    cds_dict = dict(zip(cds["Ticker"], cds["Company"]))
    matched_names = sum(1 for t in tickers if t in cds_dict)
    print(f"  CDS'te isim bulunan: {matched_names}/{N} ticker")

    # 3) Raw edges + fuzzy matching
    print(f"\n[3/4] Raw edges yukleniyor ve fuzzy matching yapiliyor...")
    print(f"  Kaynak: {RAW_EDGES_CSV}")
    edges_df = pd.read_csv(RAW_EDGES_CSV)
    graph_names = list(set(
        edges_df["SUPPLIER"].astype(str).str.strip().tolist() +
        edges_df["CUSTOMER"].astype(str).str.strip().tolist()
    ))
    print(f"  Graf dugum sayisi: {len(graph_names):,}")
    print(f"  Ham kenar sayisi : {len(edges_df):,}")
    print(f"  Fuzzy matching basliyor ({N} ticker x {len(graph_names):,} dugum)...")
    print(f"  Bu islem ~5-15 dakika surebilir...")

    name_to_ticker = {}
    match_log = []
    used_nodes = set()

    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0:
            done = sum(1 for r in match_log if r["GraphNode"] not in ("NO_MATCH","NO_CDS_NAME"))
            print(f"  {i+1}/{N} islendi | eslesen: {done}")

        cds_name = cds_dict.get(ticker, "")
        if not cds_name:
            match_log.append({
                "Ticker": ticker, "Score": 0.0,
                "GraphNode": "NO_CDS_NAME", "CDS_Name": ""
            })
            continue

        best_node, best_score = None, 0.0
        for node in graph_names:
            if node in used_nodes:
                continue
            s = _sim(cds_name, node)
            if s > best_score:
                best_score, best_node = s, node

        if best_node and best_score >= THRESHOLD:
            name_to_ticker[best_node] = ticker
            used_nodes.add(best_node)
            match_log.append({
                "Ticker": ticker, "Score": round(best_score, 3),
                "GraphNode": best_node, "CDS_Name": cds_name
            })
        else:
            match_log.append({
                "Ticker": ticker, "Score": round(best_score, 3),
                "GraphNode": "NO_MATCH", "CDS_Name": cds_name
            })

    matched = sum(1 for r in match_log if r["GraphNode"] not in ("NO_MATCH","NO_CDS_NAME"))
    print(f"\n  Esleme sonucu: {matched}/{N} ticker eslesti")
    print(f"  NO_MATCH     : {sum(1 for r in match_log if r['GraphNode']=='NO_MATCH')}")
    print(f"  NO_CDS_NAME  : {sum(1 for r in match_log if r['GraphNode']=='NO_CDS_NAME')}")

    pd.DataFrame(match_log).to_csv(MATCH_LOG_CSV, index=False)
    print(f"  [OK] {MATCH_LOG_CSV}")

    # 4) Adjacency matrisler
    print(f"\n[4/4] Adjacency matrisler olusturuluyor...")
    up_rows, up_cols = [], []
    for _, row in edges_df.iterrows():
        sup = str(row["SUPPLIER"]).strip()
        cus = str(row["CUSTOMER"]).strip()
        t_sup = name_to_ticker.get(sup)
        t_cus = name_to_ticker.get(cus)
        if t_sup and t_cus and t_sup in idx and t_cus in idx:
            up_rows.append(idx[t_sup])
            up_cols.append(idx[t_cus])

    A_up = csr_matrix(
        (np.ones(len(up_rows), dtype=np.float32), (up_rows, up_cols)),
        shape=(N, N)
    )
    A_dn   = A_up.T.tocsr()
    A_comb = (A_up + A_dn)
    A_comb.data = np.ones_like(A_comb.data)

    connected = int(((np.array(A_comb.sum(1)).flatten() > 0) |
                     (np.array(A_comb.sum(0)).flatten() > 0)).sum())
    density = A_comb.nnz / (N * N)

    print(f"  A_up  nnz     : {A_up.nnz:,}")
    print(f"  A_dn  nnz     : {A_dn.nnz:,}")
    print(f"  A_comb nnz    : {A_comb.nnz:,}")
    print(f"  Density       : {density:.4%}")
    print(f"  Baglantili    : {connected}/{N} dugum")

    save_npz(ADJ_SUP_FULL_NPZ, A_up)
    save_npz(ADJ_CUS_FULL_NPZ, A_dn)
    save_npz(ADJ_FULL_NPZ,     A_comb)

    print(f"\n  [OK] {ADJ_SUP_FULL_NPZ}")
    print(f"  [OK] {ADJ_CUS_FULL_NPZ}")
    print(f"  [OK] {ADJ_FULL_NPZ}")
    print(f"\n  Cikti klasoru: {FULL_DIR}")
    print(f"\n[DONE] 09_full_adjacency.py tamamlandi.")


if __name__ == "__main__":
    main()
