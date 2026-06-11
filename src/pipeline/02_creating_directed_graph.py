"""
src/pipeline/02_creating_directed_graph.py
==========================================
Amac:
  S&P Global Supply Chain Excel dosyalarindan tedarik zinciri iliskilerini
  cikararak yonlu bir graf kurar ve raw_edges.csv olarak kaydeder.

Gercek Excel formati (header yok, satir bazli):
  Sutun 0: Focal firma adi (Excel dosyasinin sirket)
  Sutun 1: Iliskili sirket adi
  Sutun 2: Iliski tipi ("Customer", "Supplier", "Licensee" vs.)
  Sutun 3: Sektor
  Sutun 4: Ulke
  Sutun 5: Yil
  ...

Yon mantigi:
  - Customer dosyasi: sutun 2 == "Customer" -> focal_firm -> related (customer)
  - Supplier dosyasi: sutun 2 == "Supplier" -> related (supplier) -> focal_firm

Girdi:
  Data/Supply Chain Data S&P500/Customer Data S&P500/*.xls(x)
  Data/Supply Chain Data S&P500/Supplier Data S&P500/*.xls(x)

Cikti:
  data/processed/raw_edges.csv   (SUPPLIER, CUSTOMER)
  outputs/figures/graph_top20.png

Calistirmak:
  python src/pipeline/02_creating_directed_graph.py
"""

from __future__ import annotations
import os
from pathlib import Path
import sys
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import (
    SC_CUSTOMER_DIR,
    SC_SUPPLIER_DIR,
    RAW_EDGES_CSV,
    FIG_DIR,
    make_dirs,
)

# Gecerli iliski tipleri
CUSTOMER_TYPES = {"customer", "customers"}
SUPPLIER_TYPES = {"supplier", "suppliers"}


# =============================================================================
# YARDIMCI FONKSIYONLAR
# =============================================================================

def _list_excel_files(directory: Path) -> list[Path]:
    if not directory.exists():
        print(f"  [WARN] Klasor bulunamadi: {directory}")
        return []
    return sorted(
        directory / f for f in os.listdir(directory)
        if f.lower().endswith((".xls", ".xlsx"))
    )


def _extract_edges_from_file(
    fp: Path,
    mode: str,          # "customer" veya "supplier"
    min_data_row: int = 8,
) -> list[tuple[str, str]]:
    """
    Tek bir Excel dosyasindan (SUPPLIER, CUSTOMER) kenarlari cikar.

    mode="customer":
      - Sutun 2 == "Customer" olan satirlar
      - focal_firm (sutun 0) -> related (sutun 1)  [firm -> customer]
      - edge: (focal_firm, related)  yani SUPPLIER=focal, CUSTOMER=related

    mode="supplier":
      - Sutun 2 == "Supplier" olan satirlar
      - related (sutun 1) -> focal_firm (sutun 0)  [supplier -> firm]
      - edge: (related, focal_firm)  yani SUPPLIER=related, CUSTOMER=focal
    """
    try:
        df = pd.read_excel(fp, header=None, dtype=str)
    except Exception as e:
        return []

    # Satir 0-7 genellikle baslik/filtre metni — min_data_row'dan itibaren bak
    df = df.iloc[min_data_row:].reset_index(drop=True)

    if df.shape[1] < 3:
        return []

    # Sutun 0: focal, Sutun 1: related, Sutun 2: iliski tipi
    focal   = df.iloc[:, 0].fillna("").str.strip()
    related = df.iloc[:, 1].fillna("").str.strip()
    rel_type = df.iloc[:, 2].fillna("").str.strip().str.lower()

    edges = []
    if mode == "customer":
        mask = rel_type.isin(CUSTOMER_TYPES)
        for f, r in zip(focal[mask], related[mask]):
            if f and r and f != "nan" and r != "nan":
                edges.append((f, r))   # focal -> customer
    else:  # supplier
        mask = rel_type.isin(SUPPLIER_TYPES)
        for f, r in zip(focal[mask], related[mask]):
            if f and r and f != "nan" and r != "nan":
                edges.append((r, f))   # supplier -> focal

    return edges


def _name_to_ticker_map(sc_dir: Path, mode: str) -> dict[str, str]:
    """
    Excel dosya adlarindan sirket adi -> ticker eslesmesi yap.
    Dosya adi formati: SPGlobal_<SirketAdi>_Customers_DD-Mon-YYYY.xls
    Sutun 0'daki ilk gecerli deger = o dosyanin focal firma adi.
    """
    ticker_map: dict[str, str] = {}
    files = _list_excel_files(sc_dir)

    for fp in files:
        try:
            df = pd.read_excel(fp, header=None, dtype=str, nrows=30)
        except Exception:
            continue

        # Dosya adinda ticker bilgisi yok; sutun 0'daki ilk anlamli degeri bul
        col0 = df.iloc[:, 0].dropna().str.strip()
        col0 = col0[col0.str.len() > 2]
        if col0.empty:
            continue

        # Birden fazla kolonu olan satirlari bul (veri satirlari)
        data_rows = df.dropna(thresh=2)
        if data_rows.empty:
            continue

        # Focal firma adi = veri satirlarindaki sutun 0
        focal_name = None
        for val in data_rows.iloc[:, 0]:
            v = str(val).strip()
            if v and v.lower() not in ("nan", "none", "") and len(v) > 2:
                focal_name = v
                break

        if focal_name is None:
            continue

        # Ticker: dosya adinin ikinci parcasi
        # SPGlobal_AppleInc._Customers_... -> AppleInc.
        parts = fp.stem.split("_")
        if len(parts) >= 2:
            ticker_map[focal_name] = parts[1]

    return ticker_map


def _build_ticker_lookup(sc_dir: Path) -> dict[str, str]:
    """
    Surekli isim-bazli esleme yapmak yerine:
    Her dosyadan focal_firma_adi -> dosya_tagi eslesmesi tutar.
    Ama asil amac: ham sirket adlarindan benzersiz bir anahtar uretmek.
    Burada direkt sirket adi kullanacagiz (ticker olarak).
    """
    return {}


def _draw_graph(
    G: nx.DiGraph,
    out_png: Path,
    title: str,
    node_size: int = 800,
    node_color: str = "#b7dcee",
    font_size: int = 7,
) -> None:
    if G.number_of_nodes() == 0:
        return
    pos = nx.spring_layout(G, seed=42, k=1.5)
    fig, ax = plt.subplots(figsize=(18, 12), dpi=100)
    nx.draw(
        G, pos, ax=ax,
        with_labels=True,
        node_size=node_size,
        node_color=node_color,
        font_size=font_size,
        font_color="black",
        font_weight="bold",
        arrows=True,
        arrowsize=8,
        edge_color="#aaaaaa",
        alpha=0.85,
    )
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Gorsel: {out_png.name}")


# =============================================================================
# ANA AKIS
# =============================================================================

def main() -> None:
    print("=" * 55)
    print("  02_creating_directed_graph.py")
    print("=" * 55)

    make_dirs()
    all_edges: list[tuple[str, str]] = []

    # --- 1) Customer dosyalarindan edge'ler ---
    print("\n[1/3] Customer dosyalari isleniyor...")
    cust_files = _list_excel_files(SC_CUSTOMER_DIR)
    print(f"  {len(cust_files)} dosya bulundu.")

    for i, fp in enumerate(cust_files, 1):
        edges = _extract_edges_from_file(fp, mode="customer")
        all_edges.extend(edges)
        if i % 100 == 0 or i == len(cust_files):
            print(f"  {i}/{len(cust_files)} | toplam edge: {len(all_edges)}")

    # --- 2) Supplier dosyalarindan edge'ler ---
    print("\n[2/3] Supplier dosyalari isleniyor...")
    sup_files = _list_excel_files(SC_SUPPLIER_DIR)
    print(f"  {len(sup_files)} dosya bulundu.")

    for i, fp in enumerate(sup_files, 1):
        edges = _extract_edges_from_file(fp, mode="supplier")
        all_edges.extend(edges)
        if i % 100 == 0 or i == len(sup_files):
            print(f"  {i}/{len(sup_files)} | toplam edge: {len(all_edges)}")

    # Tekrarlari kaldir
    all_edges = list(dict.fromkeys(all_edges))
    print(f"\n  Toplam benzersiz edge: {len(all_edges)}")

    # --- 3) Graf kur ve kaydet ---
    print("\n[3/3] Graf kuruluyor...")
    G = nx.DiGraph()
    G.add_edges_from(all_edges)
    print(f"  Dugum sayisi : {G.number_of_nodes()}")
    print(f"  Kenar sayisi : {G.number_of_edges()}")

    # raw_edges.csv
    df_edges = pd.DataFrame(all_edges, columns=["SUPPLIER", "CUSTOMER"])
    df_edges.to_csv(RAW_EDGES_CSV, index=False, encoding="utf-8-sig")
    print(f"  [OK] {RAW_EDGES_CSV}")

    # Top-20 gorsel
    if G.number_of_nodes() >= 2:
        top_nodes = [n for n, _ in sorted(
            G.degree, key=lambda x: x[1], reverse=True
        )[:20]]
        sub = G.subgraph(top_nodes).copy()
        _draw_graph(
            sub,
            FIG_DIR / "graph_top20.png",
            "Supply-Chain Graph — Top-20 by Degree",
            node_size=2500, node_color="#f7b7a3", font_size=8,
        )

    # Ornek cikti
    print("\n  Ilk 10 edge ornegi:")
    for sup, cus in all_edges[:10]:
        print(f"    {sup}  ->  {cus}")

    print("\n[DONE] 02_creating_directed_graph.py tamamlandi.")


if __name__ == "__main__":
    main()