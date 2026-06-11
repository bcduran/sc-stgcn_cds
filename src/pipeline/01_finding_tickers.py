"""
src/pipeline/01_finding_tickers.py
===================================
Amac:
  Supply Chain Excel dosyalarindan (Customer + Supplier) tum sirket isimlerini
  ve ticker'larini cikartir, temizler ve kaydeder.

Girdi:
  Data/Supply Chain Data S&P500/Customer Data S&P500/*.xlsx
  Data/Supply Chain Data S&P500/Supplier Data S&P500/*.xlsx

Cikti:
  data/processed/a_Customer_Tickers.xlsx
  data/processed/a_Supplier_Tickers.xlsx

Calistirmak:
  python src/pipeline/01_finding_tickers.py
"""

from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd

# configs/ klasorunu path'e ekle
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import (
    SC_CUSTOMER_DIR,
    SC_SUPPLIER_DIR,
    CUSTOMER_TICKERS_XLSX,
    SUPPLIER_TICKERS_XLSX,
    make_dirs,
)


def _extract_tickers_from_folder(folder: Path) -> pd.DataFrame:
    """
    Klasordeki her Excel dosyasini oku.
    Her dosyadan sirket adi (dosya adi) ve ticker kolonunu cikart.
    Donus: DataFrame(columns=["Company", "Ticker"])
    """
    if not folder.exists():
        raise FileNotFoundError(f"Klasor bulunamadi: {folder}")

    xlsx_files = list(folder.glob("*.xlsx")) + list(folder.glob("*.xls"))
    if not xlsx_files:
        raise FileNotFoundError(f"Klasorde Excel dosyasi yok: {folder}")

    print(f"  [INFO] {len(xlsx_files)} dosya bulundu: {folder.name}")

    rows = []
    for fpath in xlsx_files:
        try:
            df = pd.read_excel(fpath, nrows=200)
        except Exception as e:
            print(f"  [WARN] Okuma hatasi ({fpath.name}): {e}")
            continue

        # Ticker kolonunu bul (esnek: 'Ticker', 'Symbol', 'TICKER' vs.)
        ticker_col = None
        for col in df.columns:
            if str(col).strip().upper() in ("TICKER", "SYMBOL", "TKR"):
                ticker_col = col
                break

        if ticker_col is None:
            # Ilk kolonun ticker oldugunu varsay
            ticker_col = df.columns[0]

        tickers = (
            df[ticker_col]
            .astype(str)
            .str.strip()
            .str.upper()
            .dropna()
            .unique()
        )
        tickers = [t for t in tickers if t not in ("NAN", "NONE", "", "TICKER")]

        company_name = fpath.stem  # dosya adi = sirket adi
        for tkr in tickers:
            rows.append({"Company": company_name, "Ticker": tkr})

    if not rows:
        print(f"  [WARN] Hicbir ticker bulunamadi: {folder.name}")
        return pd.DataFrame(columns=["Company", "Ticker"])

    result = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["Ticker"])
        .sort_values("Ticker")
        .reset_index(drop=True)
    )
    return result


def main() -> None:
    print("=" * 55)
    print("  01_finding_tickers.py")
    print("=" * 55)

    make_dirs()

    # --- Customer tickers ---
    print("\n[1/2] Customer ticker'lari cikartiliyor...")
    cus_df = _extract_tickers_from_folder(SC_CUSTOMER_DIR)
    cus_df.to_excel(CUSTOMER_TICKERS_XLSX, index=False)
    print(f"  [OK] {len(cus_df)} ticker -> {CUSTOMER_TICKERS_XLSX}")

    # --- Supplier tickers ---
    print("\n[2/2] Supplier ticker'lari cikartiliyor...")
    sup_df = _extract_tickers_from_folder(SC_SUPPLIER_DIR)
    sup_df.to_excel(SUPPLIER_TICKERS_XLSX, index=False)
    print(f"  [OK] {len(sup_df)} ticker -> {SUPPLIER_TICKERS_XLSX}")

    # --- Ozet ---
    all_tickers = set(cus_df["Ticker"]) | set(sup_df["Ticker"])
    print(f"\n  Toplam benzersiz ticker: {len(all_tickers)}")
    print("\n[DONE] 01_finding_tickers.py tamamlandi.")


if __name__ == "__main__":
    main()
