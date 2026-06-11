"""
src/pipeline/05_cds_weekly_panel.py
=====================================
Amac:
  Ham CDS verisini (gunluk) haftalık (Cuma) ortalama panele donustur.
  TUM tickerlar icin panel uretir — Top-50 filtresi sonraki adimda yapilir.

  Islem:
    1. cds.csv oku (Date, Ticker, PX1..PX10)
    2. PX5 (5Y tenor) kolonunu al
    3. Haftalık (W-FRI) ortalama al
    4. Panel formatina cevir: satir=Ticker, sutun=hafta

Girdi:
  Data/raw/cds.csv

Cikti:
  data/processed/cds_weekly_5Y_all_by_ticker.csv
  data/processed/cds_weekly_5Y_all_by_ticker.xlsx

Calistirmak:
  python src/pipeline/05_cds_weekly_panel.py
"""

from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import (
    CDS_RAW_CSV,
    CDS_WEEKLY_CSV,
    PROCESSED_DIR,
    make_dirs,
    DataCFG,
)


def main() -> None:
    print("=" * 55)
    print("  05_cds_weekly_panel.py")
    print("=" * 55)

    make_dirs()

    # 1) Ham veriyi oku
    print(f"\n[1/4] CDS ham verisi yukleniyor...")
    print(f"  Kaynak: {CDS_RAW_CSV}")

    if not CDS_RAW_CSV.exists():
        raise FileNotFoundError(f"CDS dosyasi bulunamadi: {CDS_RAW_CSV}")

    df = pd.read_csv(CDS_RAW_CSV)
    print(f"  Ham veri: {df.shape[0]:,} satir, {df.shape[1]} kolon")
    print(f"  Kolonlar: {df.columns.tolist()}")

    # 2) Kolon normalizasyonu
    print("\n[2/4] Kolon normalizasyonu...")
    cols_lower = {c.lower(): c for c in df.columns}

    if "date" not in cols_lower or "ticker" not in cols_lower:
        raise ValueError(f"'Date' ve 'Ticker' kolonlari gerekli. Mevcut: {df.columns.tolist()}")

    # PX5 kolonunu bul
    px_col = None
    for cand in ["px5", "PX5", "Px5", "cds5y", "CDS5Y", "cds_5y"]:
        if cand.lower() in cols_lower:
            px_col = cols_lower[cand.lower()]
            break
    if px_col is None:
        px_candidates = [c for c in df.columns if c.upper().startswith("PX")]
        raise ValueError(f"PX5 kolonu bulunamadi. Mevcut PX kolonlari: {px_candidates}")

    print(f"  PX5 kolonu: '{px_col}'")

    # Normalize
    df["Date"]   = pd.to_datetime(df[cols_lower["date"]], errors="coerce")
    df["Ticker"] = df[cols_lower["ticker"]].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["Date"])
    df = df[["Date", "Ticker", px_col]].rename(columns={px_col: "PX5"})

    # Tarih filtresi
    start = pd.Timestamp(DataCFG.START_DATE)
    end   = pd.Timestamp(DataCFG.END_DATE)
    df    = df[(df["Date"] >= start) & (df["Date"] <= end)]
    print(f"  Tarih filtresi: {start.date()} - {end.date()}")
    print(f"  Kalan satir: {len(df):,} | Benzersiz ticker: {df['Ticker'].nunique()}")

    # 3) Haftalık panel
    print(f"\n[3/4] Haftalık (W-FRI) ortalama hesaplaniyor...")
    weekly = (
        df.pivot_table(index="Date", columns="Ticker", values="PX5", aggfunc="mean")
          .sort_index()
          .resample(DataCFG.WEEKLY_FREQ)
          .mean()
    )
    print(f"  Haftalik panel: {weekly.shape[0]} hafta x {weekly.shape[1]} ticker")
    print(f"  Ilk hafta: {weekly.index[0].date()}  |  Son hafta: {weekly.index[-1].date()}")

    # Panel formatina cevir: Ticker satir, haftalar sutun
    panel = weekly.T.copy()
    panel.index.name = "Ticker"
    panel.columns = pd.to_datetime(panel.columns).strftime("%Y-%m-%d")

    # NaN orani kontrolu
    nan_ratio = panel.isna().mean().mean()
    print(f"  Ortalama NaN orani: {nan_ratio:.1%}")

    # 4) Kaydet
    print(f"\n[4/4] Kaydediliyor...")
    xlsx_path = PROCESSED_DIR / "cds_weekly_5Y_all_by_ticker.xlsx"

    panel.reset_index().to_excel(xlsx_path, index=False)
    panel.reset_index().to_csv(CDS_WEEKLY_CSV, index=False, encoding="utf-8-sig")

    print(f"  [OK] {xlsx_path.name}  shape={panel.shape}")
    print(f"  [OK] {CDS_WEEKLY_CSV.name}  shape={panel.shape}")
    print(f"  Tarih araligi: {panel.columns[0]} -> {panel.columns[-1]}")

    print("\n[DONE] 05_cds_weekly_panel.py tamamlandi.")


if __name__ == "__main__":
    main()
