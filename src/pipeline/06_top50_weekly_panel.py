"""
src/pipeline/06_top50_weekly_panel.py
======================================
Amac:
  661 tickerlik haftalik CDS panelinden sadece Top-50'yi filtrele.
  NaN'lari temizle, model girdisi olan ve1.csv'yi uret.

  ve1.csv formati (SC-STGCN modeli icin):
    - Satir    : haftalar (T x N)
    - Sutun    : ticker adlari (N = 50)
    - Deger    : haftalik ortalama PX5 spread (bp)
    - Date sutunu YOK — sadece sayisal matris

Girdi:
  data/processed/cds_weekly_5Y_all_by_ticker.csv
  data/processed/e_top50_companies.xlsx

Cikti:
  data/top50/ve1.csv              <- model girdisi (T x N sayisal matris)
  data/processed/e_top50_weekly_price_info.xlsx  <- insan okunabilir versiyon

Calistirmak:
  python src/pipeline/06_top50_weekly_panel.py
"""

from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import (
    CDS_WEEKLY_CSV,
    TOP50_XLSX,
    TOP50_WEEKLY_CSV,
    PROCESSED_DIR,
    make_dirs,
    DataCFG,
)


def main() -> None:
    print("=" * 55)
    print("  06_top50_weekly_panel.py")
    print("=" * 55)

    make_dirs()

    # 1) Top-50 listesi
    print("\n[1/4] Top-50 listesi yukleniyor...")
    if not TOP50_XLSX.exists():
        raise FileNotFoundError(
            f"Top-50 dosyasi bulunamadi: {TOP50_XLSX}\n"
            "Once 03_select_top50.py calistirin."
        )
    top50_df = pd.read_excel(TOP50_XLSX)
    top50    = top50_df["Ticker"].astype(str).str.strip().str.upper().tolist()
    top50_set = set(top50)
    print(f"  {len(top50)} ticker: {top50[:8]} ...")

    # 2) Haftalık CDS paneli yukle
    print("\n[2/4] Haftalık CDS paneli yukleniyor...")
    if not CDS_WEEKLY_CSV.exists():
        raise FileNotFoundError(
            f"Haftalik panel bulunamadi: {CDS_WEEKLY_CSV}\n"
            "Once 05_cds_weekly_panel.py calistirin."
        )
    raw = pd.read_csv(CDS_WEEKLY_CSV)
    ticker_col = raw.columns[0]
    raw[ticker_col] = raw[ticker_col].astype(str).str.strip().str.upper()
    print(f"  Ham panel: {raw.shape[0]} ticker x {raw.shape[1]-1} hafta")

    # 3) Top-50 filtresi
    print("\n[3/4] Top-50 filtresi uygulanıyor...")
    raw_top = raw[raw[ticker_col].isin(top50_set)].copy()
    raw_top = raw_top.drop_duplicates(subset=[ticker_col])

    missing = top50_set - set(raw_top[ticker_col])
    if missing:
        print(f"  [WARN] Panelde olmayan Top-50 tickerlari: {sorted(missing)}")
    print(f"  Eslesen: {len(raw_top)} / {len(top50)} ticker")

    # Top-50 siralamasi koru (SP weight sirasi)
    raw_top = raw_top.set_index(ticker_col)
    raw_top = raw_top.reindex([t for t in top50 if t in raw_top.index])

    # (Ticker x Week) → (Week x Ticker)
    df = raw_top.T.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()

    print(f"  Panel: {df.shape[0]} hafta x {df.shape[1]} ticker")
    print(f"  Tarih: {df.index[0].date()} - {df.index[-1].date()}")

    # NaN temizleme
    nan_before = df.isna().sum().sum()
    df = df.ffill().bfill()      # once ileri, sonra geri doldur
    df = df.fillna(df.mean())    # kalan varsa kolon ortalamasiyla doldur
    nan_after = df.isna().sum().sum()
    print(f"  NaN: {nan_before} -> {nan_after} (temizlendi)")

    # 4) Kaydet
    print("\n[4/4] Kaydediliyor...")

    # ve1.csv — model girdisi (sadece sayisal, tarih sutunu yok)
    # Sutun isimleri = ticker adlari, satir indeksi = hafta numarasi (0, 1, 2...)
    vel_df = df.copy()
    vel_df.columns = [str(c) for c in vel_df.columns]
    vel_df.to_csv(TOP50_WEEKLY_CSV, index=False)
    print(f"  [OK] {TOP50_WEEKLY_CSV}  shape={vel_df.shape}")

    # Excel — tarihli, insan okunabilir versiyon
    df_excel = df.reset_index()
    df_excel.columns = ["Date"] + list(df.columns)
    out_xlsx = PROCESSED_DIR / "e_top50_weekly_price_info.xlsx"
    df_excel.to_excel(out_xlsx, index=False)
    print(f"  [OK] {out_xlsx.name}  shape={df_excel.shape}")

    # Ozet istatistikler
    print(f"\n  Spread istatistikleri (bp):")
    print(f"  {'Ticker':<8} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*45}")
    for col in list(df.columns)[:10]:
        s = df[col].dropna()
        print(f"  {col:<8} {s.mean():>8.1f} {s.std():>8.1f} "
              f"{s.min():>8.1f} {s.max():>8.1f}")
    if len(df.columns) > 10:
        print(f"  ... ({len(df.columns)-10} ticker daha)")

    print("\n[DONE] 06_top50_weekly_panel.py tamamlandi.")
    print(f"\n  Model girdisi hazir: {TOP50_WEEKLY_CSV}")
    print(f"  Shape: {vel_df.shape[0]} hafta x {vel_df.shape[1]} ticker")


if __name__ == "__main__":
    main()
