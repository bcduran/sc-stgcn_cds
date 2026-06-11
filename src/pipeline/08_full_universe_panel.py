"""
src/pipeline/08_full_universe_panel.py
=======================================
Amac:
  661 ticker icin tam CDS haftalik paneli olustur.
  Coverage >= 70% olan tum tickerlari dahil et.

Girdi:
  data/processed/cds_weekly_5Y_all_by_ticker.csv

Cikti:
  data/full/ve1_full.csv       <- model girdisi (T x N_full)
  data/full/tickers_full.txt   <- ticker sirasi

Calistirmak:
  python src/pipeline/08_full_universe_panel.py
"""

from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import (
    CDS_WEEKLY_CSV,
    VE1_FULL_CSV,
    TICKERS_FULL_TXT,
    FULL_DIR,
    make_dirs,
    DataCFG,
)

MIN_COV = DataCFG.MIN_COVERAGE  # 0.70


def main() -> None:
    print("=" * 55)
    print("  08_full_universe_panel.py")
    print("=" * 55)

    make_dirs()

    # 1) Haftalik panel yukle
    print(f"\n[1/3] Haftalik CDS paneli yukleniyor...")
    print(f"  Kaynak: {CDS_WEEKLY_CSV}")
    raw = pd.read_csv(CDS_WEEKLY_CSV)
    ticker_col = raw.columns[0]
    raw[ticker_col] = raw[ticker_col].astype(str).str.strip().str.upper()
    raw = raw.set_index(ticker_col)
    print(f"  Ham panel: {raw.shape[0]} ticker x {raw.shape[1]} hafta")

    # 2) Coverage filtresi
    print(f"\n[2/3] Coverage >= {MIN_COV:.0%} filtresi uygulanıyor...")
    coverage = raw.notna().mean(axis=1)
    qualified = raw[coverage >= MIN_COV].copy()
    print(f"  Gecen ticker : {len(qualified)} / {len(raw)}")
    print(f"  Coverage ozeti:")
    print(f"    %100 coverage : {(coverage == 1.0).sum()} ticker")
    print(f"    %90+ coverage : {(coverage >= 0.9).sum()} ticker")
    print(f"    %70+ coverage : {(coverage >= 0.7).sum()} ticker")
    print(f"    Elenen        : {(coverage < MIN_COV).sum()} ticker")

    # 3) (Ticker x Week) -> (Week x Ticker) transpose
    print(f"\n[3/3] Panel olusturuluyor ve kaydediliyor...")
    df = qualified.T.copy()
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df.sort_index()

    # NaN temizle
    nan_before = df.isna().sum().sum()
    df = df.ffill().bfill()
    df = df.fillna(df.mean())
    nan_after = df.isna().sum().sum()
    print(f"  NaN: {nan_before:,} -> {nan_after}")
    print(f"  Final panel shape: {df.shape[0]} hafta x {df.shape[1]} ticker")
    print(f"  Tarih: {df.index[0].date()} - {df.index[-1].date()}")

    # Kaydet
    df.columns = [str(c) for c in df.columns]
    df.to_csv(VE1_FULL_CSV, index=False)
    TICKERS_FULL_TXT.write_text("\n".join(df.columns.tolist()))

    print(f"\n  [OK] {VE1_FULL_CSV}")
    print(f"  [OK] {TICKERS_FULL_TXT}")
    print(f"\n  Cikti klasoru: {FULL_DIR}")
    print(f"\n[DONE] 08_full_universe_panel.py tamamlandi.")


if __name__ == "__main__":
    main()
