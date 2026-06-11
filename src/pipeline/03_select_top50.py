"""
src/pipeline/03_select_top50.py
================================
Amac:
  S&P 500 ilk 100 sirket (weight siralamasina gore, Nisan 2025)
  arasindan 2015-2021 CDS coverage >= 70% olanlari sec,
  weight siralamasini koruyarak Top-50 olustur.

Cikti:
  data/processed/e_top50_companies.xlsx
  outputs/figures/top50_selection.png
"""

from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import (
    CDS_RAW_CSV, TOP50_XLSX, FIG_DIR, make_dirs, DataCFG,
)

N_TOP        = 50
MIN_COVERAGE = DataCFG.MIN_COVERAGE   # 0.70

# S&P 500 Top-100 (Weight sirasi, Nisan 2025 — Slickcharts)
# Format: (SP_Rank, Ticker, Company, SP_Weight%)
CANDIDATES = [
    (  1, "NVDA",  "Nvidia",                          7.55),
    (  2, "AAPL",  "Apple Inc.",                       6.12),
    (  3, "MSFT",  "Microsoft",                        4.89),
    (  4, "AMZN",  "Amazon",                           4.23),
    (  5, "GOOGL", "Alphabet Inc. (Class A)",          3.27),
    (  6, "GOOG",  "Alphabet Inc. (Class C)",          3.03),
    (  7, "AVGO",  "Broadcom",                         2.95),
    (  8, "META",  "Meta Platforms",                   2.68),
    (  9, "TSLA",  "Tesla Inc.",                       2.29),
    ( 10, "BRK.B", "Berkshire Hathaway",               1.59),
    ( 11, "WMT",   "Walmart",                          1.54),
    ( 12, "JPM",   "JPMorgan Chase",                   1.30),
    ( 13, "LLY",   "Lilly (Eli)",                      1.28),
    ( 14, "V",     "Visa Inc.",                        0.94),
    ( 15, "XOM",   "ExxonMobil",                       0.92),
    ( 16, "JNJ",   "Johnson & Johnson",                0.87),
    ( 17, "ORCL",  "Oracle Corporation",               0.80),
    ( 18, "MU",    "Micron Technology",                0.80),
    ( 19, "MA",    "Mastercard",                       0.72),
    ( 20, "AMD",   "Advanced Micro Devices",           0.70),
    ( 21, "COST",  "Costco",                           0.68),
    ( 22, "NFLX",  "Netflix",                          0.64),
    ( 23, "BAC",   "Bank of America",                  0.60),
    ( 24, "ABBV",  "AbbVie",                           0.58),
    ( 25, "CAT",   "Caterpillar Inc.",                 0.57),
    ( 26, "CVX",   "Chevron Corporation",              0.55),
    ( 27, "PLTR",  "Palantir Technologies",            0.54),
    ( 28, "HD",    "Home Depot (The)",                 0.54),
    ( 29, "INTC",  "Intel",                            0.54),
    ( 30, "PG",    "Procter & Gamble",                 0.53),
    ( 31, "CSCO",  "Cisco",                            0.52),
    ( 32, "LRCX",  "Lam Research",                    0.51),
    ( 33, "KO",    "Coca-Cola Company",                0.50),
    ( 34, "GE",    "GE Aerospace",                     0.50),
    ( 35, "AMAT",  "Applied Materials",                0.48),
    ( 36, "MS",    "Morgan Stanley",                   0.46),
    ( 37, "UNH",   "UnitedHealth Group",               0.46),
    ( 38, "MRK",   "Merck & Co.",                      0.45),
    ( 39, "GS",    "Goldman Sachs",                    0.42),
    ( 40, "RTX",   "RTX Corporation",                  0.41),
    ( 41, "GEV",   "GE Vernova",                       0.41),
    ( 42, "WFC",   "Wells Fargo",                      0.39),
    ( 43, "PM",    "Philip Morris International",      0.38),
    ( 44, "IBM",   "IBM",                              0.37),
    ( 45, "AXP",   "American Express",                 0.36),
    ( 46, "KLAC",  "KLA Corporation",                  0.36),
    ( 47, "LIN",   "Linde plc",                        0.35),
    ( 48, "C",     "Citigroup",                        0.35),
    ( 49, "MCD",   "McDonald's",                       0.34),
    ( 50, "TMUS",  "T-Mobile US",                      0.34),
    ( 51, "PEP",   "PepsiCo",                          0.33),
    ( 52, "TXN",   "Texas Instruments",                0.32),
    ( 53, "ANET",  "Arista Networks",                  0.32),
    ( 54, "TMO",   "Thermo Fisher Scientific",         0.30),
    ( 55, "VZ",    "Verizon",                          0.30),
    ( 56, "AMGN",  "Amgen",                            0.29),
    ( 57, "NEE",   "NextEra Energy",                   0.29),
    ( 58, "DIS",   "Walt Disney Company",              0.29),
    ( 59, "APH",   "Amphenol",                         0.29),
    ( 60, "T",     "AT&T",                             0.29),
    ( 61, "BA",    "Boeing",                           0.28),
    ( 62, "ADI",   "Analog Devices",                   0.28),
    ( 63, "TJX",   "TJX Companies",                   0.27),
    ( 64, "CRM",   "Salesforce",                       0.27),
    ( 65, "GILD",  "Gilead Sciences",                  0.26),
    ( 66, "ABT",   "Abbott Laboratories",              0.26),
    ( 67, "ISRG",  "Intuitive Surgical",               0.26),
    ( 68, "BLK",   "BlackRock",                        0.25),
    ( 69, "APP",   "AppLovin Corporation",             0.25),
    ( 70, "SCHW",  "Charles Schwab Corporation",       0.25),
    ( 71, "UBER",  "Uber",                             0.25),
    ( 72, "DE",    "Deere & Company",                  0.25),
    ( 73, "ETN",   "Eaton Corporation",                0.24),
    ( 74, "PFE",   "Pfizer",                           0.24),
    ( 75, "BKNG",  "Booking Holdings",                 0.23),
    ( 76, "UNP",   "Union Pacific Corporation",        0.23),
    ( 77, "WELL",  "Welltower",                        0.23),
    ( 78, "HON",   "Honeywell",                        0.23),
    ( 79, "QCOM",  "Qualcomm",                         0.22),
    ( 80, "GLW",   "Corning Inc.",                     0.22),
    ( 81, "LOW",   "Lowe's",                           0.22),
    ( 82, "LMT",   "Lockheed Martin",                  0.22),
    ( 83, "DHR",   "Danaher Corporation",              0.21),
    ( 84, "COP",   "ConocoPhillips",                   0.21),
    ( 85, "PANW",  "Palo Alto Networks",               0.21),
    ( 86, "PLD",   "Prologis",                         0.21),
    ( 87, "SNDK",  "Sandisk Corporation",              0.21),
    ( 88, "SYK",   "Stryker Corporation",              0.20),
    ( 89, "SPGI",  "S&P Global",                       0.20),
    ( 90, "COF",   "Capital One",                      0.20),
    ( 91, "CB",    "Chubb Limited",                    0.20),
    ( 92, "DELL",  "Dell Technologies",                0.20),
    ( 93, "NEM",   "Newmont",                          0.20),
    ( 94, "WDC",   "Western Digital",                  0.19),
    ( 95, "PH",    "Parker Hannifin",                  0.19),
    ( 96, "BMY",   "Bristol Myers Squibb",             0.19),
    ( 97, "ACN",   "Accenture",                        0.19),
    ( 98, "STX",   "Seagate Technology",               0.19),
    ( 99, "PGR",   "Progressive Corporation",          0.18),
    (100, "VRT",   "Vertiv Holdings Co",               0.18),
]


def compute_cds_coverage() -> pd.DataFrame:
    """CDS dosyasindan her ticker icin PX5 coverage hesapla."""
    print("  CDS dosyasi yukleniyor...")
    df = pd.read_csv(CDS_RAW_CSV, parse_dates=["Date"])
    start = pd.Timestamp(DataCFG.START_DATE)
    end   = pd.Timestamp(DataCFG.END_DATE)
    df    = df[(df["Date"] >= start) & (df["Date"] <= end)]
    print(f"  Tarih  : {df['Date'].min().date()} - {df['Date'].max().date()}")
    print(f"  Ticker : {df['Ticker'].nunique()} benzersiz")

    total = df.groupby("Ticker")["Date"].count()
    valid = df.groupby("Ticker")["PX5"].apply(lambda x: x.notna().sum())
    cov   = (valid / total).rename("Coverage")
    mean5 = df.groupby("Ticker")["PX5"].mean().rename("PX5_mean")
    comp  = (df.groupby("Ticker")["Company"]
               .agg(lambda x: x.mode().iloc[0])
               .rename("CDS_Company"))

    result = pd.concat([comp, cov, mean5], axis=1).reset_index()
    result.columns = ["Ticker", "CDS_Company", "Coverage", "PX5_mean"]
    return result


def main() -> None:
    print("=" * 60)
    print(f"  03_select_top50.py  [{len(CANDIDATES)} aday -> Top-{N_TOP}]")
    print("=" * 60)
    make_dirs()

    # 1) Aday DataFrame
    candidates = pd.DataFrame(CANDIDATES,
                               columns=["SP_Rank","Ticker","Company","SP_Weight"])
    print(f"\n[1/3] {len(candidates)} aday hazirlandi.")

    # 2) CDS coverage
    print("\n[2/3] CDS coverage hesaplaniyor...")
    cds_cov   = compute_cds_coverage()
    cov_dict  = dict(zip(cds_cov["Ticker"], cds_cov["Coverage"]))
    mean_dict = dict(zip(cds_cov["Ticker"], cds_cov["PX5_mean"]))

    candidates["Coverage"] = candidates["Ticker"].map(cov_dict).fillna(0.0)
    candidates["PX5_mean"] = candidates["Ticker"].map(mean_dict)

    # Ozet
    above   = (candidates["Coverage"] >= MIN_COVERAGE).sum()
    below   = ((candidates["Coverage"] > 0) & (candidates["Coverage"] < MIN_COVERAGE)).sum()
    no_data = (candidates["Coverage"] == 0).sum()
    print(f"\n  Coverage >= {MIN_COVERAGE:.0%} : {above} sirket")
    print(f"  Coverage yetersiz : {below} sirket")
    print(f"  CDS verisi YOK   : {no_data} sirket")

    # CDS'i olmayan buyuk isimler
    no_cds_list = candidates[candidates["Coverage"] == 0][["SP_Rank","Ticker","Company"]].to_dict("records")
    if no_cds_list:
        print("\n  CDS verisi olmayan adaylar:")
        for r in no_cds_list:
            print(f"    SP#{r['SP_Rank']:>3}  {r['Ticker']:<8}  {r['Company']}")

    # 3) Top-50 sec
    print(f"\n[3/3] Top-{N_TOP} seciliyor...")
    qualified = (candidates[candidates["Coverage"] >= MIN_COVERAGE]
                 .sort_values("SP_Rank")
                 .head(N_TOP)
                 .reset_index(drop=True))
    qualified.index += 1

    if len(qualified) < N_TOP:
        print(f"  [WARN] Sadece {len(qualified)} sirket secilebildi.")

    # Kaydet
    save_df = qualified[["Ticker","Company","Coverage","SP_Weight","SP_Rank","PX5_mean"]]
    save_df.to_excel(TOP50_XLSX, index=False)
    print(f"  [OK] {TOP50_XLSX}")

    # Ozet tablo
    print(f"\n  {'#':<4} {'Ticker':<8} {'Coverage':>8} {'SP_Rank':>8} {'PX5_mean':>9}  Company")
    print(f"  {'-'*72}")
    for i, row in qualified.iterrows():
        print(f"  {i:<4} {row['Ticker']:<8} {row['Coverage']:>8.1%} "
              f"{int(row['SP_Rank']):>8} {row['PX5_mean']:>9.1f}  "
              f"{str(row['Company'])[:30]}")

    # Gorsel
    fig, ax = plt.subplots(figsize=(10, 16))
    all_c = candidates.sort_values("SP_Rank", ascending=False)
    selected_set = set(qualified["Ticker"])
    colors = ["#4C72B0" if t in selected_set else "#d9d9d9"
              for t in all_c["Ticker"]]
    ax.barh(
        all_c["Ticker"], all_c["Coverage"] * 100,
        color=colors, alpha=0.9, edgecolor="white", linewidth=0.3
    )
    ax.axvline(MIN_COVERAGE * 100, color="red", linestyle="--",
               linewidth=1.2)
    ax.legend(handles=[
        Patch(color="#4C72B0", label=f"Secilen Top-{N_TOP}"),
        Patch(color="#d9d9d9", label="Elendi"),
        plt.Line2D([0],[0], color="red", linestyle="--", label=f"Min {MIN_COVERAGE:.0%}"),
    ], fontsize=9, loc="lower right")
    ax.set_xlabel("PX5 Coverage (2015-2021, %)")
    ax.set_title(f"S&P 500 Top-{len(CANDIDATES)} Aday — CDS Coverage & Secim", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out_fig = FIG_DIR / "top50_selection.png"
    fig.savefig(out_fig, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"\n  [OK] Gorsel: {out_fig.name}")

    print(f"\n  Ortalama Coverage : {qualified['Coverage'].mean():.1%}")
    print(f"  Min Coverage      : {qualified['Coverage'].min():.1%}")
    print(f"  Elenen aday       : {len(candidates) - len(qualified)}")
    print("\n[DONE] 03_select_top50.py tamamlandi.")


if __name__ == "__main__":
    main()
