"""
src/analysis/firm_r2.py
========================
Her firma icin ayri ayri R² hesaplar.
Girdi: outputs/predictions/ klasorundeki y_true ve y_pred CSV'leri
Cikti: outputs/metrics/firm_r2.csv
       outputs/figures/firm_r2.png
"""

from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import PRED_DIR, METRICS_DIR, FIG_DIR, TOP50_WEEKLY_CSV, make_dirs


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot < 1e-10:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def load_predictions(tag: str) -> tuple[np.ndarray, np.ndarray] | None:
    true_path = PRED_DIR / f"{tag}_y_true.csv"
    pred_path = PRED_DIR / f"{tag}_y_pred.csv"
    if not true_path.exists() or not pred_path.exists():
        return None
    y_true = pd.read_csv(true_path).values
    y_pred = pd.read_csv(pred_path).values
    return y_true, y_pred


def main() -> None:
    make_dirs()

    # Ticker listesi
    tickers = pd.read_csv(TOP50_WEEKLY_CSV, nrows=0).columns.tolist()

    # Model tag'leri
    models = {
        "AR(1)"    : "delta_ar1",
        "LSTM"     : "delta_lstm",
        "XGBoost"  : "delta_xgb",
        "V-STGCN"  : "delta_vstgcn",
        "SC-STGCN" : "delta_scstgcn",
    }

    print("=" * 60)
    print("  Firm-by-Firm R² Analysis")
    print("=" * 60)

    results = {}

    for model_name, tag in models.items():
        data = load_predictions(tag)
        if data is None:
            print(f"  [SKIP] {model_name}: predictions not found ({tag})")
            continue

        y_true, y_pred = data
        firm_r2 = []
        for i, ticker in enumerate(tickers[:y_true.shape[1]]):
            r2 = r2_score(y_true[:, i], y_pred[:, i])
            firm_r2.append(r2)

        results[model_name] = firm_r2

        pos = sum(1 for r in firm_r2 if not np.isnan(r) and r > 0)
        mean_r2 = np.nanmean(firm_r2)
        print(f"\n  {model_name}:")
        print(f"    Mean R²       : {mean_r2:+.4f}")
        print(f"    Positive R²   : {pos}/{len(firm_r2)} firms")
        print(f"    Best firm     : {tickers[np.nanargmax(firm_r2)]} "
              f"({np.nanmax(firm_r2):+.4f})")
        print(f"    Worst firm    : {tickers[np.nanargmin(firm_r2)]} "
              f"({np.nanmin(firm_r2):+.4f})")

    # DataFrame
    df = pd.DataFrame(results, index=tickers[:len(list(results.values())[0])])
    df.index.name = "Ticker"
    df.to_csv(METRICS_DIR / "firm_r2.csv")
    print(f"\n  [OK] {METRICS_DIR / 'firm_r2.csv'}")

    # Ozet tablo
    print(f"\n  {'Ticker':<8}", end="")
    for m in results:
        print(f"  {m:>10}", end="")
    print()
    print("  " + "-" * (8 + 12 * len(results)))
    for ticker in df.index:
        print(f"  {ticker:<8}", end="")
        for m in results:
            val = df.loc[ticker, m]
            marker = " *" if not np.isnan(val) and val > 0 else "  "
            print(f"  {val:>+8.4f}{marker}", end="")
        print()

    print("\n  (* = positive R²)")

    # Gorsel
    fig, axes = plt.subplots(
        len(results), 1,
        figsize=(14, 3 * len(results)),
        sharex=True
    )
    if len(results) == 1:
        axes = [axes]

    colors_pos = "#4C72B0"
    colors_neg = "#DD8452"

    for ax, (model_name, firm_r2) in zip(axes, results.items()):
        r2_arr = np.array(firm_r2)
        colors = [colors_pos if v > 0 else colors_neg for v in r2_arr]
        bars = ax.bar(tickers[:len(r2_arr)], r2_arr,
                      color=colors, alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
        ax.axhline(np.nanmean(r2_arr), color="gray", linewidth=1,
                   linestyle="--", label=f"Mean: {np.nanmean(r2_arr):+.4f}")
        ax.set_title(model_name, fontsize=11, fontweight="bold")
        ax.set_ylabel("R²")
        ax.legend(fontsize=8, loc="upper right")
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.grid(axis="y", alpha=0.3)

        # Pozitif firma sayisi
        pos = sum(1 for v in r2_arr if v > 0)
        ax.text(0.01, 0.95, f"Positive: {pos}/{len(r2_arr)}",
                transform=ax.transAxes, fontsize=8,
                verticalalignment="top", color=colors_pos)

    from matplotlib.patches import Patch
    fig.legend(handles=[
        Patch(color=colors_pos, label="R² > 0"),
        Patch(color=colors_neg, label="R² ≤ 0"),
    ], loc="upper center", ncol=2, fontsize=9,
       bbox_to_anchor=(0.5, 1.01))

    plt.suptitle("Firm-by-Firm Out-of-Sample R² (DELTA mode)",
                 fontsize=13, y=1.03)
    plt.tight_layout()
    out_fig = FIG_DIR / "firm_r2.png"
    fig.savefig(out_fig, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  [OK] {out_fig.name}")

    # SC-STGCN icin en iyi firmalar
    if "SC-STGCN" in results:
        sc_r2 = np.array(results["SC-STGCN"])
        top5_idx = np.argsort(sc_r2)[::-1][:5]
        print("\n  SC-STGCN — En yuksek R²'li 5 firma:")
        for idx in top5_idx:
            print(f"    {tickers[idx]:<8} R²={sc_r2[idx]:+.4f}")

    print("\n[DONE] firm_r2.py tamamlandi.")


if __name__ == "__main__":
    main()
