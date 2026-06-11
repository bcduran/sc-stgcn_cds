"""
src/analysis/cds_analysis.py
CDS delta serisinin istatistiksel ozelliklerini analiz eder.
--universe top50  -> data/top50/ve1.csv
--universe full   -> data/full/ve1_full.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import kurtosis, skew

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import TOP50_WEEKLY_CSV, VE1_FULL_CSV, FIG_DIR, make_dirs

try:
    from statsmodels.tsa.stattools import acf, adfuller
    from statsmodels.stats.stattools import durbin_watson
    HAS_SM = True
except ImportError:
    HAS_SM = False
    print("[WARN] statsmodels yuklu degil, bazi testler atlanacak.")


def safe_acf(s, nlags=5):
    if not HAS_SM or s.std() < 1e-10:
        return np.zeros(nlags)
    try:
        result = acf(s, nlags=nlags, fft=True)
        result = result[1:nlags+1]
        return np.where(np.isnan(result), 0.0, result)
    except Exception:
        return np.zeros(nlags)


def safe_adf(s):
    if not HAS_SM or s.std() < 1e-10:
        return None
    try:
        return adfuller(s, maxlag=5)[1]
    except Exception:
        return None


def safe_dw(s):
    if not HAS_SM or s.std() < 1e-10:
        return None
    try:
        return durbin_watson(s)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", choices=["top50","full"], default="top50")
    args = parser.parse_args()

    make_dirs()

    if args.universe == "full":
        csv_path = VE1_FULL_CSV
        label    = "Full Universe"
    else:
        csv_path = TOP50_WEEKLY_CSV
        label    = "Top-50"

    print(f"  Universe: {label}")
    print(f"  Kaynak  : {csv_path}")

    df    = pd.read_csv(csv_path)
    level = df.values.astype(float)
    delta = np.diff(level, axis=0)
    N     = delta.shape[1]
    T     = delta.shape[0]

    print("=" * 60)
    print("  CDS DELTA SERISI ANALIZI")
    print("=" * 60)

    # Temiz veri (NaN ve extreme outlier'lar cikar)
    flat = delta.flatten()
    flat_clean = flat[~np.isnan(flat)]

    # 1. Temel istatistikler
    print(f"\n[1] TEMEL ISTATISTIKLER ({N} firma):")
    print(f"  Ortalama delta  : {np.nanmean(delta):.4f} bp")
    print(f"  Medyan delta    : {np.nanmedian(delta):.4f} bp")
    print(f"  Std             : {np.nanstd(delta):.4f} bp")
    print(f"  Skewness        : {skew(flat_clean):.4f}")
    print(f"  Kurtosis (ex.)  : {kurtosis(flat_clean):.4f}")
    print(f"  Min             : {np.nanmin(delta):.2f} bp")
    print(f"  Max             : {np.nanmax(delta):.2f} bp")
    # COVID spike olculeri
    p99 = np.nanpercentile(np.abs(flat_clean), 99)
    p95 = np.nanpercentile(np.abs(flat_clean), 95)
    print(f"  |delta| p95     : {p95:.2f} bp")
    print(f"  |delta| p99     : {p99:.2f} bp")

    # 2. Otokorelasyon
    print(f"\n[2] OTOKORELASYON (ilk 5 lag, ortalama):")
    all_acf = []
    for i in range(N):
        s = delta[:, i]
        s = s[~np.isnan(s)]
        if len(s) > 20 and s.std() > 1e-10:
            ac = safe_acf(s, nlags=5)
            all_acf.append(ac)
    if all_acf:
        mean_acf = np.nanmean(all_acf, axis=0)
        for lag, val in enumerate(mean_acf, 1):
            n_bar = int(abs(val) * 30)
            sign = "+" if val >= 0 else "-"
            bar = sign + ("*" * n_bar)
            print(f"  Lag {lag}: {val:+.4f}  {bar}")
    else:
        mean_acf = np.zeros(5)
        print("  ACF hesaplanamadi.")

    # 3. Durbin-Watson
    print(f"\n[3] DURBIN-WATSON (otokorelasyon testi):")
    dw_vals = []
    for i in range(N):
        s = delta[:, i]; s = s[~np.isnan(s)]
        if len(s) > 10:
            dw = safe_dw(s)
            if dw is not None:
                dw_vals.append(dw)
    if dw_vals:
        print(f"  Ortalama DW : {np.mean(dw_vals):.4f}")
        print(f"  (2.0=bagimsiz, <2=pozitif oto., >2=negatif oto.)")
    else:
        print("  DW hesaplanamadi (statsmodels gerekli).")

    # 4. ADF testi
    print(f"\n[4] ADF TESTI (duraganlik):")
    adf_pvals = []
    for i in range(N):
        s = delta[:, i]; s = s[~np.isnan(s)]
        if len(s) > 20:
            p = safe_adf(s)
            if p is not None:
                adf_pvals.append(p)
    if adf_pvals:
        pvals = np.array(adf_pvals)
        print(f"  %1  anlamlilikta duraganlik: {(pvals < 0.01).sum()}/{len(pvals)} firma")
        print(f"  %5  anlamlilikta duraganlik: {(pvals < 0.05).sum()}/{len(pvals)} firma")
        print(f"  Ortalama p-degeri          : {pvals.mean():.4f}")
    else:
        print("  ADF hesaplanamadi (statsmodels gerekli).")

    # 5. Neden R² negatif
    print(f"\n[5] NEDEN R2 NEGATIF:")
    te_start = int(0.9 * T)
    te = delta[te_start:]
    grand_mean = np.nanmean(te)
    SS_tot   = np.nansum((te - grand_mean)**2)
    SS_naive = np.nansum(te**2)
    naive_r2 = 1 - SS_naive / SS_tot if SS_tot > 0 else float("nan")
    print(f"  Test seti    : {te.shape[0]} hafta x {N} firma")
    print(f"  Grand mean   : {grand_mean:.4f} bp  (sifira cok yakin)")
    print(f"  SS_tot       : {SS_tot:,.0f}")
    print(f"  SS_naive     : {SS_naive:,.0f}")
    print(f"  Naive R²     : {naive_r2:+.4f}")
    print(f"\n  SONUC: Grand mean ~0 oldugu icin SS_tot ~ SS_res(naive)")
    print(f"  Bu CDS delta serisinin matematiksel ozelligi, model hatasi degil.")

    # 6. Tahmin edilebilirlik siniri
    print(f"\n[6] TAHMIN EDILEBILIRLIK SINIRI:")
    ar1_r2s = []
    for i in range(N):
        s = delta[:, i]; s = s[~np.isnan(s)]
        if len(s) > 20 and s.std() > 1e-10:
            try:
                cc = np.corrcoef(s[:-1], s[1:])
                phi = cc[0,1] if not np.isnan(cc[0,1]) else 0.0
                ar1_r2s.append(phi**2)
            except Exception:
                pass
    if ar1_r2s:
        print(f"  Teorik AR(1) R2 ust siniri (phi2): {np.mean(ar1_r2s):.4f}")
        print(f"  Lag-1 ACF (phi)                  : {mean_acf[0]:+.4f}")
        print(f"  => En iyi modelin teorik R2 <= {np.mean(ar1_r2s):.4f}")

    # 7. Gorsel
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ACF barplot
    ax = axes[0, 0]
    lags = range(1, len(mean_acf)+1)
    colors = ["#4C72B0" if v >= 0 else "#DD8452" for v in mean_acf]
    ax.bar(lags, mean_acf, color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ci = 1.96 / np.sqrt(T)
    ax.axhline(ci,  color="red", linestyle="--", linewidth=1, label=f"95% CI (±{ci:.3f})")
    ax.axhline(-ci, color="red", linestyle="--", linewidth=1)
    ax.set_title("Ortalama Otokorelasyon (ACF)", fontsize=11)
    ax.set_xlabel("Lag (hafta)"); ax.set_ylabel("ACF")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Delta dagilimi (kırpılmış)
    ax = axes[0, 1]
    clip_val = np.nanpercentile(np.abs(flat_clean), 99)
    flat_clip = np.clip(flat_clean, -clip_val, clip_val)
    ax.hist(flat_clip, bins=100, color="#4C72B0", alpha=0.75,
            edgecolor="white", linewidth=0.2)
    ax.axvline(0, color="red", linewidth=1.5, linestyle="--", label="Δ=0")
    x = np.linspace(-clip_val, clip_val, 200)
    scale = len(flat_clip) * (2*clip_val/100)
    ax.plot(x, stats.norm.pdf(x, np.nanmean(flat_clean),
            np.nanstd(flat_clean)) * scale,
            color="orange", linewidth=2, label="Normal")
    ax.set_title(f"Delta Dagilimi (p99={clip_val:.1f} bp kirpilmis)", fontsize=11)
    ax.set_xlabel("Δspread (bp)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Firma bazinda phi (AR1)
    ax = axes[1, 0]
    phis = []
    for i in range(N):
        s = delta[:, i]; s = s[~np.isnan(s)]
        if len(s) > 10 and s.std() > 1e-10:
            try:
                cc = np.corrcoef(s[:-1], s[1:])
                phi = cc[0,1] if not np.isnan(cc[0,1]) else 0.0
                phis.append(phi)
            except Exception:
                phis.append(0.0)
    phis_sorted = sorted(phis)
    colors_phi = ["#4C72B0" if p > 0 else "#DD8452" for p in phis_sorted]
    ax.bar(range(len(phis_sorted)), phis_sorted, color=colors_phi, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(np.mean(phis), color="orange", linewidth=1.5,
               linestyle="--", label=f"Ort φ={np.mean(phis):.3f}")
    ax.set_title("Firma Bazinda AR(1) Katsayisi φ₁", fontsize=11)
    ax.set_xlabel(f"Firma (sirali, N={len(phis_sorted)})")
    ax.set_ylabel("φ₁"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # SS_tot vs SS_naive
    ax = axes[1, 1]
    test_weeks = range(1, te.shape[0]+1)
    ss_tot_cum   = np.nancumsum(((te - np.nanmean(te))**2).sum(axis=1))
    ss_naive_cum = np.nancumsum((te**2).sum(axis=1))
    ax.plot(test_weeks, ss_tot_cum,   label="SS_tot (gercek varyans)",
            color="#4C72B0", linewidth=2)
    ax.plot(test_weeks, ss_naive_cum, label="SS_res (naive Δ=0)",
            color="#DD8452", linewidth=2, linestyle="--")
    ax.set_title("SS_tot ≈ SS_naive: Neden R² Negatif?", fontsize=11)
    ax.set_xlabel("Test haftasi"); ax.set_ylabel("Kumulatif SS")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.suptitle(f"CDS Delta Serisi Analizi — {label}", fontsize=13)
    plt.tight_layout()
    out = FIG_DIR / f"cds_analysis_{args.universe}.png"
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"\n  [OK] {out}")
    print(f"\n[DONE] cds_analysis.py ({label}) tamamlandi.")


if __name__ == "__main__":
    main()
