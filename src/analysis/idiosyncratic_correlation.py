"""
src/analysis/idiosyncratic_correlation.py
==========================================
Katkı 1A: Makro Faktörden Arındırılmış (Idiosyncratic) Korelasyon Analizi

Motivasyon:
  Ham CDS delta korelasyonunda supply chain sinyali, ortak piyasa
  faktörü (market-wide credit risk) tarafından bastırılıyor.
  Tüm firmalar aynı makro ortamda işlem gördüğü için bağlantılı
  vs bağlantısız çiftler benzer ham korelasyon gösteriyor.

Yöntem:
  1. Her firma için delta serisini cross-sectional ortalamaya
     (market factor) regress et
  2. Artık (residual) seriyi al — "idiosyncratic spread change"
  3. Bu artıklar üzerinden çift korelasyonu hesapla
  4. Artık korelasyonda supply chain sinyali var mı?

Beklenti:
  Makro faktör çıkarıldıktan sonra supply chain bağlantılı çiftlerin
  idiosyncratic korelasyonu bağlantısız çiftlerden anlamlı şekilde
  yüksek olmalı — eğer supply chain gerçekten kredi riskini taşıyorsa.

Çıktılar:
  outputs/metrics/idio_corr_by_group.csv
  outputs/metrics/idio_corr_by_period.csv
  outputs/figures/fig_idio_corr_groups.png
  outputs/figures/fig_idio_corr_timeseries.png

Çalıştırmak:
  python src/analysis/idiosyncratic_correlation.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import load_npz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from itertools import combinations

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import (
    TOP50_WEEKLY_CSV,
    ADJ_SUP_NPZ, ADJ_CUS_NPZ, ADJ_NPZ,
    METRICS_DIR, FIG_DIR, make_dirs,
)

# Dönemler
PRE_COVID  = ("2015-01-01", "2020-02-28")
CRISIS     = ("2020-03-01", "2020-08-31")
POST_COVID = ("2020-09-01", "2021-09-10")
COVID_START = "2020-03-01"
COVID_END   = "2020-09-01"
ROLL_WINDOW = 52


def load_data():
    df = pd.read_csv(TOP50_WEEKLY_CSV)
    n_weeks = len(df)
    dates = pd.date_range(start="2015-01-02", periods=n_weeks, freq="W-FRI")
    df.index = dates
    level = df.values.astype(np.float32)
    delta = pd.DataFrame(
        np.diff(level, axis=0),
        index=dates[1:],
        columns=df.columns,
    )
    tickers = df.columns.tolist()
    N = len(tickers)
    A_sup  = load_npz(ADJ_SUP_NPZ).toarray()
    A_cus  = load_npz(ADJ_CUS_NPZ).toarray()
    A_comb = load_npz(ADJ_NPZ).toarray()
    return delta, tickers, N, A_sup, A_cus, A_comb


def remove_market_factor(delta: pd.DataFrame,
                          method: str = "crosssectional_mean") -> pd.DataFrame:
    """
    Her haftaki ortak piyasa faktörünü çıkar.

    Yöntemler:
      'crosssectional_mean':
          Her hafta cross-sectional ortalamayı çıkar.
          Δs_i(t)_idio = Δs_i(t) - mean_j[Δs_j(t)]

      'ols_regression':
          Her firma için delta = alpha + beta * market_factor + epsilon
          OLS ile beta'yı tahmin et, artıkları al.
          Daha doğru ama aynı sonucu verebilir.

      'pca_residual':
          İlk 1-3 principal component'i çıkar.
          En kapsamlı yöntem.
    """
    arr = delta.values.copy()
    T, N = arr.shape

    if method == "crosssectional_mean":
        # Her hafta cross-sectional ortalama = piyasa faktörü
        market = np.nanmean(arr, axis=1, keepdims=True)  # (T, 1)
        resid  = arr - market

    elif method == "ols_regression":
        market = np.nanmean(arr, axis=1)  # (T,)
        resid  = np.zeros_like(arr)
        for i in range(N):
            y = arr[:, i]
            mask = ~np.isnan(y)
            if mask.sum() < 20:
                resid[:, i] = y
                continue
            X = np.column_stack([np.ones(mask.sum()), market[mask]])
            beta = np.linalg.lstsq(X, y[mask], rcond=None)[0]
            fitted = beta[0] + beta[1] * market
            resid[:, i] = y - fitted

    elif method == "pca_residual":
        from sklearn.decomposition import PCA
        # NaN'ları geçici olarak 0 ile doldur
        arr_filled = np.where(np.isnan(arr), 0, arr)
        pca = PCA(n_components=3)
        scores = pca.fit_transform(arr_filled)      # (T, 3)
        recon  = pca.inverse_transform(scores)       # (T, N) — common part
        resid  = arr - recon
    else:
        raise ValueError(f"Unknown method: {method}")

    return pd.DataFrame(resid, index=delta.index, columns=delta.columns)


def classify_pair(i, j, A_sup, A_cus):
    i_has_j_as_sup = bool(A_sup[i, j])
    j_has_i_as_sup = bool(A_sup[j, i])
    if i_has_j_as_sup and j_has_i_as_sup:
        return "bidirectional"
    elif i_has_j_as_sup:
        return "upstream"
    elif j_has_i_as_sup:
        return "downstream"
    else:
        return "unconnected"


def pairwise_corr(df: pd.DataFrame, tickers: list,
                  A_sup, A_cus, period: tuple = None) -> pd.DataFrame:
    if period:
        sub = df.loc[period[0]:period[1]]
    else:
        sub = df
    N = len(tickers)
    records = []
    for i, j in combinations(range(N), 2):
        xi = sub.iloc[:, i].values
        xj = sub.iloc[:, j].values
        mask = ~(np.isnan(xi) | np.isnan(xj))
        if mask.sum() < 10:
            continue
        if xi[mask].std() < 1e-10 or xj[mask].std() < 1e-10:
            continue
        try:
            r, p = stats.pearsonr(xi[mask], xj[mask])
            if np.isnan(r):
                continue
        except Exception:
            continue
        rel = classify_pair(i, j, A_sup, A_cus)
        records.append({
            "ticker_i": tickers[i], "ticker_j": tickers[j],
            "corr": float(r), "pvalue": float(p),
            "rel_type": rel,
        })
    return pd.DataFrame(records)


def rolling_group_corr(df: pd.DataFrame, tickers: list,
                       A_sup, A_cus, window: int = ROLL_WINDOW):
    N = len(tickers)
    pairs = {g: [] for g in
             ["upstream", "downstream", "bidirectional", "unconnected"]}
    for i, j in combinations(range(N), 2):
        g = classify_pair(i, j, A_sup, A_cus)
        pairs[g].append((i, j))

    arr = df.values
    records = []
    for t in range(window, len(arr)):
        wd = arr[t - window:t, :]
        row = {"date": df.index[t]}
        for grp, pair_list in pairs.items():
            corrs = []
            for i, j in pair_list:
                xi = wd[:, i]; xj = wd[:, j]
                mask = ~(np.isnan(xi) | np.isnan(xj))
                if mask.sum() < 10:
                    continue
                if xi[mask].std() < 1e-10 or xj[mask].std() < 1e-10:
                    continue
                try:
                    r, _ = stats.pearsonr(xi[mask], xj[mask])
                    if not np.isnan(r):
                        corrs.append(r)
                except Exception:
                    pass
            row[f"corr_{grp}"] = float(np.mean(corrs)) if corrs else np.nan
        records.append(row)
    return pd.DataFrame(records).set_index("date")


def print_group_summary(df_pairs: pd.DataFrame, label: str = ""):
    print(f"\n  {'Grup':<15} {'N çift':>7} {'Ort. Corr':>10} "
          f"{'Std':>8} {'Pozitif%':>10}")
    print(f"  {'-'*55}")
    for grp in ["upstream", "downstream", "bidirectional", "unconnected"]:
        sub = df_pairs[df_pairs["rel_type"] == grp]["corr"]
        if len(sub) == 0:
            continue
        pos = (sub > 0).mean() * 100
        print(f"  {grp:<15} {len(sub):>7} {sub.mean():>+10.4f} "
              f"{sub.std():>8.4f} {pos:>9.1f}%")

    # T-testler
    connected   = df_pairs[df_pairs["rel_type"] != "unconnected"]["corr"]
    unconnected = df_pairs[df_pairs["rel_type"] == "unconnected"]["corr"]
    t, p = stats.ttest_ind(connected, unconnected, equal_var=False)
    sig = "*** p<0.01" if p < 0.01 else "** p<0.05" if p < 0.05 else \
          "* p<0.10" if p < 0.10 else "ns"
    print(f"\n  Bağlantılı ({connected.mean():+.4f}) vs "
          f"Bağlantısız ({unconnected.mean():+.4f}): "
          f"t={t:.3f}, p={p:.4f} [{sig}]")

    for grp in ["upstream", "downstream"]:
        sub = df_pairs[df_pairs["rel_type"] == grp]["corr"]
        if len(sub) == 0:
            continue
        t2, p2 = stats.ttest_ind(sub, unconnected, equal_var=False)
        sig2 = "*** p<0.01" if p2 < 0.01 else "** p<0.05" if p2 < 0.05 else \
               "* p<0.10" if p2 < 0.10 else "ns"
        print(f"  {grp.capitalize()} ({sub.mean():+.4f}) vs "
              f"Bağlantısız ({unconnected.mean():+.4f}): "
              f"t={t2:.3f}, p={p2:.4f} [{sig2}]")


def main():
    make_dirs()

    print("=" * 65)
    print("  Idiosyncratic CDS Korelasyon Analizi")
    print("  (Makro Faktörden Arındırılmış)")
    print("=" * 65)

    # 1) Veri
    print("\n[1/5] Veri yukleniyor...")
    delta, tickers, N, A_sup, A_cus, A_comb = load_data()
    print(f"  Delta panel: {delta.shape[0]} hafta x {N} firma")

    # 2) Ham korelasyon (referans)
    print("\n[2/5] Ham korelasyon (referans - makro çıkarılmamış):")
    df_raw = pairwise_corr(delta, tickers, A_sup, A_cus)
    print_group_summary(df_raw, "Ham")

    # 3) Üç yöntemle makro faktör çıkar
    methods = {
        "Cross-sectional Mean" : "crosssectional_mean",
        "OLS Regression"       : "ols_regression",
        "PCA (3 component)"    : "pca_residual",
    }

    results_by_method = {}

    for method_name, method_key in methods.items():
        print(f"\n[3/5] {method_name} yöntemi ile idiosyncratic korelasyon:")
        resid = remove_market_factor(delta, method=method_key)

        # Artık serinin varyansı ne kadar kaldı?
        var_orig  = np.nanvar(delta.values)
        var_resid = np.nanvar(resid.values)
        r2_market = 1 - var_resid / var_orig
        print(f"  Piyasa faktörü açıklama gücü (R²): {r2_market:.4f}")
        print(f"  Artık varyans oranı: {var_resid/var_orig:.4f}")

        df_idio = pairwise_corr(resid, tickers, A_sup, A_cus)
        print_group_summary(df_idio, method_name)
        results_by_method[method_name] = df_idio

    # En iyi yöntem: OLS regression (en doğru faktör çıkarma)
    best_method = "OLS Regression"
    df_best = results_by_method[best_method]
    resid_best = remove_market_factor(delta, "ols_regression")

    # 4) Dönem bazında idio korelasyon
    print(f"\n[4/5] Dönem bazında idiosyncratic korelasyon ({best_method}):")
    periods = {
        "Pre-COVID"   : PRE_COVID,
        "COVID Crisis": CRISIS,
        "Post-COVID"  : POST_COVID,
    }
    period_records = []
    for period_name, period in periods.items():
        df_p = pairwise_corr(resid_best, tickers, A_sup, A_cus, period)
        for grp in ["upstream", "downstream", "bidirectional", "unconnected"]:
            sub = df_p[df_p["rel_type"] == grp]["corr"]
            if len(sub) > 0:
                t_vs_unc, p_vs_unc = stats.ttest_ind(
                    sub,
                    df_p[df_p["rel_type"] == "unconnected"]["corr"],
                    equal_var=False
                )
                period_records.append({
                    "period"   : period_name,
                    "group"    : grp,
                    "mean_corr": sub.mean(),
                    "std_corr" : sub.std(),
                    "n_pairs"  : len(sub),
                    "t_stat"   : t_vs_unc,
                    "p_value"  : p_vs_unc,
                })

    df_periods = pd.DataFrame(period_records)

    print(f"\n  {'Dönem':<15} {'Grup':<15} {'Ort. Corr':>10} "
          f"{'p-value':>9} {'Sig':>8}")
    print(f"  {'-'*60}")
    for _, row in df_periods.iterrows():
        sig = "***" if row["p_value"] < 0.01 else \
              "**"  if row["p_value"] < 0.05 else \
              "*"   if row["p_value"] < 0.10 else "ns"
        print(f"  {row['period']:<15} {row['group']:<15} "
              f"{row['mean_corr']:>+10.4f} {row['p_value']:>9.4f} {sig:>8}")

    # 5) Görseller
    print("\n[5/5] Goerseller olusturuluyor...")

    grp_colors = {
        "upstream"     : "#2196F3",
        "downstream"   : "#FF9800",
        "bidirectional": "#9C27B0",
        "unconnected"  : "#9E9E9E",
    }

    # --- Fig 1: Ham vs Idio karşılaştırması (3 yöntem) ---
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    datasets = {
        "Raw\n(Ham)"           : df_raw,
        "Idio:\nCS Mean"       : results_by_method["Cross-sectional Mean"],
        "Idio:\nOLS Regression": results_by_method["OLS Regression"],
        "Idio:\nPCA (3 comp)"  : results_by_method["PCA (3 component)"],
    }

    for ax, (title, df_p) in zip(axes, datasets.items()):
        grp_order = ["upstream", "downstream", "bidirectional", "unconnected"]
        grp_data  = []
        grp_names = []
        for g in grp_order:
            sub = df_p[df_p["rel_type"] == g]["corr"].values
            if len(sub) > 0:
                grp_data.append(sub)
                grp_names.append(g)

        if grp_data:
            vp = ax.violinplot(grp_data, positions=range(len(grp_names)),
                               showmedians=True, showmeans=False)
            for i, (pc, g) in enumerate(zip(vp["bodies"], grp_names)):
                pc.set_facecolor(grp_colors[g])
                pc.set_alpha(0.7)
            vp["cmedians"].set_color("black")
            vp["cmedians"].set_linewidth(2)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(len(grp_names)))
        ax.set_xticklabels([g[:3].upper() for g in grp_names], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Correlation" if ax == axes[0] else "")
        ax.grid(axis="y", alpha=0.3)

        # Ortalamalar
        for i, (d, g) in enumerate(zip(grp_data, grp_names)):
            ax.text(i, np.mean(d) + 0.02, f"{np.mean(d):+.3f}",
                    ha="center", fontsize=8, color=grp_colors[g],
                    fontweight="bold")

        # Bağlantılı vs bağlantısız
        con = df_p[df_p["rel_type"] != "unconnected"]["corr"]
        unc = df_p[df_p["rel_type"] == "unconnected"]["corr"]
        if len(con) > 0 and len(unc) > 0:
            t, p = stats.ttest_ind(con, unc, equal_var=False)
            sig = "***" if p < 0.01 else "**" if p < 0.05 else \
                  "*" if p < 0.10 else "ns"
            ax.set_xlabel(f"t={t:.2f}, p={p:.3f} [{sig}]", fontsize=8)

    plt.suptitle("Ham vs Idiosyncratic CDS Korelasyonu\n"
                 "(Makro faktör çıkarıldıktan sonra supply chain sinyali)",
                 fontsize=12)
    plt.tight_layout()
    out1 = FIG_DIR / "fig_idio_corr_methods.png"
    fig.savefig(out1, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  [OK] {out1.name}")

    # --- Fig 2: Dönem bazında idio korelasyon ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    period_names = ["Pre-COVID", "COVID Crisis", "Post-COVID"]
    grp_show = ["upstream", "downstream", "unconnected"]

    for ax, period_name in zip(axes, period_names):
        period_tup = periods[period_name]
        df_p = pairwise_corr(resid_best, tickers, A_sup, A_cus, period_tup)

        grp_data  = []
        grp_names = []
        for g in grp_show:
            sub = df_p[df_p["rel_type"] == g]["corr"].values
            if len(sub) > 0:
                grp_data.append(sub)
                grp_names.append(g)

        if grp_data:
            vp = ax.violinplot(grp_data, positions=range(len(grp_names)),
                               showmedians=True)
            for i, (pc, g) in enumerate(zip(vp["bodies"], grp_names)):
                pc.set_facecolor(grp_colors[g])
                pc.set_alpha(0.7)
            vp["cmedians"].set_color("black")
            vp["cmedians"].set_linewidth(2)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(len(grp_names)))
        ax.set_xticklabels([g[:3].upper() for g in grp_names], fontsize=9)
        ax.set_title(period_name, fontsize=11)
        ax.set_ylabel("Idio Correlation" if ax == axes[0] else "")
        ax.grid(axis="y", alpha=0.3)

        for i, (d, g) in enumerate(zip(grp_data, grp_names)):
            ax.text(i, np.mean(d) + 0.005, f"{np.mean(d):+.3f}",
                    ha="center", fontsize=9, color=grp_colors[g],
                    fontweight="bold")

    plt.suptitle("Idiosyncratic CDS Korelasyonu: COVID Dönem Analizi\n"
                 f"(OLS market factor removed)",
                 fontsize=12)
    plt.tight_layout()
    out2 = FIG_DIR / "fig_idio_corr_periods.png"
    fig.savefig(out2, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  [OK] {out2.name}")

    # --- Fig 3: Rolling idio korelasyon ---
    print("  Rolling korelasyon hesaplaniyor...")
    df_roll = rolling_group_corr(resid_best, tickers, A_sup, A_cus, ROLL_WINDOW)

    fig, ax = plt.subplots(figsize=(14, 5))
    col_map = {
        "corr_upstream"     : ("#2196F3", "Upstream",      2.0),
        "corr_downstream"   : ("#FF9800", "Downstream",    2.0),
        "corr_bidirectional": ("#9C27B0", "Bidirectional", 1.5),
        "corr_unconnected"  : ("#9E9E9E", "Unconnected",   1.5),
    }
    for col, (color, label, lw) in col_map.items():
        if col in df_roll.columns:
            ax.plot(df_roll.index, df_roll[col],
                    color=color, label=label, linewidth=lw, alpha=0.9)

    ax.axvspan(pd.Timestamp(COVID_START), pd.Timestamp(COVID_END),
               alpha=0.12, color="red", label="COVID Crisis")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Mean Idiosyncratic Correlation (rolling {ROLL_WINDOW}w)")
    ax.set_title("Rolling Idiosyncratic CDS Correlations\n"
                 "(After OLS Market Factor Removal)", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out3 = FIG_DIR / "fig_idio_corr_timeseries.png"
    fig.savefig(out3, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  [OK] {out3.name}")

    # Kaydet
    df_best.to_csv(METRICS_DIR / "idio_corr_pairwise.csv", index=False)
    df_periods.to_csv(METRICS_DIR / "idio_corr_by_period.csv", index=False)
    df_roll.to_csv(METRICS_DIR / "idio_corr_timeseries.csv")

    # Özet
    print(f"\n{'='*65}")
    print(f"  ÖZET BULGULAR (OLS Idiosyncratic Korelasyon)")
    print(f"{'='*65}")
    con = df_best[df_best["rel_type"] != "unconnected"]["corr"]
    unc = df_best[df_best["rel_type"] == "unconnected"]["corr"]
    t, p = stats.ttest_ind(con, unc, equal_var=False)
    sig = "*** p<0.01" if p < 0.01 else "** p<0.05" if p < 0.05 else \
          "* p<0.10" if p < 0.10 else "ANLAMLI DEĞİL"
    print(f"\n  Bağlantılı idio korelasyon  : {con.mean():+.4f}")
    print(f"  Bağlantısız idio korelasyon : {unc.mean():+.4f}")
    print(f"  Fark                        : {con.mean()-unc.mean():+.4f}")
    print(f"  t={t:.3f}, p={p:.4f} [{sig}]")

    for grp in ["upstream", "downstream"]:
        sub = df_best[df_best["rel_type"] == grp]["corr"]
        if len(sub) == 0:
            continue
        t2, p2 = stats.ttest_ind(sub, unc, equal_var=False)
        sig2 = "*** p<0.01" if p2 < 0.01 else "** p<0.05" if p2 < 0.05 else \
               "* p<0.10" if p2 < 0.10 else "ANLAMLI DEĞİL"
        print(f"\n  {grp.capitalize()} idio korelasyon   : {sub.mean():+.4f}")
        print(f"  Bağlantısız idio korelasyon : {unc.mean():+.4f}")
        print(f"  Fark                        : {sub.mean()-unc.mean():+.4f}")
        print(f"  t={t2:.3f}, p={p2:.4f} [{sig2}]")

    print(f"\n  Çıktılar:")
    print(f"    {METRICS_DIR / 'idio_corr_pairwise.csv'}")
    print(f"    {METRICS_DIR / 'idio_corr_by_period.csv'}")
    print(f"    {FIG_DIR / 'fig_idio_corr_methods.png'}")
    print(f"    {FIG_DIR / 'fig_idio_corr_periods.png'}")
    print(f"    {FIG_DIR / 'fig_idio_corr_timeseries.png'}")
    print(f"\n[DONE] idiosyncratic_correlation.py tamamlandi.")


if __name__ == "__main__":
    main()
