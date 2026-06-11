"""
src/analysis/cross_firm_correlation.py
========================================
Katkı 1: Cross-firm CDS Korelasyon Analizi

Araştırma soruları:
  1. Supply chain bağlantılı çiftler daha yüksek CDS delta korelasyonu
     gösteriyor mu?
  2. Upstream vs downstream asimetri var mı?
  3. Kriz döneminde (COVID) bu etki güçleniyor mu?
  4. Graph mesafesi (1-hop vs 2-hop) ile korelasyon ilişkisi?

Çıktılar:
  outputs/metrics/correlation_by_group.csv
  outputs/metrics/correlation_timeseries.csv
  outputs/metrics/correlation_pairwise.csv
  outputs/figures/fig_corr_groups.png
  outputs/figures/fig_corr_timeseries.png
  outputs/figures/fig_corr_heatmap.png

Çalıştırmak:
  python src/analysis/cross_firm_correlation.py
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
import seaborn as sns
from itertools import combinations

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import (
    TOP50_WEEKLY_CSV,
    ADJ_SUP_NPZ, ADJ_CUS_NPZ, ADJ_NPZ,
    METRICS_DIR, FIG_DIR, make_dirs,
)

# Dönem tanımları
COVID_START = "2020-03-01"
COVID_END   = "2020-09-01"
PRE_COVID   = ("2015-01-01", "2020-02-28")
CRISIS      = ("2020-03-01", "2020-08-31")
POST_COVID  = ("2020-09-01", "2021-09-10")

# Rolling korelasyon penceresi
ROLL_WINDOW = 52  # hafta


def load_data():
    """CDS delta paneli ve adjacency matrislerini yükle."""
    df = pd.read_csv(TOP50_WEEKLY_CSV)
    # Tarih index oluştur (yaklaşık — haftalık)
    n_weeks = len(df)
    dates = pd.date_range(start="2015-01-02", periods=n_weeks, freq="W-FRI")
    df.index = dates

    level  = df.values.astype(np.float32)
    delta  = pd.DataFrame(
        np.diff(level, axis=0),
        index   = dates[1:],
        columns = df.columns,
    )
    tickers = df.columns.tolist()
    N = len(tickers)

    A_sup  = load_npz(ADJ_SUP_NPZ).toarray()   # A[i,j]=1: j is supplier of i
    A_cus  = load_npz(ADJ_CUS_NPZ).toarray()   # A[i,j]=1: j is customer of i
    A_comb = load_npz(ADJ_NPZ).toarray()

    return delta, tickers, N, A_sup, A_cus, A_comb


def classify_pair(i, j, A_sup, A_cus):
    """
    Bir çifti (i,j) ilişki tipine göre sınıflandır.
    A_sup[i,j]=1: j is supplier of i  (upstream for i)
    A_sup[j,i]=1: i is supplier of j  (downstream for i)
    Döndürür: 'upstream', 'downstream', 'bidirectional', 'unconnected'
    """
    # i için upstream: j is supplier of i  OR  i is supplier of j
    i_has_j_as_sup = bool(A_sup[i, j])  # j supplies i
    j_has_i_as_sup = bool(A_sup[j, i])  # i supplies j
    
    if i_has_j_as_sup and j_has_i_as_sup:
        return "bidirectional"
    elif i_has_j_as_sup:
        return "upstream"    # j is upstream supplier of i
    elif j_has_i_as_sup:
        return "downstream"  # i is upstream supplier of j (i.e. j is customer)
    else:
        return "unconnected"


def pairwise_correlations(delta: pd.DataFrame, tickers: list,
                          A_sup, A_cus, period: tuple = None):
    """
    Tüm firm-pair'lar için Pearson korelasyonu hesapla.
    period: (start_date, end_date) string tuple veya None (tüm dönem)
    """
    if period:
        sub = delta.loc[period[0]:period[1]]
    else:
        sub = delta

    records = []
    N = len(tickers)
    for i, j in combinations(range(N), 2):
        xi = sub.iloc[:, i].values
        xj = sub.iloc[:, j].values
        # Geçerli gözlemleri filtrele
        mask = ~(np.isnan(xi) | np.isnan(xj))
        if mask.sum() < 10:
            continue
        r, p = stats.pearsonr(xi[mask], xj[mask])
        rel_type = classify_pair(i, j, A_sup, A_cus)
        records.append({
            "ticker_i"  : tickers[i],
            "ticker_j"  : tickers[j],
            "corr"      : float(r),
            "pvalue"    : float(p),
            "n_obs"     : int(mask.sum()),
            "rel_type"  : rel_type,
        })
    return pd.DataFrame(records)


def rolling_group_corr(delta: pd.DataFrame, tickers: list,
                       A_sup, A_cus, window: int = ROLL_WINDOW):
    """
    Her hafta için grup bazında ortalama çift korelasyonunu hesapla.
    """
    N = len(tickers)
    dates = delta.index[window:]

    # Çift listelerini önceden hazırla
    pairs = {g: [] for g in ["upstream","downstream","bidirectional","unconnected"]}
    for i, j in combinations(range(N), 2):
        g = classify_pair(i, j, A_sup, A_cus)
        pairs[g].append((i, j))

    records = []
    arr = delta.values  # (T, N)

    for t in range(window, len(arr)):
        window_data = arr[t - window:t, :]
        row = {"date": delta.index[t]}
        for grp, pair_list in pairs.items():
            corrs = []
            for i, j in pair_list:
                xi = window_data[:, i]
                xj = window_data[:, j]
                mask = ~(np.isnan(xi) | np.isnan(xj))
                if mask.sum() < 10:
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


def graph_distance_analysis(delta: pd.DataFrame, tickers: list, A_comb):
    """
    Graph mesafesi (hop sayısı) ile korelasyon ilişkisi.
    0-hop = same firm (skip)
    1-hop = direct neighbor
    2-hop = neighbor of neighbor
    3+ = distant
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    A_sp   = csr_matrix(A_comb)
    dist   = shortest_path(A_sp, directed=False, unweighted=True)
    N      = len(tickers)
    arr    = delta.values

    records = []
    for i, j in combinations(range(N), 2):
        raw_d = dist[i, j]
        if np.isinf(raw_d) or np.isnan(raw_d):
            hop_label = "unconnected"
        else:
            d = int(raw_d)
            if d == 0:
                continue
            hop_label = str(d) if d <= 3 else "4+"
        xi = arr[:, i]; xj = arr[:, j]
        mask = ~(np.isnan(xi) | np.isnan(xj))
        if mask.sum() < 10:
            continue
        try:
            r, _ = stats.pearsonr(xi[mask], xj[mask])
            if not np.isnan(r):
                records.append({"distance": hop_label, "corr": r})
        except Exception:
            pass

    return pd.DataFrame(records)


def ttest_groups(df_pairs: pd.DataFrame):
    """
    Bağlantılı vs bağlantısız gruplar arası t-test.
    """
    connected   = df_pairs[df_pairs["rel_type"] != "unconnected"]["corr"]
    unconnected = df_pairs[df_pairs["rel_type"] == "unconnected"]["corr"]
    t, p = stats.ttest_ind(connected, unconnected, equal_var=False)
    return t, p, connected.mean(), unconnected.mean()


def main():
    make_dirs()

    print("=" * 65)
    print("  Cross-Firm CDS Korelasyon Analizi")
    print("=" * 65)

    # 1) Veri yükle
    print("\n[1/6] Veri yukleniyor...")
    delta, tickers, N, A_sup, A_cus, A_comb = load_data()
    print(f"  Delta panel: {delta.shape[0]} hafta x {N} firma")
    print(f"  Dönem: {delta.index[0].date()} - {delta.index[-1].date()}")

    # Graph istatistikleri
    n_sup_edges = int((A_sup > 0).sum())
    n_cus_edges = int((A_cus > 0).sum())
    total_pairs = N * (N - 1) // 2
    print(f"\n  Upstream edges  : {n_sup_edges}")
    print(f"  Downstream edges: {n_cus_edges}")
    print(f"  Toplam çift     : {total_pairs}")

    # 2) Tüm dönem pairwise korelasyon
    print("\n[2/6] Pairwise korelasyonlar hesaplaniyor (tam dönem)...")
    df_pairs = pairwise_correlations(delta, tickers, A_sup, A_cus)
    df_pairs.to_csv(METRICS_DIR / "correlation_pairwise.csv", index=False)

    # Grup bazında özet
    print(f"\n  Grup bazında korelasyon özeti:")
    print(f"  {'Grup':<15} {'N çift':>7} {'Ort. Corr':>10} "
          f"{'Std':>8} {'Pozitif%':>10}")
    print(f"  {'-'*55}")

    group_stats = {}
    for grp in ["upstream", "downstream", "bidirectional", "unconnected"]:
        sub = df_pairs[df_pairs["rel_type"] == grp]["corr"]
        if len(sub) == 0:
            continue
        pos_pct = (sub > 0).mean() * 100
        group_stats[grp] = {
            "n_pairs"  : len(sub),
            "mean_corr": sub.mean(),
            "std_corr" : sub.std(),
            "pos_pct"  : pos_pct,
        }
        print(f"  {grp:<15} {len(sub):>7} {sub.mean():>+10.4f} "
              f"{sub.std():>8.4f} {pos_pct:>9.1f}%")

    # T-test
    t, p, mu_con, mu_unc = ttest_groups(df_pairs)
    print(f"\n  Bağlantılı vs Bağlantısız t-test:")
    print(f"    Bağlantılı mean : {mu_con:+.4f}")
    print(f"    Bağlantısız mean: {mu_unc:+.4f}")
    print(f"    t-statistic     : {t:.4f}")
    print(f"    p-value         : {p:.4f}  "
          f"({'*** p<0.01' if p<0.01 else '** p<0.05' if p<0.05 else '* p<0.10' if p<0.10 else 'ns'})")

    # 3) Dönem bazında korelasyon
    print("\n[3/6] Dönem bazında korelasyon (Pre/Crisis/Post COVID)...")
    periods = {
        "Pre-COVID"  : PRE_COVID,
        "COVID Crisis": CRISIS,
        "Post-COVID" : POST_COVID,
    }
    period_results = []
    for period_name, period in periods.items():
        df_p = pairwise_correlations(delta, tickers, A_sup, A_cus, period)
        for grp in ["upstream", "downstream", "unconnected"]:
            sub = df_p[df_p["rel_type"] == grp]["corr"]
            if len(sub) > 0:
                period_results.append({
                    "period"  : period_name,
                    "group"   : grp,
                    "mean_corr": sub.mean(),
                    "n_pairs" : len(sub),
                })

    df_periods = pd.DataFrame(period_results)
    print(f"\n  {'Dönem':<15} {'Grup':<15} {'Ort. Corr':>10} {'N':>6}")
    print(f"  {'-'*50}")
    for _, row in df_periods.iterrows():
        print(f"  {row['period']:<15} {row['group']:<15} "
              f"{row['mean_corr']:>+10.4f} {int(row['n_pairs']):>6}")

    df_periods.to_csv(METRICS_DIR / "correlation_by_period.csv", index=False)

    # 4) Rolling korelasyon zaman serisi
    print("\n[4/6] Rolling korelasyon zaman serisi hesaplaniyor...")
    print(f"  (Window={ROLL_WINDOW} hafta — yaklaşık 1 yıl)")
    df_roll = rolling_group_corr(delta, tickers, A_sup, A_cus, ROLL_WINDOW)
    df_roll.to_csv(METRICS_DIR / "correlation_timeseries.csv")
    print(f"  [OK] {len(df_roll)} hafta hesaplandi.")

    # 5) Graph mesafesi analizi
    print("\n[5/6] Graph mesafesi analizi...")
    df_dist = graph_distance_analysis(delta, tickers, A_comb)
    dist_summary = df_dist.groupby("distance")["corr"].agg(["mean","std","count"])
    print(f"\n  Graph mesafesi vs ortalama korelasyon:")
    print(f"  {'Mesafe':<12} {'Ort. Corr':>10} {'Std':>8} {'N çift':>8}")
    print(f"  {'-'*42}")
    for dist, row in dist_summary.iterrows():
        print(f"  {dist:<12} {row['mean']:>+10.4f} {row['std']:>8.4f} "
              f"{int(row['count']):>8}")

    # 6) Görseller
    print("\n[6/6] Goerseller olusturuluyor...")

    # --- Fig 1: Grup bazında korelasyon dağılımı ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    grp_colors = {
        "upstream"     : "#2196F3",
        "downstream"   : "#FF9800",
        "bidirectional": "#9C27B0",
        "unconnected"  : "#9E9E9E",
    }
    grp_labels = {
        "upstream"     : "Upstream",
        "downstream"   : "Downstream",
        "bidirectional": "Bidirectional",
        "unconnected"  : "Unconnected",
    }

    # Panel A: Violin plot — korelasyon dağılımı
    ax = axes[0]
    grp_order = ["upstream", "downstream", "bidirectional", "unconnected"]
    grp_data  = [df_pairs[df_pairs["rel_type"]==g]["corr"].values
                 for g in grp_order]
    grp_data  = [d for d in grp_data if len(d) > 0]
    grp_names = [g for g in grp_order
                 if len(df_pairs[df_pairs["rel_type"]==g]) > 0]

    vp = ax.violinplot(grp_data, positions=range(len(grp_names)),
                       showmedians=True, showmeans=False)
    for i, (pc, g) in enumerate(zip(vp["bodies"], grp_names)):
        pc.set_facecolor(grp_colors[g])
        pc.set_alpha(0.7)
    vp["cmedians"].set_color("black")
    vp["cmedians"].set_linewidth(2)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(grp_names)))
    ax.set_xticklabels([grp_labels[g] for g in grp_names], fontsize=9)
    ax.set_ylabel("Pearson Correlation (Δspread)")
    ax.set_title("A. Korelasyon Dağılımı by Relationship Type", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Grup ortalamalarını ekle
    for i, (d, g) in enumerate(zip(grp_data, grp_names)):
        ax.text(i, np.mean(d) + 0.02, f"{np.mean(d):+.3f}",
                ha="center", fontsize=9, fontweight="bold",
                color=grp_colors[g])

    # Panel B: Dönem karşılaştırması (bar chart)
    ax = axes[1]
    period_names = ["Pre-COVID", "COVID Crisis", "Post-COVID"]
    grp_show = ["upstream", "downstream", "unconnected"]
    x = np.arange(len(period_names))
    width = 0.25

    for k, grp in enumerate(grp_show):
        vals = []
        for pn in period_names:
            row = df_periods[(df_periods["period"]==pn) &
                             (df_periods["group"]==grp)]
            vals.append(row["mean_corr"].values[0] if len(row)>0 else 0)
        ax.bar(x + k*width, vals, width, label=grp_labels[grp],
               color=grp_colors[grp], alpha=0.8)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(period_names, fontsize=9)
    ax.set_ylabel("Mean Correlation")
    ax.set_title("B. COVID Dönemleri Karşılaştırması", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel C: Graph mesafesi vs korelasyon
    ax = axes[2]
    dist_order = [d for d in ["1","2","3","4+","unconnected"]
                  if d in dist_summary.index]
    dist_means = [dist_summary.loc[d, "mean"] for d in dist_order]
    dist_stds  = [dist_summary.loc[d, "std"]  for d in dist_order]
    dist_colors = ["#1565C0","#1E88E5","#42A5F5","#90CAF9","#9E9E9E"]
    ax.bar(range(len(dist_order)), dist_means,
           color=dist_colors[:len(dist_order)], alpha=0.85,
           yerr=dist_stds, capsize=4, error_kw={"linewidth":1})
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(dist_order)))
    ax.set_xticklabels([f"{d}-hop" if d not in ("4+","unconnected")
                        else d for d in dist_order], fontsize=9)
    ax.set_ylabel("Mean Correlation")
    ax.set_title("C. Graph Mesafesi vs Korelasyon", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Cross-Firm CDS Spread Change Correlations\n"
                 "by Supply-Chain Relationship Type", fontsize=13)
    plt.tight_layout()
    out1 = FIG_DIR / "fig_corr_groups.png"
    fig.savefig(out1, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  [OK] {out1.name}")

    # --- Fig 2: Rolling korelasyon zaman serisi ---
    fig, ax = plt.subplots(figsize=(14, 5))
    covid_start = pd.Timestamp(COVID_START)
    covid_end   = pd.Timestamp(COVID_END)

    col_map = {
        "corr_upstream"     : ("#2196F3", "Upstream",      2.0),
        "corr_downstream"   : ("#FF9800", "Downstream",    2.0),
        "corr_bidirectional": ("#9C27B0", "Bidirectional", 1.5),
        "corr_unconnected"  : ("#9E9E9E", "Unconnected",   1.5),
    }
    for col, (color, label, lw) in col_map.items():
        if col in df_roll.columns:
            ax.plot(df_roll.index, df_roll[col], color=color,
                    label=label, linewidth=lw, alpha=0.9)

    ax.axvspan(covid_start, covid_end, alpha=0.12, color="red",
               label="COVID Crisis")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Mean Pairwise Correlation (rolling {ROLL_WINDOW}w)")
    ax.set_title("Rolling Cross-Firm CDS Correlations by Supply-Chain Relationship",
                 fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out2 = FIG_DIR / "fig_corr_timeseries.png"
    fig.savefig(out2, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  [OK] {out2.name}")

    # --- Fig 3: Korelasyon heat map (bağlantılı firmalar vurgulu) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    corr_matrix = delta.corr().values
    connected_mask = (A_comb > 0).astype(float)

    im1 = axes[0].imshow(corr_matrix, cmap="RdBu_r",
                          vmin=-0.5, vmax=0.5, aspect="auto")
    plt.colorbar(im1, ax=axes[0], shrink=0.8)
    axes[0].set_title("CDS Delta Korelasyon Matrisi", fontsize=11)
    axes[0].set_xlabel("Firma"); axes[0].set_ylabel("Firma")

    # Supply chain bağlantılarını işaretle
    for i in range(N):
        for j in range(N):
            if connected_mask[i, j] > 0:
                axes[0].plot(j, i, "k.", markersize=2, alpha=0.6)

    im2 = axes[1].imshow(connected_mask, cmap="Blues",
                          vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im2, ax=axes[1], shrink=0.8)
    axes[1].set_title("Supply Chain Adjacency Matrisi", fontsize=11)
    axes[1].set_xlabel("Firma"); axes[1].set_ylabel("Firma")

    plt.suptitle("CDS Korelasyon vs Supply Chain Yapısı", fontsize=13)
    plt.tight_layout()
    out3 = FIG_DIR / "fig_corr_heatmap.png"
    fig.savefig(out3, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  [OK] {out3.name}")

    # Özet yazdır
    print(f"\n{'='*65}")
    print(f"  ÖZET BULGULAR")
    print(f"{'='*65}")
    t, p, mu_con, mu_unc = ttest_groups(df_pairs)
    print(f"\n  1. Bağlantılı çiftler ort. korelasyon : {mu_con:+.4f}")
    print(f"     Bağlantısız çiftler ort. korelasyon: {mu_unc:+.4f}")
    print(f"     Fark: {mu_con - mu_unc:+.4f}  "
          f"(t={t:.3f}, p={p:.4f})")

    for grp in ["upstream", "downstream"]:
        sub = df_pairs[df_pairs["rel_type"]==grp]["corr"]
        unc = df_pairs[df_pairs["rel_type"]=="unconnected"]["corr"]
        t2, p2 = stats.ttest_ind(sub, unc, equal_var=False)
        print(f"\n  2. {grp.capitalize()} vs Unconnected:")
        print(f"     {grp}: {sub.mean():+.4f}  Unconnected: {unc.mean():+.4f}")
        print(f"     t={t2:.3f}, p={p2:.4f}  "
              f"({'anlamlı ***' if p2<0.01 else 'anlamlı **' if p2<0.05 else 'anlamlı *' if p2<0.10 else 'anlamsız'})")

    # COVID kriz etkisi
    pre  = df_periods[(df_periods["period"]=="Pre-COVID") &
                      (df_periods["group"]=="upstream")]["mean_corr"].values
    kriz = df_periods[(df_periods["period"]=="COVID Crisis") &
                      (df_periods["group"]=="upstream")]["mean_corr"].values
    if len(pre) > 0 and len(kriz) > 0:
        print(f"\n  3. COVID kriz etkisi (Upstream):")
        print(f"     Pre-COVID   : {pre[0]:+.4f}")
        print(f"     COVID Crisis: {kriz[0]:+.4f}")
        print(f"     Artış       : {kriz[0]-pre[0]:+.4f} "
              f"({'↑ güçlendi' if kriz[0]>pre[0] else '↓ zayıfladı'})")

    print(f"\n  Çıktılar:")
    print(f"    {METRICS_DIR / 'correlation_pairwise.csv'}")
    print(f"    {METRICS_DIR / 'correlation_by_period.csv'}")
    print(f"    {METRICS_DIR / 'correlation_timeseries.csv'}")
    print(f"    {FIG_DIR / 'fig_corr_groups.png'}")
    print(f"    {FIG_DIR / 'fig_corr_timeseries.png'}")
    print(f"    {FIG_DIR / 'fig_corr_heatmap.png'}")
    print(f"\n[DONE] cross_firm_correlation.py tamamlandi.")


if __name__ == "__main__":
    main()
