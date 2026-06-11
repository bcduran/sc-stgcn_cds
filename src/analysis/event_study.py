"""
src/analysis/event_study.py
============================
Katkı 2: Supply Chain Şok Yayılım Analizi (Event Study)

Araştırma sorusu:
  Bir firmada büyük CDS spread hareketi (şok) olduğunda,
  supply chain komşuları sonraki haftalarda anormal hareket
  gösteriyor mu? Upstream vs downstream asimetri var mı?

Yöntem:
  1. Şok tanımı: |Δs_i(t)| > 2 * firm_std  (firm-specific threshold)
  2. Event window: [-4, +8] hafta
  3. Normal getiri tahmini: pre-event ortalaması ([-20, -5])
  4. Kümülatif Anormal Spread Değişimi (CASC):
       CASC_j(τ) = Σ_{t=1}^{τ} [Δs_j(t) - E[Δs_j]]
  5. Komşu grupları: upstream, downstream, unconnected (kontrol)
  6. t-test: CASC'ın sıfırdan anlamlı farklı olup olmadığı
  7. Upstream vs downstream asimetri testi

Çıktılar:
  outputs/metrics/event_study_casc.csv
  outputs/metrics/event_study_summary.csv
  outputs/figures/fig_event_study_casc.png
  outputs/figures/fig_event_study_asymmetry.png

Çalıştırmak:
  python src/analysis/event_study.py
  python src/analysis/event_study.py --threshold 2.0 --min_events 3
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import load_npz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import (
    TOP50_WEEKLY_CSV,
    ADJ_SUP_NPZ, ADJ_CUS_NPZ, ADJ_NPZ,
    METRICS_DIR, FIG_DIR, make_dirs,
)

# Event study parametreleri
PRE_WINDOW   = (-20, -5)   # normal dönem tahmini için
EVENT_PRE    = -4          # event window başlangıcı
EVENT_POST   = 8           # event window bitişi
MIN_EVENTS   = 3           # bir grup için minimum event sayısı


def load_data():
    df = pd.read_csv(TOP50_WEEKLY_CSV)
    n_weeks = len(df)
    dates = pd.date_range(start="2015-01-02", periods=n_weeks, freq="W-FRI")
    df.index = dates
    level  = df.values.astype(np.float32)
    delta  = pd.DataFrame(
        np.diff(level, axis=0),
        index=dates[1:], columns=df.columns,
    )
    tickers = df.columns.tolist()
    N = len(tickers)
    A_sup  = load_npz(ADJ_SUP_NPZ).toarray()
    A_cus  = load_npz(ADJ_CUS_NPZ).toarray()
    A_comb = load_npz(ADJ_NPZ).toarray()
    return delta, tickers, N, A_sup, A_cus, A_comb


def detect_shocks(delta: pd.DataFrame, threshold_std: float = 2.0,
                  pre_buf: int = 25) -> list[dict]:
    """
    Firm-specific threshold ile şok tespiti.
    threshold_std: kaç standart sapma üzeri şok sayılır
    pre_buf: şoktan önce en az bu kadar hafta olmalı (normal dönem için)
    Döndürür: list of {firm_idx, event_t, direction, magnitude}
    """
    arr = delta.values  # (T, N)
    T, N = arr.shape
    events = []

    for i in range(N):
        s = arr[:, i]
        # Firm-specific std (COVID hariç yaklaşık)
        firm_std = np.nanstd(s)
        if firm_std < 1e-6:
            continue

        for t in range(pre_buf, T - EVENT_POST - 1):
            val = s[t]
            if np.isnan(val):
                continue
            if abs(val) > threshold_std * firm_std:
                direction = "widen" if val > 0 else "tighten"
                events.append({
                    "firm_idx" : i,
                    "event_t"  : t,
                    "direction": direction,
                    "magnitude": float(abs(val) / firm_std),
                    "raw_delta": float(val),
                })

    return events


def get_neighbors(firm_idx: int, A_sup, A_cus, N: int) -> dict:
    """
    Bir firmanın komşularını döndür.
    upstream:   A_sup[firm_idx, j] = 1  (j is supplier of firm_idx)
    downstream: A_sup[j, firm_idx] = 1  (firm_idx is supplier of j)
    unconnected: hiçbir bağlantı yok
    """
    i = firm_idx
    upstream   = [j for j in range(N) if j != i and A_sup[i, j] > 0]
    downstream = [j for j in range(N) if j != i and A_sup[j, i] > 0]
    connected  = set(upstream + downstream)
    unconnected = [j for j in range(N)
                   if j != i and j not in connected]
    return {
        "upstream"   : upstream,
        "downstream" : downstream,
        "unconnected": unconnected,
    }


def compute_casc(delta_arr: np.ndarray, neighbor_indices: list,
                 event_t: int, pre_window: tuple,
                 event_pre: int, event_post: int) -> np.ndarray | None:
    """
    Verilen komşu grubu için kümülatif anormal spread değişimi hesapla.
    Döndürür: (event_post - event_pre + 1,) CASC dizisi veya None
    """
    if len(neighbor_indices) == 0:
        return None

    T = delta_arr.shape[0]
    pre_start = event_t + pre_window[0]
    pre_end   = event_t + pre_window[1]

    if pre_start < 0 or event_t + event_post >= T:
        return None

    casc_list = []
    for j in neighbor_indices:
        s = delta_arr[:, j]
        # Normal (beklenen) hareket = pre-event penceresi ortalaması
        pre_vals = s[pre_start:pre_end]
        if np.isnan(pre_vals).mean() > 0.5:
            continue
        expected = np.nanmean(pre_vals)

        # Event window: [event_t + event_pre, event_t + event_post]
        ev_start = event_t + event_pre
        ev_end   = event_t + event_post + 1
        if ev_start < 0 or ev_end > T:
            continue
        ev_vals = s[ev_start:ev_end]

        # Anormal değişim
        abnormal = ev_vals - expected
        # Kümülatif
        casc = np.nancumsum(abnormal)
        casc_list.append(casc)

    if len(casc_list) == 0:
        return None

    return np.mean(casc_list, axis=0)  # komşular arası ortalama


def run_event_study(delta: pd.DataFrame, tickers: list, N: int,
                    A_sup, A_cus, A_comb,
                    threshold_std: float = 2.0,
                    min_events: int = MIN_EVENTS,
                    direction_filter: str = "both"):
    """
    Ana event study fonksiyonu.
    direction_filter: 'widen', 'tighten', 'both'
    """
    arr = delta.values
    T   = arr.shape[0]
    event_len = EVENT_POST - EVENT_PRE + 1
    lags = list(range(EVENT_PRE, EVENT_POST + 1))

    # Şokları tespit et
    events = detect_shocks(delta, threshold_std=threshold_std)
    if direction_filter != "both":
        events = [e for e in events if e["direction"] == direction_filter]

    print(f"  Toplam şok: {len(events)} "
          f"({sum(1 for e in events if e['direction']=='widen')} widen, "
          f"{sum(1 for e in events if e['direction']=='tighten')} tighten)")

    # Her grup için CASC topla
    group_casc = {
        "upstream"   : [],
        "downstream" : [],
        "unconnected": [],
    }
    event_counts = {g: 0 for g in group_casc}
    firm_event_counts = {}

    for ev in events:
        i = ev["firm_idx"]
        t = ev["event_t"]
        sign = 1 if ev["direction"] == "widen" else -1

        neighbors = get_neighbors(i, A_sup, A_cus, N)

        for grp, idxs in neighbors.items():
            if grp not in group_casc:
                continue
            casc = compute_casc(arr, idxs, t, PRE_WINDOW,
                                EVENT_PRE, EVENT_POST)
            if casc is not None and len(casc) == event_len:
                # Widen şoklarında pozitif, tighten'da negatif bekliyoruz
                group_casc[grp].append(sign * casc)
                event_counts[grp] += 1

        # Firma bazında sayım
        ticker = tickers[i]
        firm_event_counts[ticker] = firm_event_counts.get(ticker, 0) + 1

    # CASC istatistikleri
    results = {}
    for grp, casc_list in group_casc.items():
        if len(casc_list) < min_events:
            print(f"  [SKIP] {grp}: yeterli event yok "
                  f"({len(casc_list)} < {min_events})")
            continue
        casc_arr = np.array(casc_list)   # (n_events, event_len)
        mean_casc = casc_arr.mean(axis=0)
        std_casc  = casc_arr.std(axis=0)
        se_casc   = std_casc / np.sqrt(len(casc_list))
        # t-test: CASC sıfırdan farklı mı?
        t_stats = mean_casc / (se_casc + 1e-10)
        p_vals  = 2 * (1 - stats.t.cdf(np.abs(t_stats),
                                        df=len(casc_list)-1))
        results[grp] = {
            "n_events" : len(casc_list),
            "mean_casc": mean_casc,
            "std_casc" : std_casc,
            "se_casc"  : se_casc,
            "t_stats"  : t_stats,
            "p_vals"   : p_vals,
            "casc_arr" : casc_arr,
        }

    return results, lags, firm_event_counts


def print_casc_table(results: dict, lags: list):
    """CASC tablosunu yazdır."""
    print(f"\n  {'Lag':>5}", end="")
    for grp in results:
        print(f"  {grp.upper()[:8]:>10} {'sig':>4}", end="")
    print()
    print(f"  {'-'*60}")
    for k, lag in enumerate(lags):
        print(f"  {lag:>5}", end="")
        for grp, res in results.items():
            val = res["mean_casc"][k]
            p   = res["p_vals"][k]
            sig = "***" if p < 0.01 else "**" if p < 0.05 else \
                  "*"   if p < 0.10 else ""
            print(f"  {val:>+10.4f} {sig:>4}", end="")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold",   type=float, default=2.0,
                        help="Şok eşiği (std multiplier, default=2.0)")
    parser.add_argument("--min_events",  type=int,   default=3,
                        help="Minimum event sayısı (default=3)")
    parser.add_argument("--direction",   type=str,   default="both",
                        choices=["both","widen","tighten"])
    args = parser.parse_args()

    make_dirs()

    print("=" * 65)
    print(f"  Event Study: Supply Chain Şok Yayılım Analizi")
    print(f"  threshold={args.threshold}σ  direction={args.direction}")
    print("=" * 65)

    # 1) Veri
    print("\n[1/4] Veri yukleniyor...")
    delta, tickers, N, A_sup, A_cus, A_comb = load_data()
    print(f"  Delta panel: {delta.shape[0]} hafta x {N} firma")

    # 2) Event study — tüm şoklar
    print(f"\n[2/4] Event study calistiriliyor...")
    print(f"  Şok eşiği     : >{args.threshold} firm-specific std")
    print(f"  Pre-window    : [{EVENT_PRE}, -1] hafta")
    print(f"  Post-window   : [0, +{EVENT_POST}] hafta")
    print(f"  Normal dönem  : [{PRE_WINDOW[0]}, {PRE_WINDOW[1]}] hafta")

    results, lags, firm_counts = run_event_study(
        delta, tickers, N, A_sup, A_cus, A_comb,
        threshold_std=args.threshold,
        min_events=args.min_events,
        direction_filter=args.direction,
    )

    # Event sayıları
    print(f"\n  Firma bazında event sayısı (top 10):")
    top_firms = sorted(firm_counts.items(), key=lambda x: -x[1])[:10]
    for ticker, cnt in top_firms:
        print(f"    {ticker}: {cnt} event")

    # CASC tablosu
    print(f"\n  CASC (Kümülatif Anormal Spread Değişimi) — bp:")
    print_casc_table(results, lags)

    # 3) Asimetri analizi — widen vs tighten ayrı
    print(f"\n[3/4] Asimetri analizi (Upstream vs Downstream)...")
    res_up = run_event_study(
        delta, tickers, N, A_sup, A_cus, A_comb,
        threshold_std=args.threshold,
        min_events=args.min_events,
        direction_filter="widen",
    )[0]
    res_dn = run_event_study(
        delta, tickers, N, A_sup, A_cus, A_comb,
        threshold_std=args.threshold,
        min_events=args.min_events,
        direction_filter="tighten",
    )[0]

    print(f"\n  WIDEN şokları için CASC @ lag=+4:")
    for grp in ["upstream","downstream","unconnected"]:
        if grp in res_up:
            v = res_up[grp]["mean_casc"][lags.index(4)]
            p = res_up[grp]["p_vals"][lags.index(4)]
            sig = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else "ns"
            print(f"    {grp:<12}: {v:>+8.4f} bp  [{sig}]")

    print(f"\n  TIGHTEN şokları için CASC @ lag=+4:")
    for grp in ["upstream","downstream","unconnected"]:
        if grp in res_dn:
            v = res_dn[grp]["mean_casc"][lags.index(4)]
            p = res_dn[grp]["p_vals"][lags.index(4)]
            sig = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else "ns"
            print(f"    {grp:<12}: {v:>+8.4f} bp  [{sig}]")

    # 4) Görseller
    print(f"\n[4/4] Goerseller olusturuluyor...")

    grp_colors = {
        "upstream"   : "#2196F3",
        "downstream" : "#FF9800",
        "unconnected": "#9E9E9E",
    }

    # --- Fig 1: Ana CASC grafiği ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    grp_titles = {
        "upstream"   : "Upstream Neighbors\n(Suppliers of shocked firm)",
        "downstream" : "Downstream Neighbors\n(Customers of shocked firm)",
        "unconnected": "Control Group\n(Unconnected firms)",
    }

    for ax, (grp, res) in zip(axes, results.items()):
        mean = res["mean_casc"]
        se   = res["se_casc"]
        n    = res["n_events"]
        t_s  = res["t_stats"]
        p_v  = res["p_vals"]

        color = grp_colors.get(grp, "#555555")

        # Güven bandı
        ax.fill_between(lags,
                         mean - 1.96 * se,
                         mean + 1.96 * se,
                         alpha=0.20, color=color)
        ax.plot(lags, mean, color=color, linewidth=2.5, marker="o",
                markersize=4, label=f"CASC (n={n})")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="red", linewidth=1.0, linestyle="--",
                   alpha=0.7, label="Shock (t=0)")

        # Anlamlı lag'ları işaretle
        for k, lag in enumerate(lags):
            p = p_v[k]
            if p < 0.01:
                ax.text(lag, mean[k] + se[k] + 0.05,
                        "***", ha="center", fontsize=9, color=color)
            elif p < 0.05:
                ax.text(lag, mean[k] + se[k] + 0.05,
                        "**", ha="center", fontsize=9, color=color)
            elif p < 0.10:
                ax.text(lag, mean[k] + se[k] + 0.05,
                        "*", ha="center", fontsize=9, color=color)

        ax.set_title(grp_titles.get(grp, grp), fontsize=10)
        ax.set_xlabel("Weeks Relative to Shock")
        ax.set_ylabel("CASC (bp)" if ax == axes[0] else "")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xticks(lags)

    plt.suptitle(
        f"Supply Chain Shock Propagation — Cumulative Abnormal Spread Change\n"
        f"(threshold={args.threshold}σ, direction={args.direction}, "
        f"shaded=95% CI, *p<0.10, **p<0.05, ***p<0.01)",
        fontsize=11
    )
    plt.tight_layout()
    tag = f"{args.direction}_{str(args.threshold).replace('.','p')}"
    out1 = FIG_DIR / f"fig_event_study_casc_{tag}.png"
    fig.savefig(out1, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  [OK] {out1.name}")

    # --- Fig 2: Upstream vs Downstream karşılaştırması ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (title, res_dir) in zip(
        axes,
        [("WIDEN Şokları", res_up), ("TIGHTEN Şokları", res_dn)]
    ):
        for grp in ["upstream", "downstream", "unconnected"]:
            if grp not in res_dir:
                continue
            mean = res_dir[grp]["mean_casc"]
            se   = res_dir[grp]["se_casc"]
            n    = res_dir[grp]["n_events"]
            color = grp_colors[grp]
            ax.fill_between(lags,
                             mean - 1.96 * se,
                             mean + 1.96 * se,
                             alpha=0.15, color=color)
            ax.plot(lags, mean, color=color, linewidth=2.0,
                    marker="o", markersize=3,
                    label=f"{grp} (n={n})")

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="red", linewidth=1.0, linestyle="--",
                   alpha=0.7)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Weeks Relative to Shock")
        ax.set_ylabel("CASC (bp)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xticks(lags)

    plt.suptitle(
        "Upstream vs Downstream Asymmetry in Shock Propagation\n"
        "(Shaded = 95% CI)",
        fontsize=12
    )
    plt.tight_layout()
    out2 = FIG_DIR / f"fig_event_study_asymmetry_{tag}.png"
    fig.savefig(out2, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  [OK] {out2.name}")

    # --- Fig 3: CASC heatmap — lag x grup ---
    fig, ax = plt.subplots(figsize=(10, 4))
    grp_list  = [g for g in ["upstream","downstream","unconnected"]
                 if g in results]
    casc_mat  = np.array([results[g]["mean_casc"] for g in grp_list])
    sig_mat   = np.array([results[g]["p_vals"]    for g in grp_list])

    im = ax.imshow(casc_mat, cmap="RdBu_r", aspect="auto",
                   vmin=-max(abs(casc_mat.min()), abs(casc_mat.max())),
                   vmax= max(abs(casc_mat.min()), abs(casc_mat.max())))
    plt.colorbar(im, ax=ax, label="CASC (bp)")
    ax.set_xticks(range(len(lags)))
    ax.set_xticklabels(lags, fontsize=9)
    ax.set_yticks(range(len(grp_list)))
    ax.set_yticklabels([g.upper() for g in grp_list])
    ax.set_xlabel("Weeks Relative to Shock")
    ax.set_title("CASC Heatmap — Supply Chain Group x Event Lag", fontsize=11)

    # Anlamlılık işaretle
    for i in range(len(grp_list)):
        for j in range(len(lags)):
            p = sig_mat[i, j]
            if p < 0.01:
                ax.text(j, i, "***", ha="center", va="center",
                        fontsize=9, color="white" if abs(casc_mat[i,j]) > 0.5
                        else "black")
            elif p < 0.05:
                ax.text(j, i, "**", ha="center", va="center",
                        fontsize=9, color="white" if abs(casc_mat[i,j]) > 0.5
                        else "black")
            elif p < 0.10:
                ax.text(j, i, "*", ha="center", va="center",
                        fontsize=9, color="white" if abs(casc_mat[i,j]) > 0.5
                        else "black")

    plt.tight_layout()
    out3 = FIG_DIR / f"fig_event_study_heatmap_{tag}.png"
    fig.savefig(out3, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  [OK] {out3.name}")

    # Kaydet
    records = []
    for grp, res in results.items():
        for k, lag in enumerate(lags):
            records.append({
                "group"    : grp,
                "lag"      : lag,
                "mean_casc": res["mean_casc"][k],
                "se_casc"  : res["se_casc"][k],
                "t_stat"   : res["t_stats"][k],
                "p_value"  : res["p_vals"][k],
                "n_events" : res["n_events"],
            })
    df_out = pd.DataFrame(records)
    out_csv = METRICS_DIR / f"event_study_casc_{tag}.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"  [OK] {out_csv.name}")

    # Özet yazdır
    print(f"\n{'='*65}")
    print(f"  ÖZET BULGULAR — Event Study")
    print(f"{'='*65}")
    print(f"\n  Şok parametreleri: threshold={args.threshold}σ, "
          f"direction={args.direction}")

    post_lags = [1, 2, 4, 8]
    for lag in post_lags:
        if lag not in lags:
            continue
        k = lags.index(lag)
        print(f"\n  Lag +{lag} haftada CASC:")
        for grp in ["upstream","downstream","unconnected"]:
            if grp not in results:
                continue
            v = results[grp]["mean_casc"][k]
            p = results[grp]["p_vals"][k]
            n = results[grp]["n_events"]
            sig = "***" if p<0.01 else "**" if p<0.05 else \
                  "*" if p<0.10 else "ns"
            print(f"    {grp:<12}: {v:>+8.4f} bp  "
                  f"[{sig}, n={n}]")

    print(f"\n[DONE] event_study.py tamamlandi.")


if __name__ == "__main__":
    main()
