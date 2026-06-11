"""
src/analysis/net_effect.py
Net supply chain etkisi hesapla.
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import METRICS_DIR

df = pd.read_csv(METRICS_DIR / "event_study_casc_both_2p0.csv")

print("=" * 60)
print("  NET SUPPLY CHAIN ETKİSİ (Downstream - Unconnected)")
print("=" * 60)

print("\nPre-shock dönemi (lag -4 ile -1):")
print(f"  {'Lag':>5} {'UP':>8} {'DN':>8} {'UNC':>8} {'UP-UNC':>9} {'DN-UNC':>9}")
print("  " + "-" * 50)
for lag in [-4, -3, -2, -1]:
    up  = df[(df["lag"]==lag) & (df["group"]=="upstream")]["mean_casc"].values[0]
    dn  = df[(df["lag"]==lag) & (df["group"]=="downstream")]["mean_casc"].values[0]
    unc = df[(df["lag"]==lag) & (df["group"]=="unconnected")]["mean_casc"].values[0]
    print(f"  {lag:>5} {up:>+8.3f} {dn:>+8.3f} {unc:>+8.3f} "
          f"{up-unc:>+9.3f} {dn-unc:>+9.3f}")

print("\nPost-shock dönemi (lag +1 ile +8):")
print(f"  {'Lag':>5} {'UP':>8} {'DN':>8} {'UNC':>8} {'UP-UNC':>9} {'DN-UNC':>9}")
print("  " + "-" * 50)
for lag in [1, 2, 3, 4, 5, 6, 7, 8]:
    up  = df[(df["lag"]==lag) & (df["group"]=="upstream")]["mean_casc"].values[0]
    dn  = df[(df["lag"]==lag) & (df["group"]=="downstream")]["mean_casc"].values[0]
    unc = df[(df["lag"]==lag) & (df["group"]=="unconnected")]["mean_casc"].values[0]
    print(f"  {lag:>5} {up:>+8.3f} {dn:>+8.3f} {unc:>+8.3f} "
          f"{up-unc:>+9.3f} {dn-unc:>+9.3f}")

# Pre-shock: DN vs UNC t-test
print("\n" + "=" * 60)
print("  İSTATİSTİKSEL TEST: Pre-shock DN vs UNC")
print("=" * 60)

# CSV'den pairwise bazında test edemeyiz — CASC array gerekli
# Ama mean fark ve SE farkından yaklaşık test yapabiliriz
print("\n  Not: Net etki = Downstream CASC - Unconnected CASC")
print("  Pre-shock lag=-2'de net downstream etkisi:")
for lag in [-3, -2, -1]:
    dn_mean  = df[(df["lag"]==lag)&(df["group"]=="downstream")]["mean_casc"].values[0]
    dn_se    = df[(df["lag"]==lag)&(df["group"]=="downstream")]["se_casc"].values[0]
    unc_mean = df[(df["lag"]==lag)&(df["group"]=="unconnected")]["mean_casc"].values[0]
    unc_se   = df[(df["lag"]==lag)&(df["group"]=="unconnected")]["se_casc"].values[0]
    net      = dn_mean - unc_mean
    net_se   = np.sqrt(dn_se**2 + unc_se**2)
    t        = net / (net_se + 1e-10)
    n_dn     = df[(df["lag"]==lag)&(df["group"]=="downstream")]["n_events"].values[0]
    p        = 2 * (1 - stats.t.cdf(abs(t), df=n_dn-1))
    sig      = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "ns"
    print(f"  lag={lag:>3}: net={net:>+7.3f} bp  SE={net_se:.3f}  "
          f"t={t:>+6.2f}  p={p:.4f}  [{sig}]")

print("\n  Pre-shock lag=-2 Upstream vs Unconnected:")
for lag in [-3, -2, -1]:
    up_mean  = df[(df["lag"]==lag)&(df["group"]=="upstream")]["mean_casc"].values[0]
    up_se    = df[(df["lag"]==lag)&(df["group"]=="upstream")]["se_casc"].values[0]
    unc_mean = df[(df["lag"]==lag)&(df["group"]=="unconnected")]["mean_casc"].values[0]
    unc_se   = df[(df["lag"]==lag)&(df["group"]=="unconnected")]["se_casc"].values[0]
    net      = up_mean - unc_mean
    net_se   = np.sqrt(up_se**2 + unc_se**2)
    t        = net / (net_se + 1e-10)
    n_up     = df[(df["lag"]==lag)&(df["group"]=="upstream")]["n_events"].values[0]
    p        = 2 * (1 - stats.t.cdf(abs(t), df=n_up-1))
    sig      = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "ns"
    print(f"  lag={lag:>3}: net={net:>+7.3f} bp  SE={net_se:.3f}  "
          f"t={t:>+6.2f}  p={p:.4f}  [{sig}]")

print("\n[DONE]")
