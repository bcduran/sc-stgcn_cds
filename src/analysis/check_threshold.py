import pandas as pd, numpy as np

df = pd.read_csv("data/top50_degree/ve1.csv")
# Ilk kolon tarih olabilir
try:
    df.iloc[:,0].astype(float)
except:
    df = df.iloc[:,1:]

level = df.values.astype(float)
delta = np.diff(level, axis=0).flatten()
delta = delta[~np.isnan(delta)]

print(f"Std        : {delta.std():.4f} bp")
print(f"Mean       : {delta.mean():.4f} bp")
print(f"N obs      : {len(delta):,}")
print()

for threshold in [0.5, 1.0, 1.5, 2.0, 2.5]:
    stable = np.abs(delta) <= threshold
    up     = delta > threshold
    down   = delta < -threshold
    print(f"|ds|<={threshold:.1f} bp:  "
          f"stable={stable.mean():.1%}  "
          f"up={up.mean():.1%}  "
          f"down={down.mean():.1%}")
