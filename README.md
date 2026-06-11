# GNN Thesis — SC-STGCN
## Supply-Chain Aware Spatio-Temporal GCN for CDS Spread Forecasting

---

## Proje Yapısı

```
GNN_Thesis/
│
├── configs/
│   └── config.py              ← TÜM yollar ve hiperparametreler (sadece bunu düzenle)
│
├── data/
│   ├── raw/                   ← Ham veriler (cds.csv buraya kopyalanabilir)
│   ├── processed/             ← Pipeline çıktıları (adj, edges, weekly panels)
│   └── top50/                 ← Model girdileri (ve1.csv, adj.npz, adj_sup.npz, adj_cus.npz)
│
├── src/
│   ├── pipeline/
│   │   ├── 01_cds_weekly_panel.py    ← cds.csv → haftalık 5Y panel
│   │   ├── 02_build_graph.py         ← SC verisi → yönlü graf → raw_edges.csv
│   │   └── 03_build_adjacency.py     ← raw_edges → adj.npz + adj_sup/cus.npz
│   │
│   ├── models/
│   │   └── sc_stgcn_train.py         ← Tüm modeller + karşılaştırma
│   │
│   ├── backtest/
│   │   └── backtest_engine.py        ← L/S CDS backtest
│   │
│   └── utils/
│       ├── seed.py                   ← Reproducibility
│       └── metrics.py                ← MSE/RMSE/R²/MAPE hesaplama
│
├── outputs/
│   ├── predictions/           ← y_true / y_pred CSV'leri
│   ├── figures/               ← PDF + PNG figürler
│   ├── metrics/               ← results_*.csv + LaTeX tabloları
│   └── checkpoints/           ← .pt model dosyaları
│
├── notebooks/                 ← EDA ve görselleştirme
├── tests/                     ← pytest testleri
│
├── requirements.txt
├── environment.yml
├── setup_env.bat              ← Windows kurulum (tek tık)
└── GNN_Thesis.code-workspace  ← VS Code workspace dosyası
```

---

## Kurulum

### Seçenek A — pip + venv (Önerilen)

```bash
# 1. Repo'yu aç
cd GNN_Thesis

# 2. Kurulum scriptini çalıştır (Windows)
setup_env.bat

# VEYA manuel:
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/Mac
pip install -r requirements.txt
```

### Seçenek B — Conda

```bash
conda env create -f environment.yml
conda activate gnn_thesis
```

### CUDA (GPU) kullanıyorsan

```bash
# requirements.txt'teki torch satırını sil, şunu çalıştır:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Çalıştırma Sırası

```bash
# Adım 0 — Yolları kontrol et
python configs/config.py

# Adım 1 — CDS haftalık panel oluştur
python src/pipeline/01_cds_weekly_panel.py

# Adım 2 — Supply chain grafını kur
python src/pipeline/02_build_graph.py

# Adım 3 — Adjacency matrisleri üret
python src/pipeline/03_build_adjacency.py

# Adım 4 — Modeli çalıştır (8 model × 2 mode)
python src/models/sc_stgcn_train.py --mode BOTH

# Hızlı test (50 epoch, CPU):
python src/models/sc_stgcn_train.py --mode DELTA --epochs 50 --device cpu
```

---

## Veri Formatları

### cds.csv (ham CDS verisi)
```
Date        | Ticker | PX1  | PX2  | PX3  | PX4  | PX5   | PX7  | PX10
2015-01-02  | AAPL   | 15.2 | 18.4 | 22.1 | 28.3 | 34.5  | 45.2 | 62.1
...
```
- `PX5` = 5Y CDS spread (ana değişken)
- Tarih aralığı: 2015-01-01 → 2021-09-10

### Supply Chain (S&P Global)
```
Customer Data S&P500/
    AAPL_customer_data.xlsx    ← "AAPL'ın müşterileri kimler?"
    MSFT_customer_data.xlsx
    ...
Supplier Data S&P500/
    AAPL_supplier_data.xlsx    ← "AAPL'ın tedarikçileri kimler?"
    ...
```

---

## Modeller

| Model       | Açıklama                              | Graph? | Temporal? |
|-------------|---------------------------------------|--------|-----------|
| Naive       | Δspread = 0 (random walk)             | ✗      | ✗         |
| AR(1)       | Firm-level AR(1) OLS                  | ✗      | ✗         |
| ARMA(1,1)   | Statsmodels ARIMA                     | ✗      | ✗         |
| LSTM        | Shared LSTM across N firms            | ✗      | ✓         |
| GRU         | Shared GRU across N firms             | ✗      | ✓         |
| XGBoost     | Per-firm XGB, window features         | ✗      | ✓         |
| V-STGCN     | Vanilla STGCN (ablation)              | ✓      | ✓         |
| **SC-STGCN**| **Dual adj + Temporal Attention**     | **✓**  | **✓**     |

---

## Çıktılar

- `outputs/metrics/results_delta.csv` — tüm modeller, tüm metrikler
- `outputs/metrics/results_delta.tex` — LaTeX tablosu (teze hazır)
- `outputs/figures/fig1_equity_delta.pdf` — equity curve karşılaştırması
- `outputs/figures/fig2_metrics_delta.pdf` — RMSE / Sharpe bar chart
- `outputs/figures/fig3_scatter_delta.pdf` — predicted vs true scatter
- `outputs/figures/fig4_attn_*.pdf` — temporal attention heatmap
- `outputs/figures/fig5_cross_mode.pdf` — DELTA vs LEVEL karşılaştırması
