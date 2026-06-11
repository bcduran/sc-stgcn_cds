# SC-STGCN: Supply-Chain Aware Spatio-Temporal Graph Convolutional Network for CDS Spread Forecasting


This repository contains the full implementation of **SC-STGCN**, a novel graph neural network architecture for forecasting weekly changes in CDS (Credit Default Swap) spreads among S&P 500 firms. The model incorporates supply-chain relationships as a structural prior, constructing a dual-adjacency graph that captures both upstream (supplier) and downstream (customer) credit-risk propagation.

The framework combines spatio-temporal graph convolution with a temporal attention mechanism, outperforming eight benchmark models on a pre-COVID panel (2015–2020) of 50 degree-ranked S&P 500 firms.


## Key Results

| Metric | SC-STGCN | Best Baseline (V-STGCN) |
|--------|----------|--------------------------|
| RMSE   | **3.950** | 4.103 |
| R²     | **+0.040** | -0.012 |
| Sharpe (L/S backtest) | **2.064** | -0.251 |

SC-STGCN achieves positive R² and a Sharpe ratio of **2.064** versus **−0.251** for the undirected baseline, demonstrating that supply-chain topology provides statistically meaningful signal for credit risk forecasting.


## Model Architecture

SC-STGCN extends the standard STGCN framework with:

- **Dual adjacency matrix** : separate supplier and customer graphs derived from S&P Global supply-chain data
- **Degree-ranked node selection** : top-50 firms selected by supply-chain connectivity degree
- **Temporal Attention layer** :  learned attention over the look-back window
- **Walk-forward cross-validation**:  time-series safe evaluation with no data leakage

### Benchmark Models

| Model | Graph | Temporal |
|-------|-------|----------|
| Naïve (random walk) | ✗ | ✗ |
| AR(1) | ✗ | ✗ |
| ARMA(1,1) | ✗ | ✗ |
| LSTM | ✗ | ✓ |
| GRU | ✗ | ✓ |
| XGBoost | ✗ | ✓ |
| V-STGCN (undirected, ablation) | ✓ | ✓ |
| **SC-STGCN (ours)** | **✓** | **✓** |

---

## Repository Structure

```
GNN_Thesis/
│
├── configs/
│   └── config.py              ← All paths and hyperparameters
│
├── data/
│   ├── raw/                   ← Raw data (place cds.csv here)
│   ├── processed/             ← Pipeline outputs (adj, edges, weekly panels)
│   └── top50/                 ← Model inputs (ve1.csv, adj.npz, adj_sup.npz, adj_cus.npz)
│
├── src/
│   ├── pipeline/              ← Data preprocessing (Steps 01–09)
│   ├── models/                ← SC-STGCN and all baselines
│   ├── analysis/              ← Post-hoc analysis scripts
│   └── utils/                 ← Metrics, seeds, helpers
│
├── outputs/
│   ├── predictions/           ← y_true / y_pred CSVs
│   ├── figures/               ← PDF + PNG figures
│   ├── metrics/               ← results_*.csv + LaTeX tables
│   └── checkpoints/           ← .pt model weights
│
├── notebooks/                 ← EDA and visualization
├── tests/                     ← pytest test suite
├── requirements.txt
├── environment.yml
└── setup_env.bat              ← One-click Windows setup
```

---

## Installation

### Option A — pip + venv (Recommended)

```bash
cd GNN_Thesis

# Windows (one-click)
setup_env.bat

# Manual
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

### Option B — Conda

```bash
conda env create -f environment.yml
conda activate gnn_thesis
```

### GPU (CUDA 12.1)

```bash
# Remove the torch line from requirements.txt, then run:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Usage

```bash
# Step 0 — Verify paths
python configs/config.py

# Step 1 — Build weekly CDS panel
python src/pipeline/05_cds_weekly_panel.py

# Step 2 — Build supply-chain graph
python src/pipeline/02_creating_directed_graph.py

# Step 3 — Build adjacency matrices
python src/pipeline/07_build_adjacency.py

# Step 4 — Train and evaluate all models
python src/models/sc_stgcn_train.py --mode BOTH

# Quick test (50 epochs, CPU)
python src/models/sc_stgcn_train.py --mode DELTA --epochs 50 --device cpu
```

---

## Data

### CDS Spreads (`cds.csv`)
Daily CDS spread data for S&P 500 firms, sourced from Bloomberg. The primary variable is the 5-year CDS spread (`PX5`). Sample period: January 2015 – September 2021 (pre-COVID window: 2015–2020).

```
Date        | Ticker | PX1  | PX2  | PX3  | PX4  | PX5   | PX7  | PX10
2015-01-02  | AAPL   | 15.2 | 18.4 | 22.1 | 28.3 | 34.5  | 45.2 | 62.1
```

### Supply-Chain Relationships (S&P Global)
Customer and supplier linkages for each S&P 500 firm, used to construct the directed graph. Raw files are not included in this repository due to data licensing restrictions.

---

## Outputs

| File | Description |
|------|-------------|
| `outputs/metrics/results_delta.csv` | Full model comparison (all metrics) |
| `outputs/metrics/results_delta.tex` | LaTeX table (thesis-ready) |
| `outputs/figures/fig1_equity_delta.pdf` | Equity curve comparison |
| `outputs/figures/fig2_metrics_delta.pdf` | RMSE / Sharpe bar chart |
| `outputs/figures/fig3_scatter_delta.pdf` | Predicted vs. true scatter |
| `outputs/figures/fig4_attn_*.pdf` | Temporal attention heatmaps |
| `outputs/figures/fig5_cross_mode.pdf` | DELTA vs. LEVEL comparison |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{duran2026scstgcn,
  author  = {Burhan Cahit Duran},
  title   = {Supply-Chain Aware Spatio-Temporal Graph Convolutional Network for CDS Spread Forecasting},
  school  = {Ozyegin University},
  year    = {2026},
  type    = {M.Sc. Thesis}
}
```

---

## License

This repository is made available for academic and research purposes. The underlying CDS and supply-chain datasets are proprietary and not redistributed here.
