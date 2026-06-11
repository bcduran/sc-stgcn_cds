"""
configs/config.py
=================
Projedeki TUM yol ve hiperparametrelerin tek merkezi kaynagi.
"""

from __future__ import annotations
from pathlib import Path

# =============================================================================
# SECTION 1 -- ROOT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CDS_RAW_CSV = PROJECT_ROOT / "Data" / "raw" / "cds.csv"

SC_BASE_DIR = PROJECT_ROOT / "Data" / "Supply Chain Data S&P500"


# =============================================================================
# SECTION 2 -- DERIVED PATHS
# =============================================================================

SC_CUSTOMER_DIR = SC_BASE_DIR / "Customer Data S&P500"
SC_SUPPLIER_DIR = SC_BASE_DIR / "Supplier Data S&P500"

DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = PROJECT_ROOT / "Data" / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TOP50_DIR     = DATA_DIR / "top50"

# ── top50_connected dizini ────────────────────────────────────────────────
TOP50_CONN_DIR     = DATA_DIR / "top50_connected"
TOP50_CONN_CSV     = TOP50_CONN_DIR / "ve1.csv"
CONN_ADJ_NPZ       = TOP50_CONN_DIR / "adj.npz"
CONN_ADJ_SUP_NPZ   = TOP50_CONN_DIR / "adj_sup.npz"
CONN_ADJ_CUS_NPZ   = TOP50_CONN_DIR / "adj_cus.npz"
CONN_FIRMS_CSV     = TOP50_CONN_DIR / "top50_connected_firms.csv"

# ── top50_degree dizini (degree + both sup&cus + kurt<100) ─────────────────
TOP50_DEGREE_DIR      = DATA_DIR / "top50_degree"
TOP50_DEGREE_CSV      = TOP50_DEGREE_DIR / "ve1.csv"
DEGREE_ADJ_NPZ        = TOP50_DEGREE_DIR / "adj.npz"
DEGREE_ADJ_SUP_NPZ    = TOP50_DEGREE_DIR / "adj_sup.npz"
DEGREE_ADJ_CUS_NPZ    = TOP50_DEGREE_DIR / "adj_cus.npz"
DEGREE_FIRMS_CSV      = TOP50_DEGREE_DIR / "top50_degree_firms.csv"
# ──────────────────────────────────────────────────────────────────────────

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PRED_DIR    = OUTPUTS_DIR / "predictions"
FIG_DIR     = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
CKPT_DIR    = OUTPUTS_DIR / "checkpoints"

FULL_DIR         = DATA_DIR / "full"
VE1_FULL_CSV     = FULL_DIR / "ve1_full.csv"
TICKERS_FULL_TXT = FULL_DIR / "tickers_full.txt"
ADJ_FULL_NPZ     = FULL_DIR / "adj_full.npz"
ADJ_SUP_FULL_NPZ = FULL_DIR / "adj_sup_full.npz"
ADJ_CUS_FULL_NPZ = FULL_DIR / "adj_cus_full.npz"
MATCH_LOG_CSV    = FULL_DIR / "adj_match_log.csv"

TOP50_XLSX          = PROCESSED_DIR / "e_top50_companies.xlsx"
CDS_WEEKLY_CSV      = PROCESSED_DIR / "cds_weekly_5Y_all_by_ticker.csv"
TOP50_WEEKLY_CSV    = TOP50_DIR / "ve1.csv"
ADJ_NPZ             = TOP50_DIR / "adj.npz"
ADJ_SUP_NPZ         = TOP50_DIR / "adj_sup.npz"
ADJ_CUS_NPZ         = TOP50_DIR / "adj_cus.npz"
RAW_EDGES_CSV       = PROCESSED_DIR / "raw_edges.csv"

CUSTOMER_TICKERS_XLSX = PROCESSED_DIR / "a_Customer_Tickers.xlsx"
SUPPLIER_TICKERS_XLSX = PROCESSED_DIR / "a_Supplier_Tickers.xlsx"
TOP50_SUPPLIER_XLSX   = PROCESSED_DIR / "e_top_50_supplier_info.xlsx"
TOP50_CUSTOMER_XLSX   = PROCESSED_DIR / "e_top_50_customer_info.xlsx"


def make_dirs() -> None:
    for d in [PROCESSED_DIR, TOP50_DIR, TOP50_CONN_DIR, TOP50_DEGREE_DIR,
              FULL_DIR, PRED_DIR, FIG_DIR, METRICS_DIR, CKPT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SECTION 3 -- DATA PARAMETERS
# =============================================================================

class DataCFG:
    PREFERRED_TENOR : str   = "5Y"
    WEEKLY_FREQ     : str   = "W-FRI"
    MIN_COVERAGE    : float = 0.70
    START_DATE      : str   = "2015-01-01"
    END_DATE        : str   = "2021-09-10"


# =============================================================================
# SECTION 4 -- MODEL PARAMETERS
# =============================================================================

class ModelCFG:
    SEED     : int   = 42
    N_HIS    : int   = 7
    N_PRED   : int   = 1
    SPLIT    : tuple = (0.80, 0.10, 0.10)
    DEVICE   : str   = "auto"

    SCSTGCN_GCN_H     : int   = 64
    SCSTGCN_ATT_H     : int   = 64
    SCSTGCN_ATT_HEADS : int   = 4
    SCSTGCN_FF_H      : int   = 128
    SCSTGCN_DROPOUT   : float = 0.10
    SCSTGCN_EPOCHS    : int   = 300
    SCSTGCN_BATCH     : int   = 64
    SCSTGCN_LR        : float = 1e-3
    SCSTGCN_WD        : float = 1e-4
    SCSTGCN_PATIENCE  : int   = 30
    SCSTGCN_CLIP      : float = 5.0

    VSTGCN_GCN_H    : int   = 64
    VSTGCN_TEMP_H   : int   = 64
    VSTGCN_EPOCHS   : int   = 300
    VSTGCN_BATCH    : int   = 64
    VSTGCN_LR       : float = 1e-3
    VSTGCN_WD       : float = 1e-4
    VSTGCN_PATIENCE : int   = 30

    RNN_HIDDEN   : int   = 128
    RNN_LAYERS   : int   = 2
    RNN_DROPOUT  : float = 0.20
    RNN_EPOCHS   : int   = 300
    RNN_BATCH    : int   = 64
    RNN_LR       : float = 1e-3
    RNN_WD       : float = 1e-4
    RNN_PATIENCE : int   = 30

    XGB_PARAMS : dict = dict(
        n_estimators=600, max_depth=5, learning_rate=0.03,
        subsample=0.80, colsample_bytree=0.80,
        min_child_weight=3, gamma=0.1,
        reg_alpha=0.05, reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42, tree_method="hist",
        n_jobs=-1, verbosity=0,
    )


# =============================================================================
# SECTION 5 -- BACKTEST PARAMETERS
# =============================================================================

class BacktestCFG:
    TOP_Q       : float = 0.20
    DV01_PER_BP : float = 100.0
    INIT_CASH   : float = 100_000.0
    ANNUALIZE   : float = 52.0


# =============================================================================
# SECTION 6 -- SANITY CHECK
# =============================================================================

def check_inputs() -> None:
    checks = {
        "CDS raw CSV"         : CDS_RAW_CSV,
        "Supply Chain dir"    : SC_BASE_DIR,
        "Customer SC dir"     : SC_CUSTOMER_DIR,
        "Supplier SC dir"     : SC_SUPPLIER_DIR,
        "Top50 connected CSV" : TOP50_CONN_CSV,
        "Top50 connected adj" : CONN_ADJ_NPZ,
        "Top50 degree CSV"    : TOP50_DEGREE_CSV,
        "Top50 degree adj"    : DEGREE_ADJ_NPZ,
    }
    ok = True
    for name, path in checks.items():
        exists = path.exists()
        status = "OK" if exists else "EKSIK"
        print(f"  [{status}] {name:26s} -> {path}")
        if not exists:
            ok = False
    if not ok:
        print("\n[!] Eksik dosyalar var.")
    else:
        print("\n[OK] Tum input dosyalari mevcut.")


if __name__ == "__main__":
    print("=" * 60)
    print("  GNN Thesis -- Config Check")
    print("=" * 60)
    check_inputs()
    print(f"\n  PROJECT_ROOT    : {PROJECT_ROOT}")
    print(f"  TOP50_DIR       : {TOP50_DIR}")
    print(f"  TOP50_CONN_DIR  : {TOP50_CONN_DIR}")
    print(f"  TOP50_DEGREE_DIR: {TOP50_DEGREE_DIR}")
    print(f"  OUTPUTS_DIR     : {OUTPUTS_DIR}")