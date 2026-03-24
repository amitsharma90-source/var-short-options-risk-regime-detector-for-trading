# -*- coding: utf-8 -*-
"""
=============================================================================
Module: config.py — Central Configuration
=============================================================================
All tunable parameters for the VaR + Options + Regime system.
Edit this file to change strategy, risk, or model parameters.
=============================================================================
"""

from pathlib import Path

# ── File Paths ─────────────────────────────────────────────────────────────
PRICES_PATH = r"C:\Users\amits\Desktop\All quant workshop\Market Risk\Value at Risk, app, Combined multi asset portfolio\data\stock_prices_30_tickers_with_rates_extended,Uber non NULL for Pre IPO.xlsx"
HOLDINGS_PATH = r"C:\Users\amits\Desktop\All quant workshop\Market Risk\Value at Risk, app, Combined multi asset portfolio\data\Portfolio_holdings.xlsx"
OUTPUT_DIR = Path(".")

# ── VaR Model Parameters ──────────────────────────────────────────────────
LAMBDA_DECAY = 0.985
LOOKBACK = 250
CONFIDENCE_LEVEL = 0.95
GARCH_DIST = 'StudentsT'
GARCH_ASYMMETRY = 1           # 0 = standard GARCH, 1 = GJR-GARCH (leverage effect)

# ── Option Strategy Parameters ────────────────────────────────────────────
OPTION_CONTRACTS = 150         # Number of contracts sold each Monday
TARGET_DELTA = 0.25            # Target delta for strike selection
SPX_MULTIPLIER = 100           # SPX option contract multiplier
DAYS_TO_EXPIRY = 5             # Weekly options: Monday to Friday
PREMIUM_CAPTURE_EXIT = 0.80    # Close when 80% of premium captured
STOP_LOSS_MULTIPLE = 2.0       # Close if loss exceeds 2x premium received

# ── Risk Alert Thresholds ─────────────────────────────────────────────────
GAMMA_THRESHOLD_PCT = 0.005    # Flag if gamma loss on 1% move > 0.5% of portfolio
DELTA_THRESHOLD_PCT = 0.10     # Flag if delta notional > 10% of portfolio

# ── Regime Model Parameters ───────────────────────────────────────────────
REGIME_N_STATES = 3            # Calm, Caution, Crisis
REGIME_HMM_ITER = 1000         # HMM fitting iterations
REGIME_RANDOM_STATE = 42       # Reproducibility
REGIME_VOL_WINDOW = 21         # Rolling window for realized vol feature
REGIME_Z_MIN_PERIODS = 30      # Minimum periods for expanding z-score
REGIME_MIN_DAYS = 1000         # ~4 years training (needs 2008/COVID crashes for good fit)

# Regime-aware VaR scaling multipliers
REGIME_VAR_MULTIPLIER = {
    0: 1.0,                    # Calm: no scaling
    1: 1.0,                    # Caution: no scaling (GARCH handles it)
    2: 1.0,                    # Crisis: widen VaR by 30% (grid-search calibrated)
}

# Regime-aware option strategy adjustments
REGIME_OPTION_CONTRACTS = {
    0: OPTION_CONTRACTS,       # Calm: full size
    1: int(OPTION_CONTRACTS * 0.67),  # Caution: reduce by 1/3
    2: 0,                      # Crisis: stop selling
}

REGIME_OPTION_DELTA = {
    0: TARGET_DELTA,           # Calm: delta 0.25
    1: 0.15,                   # Caution: further OTM
    2: 0.0,                    # Crisis: no selling
}
