# VaR System for Short Options Portfolio with Regime-Aware Trading

Production-grade Value-at-Risk system for a large-cap equity portfolio with a systematic weekly short call overlay on SPX. GJR-GARCH passes both Kupiec and Christoffersen backtests over 783 out-of-sample days. A 3-state Hidden Markov Model with causal forward filtering drives regime-aware option sizing.

## Results

| Metric | Standard GARCH | GJR-GARCH |
|--------|---------------|-----------|
| Exceptions | 37 / 783 | 34 / 783 |
| Exception rate | 4.7% | 4.3% |
| Kupiec p-value | 0.722 (PASS) | 0.388 (PASS) |
| Christoffersen p-value | 0.006 (FAIL) | **0.061 (PASS)** |
| Exception clusters | 6 | 4 |

## Key Features

- **GJR-GARCH volatility filtering** — asymmetric leverage effect spikes the vol forecast harder after negative returns, breaking the exception clustering that fails Christoffersen under standard GARCH
- **3-state HMM regime detection** — Calm / Caution / Crisis classification using SPX returns, realized vol, credit spreads, rate changes, and VIX
- **Causal forward filter** — replaces Viterbi with the HMM forward algorithm so that day t's regime uses only observations up to day t (no look-ahead bias)
- **Regime-aware option strategy** — 150 contracts at delta 0.25 in Calm, 100 contracts at delta 0.15 in Caution, no selling in Crisis
- **Greeks monitoring and risk alerts** — daily repricing with automated alerts for high gamma, unfavourable theta/gamma ratio, excessive delta, and regime transitions
- **Premium capture and stop-loss** — close at 80% premium decay or 2x premium loss
- **783-day out-of-sample backtest** — option simulation starts at day 1,001 after HMM training period, covering late 2022 through early 2026

## Architecture

```
config.py                       All tunable parameters
data_loader.py                  Excel data ingestion
option_pricer.py                Black-Scholes pricing and closed-form greeks
option_strategy.py              Weekly short call simulation with regime sizing
var_engine.py                   GJR-GARCH filtering, time-weighted VaR/CVaR, Kupiec and Christoffersen
regime_detector.py              3-state HMM with causal forward filter
risk_alerts.py                  Greeks-based and regime transition alerts
reporting.py                    Console output utilities
VaR_Stock_Options_Combined.py   Main orchestrator
```

## How It Works

**VaR Model:** GARCH-filtered Historical Simulation with Student's t innovations. Each historical return in the 250-day lookback window is scaled by the ratio of tomorrow's forecast vol to that day's conditional vol. Time-weighted percentiles (exponential decay, lambda=0.985) produce the VaR and CVaR estimates. The GJR extension adds a single asymmetry parameter that lets negative returns increase the vol forecast more than positive returns of the same magnitude.

**Regime Model:** A Gaussian HMM is trained on the first 1,000 days of data. States are ranked by a stress score (volatility z-score + OAS change z-score). Classification uses the forward algorithm instead of Viterbi — a filtering approach that is causal by construction. The regime model drives option strategy decisions (position size and delta target) but does not scale VaR, as empirical testing showed regime-based VaR scaling worsened backtest results.

**Option Strategy:** Every Monday, sell weekly SPX calls expiring Friday. Strike selected via closed-form Black-Scholes delta inversion. Positions are managed with an 80% premium capture exit and a 2x premium stop-loss. Contract count and delta target adapt to the current regime state.

## Requirements

- Python 3.12+
- numpy, pandas, scipy, arch, hmmlearn, openpyxl, matplotlib, seaborn

## Configuration

All parameters are centralised in `config.py`. Key toggles:

- `GARCH_ASYMMETRY = 1` — set to 0 for standard GARCH, 1 for GJR-GARCH
- `REGIME_MIN_DAYS = 1000` — HMM training period length
- `OPTION_CONTRACTS = 150` — base contracts per week
- `REGIME_VAR_MULTIPLIER` — all set to 1.0 (regime scaling disabled for VaR)
