# -*- coding: utf-8 -*-
"""
=============================================================================
Module: regime_detector.py - 3-State Market Regime Indicator
=============================================================================
Hidden Markov Model (HMM) using SPX returns, rolling volatility,
HY credit spread changes, 10Y rate changes, and VIX to classify each
trading day as Calm (0), Caution (1), or Crisis (2).

Approach: FROZEN MODEL with CAUSAL FORWARD FILTER
  1. Fit HMM on training period (first REGIME_MIN_DAYS observations)
  2. Rank states using Vol_Z + OAS_Diff_Z stress score (fixed once)
  3. Classify each day using the HMM forward algorithm (filtered probs)
  4. Forward filter is CAUSAL: day t's regime uses only data up to day t
     (unlike Viterbi which optimizes the global state sequence and lets
      future observations influence past classifications)

Why forward filter instead of Viterbi:
  - Viterbi solves a SMOOTHING problem: best sequence given ALL observations
  - Forward filter solves a FILTERING problem: best state given observations
    up to NOW. This is what a risk manager actually does each morning.
  - For backtesting, Viterbi creates subtle look-ahead bias because day t's
    label is influenced by observations on days t+1, t+2, ..., T.

Integration with VaR:
  - Regime state scales GARCH forecast vol (Calm=1x, Caution=1x, Crisis=1.3x)
  - Regime state adjusts option strategy (contracts and delta)
=============================================================================
"""

import pandas as pd
import numpy as np
from scipy.special import logsumexp
from hmmlearn.hmm import GaussianHMM
from config import (REGIME_N_STATES, REGIME_HMM_ITER, REGIME_RANDOM_STATE,
                    REGIME_VOL_WINDOW, REGIME_Z_MIN_PERIODS, REGIME_MIN_DAYS)


def expanding_z_score(series):
    """Expanding window z-score normalization."""
    mean = series.expanding(min_periods=REGIME_Z_MIN_PERIODS).mean()
    std = series.expanding(min_periods=REGIME_Z_MIN_PERIODS).std()
    return (series - mean) / std


def _prepare_features(spx_prices, bb_oas, rate_10y, vix_values=None):
    """
    Build the feature DataFrame from raw market data.
    Returns (df, z_cols) where df has z-scored features and z_cols lists column names.
    """
    df = pd.DataFrame({
        'SPX_Close': spx_prices,
        'BB_OAS': bb_oas,
        'Rate_10Y': rate_10y,
    }, index=spx_prices.index)

    if vix_values is not None:
        df['VIX'] = vix_values

    df['Returns'] = df['SPX_Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=REGIME_VOL_WINDOW).std()
    df['OAS_Diff'] = df['BB_OAS'].diff()
    df['Rate_Diff'] = df['Rate_10Y'].diff()

    feature_cols = ['Returns', 'Volatility', 'OAS_Diff', 'Rate_Diff']
    if vix_values is not None:
        df['VIX_Level'] = df['VIX']
        feature_cols.append('VIX_Level')

    df = df.dropna()

    z_cols = []
    for col in feature_cols:
        z_col = f'{col}_Z'
        df[z_col] = expanding_z_score(df[col])
        z_cols.append(z_col)

    df = df.dropna()

    return df, z_cols


def _rank_states(model):
    """
    Rank HMM states by stress score (Vol_Z + OAS_Diff_Z).
    Same logic as the original regime indicator.
    Returns label_map: {hmm_state_id: standardized_label}
    where 0=Calm, 1=Caution, 2=Crisis.
    """
    state_means = model.means_
    stress_score = state_means[:, 1] + state_means[:, 2]
    state_rank = stress_score.argsort()
    return {state_rank[0]: 0, state_rank[1]: 1, state_rank[2]: 2}


def _forward_filter(model, X):
    """
    Causal forward filtering: compute P(state_t | observations_1..t).

    Unlike Viterbi (model.predict) which finds the globally optimal state
    sequence using ALL observations (a smoothing problem), this returns
    the filtered state at each time step using ONLY past and current
    observations (a filtering problem). No look-ahead.

    This is the textbook HMM forward algorithm with log-space normalization
    at each step to prevent numerical underflow on long sequences.

    Parameters:
        model: fitted GaussianHMM
        X:     (n_samples, n_features) observation matrix

    Returns:
        filtered_states: (n_samples,) array of most probable state at each t
        filtered_probs:  (n_samples, n_states) array of state probabilities
    """
    n_samples = len(X)
    n_states = model.n_components

    # Emission log-likelihoods: log P(x_t | state_j)
    log_emis = model._compute_log_likelihood(X)

    log_pi = np.log(model.startprob_)
    log_A = np.log(model.transmat_)

    # Filtered log-probabilities (normalized at each step)
    log_alpha = np.zeros((n_samples, n_states))

    # t = 0: alpha_0(j) = pi(j) * P(x_0 | j)
    log_alpha[0] = log_pi + log_emis[0]
    log_alpha[0] -= logsumexp(log_alpha[0])

    # t > 0: alpha_t(j) = P(x_t | j) * sum_i[ alpha_{t-1}(i) * A(i,j) ]
    for t in range(1, n_samples):
        for j in range(n_states):
            log_alpha[t, j] = log_emis[t, j] + logsumexp(
                log_alpha[t - 1] + log_A[:, j])
        log_alpha[t] -= logsumexp(log_alpha[t])

    filtered_states = np.argmax(log_alpha, axis=1)
    filtered_probs = np.exp(log_alpha)

    return filtered_states, filtered_probs


def fit_regime_model_frozen(spx_prices, bb_oas, rate_10y, vix_values=None):
    """
    Fit HMM on training period, then classify forward with causal filter.

    Step 1: Prepare features for ALL days (z-scores use expanding window,
            so no look-ahead in feature construction)
    Step 2: Fit HMM on first REGIME_MIN_DAYS observations only
    Step 3: Rank states once using the trained model's state means
    Step 4: Classify ALL days using causal forward filter (not Viterbi)
            Day t's regime uses only observations up to day t.

    Parameters:
        spx_prices: Series of daily SPX close prices
        bb_oas:     Series of daily BB OAS values
        rate_10y:   Series of daily 10Y Treasury rate
        vix_values: Series of daily VIX (optional, adds as 5th feature)

    Returns:
        regime_states: array of regime labels (0=Calm, 1=Caution, 2=Crisis)
        regime_dates:  DatetimeIndex of all valid dates
        model:         the frozen HMM model
        regime_stats:  dict with per-regime statistics
    """
    df, z_cols = _prepare_features(spx_prices, bb_oas, rate_10y, vix_values)
    n_days = len(df)

    print(f"  Training period: first {REGIME_MIN_DAYS} days "
          f"({df.index[0].date()} to {df.index[min(REGIME_MIN_DAYS-1, n_days-1)].date()})")
    print(f"  Prediction period: remaining {n_days - REGIME_MIN_DAYS} days")

    # Step 1: Train on first REGIME_MIN_DAYS
    X_train = df[z_cols].iloc[:REGIME_MIN_DAYS].values

    model = GaussianHMM(n_components=REGIME_N_STATES, covariance_type="full",
                         n_iter=REGIME_HMM_ITER, random_state=REGIME_RANDOM_STATE)
    model.fit(X_train)

    # Step 2: Rank states once (frozen ranking)
    label_map = _rank_states(model)

    print(f"  State means (Vol_Z + OAS_Z stress score):")
    state_means = model.means_
    stress_scores = state_means[:, 1] + state_means[:, 2]
    for raw_state in range(REGIME_N_STATES):
        mapped = label_map[raw_state]
        label = {0: 'Calm', 1: 'Caution', 2: 'Crisis'}[mapped]
        print(f"    HMM State {raw_state} -> {label} "
              f"(stress={stress_scores[raw_state]:.3f})")

    # Step 3: Classify ALL days using causal forward filter
    # (replaces model.predict which uses Viterbi — a non-causal smoother)
    X_all = df[z_cols].values
    raw_states, filtered_probs = _forward_filter(model, X_all)

    # Also run Viterbi for comparison reporting
    viterbi_states = model.predict(X_all)

    # Step 4: Map to standardized labels
    regime_states = np.array([label_map[s] for s in raw_states])
    viterbi_mapped = np.array([label_map[s] for s in viterbi_states])

    # Report forward vs Viterbi agreement
    n_agree = (regime_states == viterbi_mapped).sum()
    n_differ = n_days - n_agree
    print(f"\n  Forward filter vs Viterbi: {n_agree}/{n_days} agree "
          f"({n_differ} days differ, {n_differ/n_days*100:.1f}%)")

    # Compute statistics
    regime_stats = _compute_regime_stats(df, regime_states, vix_values is not None)

    return regime_states, df.index, model, regime_stats


def _compute_regime_stats(df, regime_states, has_vix):
    """Compute per-regime statistics."""
    regime_stats = {}
    labels = {0: 'Calm', 1: 'Caution', 2: 'Crisis'}
    for state_id, label in labels.items():
        mask = regime_states == state_id
        rets = df['Returns'].values[mask]
        if len(rets) > 0:
            regime_stats[label] = {
                'count': int(mask.sum()),
                'pct_of_days': mask.sum() / len(mask) * 100,
                'ann_return': rets.mean() * 252,
                'ann_vol': rets.std() * np.sqrt(252),
                'sharpe': (rets.mean() * 252) / (rets.std() * np.sqrt(252))
                          if rets.std() > 0 else 0,
                'avg_vix': df['VIX'].values[mask].mean() if has_vix else np.nan,
                'avg_oas': df['BB_OAS'].values[mask].mean(),
            }
    return regime_stats


def get_regime_series(regime_states, regime_dates, target_dates):
    """
    Align regime states to a target date index.
    Forward-fills regime for any dates not in the HMM output.
    """
    regime_series = pd.Series(regime_states, index=regime_dates, name='Regime')
    aligned = regime_series.reindex(target_dates, method='ffill')
    aligned = aligned.fillna(0).astype(int)
    return aligned.values
