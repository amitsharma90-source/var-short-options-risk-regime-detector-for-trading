# -*- coding: utf-8 -*-
"""
=============================================================================
Module: var_engine.py — VaR Calculation Engine
=============================================================================
GARCH(1,1) volatility filtering, time-weighted VaR/CVaR,
and Kupiec/Christoffersen backtesting tests.
=============================================================================
"""

import numpy as np
from arch import arch_model
from scipy.stats import chi2
from config import GARCH_DIST, GARCH_ASYMMETRY


def garch_scale_window(raw_pnl_window):
    """
    Fit GARCH(1,1) on a raw PnL window and return scaled PnL.

    Normalizes to %-like scale for numerical stability, fits GARCH,
    then converts back. Scale ratio capped at 0.1x–10x.

    Returns (scaled_pnl, forecast_vol).
    """
    pnl_std = np.std(raw_pnl_window)
    if pnl_std < 1e-10:
        return raw_pnl_window.copy(), pnl_std

    normalized = raw_pnl_window / pnl_std * 100

    try:
        am = arch_model(normalized, vol='Garch', p=1, o=GARCH_ASYMMETRY, q=1,
                        mean='Zero', dist=GARCH_DIST)
        res = am.fit(disp='off', show_warning=False)

        cond_vol_pct = res.conditional_volatility
        forecast_vol_pct = np.sqrt(res.forecast(horizon=1).variance.iloc[-1, 0])

        cond_vol = cond_vol_pct / 100 * pnl_std
        forecast_vol = forecast_vol_pct / 100 * pnl_std

        scale_ratio = np.clip(forecast_vol / np.maximum(cond_vol, 1e-10), 0.1, 10.0)
        scaled_pnl = raw_pnl_window * scale_ratio

        return scaled_pnl, forecast_vol

    except Exception:
        return raw_pnl_window.copy(), pnl_std


def compute_time_weighted_var(pnl_window, lambda_decay, confidence_level):
    """
    Compute time-weighted VaR and CVaR with linear interpolation.
    Returns (VaR, CVaR) as positive numbers representing loss.
    """
    n = len(pnl_window)
    time_weights = np.array([(1 - lambda_decay) * lambda_decay**j
                             for j in range(n)])[::-1]
    time_weights /= time_weights.sum()

    sorted_idx = np.argsort(pnl_window)
    sorted_pnl = np.array(pnl_window)[sorted_idx]
    sorted_weights = time_weights[sorted_idx]
    cum_weights = np.cumsum(sorted_weights)

    alpha = 1 - confidence_level
    idx_right = min(np.searchsorted(cum_weights, alpha, side='left'), n - 1)
    idx_left = max(idx_right - 1, 0)

    if idx_left == idx_right or cum_weights[idx_right] == cum_weights[idx_left]:
        VaR = -sorted_pnl[idx_right]
    else:
        frac = (alpha - cum_weights[idx_left]) / (cum_weights[idx_right] - cum_weights[idx_left])
        var_pnl = sorted_pnl[idx_left] + frac * (sorted_pnl[idx_right] - sorted_pnl[idx_left])
        VaR = -var_pnl

    tail_mask = sorted_pnl < -VaR
    CVaR = -np.average(sorted_pnl[tail_mask], weights=sorted_weights[tail_mask]) \
        if tail_mask.any() else VaR

    return VaR, CVaR


def kupiec_test(n_obs, n_exceptions, confidence_level):
    """Kupiec unconditional coverage test."""
    p_hat = 1 - confidence_level
    fail_ratio = np.clip(n_exceptions / n_obs, 1e-10, 1 - 1e-10)
    LR = -2 * (np.log((1 - p_hat)**(n_obs - n_exceptions) * p_hat**n_exceptions) -
               np.log((1 - fail_ratio)**(n_obs - n_exceptions) * fail_ratio**n_exceptions))
    return LR, 1 - chi2.cdf(LR, df=1)


def christoffersen_test(exceptions_series):
    """Christoffersen independence test."""
    exc = exceptions_series.astype(int).values
    prev = np.roll(exc, 1)
    prev[0] = 0
    n00 = int(np.sum((prev == 0) & (exc == 0)))
    n01 = int(np.sum((prev == 0) & (exc == 1)))
    n10 = int(np.sum((prev == 1) & (exc == 0)))
    n11 = int(np.sum((prev == 1) & (exc == 1)))
    total_0, total_1 = n00 + n01, n10 + n11
    p01 = np.clip(n01 / total_0 if total_0 > 0 else 1e-4, 1e-10, 1 - 1e-10)
    p11 = np.clip(n11 / total_1 if total_1 > 0 else 1e-4, 1e-10, 1 - 1e-10)
    p_total = np.clip((n01 + n11) / (total_0 + total_1)
                       if (total_0 + total_1) > 0 else 1e-4, 1e-10, 1 - 1e-10)
    LR = -2 * (np.log((1 - p_total)**(n00 + n10) * p_total**(n01 + n11)) -
               np.log((1 - p01)**n00 * p01**n01 * (1 - p11)**n10 * p11**n11))
    return LR, 1 - chi2.cdf(LR, df=1)
