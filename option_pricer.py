# -*- coding: utf-8 -*-
"""
=============================================================================
Module: option_pricer.py — Black-Scholes Pricing and Greeks
=============================================================================
European call option pricing, greeks, and strike selection.
All closed-form, no numerical approximation needed for weekly SPX options.
=============================================================================
"""

import numpy as np
from scipy.stats import norm


def bs_d1_d2(S, K, T, r, sigma):
    """Compute d1 and d2 for Black-Scholes formula."""
    if T <= 0 or sigma <= 0:
        return np.nan, np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes European call price."""
    if T <= 0:
        return max(S - K, 0.0)
    d1, d2 = bs_d1_d2(S, K, T, r, sigma)
    if np.isnan(d1):
        return max(S - K, 0.0)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_greeks(S, K, T, r, sigma):
    """
    Compute all greeks for a European call option.

    Returns dict:
        price  — BS call price
        delta  — dPrice/dS
        gamma  — d²Price/dS²
        theta  — dPrice/dt per calendar day
        vega   — dPrice/dσ per 1% vol move
    """
    if T <= 0:
        intrinsic = max(S - K, 0.0)
        return {
            'delta': 1.0 if S > K else 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'price': intrinsic
        }

    d1, d2 = bs_d1_d2(S, K, T, r, sigma)
    if np.isnan(d1):
        return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0,
                'price': max(S - K, 0.0)}

    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100.0

    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega,
            'price': price}


def find_strike_for_delta(S, T, r, sigma, target_delta=0.25):
    """
    Find the strike price that gives a call delta equal to target_delta.
    Closed-form inversion of the BS delta formula, rounded to SPX strike grid.
    """
    if T <= 0 or sigma <= 0:
        return S * 1.02  # fallback: 2% OTM

    d1_target = norm.ppf(target_delta)
    K = S * np.exp((r + 0.5 * sigma**2) * T - d1_target * sigma * np.sqrt(T))

    # Round to nearest 5 points (SPX strike grid)
    K = round(K / 5) * 5
    return max(K, S * 0.90)  # safety floor


def price_current_options(option_holdings, spx_latest, vix_latest, rf_latest,
                           as_of_date, spx_multiplier=100):
    """
    Price the actual current option positions from the holdings file.
    Returns a DataFrame with greeks and current valuations.
    """
    import pandas as pd

    if option_holdings.empty:
        return pd.DataFrame()

    results = []
    for _, row in option_holdings.iterrows():
        K = row['Strike Price']
        qty = row['Quantity']
        expiry = pd.to_datetime(row['Maturity'])
        days_left = (expiry - as_of_date).days
        T = max(days_left, 0) / 365.0
        sigma = vix_latest / 100.0
        r = rf_latest / 100.0

        greeks = bs_greeks(spx_latest, K, T, r, sigma)

        results.append({
            'Security': row['Security'],
            'Strike': K,
            'Expiry': expiry,
            'Days_to_Expiry': days_left,
            'Quantity': qty,
            'BS_Price': greeks['price'],
            'Market_Price': row.get('Option Price', np.nan),
            'Delta': greeks['delta'],
            'Gamma': greeks['gamma'],
            'Theta': greeks['theta'],
            'Vega': greeks['vega'],
            'Position_Delta': greeks['delta'] * qty * spx_multiplier,
            'Position_Gamma': greeks['gamma'] * qty * spx_multiplier,
            'Position_Theta': greeks['theta'] * qty * spx_multiplier,
            'Position_Vega': greeks['vega'] * qty * spx_multiplier,
            'Notional_Value': greeks['price'] * qty * spx_multiplier,
        })

    return pd.DataFrame(results)
