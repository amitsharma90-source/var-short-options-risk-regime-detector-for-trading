# -*- coding: utf-8 -*-
"""
=============================================================================
Module: risk_alerts.py — Risk Management Alert Engine
=============================================================================
Generates alerts based on portfolio greeks thresholds and regime transitions.
=============================================================================
"""

import pandas as pd
import numpy as np
from config import GAMMA_THRESHOLD_PCT, DELTA_THRESHOLD_PCT


def generate_risk_alerts(greeks_df, spx_prices, total_portfolio_value,
                          regime_states=None):
    """
    Generate risk management alerts based on portfolio greeks and regime.

    Rules:
    1. HIGH_GAMMA: gamma loss on 1% SPX move > threshold % of portfolio
    2. LOW_THETA_VS_GAMMA: earning theta but gamma risk disproportionately high
    3. HIGH_DELTA: delta notional exposure > threshold % of portfolio
    4. REGIME_SHIFT: transition from Calm to Caution or Crisis
    """
    alerts = []

    prev_regime = None

    for i in range(len(greeks_df)):
        row = greeks_df.iloc[i]
        S = spx_prices[i] if i < len(spx_prices) else np.nan
        if np.isnan(S):
            continue

        date = row['Date']
        gamma = row['Portfolio_Gamma']
        theta = row['Portfolio_Theta']
        delta = row['Portfolio_Delta']

        # 1. Gamma risk
        gamma_loss_1pct = abs(0.5 * gamma * (0.01 * S)**2)
        if gamma_loss_1pct > GAMMA_THRESHOLD_PCT * total_portfolio_value:
            alerts.append({
                'Date': date, 'Alert_Type': 'HIGH_GAMMA',
                'Message': (f'Gamma loss on 1% SPX move: ${gamma_loss_1pct:,.0f} '
                            f'({gamma_loss_1pct/total_portfolio_value*100:.2f}% of portfolio)'),
                'Severity': 'HIGH'
            })

        # 2. Theta/Gamma ratio
        if gamma != 0 and theta > 0:
            theta_gamma_ratio = abs(theta / gamma) if gamma != 0 else np.inf
            if theta_gamma_ratio < S * 0.001:
                alerts.append({
                    'Date': date, 'Alert_Type': 'LOW_THETA_VS_GAMMA',
                    'Message': (f'Theta/Gamma ratio too low: {theta_gamma_ratio:.1f}. '
                                f'Theta=${theta:,.0f}/day, consider closing.'),
                    'Severity': 'MEDIUM'
                })

        # 3. Large delta exposure
        delta_notional = abs(delta * S)
        if delta_notional > DELTA_THRESHOLD_PCT * total_portfolio_value:
            alerts.append({
                'Date': date, 'Alert_Type': 'HIGH_DELTA',
                'Message': (f'Option delta exposure: ${delta_notional:,.0f} '
                            f'({delta_notional/total_portfolio_value*100:.1f}% of portfolio)'),
                'Severity': 'MEDIUM'
            })

        # 4. Regime transition alerts
        if regime_states is not None and i < len(regime_states):
            current_regime = int(regime_states[i])
            if prev_regime is not None and current_regime > prev_regime:
                regime_labels = {0: 'Calm', 1: 'Caution', 2: 'Crisis'}
                severity = 'HIGH' if current_regime == 2 else 'MEDIUM'
                alerts.append({
                    'Date': date, 'Alert_Type': 'REGIME_SHIFT',
                    'Message': (f'Regime shifted: {regime_labels[prev_regime]} -> '
                                f'{regime_labels[current_regime]}. '),
                    'Severity': severity
                })
            prev_regime = current_regime

    return pd.DataFrame(alerts) if alerts else pd.DataFrame(
        columns=['Date', 'Alert_Type', 'Message', 'Severity'])
