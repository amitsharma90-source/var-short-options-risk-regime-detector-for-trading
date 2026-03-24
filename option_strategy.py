# -*- coding: utf-8 -*-
"""
=============================================================================
Module: option_strategy.py — Weekly Short Call Strategy Simulator
=============================================================================
Simulates systematic weekly short call selling on SPX.
Supports regime-aware contract sizing and delta targeting.
=============================================================================
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from option_pricer import bs_call_price, bs_greeks, find_strike_for_delta
from config import (SPX_MULTIPLIER, PREMIUM_CAPTURE_EXIT, STOP_LOSS_MULTIPLE,
                    OPTION_CONTRACTS, TARGET_DELTA,
                    REGIME_OPTION_CONTRACTS, REGIME_OPTION_DELTA)


def simulate_weekly_short_calls(dates, spx_prices, vix_values, rf_rates,
                                 regime_states=None):
    """
    Simulate a systematic weekly short call strategy on SPX.

    Rules:
    - Every Monday (or next trading day), sell SPX weekly calls
    - Strike chosen to give delta ~target_delta
    - Options expire the following Friday (5 calendar days)
    - Positions held to expiry unless stop-loss or premium-capture triggers
    - If regime_states provided, adjust contracts and delta per regime

    Parameters:
        dates:         DatetimeIndex of trading days
        spx_prices:    array of daily SPX close prices
        vix_values:    array of daily VIX values (annualized %)
        rf_rates:      array of daily 3M risk-free rates (%)
        regime_states: array of regime labels (0=Calm, 1=Caution, 2=Crisis) or None

    Returns:
        option_pnl:    Series of daily option P&L
        greeks_df:     DataFrame with daily portfolio-level greeks
        trades_df:     DataFrame logging each trade entry/exit
    """
    n_days = len(dates)
    option_pnl = np.zeros(n_days)
    greeks_records = []
    trades_log = []
    active_positions = []

    for i in range(n_days):
        date = dates[i]
        S = spx_prices[i]
        sigma = vix_values[i] / 100.0
        r = rf_rates[i] / 100.0

        if np.isnan(S) or np.isnan(sigma) or np.isnan(r):
            greeks_records.append(_empty_greeks(date))
            continue

        # Determine regime-aware parameters
        if regime_states is not None and i < len(regime_states):
            regime = int(regime_states[i])
            n_contracts = REGIME_OPTION_CONTRACTS.get(regime, OPTION_CONTRACTS)
            target_delta = REGIME_OPTION_DELTA.get(regime, TARGET_DELTA)
        else:
            n_contracts = OPTION_CONTRACTS
            target_delta = TARGET_DELTA

        # ── Monday: sell new calls ──
        if date.weekday() == 0 and n_contracts > 0:
            days_to_friday = 4
            expiry_date = date + timedelta(days=days_to_friday)
            T = days_to_friday / 365.0
            K = find_strike_for_delta(S, T, r, sigma, target_delta)
            entry_price = bs_call_price(S, K, T, r, sigma)

            if entry_price > 0.01:
                position = {
                    'entry_date': date,
                    'expiry_date': expiry_date,
                    'strike': K,
                    'entry_price': entry_price,
                    'prev_price': entry_price,
                    'n_contracts': n_contracts,
                    'premium_received': entry_price * n_contracts * SPX_MULTIPLIER,
                    'active': True,
                }
                active_positions.append(position)

                greeks_at_entry = bs_greeks(S, K, T, r, sigma)
                trades_log.append({
                    'Entry_Date': date, 'Expiry_Date': expiry_date,
                    'Strike': K, 'SPX_at_Entry': S,
                    'Entry_Price': entry_price,
                    'Contracts': n_contracts,
                    'Premium_Received': position['premium_received'],
                    'Delta_at_Entry': greeks_at_entry['delta'],
                    'VIX_at_Entry': vix_values[i],
                    'Regime': regime_states[i] if regime_states is not None else None,
                })

        # ── Reprice all active positions ──
        daily_pnl = 0.0
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        total_short_contracts = 0
        positions_to_deactivate = []

        for j, pos in enumerate(active_positions):
            if not pos['active']:
                continue

            nc = pos['n_contracts']
            days_left = (pos['expiry_date'] - date).days
            T = max(days_left, 0) / 365.0

            # Expiry
            if days_left <= 0:
                intrinsic = max(S - pos['strike'], 0.0)
                daily_pnl += (pos['prev_price'] - intrinsic) * nc * SPX_MULTIPLIER
                positions_to_deactivate.append(j)
                _log_exit(trades_log, pos, date, intrinsic, 'Expiry', nc, vix_values[i], spx_prices[i])
                continue

            # Reprice
            greeks = bs_greeks(S, pos['strike'], T, r, sigma)
            current_price = greeks['price']
            daily_pnl += (pos['prev_price'] - current_price) * nc * SPX_MULTIPLIER

            # Risk management checks
            cumulative_loss = (current_price - pos['entry_price']) * nc * SPX_MULTIPLIER
            premium_captured = 1.0 - (current_price / pos['entry_price']) \
                if pos['entry_price'] > 0 else 0

            exit_reason = None
            if cumulative_loss > STOP_LOSS_MULTIPLE * pos['premium_received']:
                exit_reason = 'Stop-Loss'
            elif premium_captured >= PREMIUM_CAPTURE_EXIT and days_left > 1:
                exit_reason = 'Premium-Capture'

            if exit_reason:
                positions_to_deactivate.append(j)
                _log_exit(trades_log, pos, date, current_price, exit_reason, nc, vix_values[i], spx_prices[i])
            else:
                total_delta += -greeks['delta'] * nc * SPX_MULTIPLIER
                total_gamma += -greeks['gamma'] * nc * SPX_MULTIPLIER
                total_theta += -greeks['theta'] * nc * SPX_MULTIPLIER
                total_vega += -greeks['vega'] * nc * SPX_MULTIPLIER
                total_short_contracts += nc

            pos['prev_price'] = current_price

        for j in positions_to_deactivate:
            active_positions[j]['active'] = False

        option_pnl[i] = daily_pnl
        greeks_records.append({
            'Date': date,
            'Portfolio_Delta': total_delta,
            'Portfolio_Gamma': total_gamma,
            'Portfolio_Theta': total_theta,
            'Portfolio_Vega': total_vega,
            'Num_Active_Positions': sum(1 for p in active_positions if p['active']),
            'Total_Short_Contracts': total_short_contracts,
        })

    return (pd.Series(option_pnl, index=dates),
            pd.DataFrame(greeks_records),
            pd.DataFrame(trades_log))


def _empty_greeks(date):
    """Return empty greeks record for a date with missing data."""
    return {
        'Date': date, 'Portfolio_Delta': 0, 'Portfolio_Gamma': 0,
        'Portfolio_Theta': 0, 'Portfolio_Vega': 0,
        'Num_Active_Positions': 0, 'Total_Short_Contracts': 0,
    }


def _log_exit(trades_log, pos, date, exit_price, reason, n_contracts, vix_at_exit=None, spx_at_exit=None):
    """Update the trades log with exit information."""
    for tl in trades_log:
        if (tl['Entry_Date'] == pos['entry_date'] and tl['Strike'] == pos['strike']):
            tl['Exit_Date'] = date
            tl['Exit_Price'] = exit_price
            tl['Exit_Reason'] = reason
            tl['Realized_PnL'] = ((pos['entry_price'] - exit_price)
                                   * n_contracts * SPX_MULTIPLIER)
            if vix_at_exit is not None:
                tl['VIX_at_Exit'] = vix_at_exit
            tl['SPX_at_Exit'] = spx_at_exit
            break