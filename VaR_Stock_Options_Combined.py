# -*- coding: utf-8 -*-
"""
=============================================================================
VaR_Stock_Options_Combined.py — Main Orchestrator (v09)
=============================================================================
Combined stock + short call options VaR with regime-scaled GARCH.
No stock-only comparison — all analysis is on the combined portfolio.
=============================================================================
"""

import numpy as np
import pandas as pd
import time as timer
import warnings
warnings.filterwarnings('ignore')

from config import *
from data_loader import load_all_data, build_stock_pnl
from option_pricer import price_current_options
from option_strategy import simulate_weekly_short_calls
from var_engine import garch_scale_window, compute_time_weighted_var, kupiec_test, christoffersen_test
from regime_detector import fit_regime_model_frozen, get_regime_series
from risk_alerts import generate_risk_alerts


def run_var_model(prices_path=None, holdings_path=None, output_dir=None):

    prices_path = prices_path or PRICES_PATH
    holdings_path = holdings_path or HOLDINGS_PATH
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════
    # 1. LOAD DATA
    # ══════════════════════════════════════════════════════════════════
    print("Loading data from Excel files...")
    data = load_all_data(prices_path, holdings_path)
    dates = data['stock_prices'].index
    print(f"  Date range: {dates[0].date()} to {dates[-1].date()} ({len(dates)} days)")
    print(f"  Stocks: {len(data['stock_prices'].columns)} | "
          f"SPX: {data['spx_prices'].min():.0f}-{data['spx_prices'].max():.0f} | "
          f"VIX: {data['vix_values'].min():.1f}-{data['vix_values'].max():.1f}")

    # ══════════════════════════════════════════════════════════════════
    # 2. STOCK PnL
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("BUILDING STOCK PnL")
    print(f"{'='*60}")
    raw_stock_pnl, total_stock_value, _ = build_stock_pnl(
        data['stock_prices'], data['stock_holdings'])
    total_portfolio_value = total_stock_value + data['cash_value'] + data['bond_value']
    print(f"  Stock market value:   ${total_stock_value:>15,.2f}")
    print(f"  Cash:                 ${data['cash_value']:>15,.2f}")
    print(f"  Bonds:                ${data['bond_value']:>15,.2f}")
    print(f"  Total portfolio:      ${total_portfolio_value:>15,.2f}")
    print(f"  PnL observations:     {len(raw_stock_pnl)}")

    # ══════════════════════════════════════════════════════════════════
    # 3. REGIME MODEL (Expanding — no look-ahead bias)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("FITTING 3-STATE MARKET REGIME MODEL (FROZEN)")
    print(f"{'='*60}")
    print(f"  Features: SPX returns, 21d vol, BB OAS change, 10Y rate change, VIX")
    print(f"  Training period: first {REGIME_MIN_DAYS} days")
    t_regime = timer.time()
    regime_states, regime_dates, hmm_model, regime_stats = fit_regime_model_frozen(
        data['spx_prices'], data['bb_oas'], data['rate_10y'], data['vix_values'])
    print(f"  Total time: {timer.time() - t_regime:.1f}s")

    # Print regime stats
    labels_map = {0: 'Calm', 1: 'Caution', 2: 'Crisis'}
    print(f"\n  {'Regime':<12} {'Days':>6} {'% Time':>8} {'Ann Ret':>10} "
          f"{'Ann Vol':>10} {'Sharpe':>8} {'Avg VIX':>8} {'Avg OAS':>8}")
    print(f"  {'-'*76}")
    for label, s in regime_stats.items():
        print(f"  {label:<12} {s['count']:>6} {s['pct_of_days']:>7.1f}% "
              f"{s['ann_return']:>+9.1%} {s['ann_vol']:>9.1%} "
              f"{s['sharpe']:>8.2f} {s['avg_vix']:>7.1f} {s['avg_oas']:>7.2f}")

    sim_dates = raw_stock_pnl.index
    regime_aligned = get_regime_series(regime_states, regime_dates, sim_dates)
    current_regime_id = regime_aligned[-1]
    current_regime_label = labels_map[current_regime_id]
    print(f"\n  Current regime: {current_regime_label} "
          f"(VaR x{REGIME_VAR_MULTIPLIER[current_regime_id]}, "
          f"contracts: {REGIME_OPTION_CONTRACTS[current_regime_id]})")

    # ══════════════════════════════════════════════════════════════════
    # 4. OPTION STRATEGY (Regime-Aware)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("SIMULATING SHORT CALL STRATEGY (REGIME-AWARE)")
    print(f"{'='*60}")
    print(f"  Calm: {REGIME_OPTION_CONTRACTS[0]}c @ d{REGIME_OPTION_DELTA[0]} | "
          f"Caution: {REGIME_OPTION_CONTRACTS[1]}c @ d{REGIME_OPTION_DELTA[1]} | "
          f"Crisis: no selling")
    t_opt = timer.time()
    sim_spx = data['spx_prices'].reindex(sim_dates).values
    sim_vix = data['vix_values'].reindex(sim_dates).values
    sim_rf = data['rf_rates'].reindex(sim_dates).values

    # Start option simulation AFTER regime training period to avoid look-ahead bias.
    # Days 0..REGIME_MIN_DAYS-1 were used to train the HMM — regime labels for those
    # days are in-sample predictions and must not drive trading decisions.
    opt_start = REGIME_MIN_DAYS
    print(f"  Option backtest starts at day {opt_start} "
          f"({sim_dates[opt_start].date()}) — "
          f"first {opt_start} days are regime training period")
    option_pnl, greeks_df, trades_df = simulate_weekly_short_calls(
        sim_dates[opt_start:], sim_spx[opt_start:], sim_vix[opt_start:],
        sim_rf[opt_start:], regime_states=regime_aligned[opt_start:])
    print(f"  Completed in {timer.time() - t_opt:.1f}s")

    # Strategy summary
    if not trades_df.empty:
        total_premium = trades_df['Premium_Received'].sum()
        print(f"  Total trades: {len(trades_df)}")
        print(f"  Avg premium/trade:      ${total_premium/len(trades_df):>12,.2f}")
        if 'Realized_PnL' in trades_df.columns:
            realized = trades_df.dropna(subset=['Realized_PnL'])
            print(f"  Total premium received: ${total_premium:>12,.2f}")
            print(f"  Total realized P&L:     ${realized['Realized_PnL'].sum():>12,.2f}")
            n_win = (realized['Realized_PnL'] > 0).sum()
            n_loss = (realized['Realized_PnL'] <= 0).sum()
            print(f"  Win/Loss:               {n_win}/{n_loss}")
            if 'Exit_Reason' in realized.columns:
                print(f"  Exit reasons:")
                for reason, count in realized['Exit_Reason'].value_counts().items():
                    print(f"    {reason}: {count}")

        if 'Regime' in trades_df.columns:
            print(f"\n  Trades by regime:")
            for r_id, r_label in labels_map.items():
                r_trades = trades_df[trades_df['Regime'] == r_id]
                if not r_trades.empty:
                    print(f"    {r_label}: {len(r_trades)} trades, "
                          f"avg premium ${r_trades['Premium_Received'].mean():,.0f}")

    # ══════════════════════════════════════════════════════════════════
    # 5. COMBINED PnL
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("COMBINED PORTFOLIO PnL (STOCK + OPTIONS)")
    print(f"{'='*60}")
    common_idx = raw_stock_pnl.index.intersection(option_pnl.index)
    stock_pnl_aligned = raw_stock_pnl.reindex(common_idx)
    option_pnl_aligned = option_pnl.reindex(common_idx)
    combined_pnl = stock_pnl_aligned + option_pnl_aligned
    print(f"  Combined period: {common_idx[0].date()} to {common_idx[-1].date()}")
    print(f"  (First {opt_start} days excluded — used for regime training, no option simulation)")
    print(f"  Observations: {len(combined_pnl)}")
    print(f"  Mean:     ${combined_pnl.mean():>12,.2f}/day")
    print(f"  Std Dev:  ${combined_pnl.std():>12,.2f}")
    print(f"  Skewness: {combined_pnl.skew():>12.3f}")
    print(f"  Kurtosis: {combined_pnl.kurtosis():>12.3f}")
    print(f"  Min:      ${combined_pnl.min():>12,.2f}")
    print(f"  Max:      ${combined_pnl.max():>12,.2f}")

    # ══════════════════════════════════════════════════════════════════
    # 6. CURRENT OPTION SNAPSHOT
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("CURRENT OPTION POSITIONS")
    print(f"{'='*60}")
    current_opts = pd.DataFrame()
    if not data['option_holdings'].empty:
        as_of = dates[-1]
        current_opts = price_current_options(
            data['option_holdings'], data['spx_prices'].iloc[-1],
            data['vix_values'].iloc[-1], data['rf_rates'].iloc[-1], as_of)
        if not current_opts.empty:
            print(f"  As of {as_of.date()}, SPX={data['spx_prices'].iloc[-1]:.2f}")
            for _, opt in current_opts.iterrows():
                print(f"  {opt['Security']}: K={opt['Strike']:.0f} DTE={opt['Days_to_Expiry']}d "
                      f"Qty={opt['Quantity']:.0f} BS=${opt['BS_Price']:.2f} Mkt=${opt['Market_Price']:.2f}")
            print(f"  Totals: D=${current_opts['Position_Delta'].sum():,.0f} "
                  f"G=${current_opts['Position_Gamma'].sum():,.2f} "
                  f"T=${current_opts['Position_Theta'].sum():,.0f}/day "
                  f"V=${current_opts['Position_Vega'].sum():,.0f}/1%vol")

    # ══════════════════════════════════════════════════════════════════
    # 7. FORWARD-LOOKING VaR (Regime-Scaled)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"VaR ESTIMATE — REGIME: {current_regime_label}")
    print(f"{'='*60}")
    var_combined = cvar_combined = np.nan
    if len(combined_pnl) >= LOOKBACK:
        recent = combined_pnl.iloc[-LOOKBACK:].values
        scaled, fwd_vol = garch_scale_window(recent)
        r_mult = REGIME_VAR_MULTIPLIER[current_regime_id]
        if r_mult != 1.0:
            scaled = scaled * r_mult
            fwd_vol = fwd_vol * r_mult
        var_combined, cvar_combined = compute_time_weighted_var(
            scaled, LAMBDA_DECAY, CONFIDENCE_LEVEL)

        print(f"  GARCH Forecast Vol: ${fwd_vol:>12,.2f}/day")
        print(f"  Regime multiplier:  {r_mult}x")
        print(f"  1-Day VaR  (95%):   ${var_combined:>12,.2f}")
        print(f"  1-Day CVaR (95%):   ${cvar_combined:>12,.2f}")
        print(f"  VaR % of portfolio: {var_combined/total_portfolio_value*100:.2f}%")

    # ══════════════════════════════════════════════════════════════════
    # 8. BACKTESTING
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("BACKTESTING — Regime-Scaled Rolling GARCH")
    print(f"{'='*60}")
    backtest_results = []
    combined_values = combined_pnl.values
    combined_index = combined_pnl.index
    regime_for_bt = regime_aligned[:len(combined_values)]
    n_total = len(combined_values)
    n_backtest = n_total - LOOKBACK - 1
    print(f"  Running {n_backtest} rolling GARCH fits...")
    t_start = timer.time()

    for i in range(LOOKBACK, n_total - 1):
        window = combined_values[i - LOOKBACK:i]
        scaled, fvol = garch_scale_window(window)
        r_m = REGIME_VAR_MULTIPLIER.get(int(regime_for_bt[i]), 1.0)
        VaR, CVaR = compute_time_weighted_var(
            scaled * r_m, LAMBDA_DECAY, CONFIDENCE_LEVEL)

        actual = combined_values[i + 1]
        backtest_results.append({
            'Date': combined_index[i + 1],
            'VaR_95': VaR,
            'CVaR_95': CVaR,
            'Actual_PnL': actual,
            'Exception': (-actual) > VaR,
            'GARCH_Forecast_Vol': fvol,
            'Regime': int(regime_for_bt[i]),
            'Regime_Multiplier': r_m,
        })
        done = i - LOOKBACK + 1
        if done % 100 == 0:
            print(f"    {done}/{n_backtest} done ({timer.time() - t_start:.1f}s)")

    elapsed = timer.time() - t_start
    print(f"  Completed {n_backtest} fits in {elapsed:.1f}s "
          f"({elapsed/n_backtest*1000:.0f}ms/fit)")
    backtest_df = pd.DataFrame(backtest_results)

    # ══════════════════════════════════════════════════════════════════
    # 9. STATISTICAL TESTS
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("BACKTEST RESULTS")
    print(f"{'='*60}")
    n_obs = len(backtest_df)
    n_exc = int(backtest_df['Exception'].sum())
    expected = (1 - CONFIDENCE_LEVEL) * n_obs
    LR_uc, pval_uc = kupiec_test(n_obs, n_exc, CONFIDENCE_LEVEL)
    LR_ind, pval_ind = christoffersen_test(backtest_df['Exception'])

    print(f"  Period: {backtest_df['Date'].iloc[0].date()} to "
          f"{backtest_df['Date'].iloc[-1].date()}")
    print(f"  Observations: {n_obs}")
    print(f"  Exceptions: {n_exc} (expected: {expected:.0f})")
    print(f"  Exception Rate: {n_exc/n_obs*100:.1f}% (target: 5.0%)")
    print(f"  Kupiec:         LR={LR_uc:.3f}, p={pval_uc:.3f} "
          f"{'PASS' if pval_uc > 0.05 else 'FAIL'}")
    print(f"  Christoffersen: LR={LR_ind:.3f}, p={pval_ind:.3f} "
          f"{'PASS' if pval_ind > 0.05 else 'FAIL'}")

    # Exceptions by regime
    print(f"\n  Exceptions by regime:")
    for r_id, r_label in labels_map.items():
        r_mask = backtest_df['Regime'] == r_id
        r_exc = backtest_df.loc[r_mask, 'Exception'].sum()
        r_total = r_mask.sum()
        if r_total > 0:
            print(f"    {r_label}: {r_exc}/{r_total} ({r_exc/r_total*100:.1f}%)")

    # ══════════════════════════════════════════════════════════════════
    # 10. RISK ALERTS
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("RISK ALERTS")
    print(f"{'='*60}")
    alerts_df = generate_risk_alerts(greeks_df, sim_spx[opt_start:], total_portfolio_value,
                                      regime_states=regime_aligned[opt_start:])
    if not alerts_df.empty:
        for sev in ['HIGH', 'MEDIUM']:
            c = (alerts_df['Severity'] == sev).sum()
            if c > 0:
                print(f"  {sev}: {c}")
        regime_shifts = alerts_df[alerts_df['Alert_Type'] == 'REGIME_SHIFT']
        if not regime_shifts.empty:
            print(f"\n  Recent regime transitions:")
            for _, a in regime_shifts.tail(5).iterrows():
                print(f"    {a['Date'].date()} — {a['Message']}")
    else:
        print("  No alerts.")

    # ══════════════════════════════════════════════════════════════════
    # 11. CHARTS
    # ══════════════════════════════════════════════════════════════════
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(18, 22))

    # 1a: VaR Backtest
    ax = axes[0, 0]
    ax.plot(backtest_df['Date'], backtest_df['Actual_PnL'],
            color='steelblue', linewidth=0.7, alpha=0.7, label='Actual PnL')
    ax.plot(backtest_df['Date'], -backtest_df['VaR_95'],
            color='orange', linewidth=1.5, linestyle='--', label='VaR 95%')
    ax.plot(backtest_df['Date'], -backtest_df['CVaR_95'],
            color='red', linewidth=1.0, linestyle=':', alpha=0.7, label='CVaR 95%')
    exc_mask = backtest_df['Exception']
    ax.scatter(backtest_df.loc[exc_mask, 'Date'],
               backtest_df.loc[exc_mask, 'Actual_PnL'],
               color='red', s=30, zorder=5, label=f'Exceptions ({n_exc})')
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax.set_title('Combined Portfolio VaR Backtest (Regime-Scaled)', fontsize=12)
    ax.set_ylabel('P&L ($)')
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 1b: Regime Overlay on SPX
    ax = axes[0, 1]
    ax.plot(sim_dates, sim_spx, color='black', linewidth=0.8, label='SPX')
    y_min, y_max = np.nanmin(sim_spx), np.nanmax(sim_spx)
    colors = {0: 'green', 1: 'orange', 2: 'red'}
    for s_id in [0, 1, 2]:
        mask = regime_aligned == s_id
        ax.fill_between(sim_dates, y_min, y_max, where=mask,
                         color=colors[s_id], alpha=0.25, label=labels_map[s_id])
    ax.set_title('Market Regime Overlay on SPX', fontsize=12)
    ax.set_ylabel('SPX Level')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2a: GARCH Forecast Vol
    ax = axes[1, 0]
    ax.plot(backtest_df['Date'], backtest_df['GARCH_Forecast_Vol'],
            color='purple', linewidth=1.0, label='GARCH Forecast Vol')
    ax.set_title('Rolling GARCH Forecast Volatility', fontsize=12)
    ax.set_ylabel('Forecast Vol ($/day)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2b: Regime Multiplier Over Time
    ax = axes[1, 1]
    ax.plot(backtest_df['Date'], backtest_df['Regime_Multiplier'],
            color='darkred', linewidth=1.0, label='Regime Multiplier')
    ax.set_title('Regime VaR Multiplier Over Time', fontsize=12)
    ax.set_ylabel('Multiplier')
    ax.set_ylim(0.8, max(REGIME_VAR_MULTIPLIER.values()) + 0.3)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3a: Portfolio Greeks — Delta and Gamma
    ax = axes[2, 0]
    greeks_bt = greeks_df.set_index('Date')
    ax.plot(greeks_bt.index, greeks_bt['Portfolio_Delta'], label='Delta ($)',
            linewidth=0.8, color='steelblue')
    ax.set_title('Portfolio Option Delta', fontsize=12)
    ax.set_ylabel('Delta ($)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(greeks_bt.index, greeks_bt['Portfolio_Gamma'],
             color='red', linewidth=0.6, alpha=0.6, label='Gamma')
    ax2.set_ylabel('Gamma', color='red')
    ax2.legend(loc='lower right', fontsize=8)

    # 3b: Theta Income
    ax = axes[2, 1]
    ax.plot(greeks_bt.index, greeks_bt['Portfolio_Theta'],
            color='green', linewidth=0.8, label='Theta ($/day)')
    ax.set_title('Daily Theta Income', fontsize=12)
    ax.set_ylabel('Theta ($/day)')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 4a: Cumulative PnL Decomposition
    ax = axes[3, 0]
    ax.plot(stock_pnl_aligned.cumsum().index, stock_pnl_aligned.cumsum(),
            label='Stock', color='steelblue', linewidth=1.0)
    ax.plot(option_pnl_aligned.cumsum().index, option_pnl_aligned.cumsum(),
            label='Option', color='green', linewidth=1.0)
    ax.plot(combined_pnl.cumsum().index, combined_pnl.cumsum(),
            label='Combined', color='darkred', linewidth=1.2)
    ax.set_title('Cumulative P&L Decomposition', fontsize=12)
    ax.set_ylabel('Cumulative P&L ($)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 4b: PnL Distribution
    ax = axes[3, 1]
    ax.hist(combined_pnl.values, bins=60, alpha=0.6, density=True,
            color='darkred', label='Combined PnL')
    if not np.isnan(var_combined):
        ax.axvline(-var_combined, color='orange', linestyle='--', linewidth=2,
                   label=f'VaR 95% (${var_combined:,.0f})')
    if not np.isnan(cvar_combined):
        ax.axvline(-cvar_combined, color='red', linestyle=':', linewidth=1.5,
                   label=f'CVaR 95% (${cvar_combined:,.0f})')
    ax.set_title('Combined P&L Distribution', fontsize=12)
    ax.set_xlabel('P&L ($)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'var_backtest_stock_options.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ══════════════════════════════════════════════════════════════════
    # 12. EXPORT
    # ══════════════════════════════════════════════════════════════════
    backtest_df.to_excel(output_dir / 'var_backtest_combined.xlsx', index=False)
    greeks_df.to_excel(output_dir / 'portfolio_greeks_daily.xlsx', index=False)
    if not trades_df.empty:
        trades_df.to_excel(output_dir / 'option_trades_log.xlsx', index=False)
    if not alerts_df.empty:
        alerts_df.to_excel(output_dir / 'risk_alerts.xlsx', index=False)
    if not current_opts.empty:
        current_opts.to_excel(output_dir / 'current_option_greeks.xlsx', index=False)
    exc_df = backtest_df[backtest_df['Exception']]
    if not exc_df.empty:
        exc_df.to_excel(output_dir / 'exceptions.xlsx', index=False)
    pd.DataFrame({'Date': sim_dates, 'Regime': regime_aligned,
                   'Label': [labels_map[r] for r in regime_aligned]}
                  ).to_excel(output_dir / 'regime_history.xlsx', index=False)

    print(f"\nAll outputs exported to: {output_dir.absolute()}")
    return {'backtest_df': backtest_df, 'combined_pnl': combined_pnl,
            'greeks_df': greeks_df, 'trades_df': trades_df,
            'alerts_df': alerts_df, 'regime_states': regime_aligned,
            'regime_stats': regime_stats}


if __name__ == "__main__":
    results = run_var_model()
