# -*- coding: utf-8 -*-
"""
=============================================================================
Module: data_loader.py — Data Loading and Preparation
=============================================================================
Reads all market data and holdings from Excel files.
No yfinance or external API calls.
=============================================================================
"""

import pandas as pd
import numpy as np


def load_all_data(prices_path, holdings_path):
    """
    Load all required data from the two Excel files.

    Returns dict with keys:
        stock_prices, spx_prices, vix_values, rf_rates, rate_10y,
        bb_oas, b_oas, stock_holdings, option_holdings,
        cash_value, bond_value
    """
    # ── Market data ──
    market = pd.read_excel(prices_path)
    market['Date'] = pd.to_datetime(market['Date'])
    market = market.sort_values('Date').reset_index(drop=True)

    non_stock_cols = ['Date', '1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y',
                      '10Y', '20Y', '30Y', 'AAA_OAS', 'AA_OAS', 'A_OAS',
                      'BBB_OAS', 'BB_OAS', 'B_OAS', 'VIXCLS', 'SPX_Close']
    stock_tickers = [c for c in market.columns if c not in non_stock_cols]

    stock_prices = market[['Date'] + stock_tickers].set_index('Date')
    spx_prices = market.set_index('Date')['SPX_Close']
    vix_values = market.set_index('Date')['VIXCLS']
    rf_rates = market.set_index('Date')['3M']
    rate_10y = market.set_index('Date')['10Y']
    bb_oas = market.set_index('Date')['BB_OAS']
    b_oas = market.set_index('Date')['B_OAS']

    # ── Holdings ──
    holdings = pd.read_excel(holdings_path, sheet_name='Holdings')

    # Stocks
    equity_mask = holdings['Asset class'].str.lower() == 'equity'
    stock_holdings = holdings[equity_mask][['Ticker', 'Quantity', 'Market value']].copy()
    stock_holdings = stock_holdings.rename(columns={'Quantity': 'Shares Held'})
    stock_holdings['Ticker'] = stock_holdings['Ticker'].str.replace('.', '_', regex=False)

    # Options (current snapshot)
    option_mask = holdings['Asset class'].str.lower() == 'options'
    option_holdings = holdings[option_mask].copy() if option_mask.any() else pd.DataFrame()

    # Cash and Bonds
    cash_mask = holdings['Asset class'].str.lower() == 'cash'
    cash_value = holdings[cash_mask]['Quantity'].sum()

    bond_mask = holdings['Asset class'].str.lower() == 'bond'
    bond_value = holdings[bond_mask]['Quantity'].sum()

    return {
        'stock_prices': stock_prices,
        'spx_prices': spx_prices,
        'vix_values': vix_values,
        'rf_rates': rf_rates,
        'rate_10y': rate_10y,
        'bb_oas': bb_oas,
        'b_oas': b_oas,
        'stock_holdings': stock_holdings,
        'option_holdings': option_holdings,
        'cash_value': cash_value,
        'bond_value': bond_value,
    }


def build_stock_pnl(stock_prices, stock_holdings):
    """
    Build raw stock portfolio PnL from log returns × market values.

    Returns:
        raw_stock_pnl: Series indexed by date
        total_stock_value: float (latest market value)
        latest_market_value: Series (per-ticker market values)
    """
    returns = np.log(stock_prices / stock_prices.shift(1)).dropna()

    shares = stock_holdings.set_index('Ticker')['Shares Held']
    common_tickers = returns.columns.intersection(shares.index)
    returns = returns[common_tickers]
    shares = shares[common_tickers]

    latest_prices = stock_prices[common_tickers].iloc[-1]
    latest_market_value = latest_prices * shares
    total_stock_value = latest_market_value.sum()

    raw_stock_pnl = returns.dot(latest_market_value)

    return raw_stock_pnl, total_stock_value, latest_market_value
