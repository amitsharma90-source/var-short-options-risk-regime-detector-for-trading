# -*- coding: utf-8 -*-
"""
=============================================================================
Module: reporting.py — Console Reporting Utilities
=============================================================================
Helper functions for formatted console output.
Charts are now inline in the orchestrator for simplicity.
=============================================================================
"""


def print_portfolio_summary(total_stock_value, cash_value, bond_value, n_pnl_obs):
    """Print portfolio composition summary. Returns total value."""
    total = total_stock_value + cash_value + bond_value
    print(f"  Stock market value:   ${total_stock_value:>15,.2f}")
    print(f"  Cash:                 ${cash_value:>15,.2f}")
    print(f"  Bonds:                ${bond_value:>15,.2f}")
    print(f"  Total portfolio:      ${total:>15,.2f}")
    print(f"  PnL observations:     {n_pnl_obs}")
    return total
