"""Vectorized backtesting engine shared by all strategies.

A strategy is represented by a *position* series: the fraction of capital
held in the asset at each bar (0 = flat, 1 = fully long). Positions are
assumed to be decided at the close of bar t and held during bar t+1, so
strategy returns are ``position.shift(1) * asset_returns`` — this avoids
look-ahead bias by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Binance spot taker fee (0.1%) charged on every change in position.
DEFAULT_FEE = 0.001


def backtest(
    close: pd.Series,
    position: pd.Series,
    fee: float = DEFAULT_FEE,
) -> pd.DataFrame:
    """Backtest a position series against a close-price series.

    Parameters
    ----------
    close:
        Close prices indexed by timestamp.
    position:
        Target position (fraction of capital in the asset, e.g. 0 or 1)
        decided at the close of each bar. Reindexed to `close` and
        forward-filled, so sparse signals are allowed.
    fee:
        Proportional transaction cost applied to each unit of turnover.

    Returns
    -------
    DataFrame with columns: position, asset_returns, strategy_returns
    (net of fees) and equity (cumulative growth of 1 unit of capital).
    """
    position = position.reindex(close.index).ffill().fillna(0.0)

    asset_returns = close.pct_change().fillna(0.0)
    held_position = position.shift(1).fillna(0.0)

    turnover = position.diff().abs().fillna(position.abs())
    costs = turnover * fee

    strategy_returns = held_position * asset_returns - costs
    equity = (1 + strategy_returns).cumprod()

    return pd.DataFrame({
        "position": position,
        "asset_returns": asset_returns,
        "strategy_returns": strategy_returns,
        "equity": equity,
    })


def backtest_portfolio(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    fee: float = DEFAULT_FEE,
) -> pd.DataFrame:
    """Backtest a portfolio given per-asset target weights over time.

    Parameters
    ----------
    prices:
        Close prices, one column per asset.
    weights:
        Target portfolio weights, same columns as `prices`. Rows are
        forward-filled, so weights only need to be specified on
        rebalancing dates. Weights decided at the close of bar t apply
        to bar t+1.
    fee:
        Proportional cost on turnover (sum of absolute weight changes).

    Returns
    -------
    DataFrame with portfolio_returns (net of fees), turnover and equity.
    """
    weights = weights.reindex(prices.index).ffill().fillna(0.0)
    weights = weights.reindex(columns=prices.columns, fill_value=0.0)

    asset_returns = prices.pct_change().fillna(0.0)
    held_weights = weights.shift(1).fillna(0.0)

    gross_returns = (held_weights * asset_returns).sum(axis=1)

    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    costs = turnover * fee

    portfolio_returns = gross_returns - costs
    equity = (1 + portfolio_returns).cumprod()

    return pd.DataFrame({
        "portfolio_returns": portfolio_returns,
        "turnover": turnover,
        "equity": equity,
    })
