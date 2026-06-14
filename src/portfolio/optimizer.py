"""Level 3: static portfolio optimization on historical data.

Given a lookback window of daily prices, computes optimal weights under
several classic schemes (Markowitz max-Sharpe, minimum volatility, risk
parity) plus the equal-weight benchmark, so they can be compared
out-of-sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models

# Default universe for levels 3-4: liquid, large-cap, with full history.
DEFAULT_UNIVERSE = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "LTC/USDT",
]


def equal_weights(prices: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0 / prices.shape[1], index=prices.columns)


def max_sharpe_weights(prices: pd.DataFrame, risk_free_rate: float = 0.0) -> pd.Series:
    """Markowitz tangency portfolio (long-only) from historical prices."""
    mu = expected_returns.mean_historical_return(prices, frequency=365)
    cov = risk_models.CovarianceShrinkage(prices, frequency=365).ledoit_wolf()

    ef = EfficientFrontier(mu, cov, weight_bounds=(0, 1))
    ef.max_sharpe(risk_free_rate=risk_free_rate)
    return pd.Series(ef.clean_weights())


def min_volatility_weights(prices: pd.DataFrame) -> pd.Series:
    """Long-only minimum-variance portfolio."""
    mu = expected_returns.mean_historical_return(prices, frequency=365)
    cov = risk_models.CovarianceShrinkage(prices, frequency=365).ledoit_wolf()

    ef = EfficientFrontier(mu, cov, weight_bounds=(0, 1))
    ef.min_volatility()
    return pd.Series(ef.clean_weights())


def risk_parity_weights(prices: pd.DataFrame) -> pd.Series:
    """Inverse-volatility weights — a simple, robust risk-parity proxy."""
    vol = prices.pct_change().std()
    inverse_vol = 1.0 / vol
    return inverse_vol / inverse_vol.sum()


WEIGHT_SCHEMES = {
    "equal_weight": equal_weights,
    "max_sharpe": max_sharpe_weights,
    "min_volatility": min_volatility_weights,
    "risk_parity": risk_parity_weights,
}


def static_weights_table(prices: pd.DataFrame) -> pd.DataFrame:
    """Weights of every scheme fitted on `prices`, side by side."""
    return pd.DataFrame({name: scheme(prices) for name, scheme in WEIGHT_SCHEMES.items()})
