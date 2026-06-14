"""Level 4: dynamic portfolio rebalancing.

All schemes are walk-forward: weights applied at time t are computed
from a `lookback`-day window ending at t. Three rebalancing triggers are
implemented and compared:

- **Time-based** — re-optimize on a fixed calendar (every N days).
- **Threshold-based** — rebalance only when drifted weights deviate from
  the last targets by more than a tolerance (saves costs in calm markets).
- **Regime-based** — re-optimize when the market's volatility regime
  shifts (rebalance exactly when conditions change, not by the calendar).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.portfolio.optimizer import risk_parity_weights


def _drifted_weights(target: pd.Series, prices: pd.DataFrame, start: int, end: int) -> pd.Series:
    """Weights at `end` if `target` was set at `start` and never touched."""
    growth = prices.iloc[end] / prices.iloc[start]
    value = target * growth
    return value / value.sum()


def time_based_weights(
    prices: pd.DataFrame,
    weight_fn=risk_parity_weights,
    lookback: int = 90,
    rebalance_every: int = 30,
) -> pd.DataFrame:
    """Re-optimize on a fixed calendar; weights are NaN-free only on rebalance dates."""
    rows = {}
    for i in range(lookback, len(prices), rebalance_every):
        window = prices.iloc[i - lookback : i]
        rows[prices.index[i]] = weight_fn(window)
    return pd.DataFrame(rows).T


def threshold_based_weights(
    prices: pd.DataFrame,
    weight_fn=risk_parity_weights,
    lookback: int = 90,
    tolerance: float = 0.05,
) -> pd.DataFrame:
    """Rebalance only when any drifted weight deviates from target by > `tolerance`."""
    rows = {}
    last_target: pd.Series | None = None
    last_index = lookback

    for i in range(lookback, len(prices)):
        if last_target is None:
            last_target = weight_fn(prices.iloc[i - lookback : i])
            last_index = i
            rows[prices.index[i]] = last_target
            continue

        current = _drifted_weights(last_target, prices, last_index, i)
        if (current - last_target).abs().max() > tolerance:
            last_target = weight_fn(prices.iloc[i - lookback : i])
            last_index = i
            rows[prices.index[i]] = last_target

    return pd.DataFrame(rows).T


def regime_based_weights(
    prices: pd.DataFrame,
    weight_fn=risk_parity_weights,
    lookback: int = 90,
    vol_window: int = 21,
    regime_change: float = 0.5,
    min_gap: int = 7,
) -> pd.DataFrame:
    """Re-optimize when market volatility shifts regime.

    Tracks the rolling `vol_window`-day volatility of the equal-weight
    basket; a rebalance triggers when volatility moves more than
    `regime_change` (relative) away from its level at the last rebalance.
    `min_gap` enforces a minimum number of days between rebalances.
    """
    basket_returns = prices.pct_change().mean(axis=1)
    rolling_vol = basket_returns.rolling(vol_window).std()

    rows = {}
    last_vol: float | None = None
    last_i = -np.inf

    for i in range(lookback, len(prices)):
        vol = rolling_vol.iloc[i]
        if np.isnan(vol):
            continue

        first = last_vol is None
        shifted = not first and abs(vol - last_vol) / last_vol > regime_change
        if (first or shifted) and i - last_i >= min_gap:
            rows[prices.index[i]] = weight_fn(prices.iloc[i - lookback : i])
            last_vol = vol
            last_i = i

    return pd.DataFrame(rows).T
