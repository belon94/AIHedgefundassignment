"""Level 5: portfolio management over a large (100+) universe of pairs.

Scaling from 7 coins to 100+ changes the problem:

- **Pair selection** becomes a pipeline stage of its own: at each
  rebalance, pairs are screened for sufficient history and liquidity
  (rolling dollar volume) using only past data.
- **Signal prioritization**: with hundreds of candidate signals, capital
  is allocated only to the strongest — here, cross-sectional momentum
  ranks the screened universe and the top `top_n` make the book.
- **Risk** is managed at the portfolio level: inverse-volatility sizing
  with a per-asset weight cap prevents concentration in any single coin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import available_symbols, load_symbol


def load_universe(min_history_days: int = 365) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load close prices and dollar volumes for every downloaded symbol.

    Returns (closes, dollar_volumes), one column per symbol. Symbols with
    fewer than `min_history_days` rows are dropped.
    """
    closes, volumes = {}, {}

    for symbol in available_symbols():
        df = load_symbol(symbol, "full")
        if len(df) < min_history_days:
            continue
        closes[symbol] = df["close"]
        volumes[symbol] = df["close"] * df["volume"]

    return pd.DataFrame(closes), pd.DataFrame(volumes)


def screen_universe(
    closes: pd.DataFrame,
    dollar_volumes: pd.DataFrame,
    as_of: pd.Timestamp,
    min_history_days: int = 180,
    liquidity_window: int = 30,
    min_dollar_volume: float = 1e6,
) -> list[str]:
    """Symbols tradeable at `as_of`: enough history and enough recent liquidity."""
    past_closes = closes[closes.index < as_of]
    past_volumes = dollar_volumes[dollar_volumes.index < as_of]

    history = past_closes.notna().sum()
    recent_liquidity = past_volumes.tail(liquidity_window).mean()

    eligible = (history >= min_history_days) & (recent_liquidity >= min_dollar_volume)
    return list(eligible[eligible].index)


def momentum_scores(closes: pd.DataFrame, as_of: pd.Timestamp, lookback: int = 90, skip: int = 7) -> pd.Series:
    """Cross-sectional momentum: return over [t-lookback, t-skip].

    The most recent `skip` days are excluded to avoid short-term reversal
    effects (standard momentum construction).
    """
    past = closes[closes.index < as_of]
    return past.iloc[-skip] / past.iloc[-lookback] - 1


def select_portfolio(
    closes: pd.DataFrame,
    dollar_volumes: pd.DataFrame,
    as_of: pd.Timestamp,
    top_n: int = 20,
    weight_cap: float = 0.10,
    vol_window: int = 30,
    **screen_kwargs,
) -> pd.Series:
    """One rebalance decision for the large universe at `as_of`.

    Pipeline: liquidity/history screen -> rank by momentum -> keep the
    top `top_n` with positive momentum -> size by inverse volatility,
    capped at `weight_cap` per asset. Unallocated weight stays in cash.
    """
    tradeable = screen_universe(closes, dollar_volumes, as_of, **screen_kwargs)
    if not tradeable:
        return pd.Series(dtype=float)

    momentum = momentum_scores(closes[tradeable], as_of).dropna()
    winners = momentum[momentum > 0].nlargest(top_n).index.tolist()
    if not winners:
        return pd.Series(dtype=float)

    past = closes[winners][closes.index < as_of]
    vol = past.pct_change().tail(vol_window).std()

    weights = (1.0 / vol).replace([np.inf, -np.inf], np.nan).dropna()
    weights = weights / weights.sum()

    # Iteratively cap and redistribute so no single asset dominates.
    for _ in range(10):
        over = weights > weight_cap
        if not over.any():
            break
        excess = (weights[over] - weight_cap).sum()
        weights[over] = weight_cap
        under = ~over
        if weights[under].sum() > 0:
            weights[under] += excess * weights[under] / weights[under].sum()
        else:
            break

    return weights


def large_universe_weights(
    closes: pd.DataFrame,
    dollar_volumes: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    **kwargs,
) -> pd.DataFrame:
    """Weight matrix over `rebalance_dates` for the full pipeline."""
    rows = {
        date: select_portfolio(closes, dollar_volumes, date, **kwargs)
        for date in rebalance_dates
    }
    return pd.DataFrame(rows).T.fillna(0.0)
