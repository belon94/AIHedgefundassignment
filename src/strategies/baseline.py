"""Level 1 baseline: simple moving-average crossover strategy.

Long when the fast SMA is above the slow SMA, flat otherwise. This is the
benchmark every AI/ML strategy in later levels must beat.
"""

from __future__ import annotations

import pandas as pd

from src.indicators import sma


def sma_crossover_positions(close: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    """Return a 0/1 position series: long while SMA(fast) > SMA(slow)."""
    fast_sma = sma(close, fast)
    slow_sma = sma(close, slow)

    position = (fast_sma > slow_sma).astype(float)
    # No position until both SMAs exist.
    position[slow_sma.isna()] = 0.0
    return position


def buy_and_hold_positions(close: pd.Series) -> pd.Series:
    """Always fully invested — the passive benchmark."""
    return pd.Series(1.0, index=close.index)
