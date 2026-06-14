"""Level 2 AI agent layer.

Two cooperating agents sit on top of the individual strategies:

- **Signal agent** — aggregates the position signals of all underlying
  strategies (baseline, econometric, ML, deep learning) into a single
  consensus position via (optionally weighted) voting.
- **Risk agent** — supervises the signal agent. It scales exposure down
  when realized volatility exceeds the target (volatility targeting) and
  cuts the position entirely after a deep drawdown (circuit breaker),
  re-entering once the market stabilizes.

This mirrors the multi-agent architecture from Part 1: signal generation
and risk management are separate modules with a clear interface — the
risk agent can override the signal agent but never the other way around.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def consensus_positions(
    signals: dict[str, pd.Series],
    weights: dict[str, float] | None = None,
    threshold: float = 0.5,
) -> pd.Series:
    """Combine strategy position series via weighted voting.

    Each signal is a 0/1 position series. The consensus is long when the
    weighted average vote exceeds `threshold` (default: simple majority).
    """
    votes = pd.DataFrame(signals)

    if weights is None:
        weight_vec = pd.Series(1.0, index=votes.columns)
    else:
        weight_vec = pd.Series(weights).reindex(votes.columns).fillna(0.0)
    weight_vec = weight_vec / weight_vec.sum()

    score = votes.fillna(0.0).mul(weight_vec, axis=1).sum(axis=1)
    return (score > threshold).astype(float)


def volatility_target_scaling(
    close: pd.Series,
    target_vol: float = 0.40,
    lookback: int = 21,
    max_leverage: float = 1.0,
) -> pd.Series:
    """Exposure multiplier from volatility targeting.

    Scales exposure by target_vol / realized_vol (annualized, rolling
    `lookback` days), capped at `max_leverage`. In calm markets the
    multiplier is ~1; in turbulent markets it shrinks toward 0.
    """
    returns = close.pct_change()
    realized_vol = returns.rolling(lookback).std() * np.sqrt(365)
    scaling = (target_vol / realized_vol).clip(upper=max_leverage)
    return scaling.fillna(0.0)


def drawdown_circuit_breaker(
    close: pd.Series,
    max_drawdown: float = 0.25,
    reentry_drawdown: float = 0.10,
    lookback: int = 90,
) -> pd.Series:
    """1/0 kill-switch based on the asset's rolling drawdown.

    Trips to 0 when the drawdown from the rolling `lookback`-day high
    exceeds `max_drawdown`; re-arms once the drawdown recovers above
    `reentry_drawdown` (hysteresis prevents rapid on/off flipping).
    """
    rolling_high = close.rolling(lookback, min_periods=1).max()
    drawdown = close / rolling_high - 1

    allowed = np.ones(len(close))
    tripped = False
    for i, dd in enumerate(drawdown):
        if tripped and dd > -reentry_drawdown:
            tripped = False
        elif not tripped and dd < -max_drawdown:
            tripped = True
        allowed[i] = 0.0 if tripped else 1.0

    return pd.Series(allowed, index=close.index)


def meta_agent_positions(
    signals: dict[str, pd.Series],
    close: pd.Series,
    weights: dict[str, float] | None = None,
    target_vol: float = 0.40,
    max_drawdown: float = 0.25,
) -> pd.Series:
    """Full agent pipeline: consensus signal, then risk-agent overlays."""
    consensus = consensus_positions(signals, weights)
    vol_scale = volatility_target_scaling(close, target_vol)
    breaker = drawdown_circuit_breaker(close, max_drawdown)

    return (consensus * vol_scale * breaker).reindex(close.index).fillna(0.0)
