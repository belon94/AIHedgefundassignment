"""Statistical significance testing for strategy performance.

Answers the assignment question "how can you verify that the strategy's
performance is not due to random chance?" via a circular-shift
permutation test: the position series is rotated by random offsets,
which preserves its autocorrelation structure (holding spells, trade
count, fraction of time long) while destroying any alignment with
returns. If the real strategy's Sharpe ratio is not clearly above this
null distribution, its timing adds no detectable value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.risk.metrics import PERIODS_PER_YEAR, sharpe_ratio


def permutation_test_sharpe(
    position: pd.Series,
    asset_returns: pd.Series,
    n_permutations: int = 1000,
    periods_per_year: int = PERIODS_PER_YEAR,
    seed: int = 42,
) -> dict:
    """Circular-shift permutation test of a strategy's Sharpe ratio.

    Parameters
    ----------
    position:
        The strategy's position series (as fed to the backtester).
    asset_returns:
        The asset's simple returns over the same index.

    Returns
    -------
    dict with the observed Sharpe, the null distribution of permuted
    Sharpes, and the one-sided p-value P(random timing >= observed).
    """
    position, asset_returns = position.align(asset_returns, join="inner")

    held = position.shift(1).fillna(0.0).to_numpy()
    returns = asset_returns.to_numpy()
    observed = sharpe_ratio(pd.Series(held * returns), periods_per_year=periods_per_year)

    rng = np.random.default_rng(seed)
    n = len(held)
    null_sharpes = np.empty(n_permutations)

    for i in range(n_permutations):
        shift = rng.integers(1, n)
        shifted = np.roll(held, shift)
        null_sharpes[i] = sharpe_ratio(pd.Series(shifted * returns), periods_per_year=periods_per_year)

    p_value = float(np.mean(null_sharpes >= observed))

    return {
        "observed_sharpe": observed,
        "null_sharpes": null_sharpes,
        "p_value": p_value,
    }
