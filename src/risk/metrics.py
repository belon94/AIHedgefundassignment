"""Risk and performance metrics for evaluating trading strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Crypto markets trade every day of the year.
PERIODS_PER_YEAR = 365


def annualized_return(returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Compound annual growth rate implied by a series of period returns."""
    returns = returns.dropna()
    if returns.empty:
        return np.nan

    cumulative = (1 + returns).prod()
    return cumulative ** (periods_per_year / len(returns)) - 1


def annualized_volatility(returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Annualized standard deviation of returns."""
    return returns.dropna().std() * np.sqrt(periods_per_year)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Annualized Sharpe ratio."""
    returns = returns.dropna()
    excess = returns - risk_free_rate / periods_per_year

    std = excess.std()
    if std == 0 or np.isnan(std):
        return np.nan

    return (excess.mean() / std) * np.sqrt(periods_per_year)


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
    returns = returns.dropna()
    excess = returns - risk_free_rate / periods_per_year

    downside_std = excess[excess < 0].std()
    if downside_std == 0 or np.isnan(downside_std):
        return np.nan

    return (excess.mean() / downside_std) * np.sqrt(periods_per_year)


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown of the cumulative return curve, as a negative fraction."""
    returns = returns.dropna()
    if returns.empty:
        return np.nan

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    return drawdown.min()


def calmar_ratio(returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Annualized return divided by the absolute maximum drawdown."""
    mdd = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return np.nan

    return annualized_return(returns, periods_per_year) / abs(mdd)


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical Value at Risk at the given confidence level (negative number = loss)."""
    returns = returns.dropna()
    if returns.empty:
        return np.nan

    return np.percentile(returns, (1 - confidence) * 100)


def conditional_value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Expected shortfall: mean return in the tail beyond the VaR threshold."""
    returns = returns.dropna()
    if returns.empty:
        return np.nan

    var = value_at_risk(returns, confidence)
    tail = returns[returns <= var]
    return tail.mean() if not tail.empty else var


def compute_metrics(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = PERIODS_PER_YEAR) -> dict:
    """Compute a standard set of performance and risk metrics for a return series."""
    return {
        "annualized_return": annualized_return(returns, periods_per_year),
        "annualized_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "calmar_ratio": calmar_ratio(returns, periods_per_year),
        "var_95": value_at_risk(returns, 0.95),
        "cvar_95": conditional_value_at_risk(returns, 0.95),
    }


def compare_strategies(
    strategy_returns: dict[str, pd.Series],
    risk_free_rate: float = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> pd.DataFrame:
    """Compute metrics for multiple strategies and return them side by side.

    Parameters
    ----------
    strategy_returns:
        Mapping of strategy name to a series of period returns.

    Returns
    -------
    A DataFrame with metrics as rows and strategy names as columns.
    """
    results = {
        name: compute_metrics(returns, risk_free_rate, periods_per_year)
        for name, returns in strategy_returns.items()
    }
    return pd.DataFrame(results)
