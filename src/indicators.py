"""Technical indicators and feature engineering for OHLCV data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Moving Average Convergence Divergence: macd line, signal line and histogram."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)

    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line

    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": histogram})


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands: middle (SMA), upper and lower bands."""
    middle = sma(series, window)
    std = series.rolling(window=window).std()

    return pd.DataFrame({
        "bb_middle": middle,
        "bb_upper": middle + num_std * std,
        "bb_lower": middle - num_std * std,
    })


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range using Wilder's smoothing. Expects high, low, close columns."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)

    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    return true_range.ewm(alpha=1 / window, min_periods=window).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the standard set of technical indicators and return features to `df`.

    Expects a DataFrame with at least open, high, low, close, volume columns.
    """
    out = df.copy()

    out["sma_10"] = sma(out["close"], 10)
    out["sma_50"] = sma(out["close"], 50)
    out["ema_12"] = ema(out["close"], 12)
    out["ema_26"] = ema(out["close"], 26)
    out["rsi_14"] = rsi(out["close"], 14)

    out = out.join(macd(out["close"]))
    out = out.join(bollinger_bands(out["close"]))

    out["atr_14"] = atr(out)

    out["returns"] = out["close"].pct_change()
    out["log_returns"] = np.log(out["close"] / out["close"].shift(1))

    return out
