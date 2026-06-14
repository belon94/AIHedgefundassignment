"""Level 2 econometric strategy: ARIMA return forecasts + GARCH volatility filter.

Walk-forward design: models are fit only on data *before* each forecast
block, then produce one-step-ahead forecasts inside the block with fixed
parameters. Parameters are refit every `refit_every` days, mimicking how
the system would be retrained in production.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA

# Returns are scaled to percent for numerical stability of the optimizers.
SCALE = 100


def _log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1)).dropna() * SCALE


def arima_forecasts(
    close: pd.Series,
    test_index: pd.DatetimeIndex,
    order: tuple[int, int, int] = (1, 0, 1),
    refit_every: int = 21,
) -> pd.Series:
    """One-step-ahead ARIMA forecasts of (scaled) log returns over `test_index`.

    The model is refit every `refit_every` days on all data before the
    block; inside a block, forecasts condition on realized data up to the
    previous day with fixed parameters (no look-ahead).
    """
    returns = _log_returns(close)
    forecasts = {}

    for block_start in range(0, len(test_index), refit_every):
        block = test_index[block_start : block_start + refit_every]

        history = returns[returns.index < block[0]]
        fitted = ARIMA(history, order=order).fit()

        # Append the block's realized returns without refitting, then take
        # one-step-ahead in-sample predictions: each uses data up to t-1 only.
        block_returns = returns[returns.index.isin(block)]
        extended = fitted.append(block_returns, refit=False)
        preds = extended.get_prediction(start=block[0], end=block[-1]).predicted_mean

        forecasts.update(preds.to_dict())

    return pd.Series(forecasts).sort_index()


def garch_volatility_forecasts(
    close: pd.Series,
    test_index: pd.DatetimeIndex,
    refit_every: int = 21,
) -> pd.Series:
    """One-step-ahead GARCH(1,1) conditional volatility forecasts (in % per day)."""
    returns = _log_returns(close)
    forecasts = {}

    for block_start in range(0, len(test_index), refit_every):
        block = test_index[block_start : block_start + refit_every]

        # Fit only on data before the block, then roll 1-step forecasts
        # through the block with fixed parameters.
        model = arch_model(returns, vol="GARCH", p=1, q=1, mean="Constant")
        fitted = model.fit(last_obs=block[0], disp="off")

        variance = fitted.forecast(start=block[0], horizon=1, reindex=True).variance["h.1"]
        vol = np.sqrt(variance.loc[variance.index.isin(block)])

        forecasts.update(vol.to_dict())

    return pd.Series(forecasts).sort_index()


def arima_garch_positions(
    close: pd.Series,
    test_index: pd.DatetimeIndex,
    order: tuple[int, int, int] = (1, 0, 1),
    refit_every: int = 21,
    vol_quantile: float = 0.90,
) -> pd.Series:
    """Long when ARIMA predicts a positive return and GARCH volatility is not extreme.

    The volatility ceiling is the `vol_quantile` quantile of conditional
    volatility estimated on the training period — a simple "stand aside
    in panic regimes" rule.

    `close` must contain the full history (train + test) so that models
    can be fit on the data preceding each test block.
    """
    mean_forecast = arima_forecasts(close, test_index, order, refit_every)
    vol_forecast = garch_volatility_forecasts(close, test_index, refit_every)

    train_returns = _log_returns(close[close.index < test_index[0]])
    train_garch = arch_model(train_returns, vol="GARCH", p=1, q=1, mean="Constant").fit(disp="off")
    vol_ceiling = train_garch.conditional_volatility.quantile(vol_quantile)

    long_signal = (mean_forecast > 0) & (vol_forecast < vol_ceiling)
    return long_signal.astype(float).reindex(test_index).fillna(0.0)
