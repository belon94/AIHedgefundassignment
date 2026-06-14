"""Level 2 machine-learning strategy: XGBoost direction classifier.

Features: the technical indicators from ``src.indicators.build_features``.
Target: whether the next day's return is positive.
Training: walk-forward — the model is retrained every `retrain_every`
days on an expanding window of all data available up to that point, then
predicts the next block. The test period is never seen during training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.indicators import build_features

RANDOM_STATE = 42

# Indicator columns used as model inputs. Price-level columns (close, sma,
# bands) are excluded in favor of stationary, scale-free features.
FEATURE_COLUMNS = [
    "rsi_14",
    "macd_hist",
    "returns",
    "log_returns",
    "sma_ratio",
    "bb_position",
    "atr_pct",
    "ret_5d",
    "ret_21d",
    "vol_21d",
]


def make_feature_matrix(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Build a stationary feature matrix plus the `target` column (next day up)."""
    feat = build_features(ohlcv)

    # Scale-free transformations of the price-level indicators.
    feat["sma_ratio"] = feat["sma_10"] / feat["sma_50"] - 1
    band_width = feat["bb_upper"] - feat["bb_lower"]
    feat["bb_position"] = (feat["close"] - feat["bb_lower"]) / band_width
    feat["atr_pct"] = feat["atr_14"] / feat["close"]

    # Momentum and realized volatility over several horizons.
    feat["ret_5d"] = feat["close"].pct_change(5)
    feat["ret_21d"] = feat["close"].pct_change(21)
    feat["vol_21d"] = feat["returns"].rolling(21).std()

    feat["target"] = (feat["returns"].shift(-1) > 0).astype(int)

    return feat[FEATURE_COLUMNS + ["target"]].dropna(subset=FEATURE_COLUMNS)


def _new_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )


def xgboost_positions(
    ohlcv: pd.DataFrame,
    test_index: pd.DatetimeIndex,
    retrain_every: int = 21,
    threshold: float = 0.5,
) -> pd.Series:
    """Walk-forward XGBoost positions over `test_index`.

    `ohlcv` must contain the full history (train + test) so each retrain
    can use an expanding window ending just before its prediction block.
    Long (1.0) when the predicted probability of an up-day exceeds
    `threshold`, flat (0.0) otherwise.
    """
    data = make_feature_matrix(ohlcv)
    positions = {}

    for block_start in range(0, len(test_index), retrain_every):
        block = test_index[block_start : block_start + retrain_every]

        train = data[data.index < block[0]].dropna(subset=["target"])
        # The final training row's target uses the first block day's return —
        # drop it to keep the split strictly out-of-sample.
        train = train.iloc[:-1]

        model = _new_model()
        model.fit(train[FEATURE_COLUMNS], train["target"])

        block_features = data.loc[data.index.isin(block), FEATURE_COLUMNS]
        prob_up = model.predict_proba(block_features)[:, 1]

        for ts, p in zip(block_features.index, prob_up):
            positions[ts] = 1.0 if p > threshold else 0.0

    return pd.Series(positions).sort_index().reindex(test_index).fillna(0.0)


def feature_importances(ohlcv: pd.DataFrame, test_start: pd.Timestamp) -> pd.Series:
    """Feature importances of a model trained on all data before `test_start`."""
    data = make_feature_matrix(ohlcv)
    train = data[data.index < test_start].dropna(subset=["target"]).iloc[:-1]

    model = _new_model()
    model.fit(train[FEATURE_COLUMNS], train["target"])

    return pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
