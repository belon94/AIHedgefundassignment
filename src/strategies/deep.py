"""Level 2 deep-learning strategy: LSTM direction classifier.

The network sees a window of the last `seq_len` days of stationary
features and predicts the probability that the next day's return is
positive. It is trained once on the training period (2020-2023) and
evaluated on the untouched test period. All seeds are fixed so results
are reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.strategies.ml import FEATURE_COLUMNS, make_feature_matrix

SEED = 42


class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 32, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :]).squeeze(-1)


def _make_sequences(features: np.ndarray, targets: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Stack rolling windows: X[i] = features[i : i+seq_len], y[i] = target at window end."""
    xs = np.stack([features[i : i + seq_len] for i in range(len(features) - seq_len + 1)])
    ys = targets[seq_len - 1 :]
    return xs, ys


def lstm_positions(
    ohlcv: pd.DataFrame,
    test_index: pd.DatetimeIndex,
    seq_len: int = 30,
    epochs: int = 30,
    threshold: float = 0.5,
) -> pd.Series:
    """Train an LSTM on data before `test_index` and emit 0/1 positions over it.

    `ohlcv` must contain the full history (train + test); the test rows
    are used only for prediction, never for fitting or normalization.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data = make_feature_matrix(ohlcv)
    is_test = data.index.isin(test_index)
    train_data = data[~is_test & (data.index < test_index[0])]

    # Normalize with training statistics only.
    mean = train_data[FEATURE_COLUMNS].mean()
    std = train_data[FEATURE_COLUMNS].std()
    normalized = (data[FEATURE_COLUMNS] - mean) / std

    train_features = normalized[normalized.index < test_index[0]].to_numpy(dtype=np.float32)
    train_targets = train_data["target"].to_numpy(dtype=np.float32)
    # Last row's target is the first test day's return — exclude it.
    x_train, y_train = _make_sequences(train_features[:-1], train_targets[:-1], seq_len)

    model = LSTMClassifier(n_features=len(FEATURE_COLUMNS))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(x_train_t), y_train_t)
        loss.backward()
        optimizer.step()

    # Predict each test day from the window of `seq_len` days ending on it.
    normalized_np = normalized.to_numpy(dtype=np.float32)
    index_positions = {ts: i for i, ts in enumerate(normalized.index)}

    model.eval()
    positions = {}
    with torch.no_grad():
        for ts in test_index:
            i = index_positions.get(ts)
            if i is None or i + 1 < seq_len:
                continue
            window = torch.from_numpy(normalized_np[i + 1 - seq_len : i + 1]).unsqueeze(0)
            prob_up = torch.sigmoid(model(window)).item()
            positions[ts] = 1.0 if prob_up > threshold else 0.0

    return pd.Series(positions).sort_index().reindex(test_index).fillna(0.0)
