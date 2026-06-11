# AI Crypto Hedge Fund

CMF Summer School assignment: building and evaluating algorithmic crypto trading strategies.

## Project structure

```
.
├── data/
│   ├── raw/           # Raw OHLCV data per symbol (full history)
│   └── processed/     # Train (2020-2023) / test (2024) splits per symbol
├── src/
│   ├── data_loader.py # Fetches OHLCV data from Binance via ccxt
│   ├── indicators.py  # Technical indicators and feature engineering
│   ├── strategies/    # Trading strategy implementations
│   ├── portfolio/     # Portfolio construction and optimization
│   └── risk/
│       └── metrics.py # Performance and risk metrics (Sharpe, Sortino, VaR, ...)
├── Dockerfile
└── pyproject.toml
```

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

## Downloading data

Fetch daily OHLCV data for the top N USDT pairs on Binance by 24h volume,
and split it chronologically into train (2020-2023) and test (2024) sets:

```bash
uv run python src/data_loader.py --top 120
```

Data is written to `data/raw/` (full history) and `data/processed/`
(`<symbol>_train.csv` and `<symbol>_test.csv`).

## Running with Docker

```bash
docker build -t ai-crypto-hedge-fund .
docker run --rm -v "$(pwd)/data:/app/data" ai-crypto-hedge-fund
```
