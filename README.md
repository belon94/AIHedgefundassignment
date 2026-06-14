# AI Crypto Hedge Fund

CMF Summer School assignment — an AI agent-based automated cryptocurrency trading and
risk-management system. This repository contains **Part 2 (technical implementation)**:
a modular, reproducible codebase plus a single self-contained notebook
([ai_hedge_fund.ipynb](ai_hedge_fund.ipynb)) that presents all results.

## The five levels (Part 2)

The notebook is structured exactly as the assignment requires. Every result is reported
**out-of-sample**: models are trained on **2020–2023** and evaluated on **2024**, net of
Binance taker fees (0.1%), with no look-ahead (a position decided at the close of day *t*
earns day *t+1*'s return).

| Level | What | Module |
|---|---|---|
| 1 | Baseline SMA-crossover strategy on BTC/USDT vs. buy-and-hold | [src/strategies/baseline.py](src/strategies/baseline.py) |
| 2 | Econometric (ARIMA + GARCH), ML (XGBoost), deep learning (LSTM), and a two-agent layer (signal consensus + risk agent), with a permutation significance test | [src/strategies/](src/strategies/), [src/risk/significance.py](src/risk/significance.py) |
| 3 | Static portfolio optimization of 7 coins (max-Sharpe, min-vol, risk parity, equal weight) | [src/portfolio/optimizer.py](src/portfolio/optimizer.py) |
| 4 | Dynamic rebalancing — time-based, threshold-based, regime-based | [src/portfolio/rebalancing.py](src/portfolio/rebalancing.py) |
| 5 | Expansion to 100+ pairs: liquidity screening, momentum prioritization, risk-capped sizing | [src/portfolio/universe.py](src/portfolio/universe.py) |

## Project structure

```
.
├── ai_hedge_fund.ipynb     # Final self-contained deliverable (run this)
├── data/
│   ├── raw/                # Full-history OHLCV per symbol
│   └── processed/          # Train (2020-2023) / test (2024) splits
├── src/
│   ├── data_loader.py      # Binance OHLCV download + load helpers
│   ├── indicators.py       # SMA, EMA, RSI, MACD, Bollinger, ATR, build_features
│   ├── backtest.py         # Vectorized single-asset + portfolio backtester
│   ├── strategies/
│   │   ├── baseline.py     # Level 1: SMA crossover, buy & hold
│   │   ├── econometric.py  # Level 2: ARIMA mean + GARCH vol filter
│   │   ├── ml.py           # Level 2: XGBoost direction classifier
│   │   ├── deep.py         # Level 2: LSTM direction classifier
│   │   └── agent.py        # Level 2: signal consensus + risk agent
│   ├── portfolio/
│   │   ├── optimizer.py    # Level 3: static weight schemes
│   │   ├── rebalancing.py  # Level 4: dynamic rebalancing triggers
│   │   └── universe.py     # Level 5: large-universe selection pipeline
│   └── risk/
│       ├── metrics.py      # Sharpe, Sortino, Calmar, drawdown, VaR, CVaR
│       └── significance.py # Permutation test for performance significance
├── scripts/make_notebook.py # Regenerates ai_hedge_fund.ipynb
├── Dockerfile
└── pyproject.toml
```

## Setup

The project uses [uv](https://docs.astral.sh/uv/) for reproducible dependency management.

```bash
uv sync
```

On macOS, XGBoost needs the OpenMP runtime: `brew install libomp`.

## Reproducing the results

The data needed to run the notebook is included in `data/`. Just launch it:

```bash
uv run jupyter notebook ai_hedge_fund.ipynb
```

or execute it headlessly end-to-end:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace ai_hedge_fund.ipynb
```

To re-download fresh data from Binance (top 300 USDT pairs by volume → ~140 with full
history, satisfying the 100+ requirement for Level 5):

```bash
uv run python src/data_loader.py --top 300
```

After editing notebook content in `scripts/make_notebook.py`, regenerate with
`uv run python scripts/make_notebook.py`.

## Docker

```bash
docker build -t ai-crypto-hedge-fund .
docker run --rm -v "$(pwd)/data:/app/data" ai-crypto-hedge-fund \
  jupyter nbconvert --to notebook --execute --inplace ai_hedge_fund.ipynb
```
