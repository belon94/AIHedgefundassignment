"""Generate the final assignment notebook (ai_hedge_fund.ipynb).

The notebook is the single, self-contained deliverable required by the
assignment. All heavy lifting lives in the `src` package; the notebook
orchestrates it and explains the results, structured by the five levels
of Part 2. Re-run this script after changing cell content.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source.strip()))


md("""
# AI Crypto Hedge Fund — Technical Implementation (Part 2)

**CMF Summer School assignment.**

This notebook is the single, self-contained presentation of the technical implementation. It is structured according to the five levels required by the assignment:

1. **Baseline strategy** for a single cryptocurrency (BTC/USDT)
2. **Econometric, ML and AI-agent strategies** for the same pair
3. **Static portfolio management** of 7 coins on historical data
4. **Dynamic portfolio rebalancing**
5. **Portfolio expansion** to 100+ pairs

**Methodology.** All data is split chronologically: models are trained on **2020–2023** and every reported result is **out-of-sample on 2024**. Backtests deduct Binance taker fees (0.1%) on every position change and avoid look-ahead by construction (a position decided at the close of day *t* earns the return of day *t+1*). All seeds are fixed, so the notebook is fully reproducible.

**Reproducibility.** `uv sync` installs the exact locked environment; the data needed is included in the repository (`data/`). To re-download fresh data: `uv run python src/data_loader.py --top 300`.
""")

code("""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

# Make `src` importable when the notebook runs from the repo root.
ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import load_symbol, load_close_prices, available_symbols
from src.backtest import backtest, backtest_portfolio
from src.risk.metrics import compare_strategies, compute_metrics

TEST_START = pd.Timestamp("2024-01-01", tz="UTC")
TEST_END = pd.Timestamp("2024-12-31", tz="UTC")

pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
""")

md("""
## 0. Data

Daily OHLCV candles from **Binance** (via `ccxt`), for the top USDT spot pairs by 24h volume. The loader (`src/data_loader.py`) stores the raw history in `data/raw/` and a chronological train (2020–2023) / test (2024) split in `data/processed/`.
""")

code("""
btc_full = load_symbol("BTC/USDT", "full")
btc_test = load_symbol("BTC/USDT", "test")
btc_hist = btc_full[btc_full.index <= TEST_END]  # nothing after the test period

print(f"Symbols downloaded: {len(available_symbols())}")
print(f"BTC/USDT history: {btc_full.index[0].date()} → {btc_full.index[-1].date()} ({len(btc_full)} days)")
btc_full.tail(3)
""")

code("""
fig = go.Figure()
fig.add_scatter(x=btc_hist.index, y=btc_hist["close"], name="BTC/USDT close")
fig.add_vline(x=TEST_START, line_dash="dash", line_color="red")
fig.add_annotation(x=TEST_START, y=btc_hist["close"].max(), text="train | test", showarrow=False, xshift=45)
fig.update_layout(title="BTC/USDT — train/test split", yaxis_type="log", height=400)
fig.show()
""")

md("""
## 1. Baseline strategy for a single cryptocurrency

A **20/50-day SMA crossover** on BTC/USDT: long while the fast SMA is above the slow SMA, in cash otherwise. Buy-and-hold is shown as the passive benchmark.

**Metrics.** Beyond the ROI / Sharpe / drawdown asked for in the assignment, we report Sortino (penalizes only downside volatility — appropriate for the fat-tailed, skewed return distributions of crypto), Calmar (return per unit of worst-case pain), and VaR/CVaR at 95% (tail-risk per day — what a risk-management module would monitor live).
""")

code("""
from src.strategies.baseline import sma_crossover_positions, buy_and_hold_positions

close_test = btc_test["close"]

baseline_pos = sma_crossover_positions(close_test)
results_l1 = {
    "sma_crossover": backtest(close_test, baseline_pos),
    "buy_and_hold": backtest(close_test, buy_and_hold_positions(close_test)),
}

compare_strategies({name: r["strategy_returns"] for name, r in results_l1.items()})
""")

code("""
fig = go.Figure()
for name, res in results_l1.items():
    fig.add_scatter(x=res.index, y=res["equity"], name=name)
fig.update_layout(title="Level 1 — equity curves, out-of-sample 2024 (net of 0.1% fees)",
                  yaxis_title="growth of 1 USDT", height=400)
fig.show()
""")

md("""
**Reading the result.** 2024 was a strong bull year for BTC, so buy-and-hold is a hard benchmark: the crossover sacrifices upside in exchange for sitting out downtrends (visible as the flat stretches of its equity curve). The value of trend filters shows up in bear regimes — which is exactly why a *single static rule* is not enough, motivating everything below.

**How does this evolve into an AI agent?** The crossover is already a degenerate agent: it *perceives* (prices → SMAs), *decides* (compare), and *acts* (position). Evolving it means upgrading each part: perception from two indicators to a learned feature representation (Level 2 ML), decision from a fixed threshold to a trained model with uncertainty (probabilities), and adding *self-supervision* — agents that monitor risk and veto the signal agent (our risk agent below), plus retraining loops that adapt parameters as the market changes.
""")

md("""
## 2. Adding econometric models, ML models and AI agents

All approaches predict the **same target** — whether BTC's next daily return is positive — from data available strictly before the prediction day, then map predictions to a long/flat position. They are evaluated on identical terms (same test year, same costs), so the comparison table at the end is apples-to-apples.

| Approach | Model | Features | Retraining |
|---|---|---|---|
| Econometric | ARIMA(1,0,1) mean + GARCH(1,1) volatility filter | log returns | refit every 21 days, walk-forward |
| Machine learning | XGBoost classifier | 10 stationary technical features | retrained every 21 days, expanding window |
| Deep learning | 2-layer LSTM | same features, 30-day windows | trained once on 2020–2023 |
| AI agents | consensus vote + risk-agent overlay | the three signals above + baseline | inherits the above |

**Why these choices?**
- *Features*: stationary, scale-free transforms (RSI, MACD histogram, SMA ratio, Bollinger position, ATR%, multi-horizon momentum and realized volatility). Raw price levels are excluded — a model trained on 2020 price levels would be extrapolating in 2024.
- *Target*: next-day direction rather than magnitude — classification is more robust to crypto's heavy tails than regression on returns.
- *Train/test discipline*: the 2024 test year is never touched during fitting or feature normalization. Walk-forward retraining (every 21 days) mimics production: in live trading we would retrain on a schedule of the same order — frequent enough to track regime change, rare enough to avoid refitting noise.
""")

code("""
from src.strategies.econometric import arima_garch_positions
from src.strategies.ml import xgboost_positions, feature_importances
from src.strategies.deep import lstm_positions

test_index = btc_test.index

econ_pos = arima_garch_positions(btc_hist["close"], test_index)
xgb_pos = xgboost_positions(btc_hist, test_index)
lstm_pos = lstm_positions(btc_hist, test_index)

print("Days long — econometric: %d, xgboost: %d, lstm: %d (of %d)" % (
    econ_pos.sum(), xgb_pos.sum(), lstm_pos.sum(), len(test_index)))
""")

code("""
feature_importances(btc_hist, TEST_START).to_frame("xgboost feature importance")
""")

md("""
### The AI agent layer

Two agents sit on top of the individual models, mirroring the system architecture from Part 1:

- The **signal agent** aggregates all four strategies (baseline, ARIMA-GARCH, XGBoost, LSTM) by majority vote — positions are taken only when independent methodologies agree, which filters out single-model noise.
- The **risk agent** supervises it with two overlays: **volatility targeting** (exposure is scaled by target/realized volatility, shrinking positions in turbulent markets) and a **drawdown circuit breaker** (the book is cut entirely after a >25% drawdown from the 90-day high and re-armed only after recovery). The risk agent can override the signal agent; never the reverse.
""")

code("""
from src.strategies.agent import meta_agent_positions

signals = {
    "baseline": baseline_pos,
    "econometric": econ_pos,
    "xgboost": xgb_pos,
    "lstm": lstm_pos,
}
agent_pos = meta_agent_positions(signals, close_test)

results_l2 = {name: backtest(close_test, pos) for name, pos in {
    **signals, "ai_agent": agent_pos, "buy_and_hold": buy_and_hold_positions(close_test),
}.items()}

comparison_l2 = compare_strategies({n: r["strategy_returns"] for n, r in results_l2.items()})
comparison_l2
""")

code("""
fig = go.Figure()
for name, res in results_l2.items():
    fig.add_scatter(x=res.index, y=res["equity"], name=name,
                    line=dict(width=3 if name == "ai_agent" else 1.2))
fig.update_layout(title="Level 2 — all single-pair strategies, out-of-sample 2024",
                  yaxis_title="growth of 1 USDT", height=450)
fig.show()
""")

md("""
### Is the performance distinguishable from luck?

A backtest can look good by accident. We test this with a **circular-shift permutation test**: the position series is rotated by 1,000 random offsets — preserving its trade count, holding-spell lengths and time-in-market, but destroying any alignment with returns. If the strategy's real Sharpe is not in the right tail of this null distribution, its *timing* adds nothing beyond exposure.
""")

code("""
from src.risk.significance import permutation_test_sharpe

asset_returns = close_test.pct_change().fillna(0.0)
rows = {}
for name, pos in {**signals, "ai_agent": agent_pos}.items():
    t = permutation_test_sharpe(pos, asset_returns)
    rows[name] = {"observed_sharpe": t["observed_sharpe"],
                  "null_mean": t["null_sharpes"].mean(),
                  "null_95pct": np.percentile(t["null_sharpes"], 95),
                  "p_value": t["p_value"]}
pd.DataFrame(rows).T
""")

md("""
**Interpretation.** p-values below 0.05 mean the strategy's timing beats >95% of random timings with identical trading style. In a single bull year, exposure alone explains a lot — strategies whose p-value is large are earning beta, not alpha. This test (plus reporting *all* attempted strategies rather than the best one, and using a single untouched test year) is our defense against data-mining bias.

**How often to retrain?** Our walk-forward grid uses 21 days. More frequent retraining chases noise and increases operational risk; less frequent retraining lags regime changes. In production we would monitor *live* prediction quality (rolling hit-rate vs. its backtest distribution) and retrain on degradation rather than purely on a calendar.
""")

md("""
## 3. Portfolio management on historical data (7 coins)

Universe: **BTC, ETH, BNB, SOL, XRP, ADA, LTC** — popular, liquid pairs with full history. Weights are optimized on the **last 12 months of training data (2023)** and then held fixed through 2024 — a strictly out-of-sample test of each allocation scheme:

- **Equal weight** — the no-information benchmark.
- **Max Sharpe** (Markowitz tangency, Ledoit–Wolf shrunk covariance, long-only).
- **Minimum volatility** — ignores the noisy expected-return estimates entirely.
- **Risk parity** (inverse volatility) — equalizes risk contributions.

An *optimal portfolio* is the one maximizing compensation per unit of risk **given estimation error**: mean-variance is optimal in-sample by construction, but expected returns are so noisy in crypto that schemes ignoring them often dominate out-of-sample. That is precisely what the table below shows.
""")

code("""
from src.portfolio.optimizer import DEFAULT_UNIVERSE, WEIGHT_SCHEMES, static_weights_table

prices_full = load_close_prices(DEFAULT_UNIVERSE, "full")
prices_2023 = prices_full[(prices_full.index >= TEST_START - pd.Timedelta(days=365))
                          & (prices_full.index < TEST_START)]
prices_test = prices_full[(prices_full.index >= TEST_START) & (prices_full.index <= TEST_END)]

weights_l3 = static_weights_table(prices_2023)
weights_l3
""")

code("""
returns_l3 = {}
for scheme in WEIGHT_SCHEMES:
    w = pd.DataFrame([weights_l3[scheme]], index=[prices_test.index[0]])
    returns_l3[scheme] = backtest_portfolio(prices_test, w)["portfolio_returns"]

compare_strategies(returns_l3)
""")

code("""
fig = px.imshow(prices_2023.pct_change().corr().round(2), text_auto=True,
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                title="Daily return correlations, 2023 — why diversification within crypto is limited")
fig.update_layout(height=450)
fig.show()
""")

md("""
**Reading the result.** Max-Sharpe concentrates the entire book in the single best 2023 performer — the classic Markowitz failure mode: maximizing over noisy mean estimates selects the largest estimation error. Min-volatility and risk parity, which don't use mean estimates, deliver better out-of-sample Sharpe. The correlation heatmap explains why diversification *within* crypto has limits: everything is 0.5–0.8 correlated with BTC.

**Real-trading application.** These weights are directly executable: at the start of the period, allocate capital in these proportions (all pairs trade against USDT, so implementation is seven market orders), then leave it. The 0.1% fee on initial deployment is included. What static weights ignore — drift, regime change — is Level 4's subject.
""")

md("""
## 4. Dynamic portfolio rebalancing

Static weights drift: a winner grows until it dominates the book. We compare three rebalancing triggers (all walk-forward — each decision uses only the trailing 90 days, risk-parity weights):

- **Time-based** — re-optimize every 30 days. Simple, predictable, but trades whether needed or not.
- **Threshold-based** — rebalance only when any weight drifts >5% from target. Trades exactly when the portfolio has deviated.
- **Regime-based** — re-optimize when basket volatility shifts >50% from its level at the last rebalance: rebalances when *market conditions* change, not when the calendar says so.

**How to choose?** Out-of-sample risk-adjusted return **net of costs**, with turnover as the tiebreaker — more rebalancing is only worth it if it buys more than it costs in fees. (With more history, we would validate the choice across multiple market regimes rather than a single year.)
""")

code("""
from src.portfolio.rebalancing import time_based_weights, threshold_based_weights, regime_based_weights

hist = prices_full[prices_full.index <= TEST_END]
schemes_l4 = {
    "time_30d": time_based_weights(hist),
    "threshold_5pct": threshold_based_weights(hist),
    "regime_vol": regime_based_weights(hist),
}

results_l4, returns_l4 = {}, {}
for name, w in schemes_l4.items():
    w_recent = w[w.index >= TEST_START - pd.Timedelta(days=40)]  # carry latest pre-2024 weights in
    res = backtest_portfolio(prices_test, w_recent)
    results_l4[name] = res
    returns_l4[name] = res["portfolio_returns"]
    print(f"{name:16s} rebalances in 2024: {int((w.index >= TEST_START).sum()):3d}   "
          f"total turnover: {res['turnover'].sum():.2f}x")

returns_l4["static_risk_parity"] = returns_l3["risk_parity"]
compare_strategies(returns_l4)
""")

code("""
fig = go.Figure()
for name, res in results_l4.items():
    fig.add_scatter(x=res.index, y=res["equity"], name=name)
fig.add_scatter(x=prices_test.index, y=(1 + returns_l3["risk_parity"]).cumprod(),
                name="static_risk_parity", line=dict(dash="dot"))
fig.update_layout(title="Level 4 — dynamic rebalancing vs static, out-of-sample 2024",
                  yaxis_title="growth of 1 USDT", height=450)
fig.show()
""")

md("""
**Result.** Threshold-based rebalancing achieves the best risk-adjusted performance with roughly *half* the rebalances of the calendar scheme — it trades only when drift is material. This cost-awareness compounds with universe size, which is exactly the regime of Level 5.
""")

md("""
## 5. Portfolio expansion to 100+ pairs

Scaling to 100+ pairs changes the architecture more than the math. The pipeline (`src/portfolio/universe.py`) is staged, and every stage uses only past data:

1. **Pair selection** — at each monthly rebalance, screen for ≥180 days of history and ≥$1M average daily dollar volume over the last 30 days. Liquidity screening is non-negotiable at this scale: a signal you cannot exit is a liability.
2. **Signal prioritization** — cross-sectional **momentum** (90-day return, skipping the last 7 days) ranks the screened universe; only the top 20 with positive momentum get capital. With hundreds of agent signals, ranking + a capital budget *is* the prioritization mechanism.
3. **Risk management** — inverse-volatility sizing with a **10% per-asset cap**; unallocated weight stays in USDT. Portfolio VaR/CVaR are monitored below.

**Operating at this scale (beyond the backtest):** signals from per-pair agents would be normalized to a common score, netted against existing positions, and filtered by the risk agent before execution. Monitoring goes beyond trading KPIs: data-feed freshness, model prediction drift vs. backtest distributions, exchange API health, and slippage-vs-model tracking. Long-term quality is tracked by comparing live rolling Sharpe/hit-rate against the backtest's bootstrap distribution — sustained deviation triggers retraining or shutdown. Fail-safes: the drawdown circuit breaker from Level 2, a global kill-switch flattening the book to USDT, per-asset and gross-exposure caps, and order-size sanity checks vetoing any single trade above a fraction of average volume.
""")

code("""
from src.portfolio.universe import load_universe, large_universe_weights, screen_universe

closes_all, dollar_vol_all = load_universe(min_history_days=365)
closes_all = closes_all[closes_all.index <= TEST_END]
dollar_vol_all = dollar_vol_all[dollar_vol_all.index <= TEST_END]

print(f"Universe loaded: {closes_all.shape[1]} pairs")
print(f"Tradeable on 2024-01-01 after screening: "
      f"{len(screen_universe(closes_all, dollar_vol_all, TEST_START))}")
""")

code("""
rebalance_dates = pd.date_range(TEST_START, TEST_END, freq="30D", tz="UTC")
rebalance_dates = pd.DatetimeIndex([d for d in rebalance_dates if d in closes_all.index])

weights_l5 = large_universe_weights(closes_all, dollar_vol_all, rebalance_dates)
prices_test_all = closes_all[(closes_all.index >= TEST_START) & (closes_all.index <= TEST_END)]

res_l5 = backtest_portfolio(prices_test_all.fillna(method="ffill"), weights_l5)

btc_2024 = btc_test["close"].pct_change().fillna(0.0)
compare_strategies({
    "large_universe_momentum": res_l5["portfolio_returns"],
    "small_universe_threshold (L4)": returns_l4["threshold_5pct"],
    "btc_buy_and_hold": btc_2024,
})
""")

code("""
held = weights_l5.gt(0).sum(axis=1)
fig = go.Figure()
fig.add_scatter(x=res_l5.index, y=res_l5["equity"], name="large-universe portfolio")
fig.add_scatter(x=btc_test.index, y=(1 + btc_2024).cumprod(), name="BTC buy & hold", line=dict(dash="dot"))
fig.update_layout(title=f"Level 5 — 100+ pair universe, monthly momentum rebalance "
                        f"(avg {held.mean():.0f} positions held)",
                  yaxis_title="growth of 1 USDT", height=450)
fig.show()
""")

md("""
## Conclusions

| Level | Key finding |
|---|---|
| 1 | A static SMA rule is a real but fragile edge — strong benchmark discipline (buy-and-hold, fees, OOS year) matters more than the rule itself. |
| 2 | No single model dominates; the agent layer's value is *risk shaping* (vol targeting + circuit breaker), and the permutation test separates timing skill from bull-market beta. |
| 3 | Mean-variance optimization fails out-of-sample through estimation error; schemes that ignore expected returns (min-vol, risk parity) are more robust. |
| 4 | *When* you rebalance matters as much as *how*: threshold-based rebalancing matched calendar rebalancing with half the trades. |
| 5 | At 100+ pairs the problem becomes systems engineering: screening, signal prioritization, risk caps and fail-safes — the architecture from Part 1. |

**Honest limitations.** One out-of-sample year (a bull market) is a small sample; results would need validation across regimes. Execution is modeled with fixed 0.1% fees and no slippage or funding. Hyperparameters were chosen a priori rather than tuned — which protects against overfitting but leaves performance on the table.
""")

nb.cells = cells
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}

out = "ai_hedge_fund.ipynb"
nbf.write(nb, out)
print(f"wrote {out} with {len(cells)} cells")
