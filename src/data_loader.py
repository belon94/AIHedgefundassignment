"""Download OHLCV data for top crypto pairs from Binance via ccxt.


Saves raw daily OHLCV data to ``data/raw`` and a chronological
train/test split to ``data/processed``:

- Train: 2020-01-01 to 2023-12-31
- Test:  2024-01-01 to 2024-12-31
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import ccxt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

TIMEFRAME = "1d"
TRAIN_START = "2020-01-01"
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2024-12-31"


def get_top_symbols(exchange: ccxt.Exchange, top_n: int, quote: str = "USDT") -> list[str]:
    """Return the `top_n` spot symbols quoted in `quote` ranked by 24h quote volume."""
    markets = exchange.load_markets()
    tickers = exchange.fetch_tickers()

    candidates = []
    for symbol, market in markets.items():
        if not market.get("spot") or not market.get("active"):
            continue
        if market.get("quote") != quote:
            continue
        ticker = tickers.get(symbol)
        if ticker is None:
            continue
        volume = ticker.get("quoteVolume") or 0
        candidates.append((symbol, volume))

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [symbol for symbol, _ in candidates[:top_n]]


def fetch_ohlcv(exchange: ccxt.Exchange, symbol: str, timeframe: str, since_ms: int, until_ms: int) -> pd.DataFrame:
    """Fetch OHLCV candles for `symbol` between `since_ms` and `until_ms` (ms epoch)."""
    rows = []
    cursor = since_ms

    while cursor < until_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=1000)
        if not batch:
            break

        rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts == cursor:
            break

        cursor = last_ts + 1
        if len(batch) < 1000:
            break

        time.sleep(exchange.rateLimit / 1000)

    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="timestamp").set_index("timestamp")

    start = pd.to_datetime(since_ms, unit="ms", utc=True)
    end = pd.to_datetime(until_ms, unit="ms", utc=True)
    return df[(df.index >= start) & (df.index < end)]


def save_and_split(df: pd.DataFrame, symbol: str) -> None:
    """Persist raw data and a chronological train/test split for `symbol`."""
    safe_name = symbol.replace("/", "_")

    df.to_csv(RAW_DIR / f"{safe_name}.csv")

    train = df.loc[TRAIN_START:TRAIN_END]
    test = df.loc[TEST_START:TEST_END]

    train.to_csv(PROCESSED_DIR / f"{safe_name}_train.csv")
    test.to_csv(PROCESSED_DIR / f"{safe_name}_test.csv")


def load_symbol(symbol: str, split: str = "full") -> pd.DataFrame:
    """Load saved OHLCV data for `symbol` ("BTC/USDT" or "BTC_USDT").

    `split` is one of "full" (raw history), "train" or "test".
    """
    safe_name = symbol.replace("/", "_")

    if split == "full":
        path = RAW_DIR / f"{safe_name}.csv"
    elif split in ("train", "test"):
        path = PROCESSED_DIR / f"{safe_name}_{split}.csv"
    else:
        raise ValueError(f"split must be 'full', 'train' or 'test', got {split!r}")

    df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    return df


def load_close_prices(symbols: list[str], split: str = "full") -> pd.DataFrame:
    """Load close prices for several symbols into one DataFrame (columns = symbols)."""
    closes = {symbol: load_symbol(symbol, split)["close"] for symbol in symbols}
    return pd.DataFrame(closes)


def available_symbols() -> list[str]:
    """List all symbols with downloaded raw data, e.g. ["BTC/USDT", ...]."""
    return sorted(
        path.stem.replace("_", "/", 1) for path in RAW_DIR.glob("*.csv")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OHLCV data from Binance via ccxt")
    parser.add_argument("--top", type=int, default=120, help="Number of top symbols by quote volume to download")
    parser.add_argument("--quote", type=str, default="USDT", help="Quote currency to filter pairs")
    parser.add_argument("--timeframe", type=str, default=TIMEFRAME, help="OHLCV timeframe")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    exchange = ccxt.binance({"enableRateLimit": True})

    print(f"Fetching top {args.top} {args.quote} pairs by quote volume...")
    symbols = get_top_symbols(exchange, args.top, args.quote)
    print(f"Found {len(symbols)} symbols.")

    since_ms = exchange.parse8601(f"{TRAIN_START}T00:00:00Z")
    until_ms = exchange.parse8601(f"{TEST_END}T00:00:00Z") + 24 * 60 * 60 * 1000

    for i, symbol in enumerate(symbols, start=1):
        print(f"[{i}/{len(symbols)}] {symbol}")
        try:
            df = fetch_ohlcv(exchange, symbol, args.timeframe, since_ms, until_ms)
            if df.empty:
                print(f"  no data returned, skipping")
                continue
            save_and_split(df, symbol)
            print(f"  saved {len(df)} rows")
        except ccxt.BaseError as exc:
            print(f"  failed: {exc}")


if __name__ == "__main__":
    main()
