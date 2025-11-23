"""下載 Binance BTC/USDT 1 分鐘 K 線資料並存成 CSV。
此腳本使用 ccxt 從公開端點抓取資料，供後續做市 RL 環境使用。
"""
from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path
from typing import List

import ccxt
import pandas as pd

# 預設參數設定
DEFAULT_SYMBOL = "BTC/USDT"
DEFAULT_TIMEFRAME = "1m"
DEFAULT_SINCE = "2023-01-01 00:00:00"
MAX_LIMIT = 1000
SLEEP_SECONDS = 0.2
TIMEZONE = "Asia/Taipei"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="抓取 Binance OHLCV 資料")
    parser.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL, help="交易對，例如 BTC/USDT")
    parser.add_argument("--timeframe", type=str, default=DEFAULT_TIMEFRAME, help="時間週期，例如 1m")
    parser.add_argument(
        "--since",
        type=str,
        default=DEFAULT_SINCE,
        help="開始時間（UTC），格式 YYYY-MM-DD HH:MM:SS",
    )
    parser.add_argument(
        "--until",
        type=str,
        default=None,
        help="結束時間（UTC，可選），格式 YYYY-MM-DD HH:MM:SS",
    )
    return parser.parse_args()


def utc_str_to_ms(timestamp_str: str) -> int:
    """將 UTC 字串轉為毫秒 timestamp。"""
    dt_obj = dt.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    dt_obj = dt_obj.replace(tzinfo=dt.timezone.utc)
    return int(dt_obj.timestamp() * 1000)


def format_filename(symbol: str, timeframe: str, since: str) -> str:
    year = since[:4]
    clean_symbol = symbol.replace("/", "_").lower()
    return f"{clean_symbol}_{timeframe}_{year}.csv"


def fetch_ohlcv(symbol: str, timeframe: str, since_ms: int, until_ms: int | None) -> pd.DataFrame:
    exchange = ccxt.binance({"enableRateLimit": True})
    all_ohlcv: List[List[float]] = []
    current_since = since_ms

    print(f"開始抓取 {symbol} {timeframe}，起始時間 {dt.datetime.utcfromtimestamp(since_ms/1000)} UTC")

    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=current_since, limit=MAX_LIMIT)
        if not ohlcv:
            print("無更多資料，結束下載。")
            break

        all_ohlcv.extend(ohlcv)
        last_ts = ohlcv[-1][0]
        last_dt = dt.datetime.utcfromtimestamp(last_ts / 1000)
        print(f"抓取到 {last_dt} UTC，共累積 {len(all_ohlcv)} 筆")

        current_since = last_ts + 60_000  # 下一分鐘
        if until_ms and current_since >= until_ms:
            print("已達指定結束時間，停止抓取。")
            break

        time.sleep(SLEEP_SECONDS)

    if not all_ohlcv:
        raise RuntimeError("沒有抓到任何資料，請檢查參數或網路連線。")

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(TIMEZONE)
    df = df[["timestamp", "datetime", "open", "high", "low", "close", "volume"]]
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def main() -> None:
    args = parse_args()
    since_ms = utc_str_to_ms(args.since)
    until_ms = utc_str_to_ms(args.until) if args.until else None

    df = fetch_ohlcv(symbol=args.symbol, timeframe=args.timeframe, since_ms=since_ms, until_ms=until_ms)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filename = format_filename(args.symbol, args.timeframe, args.since)
    output_path = DATA_DIR / filename
    df.to_csv(output_path, index=False)

    print("下載完成，前幾列：")
    print(df.head())
    print(f"總筆數：{len(df)}，輸出檔案：{output_path}")


if __name__ == "__main__":
    main()
