"""
fetch_data.py — собирает исторические дневные OHLCV по всем инструментам
(живой список А/B + исследовательский набор) через FMP REST API и
сохраняет в data/*.csv для последующего бэктеста.

Запускается вручную через workflow_dispatch, не по расписанию —
это подготовка данных, а не ежедневный процесс.

Требует переменную окружения FMP_API_KEY (секрет репозитория).
"""

import os
import time
import requests
import pandas as pd
from datetime import date, timedelta

API_KEY = os.environ["FMP_API_KEY"]
BASE_URL = "https://financialmodelingprep.com/stable/historical-price-eod/full"

# (symbol, тип, лет истории)
# тип: "equity" | "forex" | "crypto" — влияет только на глубину истории здесь,
# сам REST-запрос одинаковый для всех классов (просто другой формат symbol)
INSTRUMENTS = [
    # живой список А/B
    ("SPY", "equity", 7),
    ("QQQ", "equity", 7),
    ("EEM", "equity", 7),
    ("TLT", "equity", 7),
    ("XLE", "equity", 7),
    ("EURUSD", "forex", 7),
    ("USDJPY", "forex", 7),
    ("BTCUSD", "crypto", 4),
    # исследовательский набор (+4, не входит в живую торговлю)
    ("IWM", "equity", 7),
    ("GBPUSD", "forex", 7),
    ("ETHUSD", "crypto", 4),
    ("GLD", "equity", 7),
]

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)


def fetch_chunk(symbol: str, from_date: str, to_date: str) -> list[dict]:
    """Один запрос к FMP, не более 5 лет диапазона (ограничение free/stable плана).
    Поднимает FetchError с текстом ответа FMP при HTTP-ошибке (напр. 402) —
    вызывающий код решает, останавливаться на этом тикере или нет."""
    resp = requests.get(
        BASE_URL,
        params={"symbol": symbol, "from": from_date, "to": to_date, "apikey": API_KEY},
        timeout=30,
    )
    if not resp.ok:
        raise FetchError(f"HTTP {resp.status_code} for {symbol}: {resp.text}")
    data = resp.json()
    if not isinstance(data, list):
        print(f"  WARNING: unexpected response for {symbol} {from_date}..{to_date}: {data}")
        return []
    return data


class FetchError(Exception):
    pass


def fetch_full_history(symbol: str, years: int) -> pd.DataFrame:
    """Собирает всю историю кусками по <=5 лет (лимит FMP на один запрос).
    Если тикер недоступен (напр. 402 Payment Required), пробрасывает FetchError —
    ошибка одного тикера не должна портить данные, уже собранные для него частично."""
    end = date.today()
    start = end - timedelta(days=years * 366)
    all_rows: list[dict] = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=5 * 365 - 1), end)
        rows = fetch_chunk(symbol, chunk_start.isoformat(), chunk_end.isoformat())
        all_rows.extend(rows)
        print(f"  {symbol}: {chunk_start} .. {chunk_end} -> {len(rows)} rows")
        chunk_start = chunk_end + timedelta(days=1)
        time.sleep(0.5)  # вежливая пауза между запросами, лимит 250/день не проблема, но не нужно спешить

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    keep_cols = [c for c in ["symbol", "date", "open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep_cols].drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    return df


def main():
    summary = []
    for symbol, kind, years in INSTRUMENTS:
        print(f"Fetching {symbol} ({kind}, {years}y)...")
        try:
            df = fetch_full_history(symbol, years)
        except FetchError as e:
            print(f"  FAILED: {e}")
            summary.append((symbol, kind, 0, None, None, str(e)))
            continue
        except requests.RequestException as e:
            print(f"  FAILED: network error for {symbol}: {e}")
            summary.append((symbol, kind, 0, None, None, str(e)))
            continue
        if df.empty:
            print(f"  FAILED: no data for {symbol}")
            summary.append((symbol, kind, 0, None, None, "empty response"))
            continue
        out_path = os.path.join(OUT_DIR, f"{symbol}.csv")
        df.to_csv(out_path, index=False)
        summary.append((symbol, kind, len(df), df.date.min(), df.date.max(), None))
        print(f"  saved {len(df)} rows -> {out_path}")

    print("\n=== SUMMARY ===")
    for symbol, kind, n, dmin, dmax, error in summary:
        status = "OK" if n > 0 else "FAILED"
        line = f"{symbol:10s} {kind:8s} {n:5d} rows  {dmin} .. {dmax}  [{status}]"
        if error:
            line += f"  -- {error}"
        print(line)


if __name__ == "__main__":
    main()
