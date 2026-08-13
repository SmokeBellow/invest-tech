"""
fetch_data.py — собирает исторические дневные OHLCV по всем инструментам
(живой список А/B + исследовательский набор) через FMP REST API и
сохраняет в data/*.csv для последующего бэктеста.

Запускается вручную через workflow_dispatch, не по расписанию —
это подготовка данных, а не ежедневный процесс.

Требует переменную окружения FMP_API_KEY (секрет репозитория).

Опционально: TIINGO_API_KEY — фоллбэк для equity/ETF-тикеров, которые FMP
free-план отдаёт с 402 Payment Required (эмпирически: QQQ, EEM, TLT, XLE,
IWM, GLD — весь набор кроме SPY). Форекс и крипта на FMP проходят без
проблем, поэтому фоллбэк применяется только к kind == "equity". Если
TIINGO_API_KEY не задан, тикеры, недоступные на FMP, просто помечаются
FAILED, как раньше.
"""

import os
import time
import requests
import pandas as pd
from datetime import date, timedelta

API_KEY = os.environ["FMP_API_KEY"]
BASE_URL = "https://financialmodelingprep.com/stable/historical-price-eod/full"

TIINGO_API_KEY = os.environ.get("TIINGO_API_KEY")
TIINGO_BASE_URL = "https://api.tiingo.com/tiingo/daily"

# (symbol, тип, лет истории)
# тип: "equity" | "forex" | "crypto" — влияет только на глубину истории здесь,
# сам REST-запрос одинаковый для всех классов (просто другой формат symbol)
#
# Глубина истории для equity/forex расширена с 7 до 20 лет (12.08.2026) —
# честный способ увеличить число walk-forward окон и статистическую мощность
# без размытия торгуемого набора или ослабления входа (см. CLAUDE.md 5.1, 7).
# Крипта оставлена на 4 годах — старая история (2018-2021) структурно
# нерепрезентативна для сегодняшнего рынка, решение не пересматривается.
INSTRUMENTS = [
    # живой список А/B
    ("SPY", "equity", 20),
    ("QQQ", "equity", 20),
    ("EEM", "equity", 20),
    ("TLT", "equity", 20),
    ("XLE", "equity", 20),
    ("EURUSD", "forex", 20),
    ("USDJPY", "forex", 20),
    ("BTCUSD", "crypto", 4),
    # исследовательский набор (+4, не входит в живую торговлю)
    ("IWM", "equity", 20),
    ("GBPUSD", "forex", 20),
    ("ETHUSD", "crypto", 4),
    ("GLD", "equity", 20),
    # расширение вотч-листа для статистической мощности бэктеста (12.08.2026) —
    # НЕ входит в живую торговлю, только пул сделок в backtest.py. Секторальные
    # ETF и пары для диверсификации набора без изменения самой методологии.
    ("XLF", "equity", 20),
    ("XLK", "equity", 20),
    ("XLV", "equity", 20),
    ("DIA", "equity", 20),
    ("VNQ", "equity", 20),
    ("AUDUSD", "forex", 20),
    ("USDCHF", "forex", 20),
    # кросс-классовое расширение (13.08.2026) — предыдущее расширение (XLF..USDCHF)
    # добавляло в основном ещё секторов акций и валютных пар, т.е. диверсификацию
    # ВНУТРИ уже представленных классов (акции, FX). Академические работы по
    # time-series momentum (Moskowitz/Ooi/Pedersen) тестируют край на портфеле из
    # НЕСКОЛЬКИХ классов активов одновременно (акции, бонды, сырьё, валюты) — этого
    # у нас не хватало. Добавлены бонды другой дюрации и сырьевые ETF, которых
    # раньше не было вовсе (TLT — единственный бонд, GLD — единственное сырьё).
    # НЕ входит в живую торговлю, только пул сделок в backtest.py.
    ("IEF", "equity", 20),   # казначейские облигации США 7-10 лет
    ("SHY", "equity", 20),   # казначейские облигации США 1-3 года
    ("DBC", "equity", 20),   # широкий сырьевой индекс (не только золото)
    ("USO", "equity", 20),   # нефть
    ("EFA", "equity", 20),   # развитые рынки вне США
    ("VGK", "equity", 20),   # Европа
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


def fetch_tiingo_history(symbol: str, years: int) -> pd.DataFrame:
    """Фоллбэк через Tiingo для тикеров, недоступных на FMP free-плане.
    Один запрос покрывает весь диапазон — у Tiingo нет ограничения в 5 лет
    на запрос, в отличие от FMP."""
    end = date.today()
    start = end - timedelta(days=years * 366)
    resp = requests.get(
        f"{TIINGO_BASE_URL}/{symbol.lower()}/prices",
        params={"startDate": start.isoformat(), "endDate": end.isoformat(),
                "token": TIINGO_API_KEY, "format": "json"},
        timeout=30,
    )
    if not resp.ok:
        raise FetchError(f"Tiingo HTTP {resp.status_code} for {symbol}: {resp.text}")
    rows = resp.json()
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    keep_cols = [c for c in ["symbol", "date", "open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep_cols].drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    return df


def main():
    summary = []
    for symbol, kind, years in INSTRUMENTS:
        print(f"Fetching {symbol} ({kind}, {years}y)...")
        source, error = "FMP", None
        try:
            df = fetch_full_history(symbol, years)
        except FetchError as e:
            print(f"  FMP FAILED: {e}")
            df, error = pd.DataFrame(), str(e)
        except requests.RequestException as e:
            print(f"  FMP FAILED: network error for {symbol}: {e}")
            df, error = pd.DataFrame(), str(e)

        if df.empty and kind == "equity" and TIINGO_API_KEY:
            print(f"  Falling back to Tiingo for {symbol}...")
            try:
                df = fetch_tiingo_history(symbol, years)
                if not df.empty:
                    source, error = "Tiingo", None
                    print(f"  Tiingo: {len(df)} rows")
            except FetchError as e:
                print(f"  Tiingo FAILED: {e}")
                error = f"FMP: {error} | Tiingo: {e}"
            except requests.RequestException as e:
                print(f"  Tiingo FAILED: network error for {symbol}: {e}")
                error = f"FMP: {error} | Tiingo: network error: {e}"

        if df.empty:
            print(f"  FAILED: no data for {symbol} from any source")
            summary.append((symbol, kind, 0, None, None, "FAILED", error or "empty response"))
            continue
        out_path = os.path.join(OUT_DIR, f"{symbol}.csv")
        df.to_csv(out_path, index=False)
        summary.append((symbol, kind, len(df), df.date.min(), df.date.max(), source, None))
        print(f"  saved {len(df)} rows -> {out_path} (source: {source})")

    print("\n=== SUMMARY ===")
    for symbol, kind, n, dmin, dmax, source, error in summary:
        status = "OK" if n > 0 else "FAILED"
        line = f"{symbol:10s} {kind:8s} {n:5d} rows  {dmin} .. {dmax}  [{status}, {source}]"
        if error:
            line += f"  -- {error}"
        print(line)


if __name__ == "__main__":
    main()
