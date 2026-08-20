"""
live_fetch_incremental.py — ежедневный ИНКРЕМЕНТАЛЬНЫЙ сбор свежих баров
для live paper-trading боевой стратегии (см. CLAUDE.md, раздел 9).

В отличие от fetch_data.py (полный сбор истории, кэшируется — если файл
уже есть, пропускается), этот скрипт КАЖДЫЙ РАЗ дозаписывает только новые
строки (от последней даты в data/{symbol}.csv до сегодня) в уже
существующие CSV. Не трогает историю до последней сохранённой даты.

Источник тот же, что и в fetch_data.py: FMP (SPY + форекс/крипта проходят
без проблем), автоматический фоллбэк на Tiingo для equity-тикеров, которые
FMP free-план отдаёт с 402 (QQQ, EEM, TLT, XLE, GLD, SHY — см. CLAUDE.md 4).

Список тикеров — живой список А/B (3.1 CLAUDE.md) плюс GLD/SHY (нужны для
вечного портфеля внутри блендинга, см. 5.17).

Запускается ежедневно из GitHub Actions (live_review.yml), не вручную.
Требует FMP_API_KEY (обязательно), TIINGO_API_KEY (опционально, но нужен
для большинства equity-тикеров, см. CLAUDE.md 4).
"""
import os
import sys
from datetime import date, timedelta

import pandas as pd
import requests

API_KEY = os.environ["FMP_API_KEY"]
BASE_URL = "https://financialmodelingprep.com/stable/historical-price-eod/full"
TIINGO_API_KEY = os.environ.get("TIINGO_API_KEY")
TIINGO_BASE_URL = "https://api.tiingo.com/tiingo/daily"

OUT_DIR = "data"

# (symbol, kind) — живые А/B инструменты + GLD/SHY для вечного портфеля внутри блендинга
# + SGOV (19.08.2026) — парковка свободного кэша Комбо C, см. CLAUDE.md §9
LIVE_FETCH_LIST = [
    ("SPY", "equity"), ("QQQ", "equity"), ("EEM", "equity"), ("TLT", "equity"), ("XLE", "equity"),
    ("EURUSD", "forex"), ("USDJPY", "forex"), ("BTCUSD", "crypto"),
    ("GLD", "equity"), ("SHY", "equity"), ("SGOV", "equity"),
]


class FetchError(Exception):
    pass


def fetch_fmp(symbol, from_date, to_date):
    resp = requests.get(BASE_URL, params={"symbol": symbol, "from": from_date, "to": to_date, "apikey": API_KEY},
                         timeout=30)
    if not resp.ok:
        raise FetchError(f"HTTP {resp.status_code} for {symbol}: {resp.text}")
    data = resp.json()
    if not isinstance(data, list):
        return []
    return data


def fetch_tiingo(symbol, from_date, to_date):
    resp = requests.get(f"{TIINGO_BASE_URL}/{symbol.lower()}/prices",
                         params={"startDate": from_date, "endDate": to_date, "token": TIINGO_API_KEY,
                                 "format": "json"}, timeout=30)
    if not resp.ok:
        raise FetchError(f"Tiingo HTTP {resp.status_code} for {symbol}: {resp.text}")
    rows = resp.json()
    if not isinstance(rows, list):
        return []
    return rows


def update_symbol(symbol, kind):
    path = os.path.join(OUT_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        print(f"  {symbol}: SKIP — {path} не существует, сначала запустите fetch_data.py")
        return "missing", 0

    existing = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    last_date = existing["date"].max().date()
    today = date.today()
    if last_date >= today:
        print(f"  {symbol}: уже актуален ({last_date})")
        return "up_to_date", 0

    from_date = (last_date + timedelta(days=1)).isoformat()
    to_date = today.isoformat()

    source = "FMP"
    try:
        rows = fetch_fmp(symbol, from_date, to_date)
    except FetchError as e:
        print(f"  {symbol}: FMP FAILED ({e})")
        rows = []

    if not rows and kind == "equity" and TIINGO_API_KEY:
        try:
            rows = fetch_tiingo(symbol, from_date, to_date)
            source = "Tiingo"
        except FetchError as e:
            print(f"  {symbol}: Tiingo FAILED ({e})")
            rows = []

    if not rows:
        print(f"  {symbol}: нет новых строк ({from_date}..{to_date})")
        return "no_new_data", 0

    new_df = pd.DataFrame(rows)
    if "date" not in new_df.columns:
        print(f"  {symbol}: неожиданный формат ответа, пропуск")
        return "bad_response", 0
    new_df["date"] = pd.to_datetime(new_df["date"]).dt.strftime("%Y-%m-%d")
    keep_cols = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in new_df.columns]
    new_df = new_df[keep_cols]

    combined = pd.concat([existing.assign(date=existing["date"].dt.strftime("%Y-%m-%d")), new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    combined.to_csv(path, index=False)
    added = len(combined) - len(existing)
    print(f"  {symbol}: +{added} новых строк из {source} ({from_date}..{to_date}), теперь до {combined.date.max()}")
    return "updated", added


def main():
    print(f"Инкрементальное обновление данных, {date.today().isoformat()}")
    summary = []
    for symbol, kind in LIVE_FETCH_LIST:
        status, added = update_symbol(symbol, kind)
        summary.append((symbol, status, added))

    print("\n=== SUMMARY ===")
    any_updated = False
    for symbol, status, added in summary:
        print(f"{symbol:10s} {status:12s} +{added}")
        if status == "updated":
            any_updated = True
    # код возврата для GH Actions — не считаем отсутствие новых строк ошибкой
    # (выходные, рынок закрыт), только реальный сбой источника
    hard_fail = any(s == "bad_response" for _, s, _ in summary)
    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
