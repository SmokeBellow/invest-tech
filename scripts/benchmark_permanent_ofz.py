"""
Бенчмарки для сравнения с System RSI2+TOM (см. combo_rsi2_tom.py):
- Вечный портфель (Harry Browne): 25% SPY / 25% TLT / 25% SHY / 25% GLD,
  ребаланс раз в год на последний торговый день года.
- ОФЗ: индекс полной доходности гособлигаций РФ RGBITR (MOEX ISS,
  data/reference/RGBITR.csv) — реальный держатель облигаций получает и
  переоценку тела, и купон, поэтому берём индекс ПОЛНОЙ доходности, не
  ценовой RGBI. Рублёвый инструмент — переводим в USD/RUB по курсу MOEX
  ISS (тот же источник и та же оговорка про приостановку торгов USD/RUB
  12.06.2024-12.05.2026, что и в equity_ibs_rsi2.py).

Запуск: python scripts/benchmark_permanent_ofz.py [--start-year YYYY]
Выход: results/benchmark_permanent_portfolio.csv, results/benchmark_ofz.csv
"""
import argparse
import os

import pandas as pd

DATA_DIR = "data"
REFERENCE_DIR = "data/reference"
RESULTS_DIR = "results"

START_CAPITAL = 1000.0
PERMANENT_ASSETS = ["SPY", "TLT", "SHY", "GLD"]


def permanent_portfolio_daily(start_date):
    """Дневная кривая вечного портфеля с ФАКТИЧЕСКИМ ежегодным ребалансом.

    Баг, исправленный 17.08.2026: раньше вся кривая строилась ОДНИМ проходом
    по стартовым долям, а цикл ребаланса запускался ПОСЛЕ и пересчитывал
    shares в пустоту — ни на что уже не влияя. Портфель фактически был
    buy&hold от первой даты, несмотря на то что везде документирован как
    "ребаланс раз в год". Правильно: пересчитывать доли ВНУТРИ прохода по
    датам, сразу после последнего дня каждого года."""
    prices = {}
    for a in PERMANENT_ASSETS:
        df = pd.read_csv(os.path.join(DATA_DIR, f"{a}.csv"), parse_dates=["date"]).sort_values("date")
        df = df[df.date >= start_date].reset_index(drop=True)
        prices[a] = df.set_index("date")["close"]

    common_dates = pd.DatetimeIndex(sorted(set.intersection(*[set(p.index) for p in prices.values()])))
    first_date = common_dates[0]
    shares = {a: (START_CAPITAL / len(PERMANENT_ASSETS)) / prices[a].loc[first_date] for a in PERMANENT_ASSETS}
    years = pd.Series(common_dates).dt.year
    n = len(common_dates)

    curve = []
    for i, d in enumerate(common_dates):
        value = sum(shares[a] * prices[a].loc[d] for a in PERMANENT_ASSETS)
        curve.append((d, value))
        is_last_of_year = (i == n - 1) or (years.iloc[i + 1] != years.iloc[i])
        if is_last_of_year and i != n - 1:  # не ребалансируем на самом последнем дне истории
            shares = {a: (value / len(PERMANENT_ASSETS)) / prices[a].loc[d] for a in PERMANENT_ASSETS}

    curve_df = pd.DataFrame(curve, columns=["date", "equity"]).set_index("date")
    return curve_df["equity"]


def permanent_portfolio(start_date):
    daily = permanent_portfolio_daily(start_date)
    curve_df = daily.reset_index()
    curve_df.columns = ["date", "equity"]
    curve_df["year"] = curve_df.date.dt.year
    yearly = curve_df.groupby("year")["equity"].last().reset_index()
    yearly["equity_prev"] = yearly["equity"].shift(1).fillna(START_CAPITAL)
    yearly["return_pct"] = (yearly["equity"] / yearly["equity_prev"] - 1) * 100
    return yearly


def ofz_benchmark(start_date):
    usdrub = pd.read_csv(os.path.join(REFERENCE_DIR, "USDRUB.csv"), parse_dates=["date"]).sort_values("date")
    usdrub = usdrub[usdrub["close"] > 0].reset_index(drop=True)

    rgbi = pd.read_csv(os.path.join(REFERENCE_DIR, "RGBITR.csv"), parse_dates=["date"]).sort_values("date")
    rgbi = rgbi[rgbi.date >= start_date].reset_index(drop=True)
    rgbi_start_price = rgbi.iloc[0]["close"]
    rgbi_start_date = rgbi.iloc[0]["date"]
    rate_at_start = usdrub[usdrub.date <= rgbi_start_date].iloc[-1]["close"]
    start_rub = START_CAPITAL * rate_at_start
    rgbi["year"] = rgbi.date.dt.year
    rgbi_equity_rub = (rgbi.groupby("year")["close"].last() / rgbi_start_price) * start_rub

    def rate_asof(year):
        cutoff = pd.Timestamp(f"{year}-12-31")
        sub = usdrub[usdrub.date <= cutoff]
        if sub.empty:
            return None, None
        row = sub.iloc[-1]
        return row["close"], row["date"]

    rows = []
    for y in sorted(rgbi_equity_rub.index):
        rate, rdate = rate_asof(y)
        stale = rdate is not None and rdate.year < y
        rub = rgbi_equity_rub.get(y, float("nan"))
        usd = rub / rate if rate else float("nan")
        rows.append({"year": y, "ofz_rub": rub, "ofz_usd": usd, "usdrub_rate": rate, "usdrub_stale": stale})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2016)
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    start_date = pd.Timestamp(f"{args.start_year}-01-01")

    perm = permanent_portfolio(start_date)
    perm.to_csv(os.path.join(RESULTS_DIR, "benchmark_permanent_portfolio.csv"), index=False)
    print("=== Вечный портфель (25% SPY/TLT/SHY/GLD, ребаланс раз в год) ===")
    print(perm.to_string(index=False))

    ofz = ofz_benchmark(start_date)
    ofz.to_csv(os.path.join(RESULTS_DIR, "benchmark_ofz.csv"), index=False)
    print("\n=== ОФЗ (RGBITR, индекс полной доходности гособлигаций РФ) ===")
    print(ofz.to_string(index=False))


if __name__ == "__main__":
    main()
