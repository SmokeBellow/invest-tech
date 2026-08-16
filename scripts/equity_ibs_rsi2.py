"""
Эквити-кривая для System IBS и System RSI2 (см. system_ibs_rsi2.py, CLAUDE.md
5.13) — по годам, на live-8, $1000 старт, та же риск-механика, что и в
equity_curve.py (1% риска на сделку от текущего капитала, потолок открытого
риска 4%, макс. 5 позиций одновременно, компаундинг).

Считает четыре отдельных варианта (IBS<0.1, IBS<0.2, RSI2<5, RSI2<10) плюс
КОМБИНИРОВАННЫЙ портфель, где "боевые" по итоговой доходности IBS<0.2 и
RSI2<10 делят ОДИН капитал и ОДИН риск-бюджет (не два раздельных портфеля).
Приоритет при конфликте бюджета — по entry_date без сортировки по силе
сигнала (тот же упрощённый порядок, что и в equity_curve.py для A/B, см.
открытый вопрос 7.6 CLAUDE.md).

Также считает бенчмарки за тот же период (buy&hold QQQ, buy&hold фонда,
повторяющего индекс МосБиржи IMOEX) в USD и RUB — курс USD/RUB и IMOEX
берутся из data/reference/ (MOEX ISS, публичный API, см. fetch_data.py
комментарий о том, что это НЕ часть боевой 25-инструментной вселенной).

ВАЖНАЯ ОГОВОРКА (см. CLAUDE.md 5.13): торги USD/RUB на MOEX были
приостановлены с 12.06.2024 по 12.05.2026 (санкции против самой биржи) —
для лет в этом окне используется последний доступный биржевой курс ДО
приостановки, не официальный курс ЦБ на дату.

Запуск: python scripts/equity_ibs_rsi2.py [--start-year YYYY]
Выход: results/equity_ibs_rsi2_summary.md
"""
import argparse
import glob
import heapq
import os

import numpy as np
import pandas as pd

import backtest as bt
from system_ibs_rsi2 import add_extra_indicators, backtest_system_ibs, backtest_system_rsi2

DATA_DIR = "data"
REFERENCE_DIR = "data/reference"
RESULTS_DIR = "results"

START_CAPITAL = 1000.0
RISK_PER_TRADE = 0.01
MAX_OPEN_RISK = 0.04
MAX_POSITIONS = 5

BEST_IBS_THRESHOLD = 0.2
BEST_RSI2_THRESHOLD = 10


def trades_with_dates(data, kind, threshold, symbol):
    """Та же логика бэктеста, что backtest_system_ibs/backtest_system_rsi2,
    но с датами входа/выхода вместо одного r_mult — нужно для событийной
    симуляции портфеля (даты определяют, когда сделка занимает/освобождает
    риск-бюджет)."""
    d = data.reset_index(drop=True)
    trades = []
    position = None
    for i in range(1, len(d)):
        row, prev = d.iloc[i], d.iloc[i - 1]
        if position is None:
            if kind == "ibs":
                cond_now = (row.close > row.ema200) and (row.ibs < threshold)
                cond_prev = (prev.close > prev.ema200) and (prev.ibs < threshold)
            else:
                cond_now = (row.close > row.ema200) and (row.rsi2 < threshold)
                cond_prev = (prev.close > prev.ema200) and (prev.rsi2 < threshold)
            if cond_now and not cond_prev and i + 1 < len(d):
                entry_price = d.iloc[i + 1].open
                stop = entry_price - 2.0 * row.atr14
                if not np.isnan(stop) and stop < entry_price:
                    position = {"entry_date": d.iloc[i + 1].date, "entry_price": entry_price,
                                "stop": stop, "bars": 0}
        else:
            rowj = d.iloc[i]
            position["bars"] += 1
            exit_price, reason = None, None
            if rowj.low <= position["stop"]:
                exit_price = rowj.open if rowj.open < position["stop"] else position["stop"]
                reason = "stop"
            elif kind == "ibs" and rowj.ibs > 0.8:
                exit_price, reason = rowj.close, "ibs_exit"
            elif kind == "rsi2" and prev.close <= prev.sma5 and rowj.close > rowj.sma5:
                exit_price, reason = rowj.close, "sma5_cross"
            elif position["bars"] >= 10:
                exit_price, reason = rowj.close, "time_stop"
            if reason:
                ret_pct = (exit_price - position["entry_price"]) / position["entry_price"] * 100
                risk_pct = (position["entry_price"] - position["stop"]) / position["entry_price"] * 100
                r_mult = ret_pct / risk_pct if risk_pct > 0 else np.nan
                trades.append({"symbol": symbol, "entry_date": position["entry_date"],
                               "exit_date": rowj.date, "r_mult": r_mult, "reason": reason})
                position = None
    return trades


def simulate_portfolio(trades, start_capital=START_CAPITAL):
    trades = sorted(trades, key=lambda t: t["entry_date"])
    equity = start_capital
    open_heap = []
    open_risk = 0.0
    curve = [(trades[0]["entry_date"] if trades else None, equity)]
    skipped, taken = 0, 0

    def flush_until(cutoff_date):
        nonlocal equity, open_risk
        while open_heap and open_heap[0][0] <= cutoff_date:
            exit_date, risked, r_mult = heapq.heappop(open_heap)
            equity += r_mult * risked
            open_risk -= RISK_PER_TRADE
            curve.append((exit_date, equity))

    for t in trades:
        flush_until(t["entry_date"])
        if open_risk + RISK_PER_TRADE > MAX_OPEN_RISK + 1e-9 or len(open_heap) >= MAX_POSITIONS:
            skipped += 1
            continue
        risked_dollars = RISK_PER_TRADE * equity
        heapq.heappush(open_heap, (t["exit_date"], risked_dollars, t["r_mult"]))
        open_risk += RISK_PER_TRADE
        taken += 1

    while open_heap:
        exit_date, risked, r_mult = heapq.heappop(open_heap)
        equity += r_mult * risked
        curve.append((exit_date, equity))

    return curve, equity, skipped, taken


def annual_returns(curve, start_capital=START_CAPITAL):
    if not curve or curve[0][0] is None:
        return pd.DataFrame()
    df = pd.DataFrame(curve, columns=["date", "equity"]).dropna(subset=["date"])
    df["year"] = df["date"].apply(lambda d: d.year)
    yearly = df.groupby("year")["equity"].last().reset_index()
    yearly["equity_prev"] = yearly["equity"].shift(1).fillna(start_capital)
    yearly["return_pct"] = (yearly["equity"] / yearly["equity_prev"] - 1) * 100
    return yearly


def load_instrument_data(start_date=None):
    dfs = {}
    for symbol in sorted(bt.LIVE_INSTRUMENTS):
        path = os.path.join(DATA_DIR, f"{symbol}.csv")
        raw = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
        d = bt.add_indicators(raw)
        d = add_extra_indicators(d)
        d = d.iloc[60:].reset_index(drop=True)
        dfs[symbol] = d
    return dfs


def benchmark_table(start_date, combined_curve_csv):
    """QQQ buy&hold и IMOEX-фонд buy&hold в USD/RUB за тот же период, что и
    комбинированный портфель — для сравнения (см. CLAUDE.md 5.13)."""
    usdrub = pd.read_csv(os.path.join(REFERENCE_DIR, "USDRUB.csv"), parse_dates=["date"]).sort_values("date")
    usdrub = usdrub[usdrub["close"] > 0].reset_index(drop=True)

    combined = pd.read_csv(combined_curve_csv).set_index("year")["equity"]

    qqq = pd.read_csv(os.path.join(DATA_DIR, "QQQ.csv"), parse_dates=["date"]).sort_values("date")
    qqq = qqq[qqq.date >= start_date]
    qqq_start_price = qqq.iloc[0]["close"]
    qqq["year"] = qqq.date.dt.year
    qqq_equity = (qqq.groupby("year")["close"].last() / qqq_start_price) * START_CAPITAL

    imoex = pd.read_csv(os.path.join(REFERENCE_DIR, "IMOEX.csv"), parse_dates=["date"]).sort_values("date")
    imoex = imoex[imoex.date >= start_date]
    imoex_year_end = imoex.groupby(imoex.date.dt.year)["close"].last()
    imoex_start_price = imoex.iloc[0]["close"]
    imoex_start_date = imoex.iloc[0]["date"]
    rate_at_start = usdrub[usdrub.date <= imoex_start_date].iloc[-1]["close"]
    start_rub = START_CAPITAL * rate_at_start
    imoex_equity_rub = (imoex_year_end / imoex_start_price) * start_rub

    def rate_asof(year):
        cutoff = pd.Timestamp(f"{year}-12-31")
        sub = usdrub[usdrub.date <= cutoff]
        if sub.empty:
            return None, None
        row = sub.iloc[-1]
        return row["close"], row["date"]

    rows = []
    for y in sorted(combined.index):
        p_usd = combined.get(y, float("nan"))
        rate, rdate = rate_asof(y)
        stale = rdate is not None and rdate.year < y
        p_rub = p_usd * rate if rate else float("nan")
        q_usd = qqq_equity.get(y, float("nan"))
        q_rub = q_usd * rate if rate else float("nan")
        im_rub = imoex_equity_rub.get(y, float("nan"))
        im_usd = im_rub / rate if rate else float("nan")
        rows.append({"year": y, "portfolio_usd": p_usd, "portfolio_rub": p_rub,
                     "qqq_usd": q_usd, "qqq_rub": q_rub,
                     "imoex_fund_rub": im_rub, "imoex_fund_usd": im_usd,
                     "usdrub_rate": rate, "usdrub_stale": stale})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=None,
                         help="Год старта портфеля (по умолчанию — вся доступная история)")
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    dfs = load_instrument_data()
    start_date = pd.Timestamp(f"{args.start_year}-01-01") if args.start_year else pd.Timestamp("1970-01-01")

    variants = [("IBS<0.1", "ibs", 0.1), ("IBS<0.2", "ibs", 0.2),
                ("RSI2<5", "rsi2", 5), ("RSI2<10", "rsi2", 10)]

    summary_lines = ["# System IBS / RSI2 — эквити по годам (live-8, $1000 старт)\n"]
    if args.start_year:
        summary_lines.append(f"Портфель стартует {args.start_year} годом (не переносит капитал с более ранних лет).\n")

    for label, kind, th in variants:
        all_trades = []
        for symbol, d in dfs.items():
            all_trades.extend([t for t in trades_with_dates(d, kind, th, symbol) if t["entry_date"] >= start_date])
        curve, final_equity, skipped, taken = simulate_portfolio(all_trades)
        yearly = annual_returns(curve)
        summary_lines.append(f"## {label}\n\nИтог: ${final_equity:.2f} ({(final_equity/START_CAPITAL-1)*100:+.1f}%), "
                              f"сделок взято/пропущено={taken}/{skipped}\n")
        summary_lines.append(yearly.to_markdown(index=False) + "\n")

    # комбинированный портфель IBS<0.2 + RSI2<10 на одном капитале
    combined_trades = []
    for symbol, d in dfs.items():
        for t in trades_with_dates(d, "ibs", BEST_IBS_THRESHOLD, symbol):
            if t["entry_date"] >= start_date:
                combined_trades.append(t)
        for t in trades_with_dates(d, "rsi2", BEST_RSI2_THRESHOLD, symbol):
            if t["entry_date"] >= start_date:
                combined_trades.append(t)
    curve, final_equity, skipped, taken = simulate_portfolio(combined_trades)
    yearly = annual_returns(curve)
    combined_csv = os.path.join(RESULTS_DIR, "equity_ibs_rsi2_combined.csv")
    yearly.to_csv(combined_csv, index=False)
    summary_lines.append(f"## Комбинированный портфель (IBS<{BEST_IBS_THRESHOLD} + RSI2<{BEST_RSI2_THRESHOLD}, один капитал)\n\n"
                          f"Итог: ${final_equity:.2f} ({(final_equity/START_CAPITAL-1)*100:+.1f}%), "
                          f"сделок взято/пропущено={taken}/{skipped}\n")
    summary_lines.append(yearly.to_markdown(index=False) + "\n")

    # бенчмарки — buy&hold с той же стартовой даты, что и первая сделка комбинированного портфеля
    first_trade_date = min(t["entry_date"] for t in combined_trades) if combined_trades else start_date
    bench = benchmark_table(first_trade_date, combined_csv)
    bench_csv = os.path.join(RESULTS_DIR, "equity_ibs_rsi2_benchmarks.csv")
    bench.to_csv(bench_csv, index=False)
    summary_lines.append("## Бенчмарки: QQQ buy&hold и IMOEX-фонд buy&hold, USD/RUB\n\n"
                          "Курс USD/RUB и IMOEX — MOEX ISS (data/reference/). Торги USD/RUB на MOEX были "
                          "приостановлены с 12.06.2024 по 12.05.2026 (санкции против биржи) — годы с "
                          "`usdrub_stale=True` используют последний биржевой курс до приостановки, "
                          "не официальный курс ЦБ на дату.\n\n")
    summary_lines.append(bench.to_markdown(index=False) + "\n")

    with open(os.path.join(RESULTS_DIR, "equity_ibs_rsi2_summary.md"), "w") as f:
        f.write("\n".join(summary_lines))
    print(f"Wrote results/equity_ibs_rsi2_summary.md, {combined_csv}, {bench_csv}")


if __name__ == "__main__":
    main()
