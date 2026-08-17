"""
Финальное боевое решение сессии (15.08.2026, см. CLAUDE.md 5.17):
40% капитала в Комбо C (System RSI2<10 + System TOM offset=5,
leftover_only, потолок риска 8%), 60% в вечном портфеле (25%
SPY/TLT/SHY/GLD), ежегодный ребаланс между двумя долями.

Сетка по доле Комбо C (0-100%) показала эффект диверсификации: 0%→30%
доходность растёт, а просадка ОДНОВРЕМЕННО падает (не размен, а бесплатное
улучшение) — Комбо C и вечный портфель не проседают одновременно. 30% —
точка максимума Calmar-отношения (доходность/просадка). Пользователь выбрал
40% — на шаге 30%→40% курс обмена (доп. доходность за 1 доп.% просадки)
всё ещё один из лучших (7.13), дальше он монотонно ухудшается.

Запуск: python scripts/blend_combo_permanent.py [--weight 0.4]
Выход: results/blend_combo_permanent.csv (дневная кривая), печатает сводку.
"""
import argparse
import os

import numpy as np
import pandas as pd

import backtest as bt
from system_ibs_rsi2 import add_extra_indicators
from combo_rsi2_tom import gen_rsi2_trades, gen_tom_trades, simulate_shared
from benchmark_permanent_ofz import PERMANENT_ASSETS

DATA_DIR = "data"
RESULTS_DIR = "results"
START_CAPITAL = 1000.0


def combo_c_curve(start_date, end_date):
    dfs, close_lookup = {}, {}
    for symbol in sorted(bt.LIVE_INSTRUMENTS):
        raw = pd.read_csv(os.path.join(DATA_DIR, f"{symbol}.csv"), parse_dates=["date"]).sort_values("date")
        d = bt.add_indicators(raw)
        d2 = add_extra_indicators(d)
        d2 = d2.iloc[60:].reset_index(drop=True)
        dfs[symbol] = d2
        close_lookup[symbol] = raw.set_index("date")["close"].sort_index()

    all_rsi2, all_tom = [], []
    for symbol, d in dfs.items():
        all_rsi2.extend([t for t in gen_rsi2_trades(d, symbol) if start_date <= t["entry_date"] <= end_date])
        all_tom.extend([t for t in gen_tom_trades(d, symbol) if start_date <= t["entry_date"] <= end_date])

    curve, _, _, _, _ = simulate_shared(all_rsi2, all_tom, close_lookup, "leftover_only")
    df = pd.DataFrame([c for c in curve if c[0] is not None], columns=["date", "equity"])
    df = df.drop_duplicates("date", keep="last").set_index("date").sort_index()
    return df["equity"]


def permanent_daily_curve(start_date):
    prices = {}
    for a in PERMANENT_ASSETS:
        df = pd.read_csv(os.path.join(DATA_DIR, f"{a}.csv"), parse_dates=["date"]).sort_values("date")
        df = df[df.date >= start_date].reset_index(drop=True)
        prices[a] = df.set_index("date")["close"]
    common_dates = pd.DatetimeIndex(sorted(set.intersection(*[set(p.index) for p in prices.values()])))
    shares = {a: (START_CAPITAL / len(PERMANENT_ASSETS)) / prices[a].loc[common_dates[0]] for a in PERMANENT_ASSETS}
    curve = []
    for d in common_dates:
        val = sum(shares[a] * prices[a].loc[d] for a in PERMANENT_ASSETS)
        curve.append((d, val))
    by_year = pd.Series(common_dates).groupby(pd.Series(common_dates).dt.year)
    for year, dates_in_year in by_year:
        last_date = dates_in_year.iloc[-1]
        if last_date == common_dates[-1]:
            continue
        val = sum(shares[a] * prices[a].loc[last_date] for a in PERMANENT_ASSETS)
        shares = {a: (val / len(PERMANENT_ASSETS)) / prices[a].loc[last_date] for a in PERMANENT_ASSETS}
    df = pd.DataFrame(curve, columns=["date", "equity"]).set_index("date").sort_index()
    return df["equity"]


def blend(weight, start_date, end_date):
    combo_s = combo_c_curve(start_date, end_date)
    perm_s = permanent_daily_curve(start_date)

    all_days = pd.DatetimeIndex(sorted(set(combo_s.index) | set(perm_s.index)))
    combo_mult = combo_s.reindex(all_days).ffill().bfill()
    combo_mult = combo_mult / combo_mult.iloc[0]
    perm_mult = perm_s.reindex(all_days).ffill().bfill()
    perm_mult = perm_mult / perm_mult.iloc[0]

    years = pd.Series(all_days).dt.year
    equity = []
    combo_dollars = START_CAPITAL * weight
    perm_dollars = START_CAPITAL * (1 - weight)
    combo_base = combo_mult.iloc[0]
    perm_base = perm_mult.iloc[0]
    cur_year = years.iloc[0]
    for i, d in enumerate(all_days):
        y = years.iloc[i]
        if y != cur_year:
            total = combo_dollars * (combo_mult.loc[d] / combo_base) + perm_dollars * (perm_mult.loc[d] / perm_base)
            combo_dollars = total * weight
            perm_dollars = total * (1 - weight)
            combo_base = combo_mult.loc[d]
            perm_base = perm_mult.loc[d]
            cur_year = y
        val = combo_dollars * (combo_mult.loc[d] / combo_base) + perm_dollars * (perm_mult.loc[d] / perm_base)
        equity.append(val)
    return pd.Series(equity, index=all_days)


def max_drawdown_pct(series):
    peak = series.iloc[0]
    max_dd = 0.0
    for v in series:
        if v > peak:
            peak = v
        dd = (v - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd
    return max_dd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=float, default=0.4, help="Доля капитала в Комбо C (по умолчанию 0.4)")
    parser.add_argument("--start-year", type=int, default=2007)
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    start_date = pd.Timestamp(f"{args.start_year}-01-01")
    end_date = pd.Timestamp("2026-12-31")

    series = blend(args.weight, start_date, end_date)
    final = series.iloc[-1]
    yearly = series.groupby(series.index.year).last()
    yearly_prev = yearly.shift(1)
    yearly_prev.iloc[0] = START_CAPITAL
    ret = (yearly / yearly_prev - 1) * 100
    dd = max_drawdown_pct(series)

    print(f"Комбо C {args.weight*100:.0f}% / Вечный портфель {(1-args.weight)*100:.0f}%, ребаланс раз в год")
    print(f"Итог: ${final:.2f} ({(final/START_CAPITAL-1)*100:+.1f}%), средняя/год={ret.mean():+.1f}%, "
          f"медиана/год={ret.median():+.1f}%, макс. просадка={dd:.1f}%")

    series.to_frame("equity").to_csv(os.path.join(RESULTS_DIR, "blend_combo_permanent.csv"))


if __name__ == "__main__":
    main()
