"""
Диагностика (17.08.2026, по запросу пользователя): что если заменить вторую
долю финального блендинга (вечный портфель, см. blend_combo_permanent.py,
CLAUDE.md 5.17/5.18) на простой buy&hold индекса — QQQ или SPY? Тот же
механизм смешивания (ежегодный ребаланс между двумя долями), но вторая доля
не диверсифицированный вечный портфель, а один инструмент.

НЕ меняет боевое решение (40% Комбо C / 60% вечный портфель остаётся в
силе, см. blend_combo_permanent.py) — это отдельное исследование
"что если", результаты см. CLAUDE.md 5.19.

Запуск: python scripts/blend_combo_benchmarks.py --benchmark QQQ
        python scripts/blend_combo_benchmarks.py --benchmark SPY
Выход: results/blend_combo_{benchmark}_sweep.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

from blend_combo_permanent import combo_c_curve, max_drawdown_pct, START_CAPITAL

DATA_DIR = "data"
RESULTS_DIR = "results"
WEIGHT_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def benchmark_daily_curve(symbol, start_date):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{symbol}.csv"), parse_dates=["date"]).sort_values("date")
    df = df[df.date >= start_date].reset_index(drop=True)
    shares = START_CAPITAL / df.close.iloc[0]
    return pd.Series((df.close * shares).values, index=df.date)


def blend_generic(weight, sleeve_a, sleeve_b):
    """sleeve_a получает `weight` капитала, sleeve_b — (1-weight), ежегодный ребаланс."""
    all_days = pd.DatetimeIndex(sorted(set(sleeve_a.index) | set(sleeve_b.index)))
    a_mult = sleeve_a.reindex(all_days).ffill().bfill()
    a_mult = a_mult / a_mult.iloc[0]
    b_mult = sleeve_b.reindex(all_days).ffill().bfill()
    b_mult = b_mult / b_mult.iloc[0]

    years = pd.Series(all_days).dt.year
    equity = []
    a_dollars, b_dollars = START_CAPITAL * weight, START_CAPITAL * (1 - weight)
    a_base, b_base = a_mult.iloc[0], b_mult.iloc[0]
    cur_year = years.iloc[0]
    for i, d in enumerate(all_days):
        y = years.iloc[i]
        if y != cur_year:
            total = a_dollars * (a_mult.loc[d] / a_base) + b_dollars * (b_mult.loc[d] / b_base)
            a_dollars, b_dollars = total * weight, total * (1 - weight)
            a_base, b_base = a_mult.loc[d], b_mult.loc[d]
            cur_year = y
        val = a_dollars * (a_mult.loc[d] / a_base) + b_dollars * (b_mult.loc[d] / b_base)
        equity.append(val)
    return pd.Series(equity, index=all_days)


def stats(series):
    final = series.iloc[-1]
    yearly = series.groupby(series.index.year).last()
    yearly_prev = yearly.shift(1)
    yearly_prev.iloc[0] = START_CAPITAL
    ret = (yearly / yearly_prev - 1) * 100
    dd = max_drawdown_pct(series)
    calmar = ((final / START_CAPITAL - 1) * 100) / abs(dd) if dd != 0 else np.nan
    return {"ret_20y": (final / START_CAPITAL - 1) * 100, "mean_annual": ret.mean(),
            "median_annual": ret.median(), "max_dd": dd, "calmar": calmar, "final": final}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="QQQ", help="Тикер для второй доли (QQQ, SPY, ...)")
    parser.add_argument("--start-year", type=int, default=2007)
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    start_date = pd.Timestamp(f"{args.start_year}-01-01")
    end_date = pd.Timestamp("2026-12-31")

    combo_s = combo_c_curve(start_date, end_date)
    bench_s = benchmark_daily_curve(args.benchmark, start_date)

    rows = []
    for w in WEIGHT_GRID:
        s = blend_generic(w, combo_s, bench_s)
        r = stats(s)
        rows.append({"weight_combo_c": w, **r})
        print(f"{w*100:>5.0f}% Combo C: итог {r['ret_20y']:+.1f}%, сред/год {r['mean_annual']:+.1f}%, "
              f"медиана {r['median_annual']:+.1f}%, макс.просадка {r['max_dd']:.1f}%, Calmar {r['calmar']:.2f}")

    out = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, f"blend_combo_{args.benchmark.lower()}_sweep.csv")
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
