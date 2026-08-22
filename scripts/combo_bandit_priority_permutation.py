"""
Перестановочный тест (permutation test, разновидность bootstrap) для находки
§10.2/10.4/10.5 CLAUDE.md — по запросу пользователя, 22.08.2026.

Мотивация: t-тест (10.5) показал, что бандит НЕ отбирает сделки более
высокого КАЧЕСТВА (r_mult), чем baseline — разница в среднем r_mult не
значима (p=0.72). Но $-превосходство бандита реально и устойчиво (10.4).
Правильная гипотеза для проверки — НЕ "лучше ли бандит baseline", а "лучше
ли КОНКРЕТНЫЙ порядок допуска бандита (и хуже ли конкретный порядок
baseline), чем ТИПИЧНЫЙ порядок допуска при конфликте бюджета?" Baseline —
сам по себе произвольный порядок (список TOM, затем список RSI2, без
приоритизации внутри) — не обязательно "средний" случай.

Механика: при конфликте бюджета в один день кандидаты допускаются в
СЛУЧАЙНОМ порядке (вместо UCB1-скора или baseline-порядка), N=300 прогонов
с разными seed. Даёт эмпирическое распределение итоговой $-эквити для
"типичного" порядка допуска. Затем: на каком перцентиле этого распределения
оказываются (а) фактический baseline и (б) фактический бандит (c=0.5)?

Запуск: python scripts/combo_bandit_priority_permutation.py
Выход: results/combo_bandit_priority_permutation.csv, печатает сводку.
"""
import glob
import os
import random

import numpy as np
import pandas as pd

from combo_rsi2_tom import (gen_rsi2_trades, gen_tom_trades, r_mult_of, simulate_shared,
                             load_sgov_returns, RISK_PER_TRADE, MAX_OPEN_RISK, MAX_POSITIONS)
from combo_bandit_priority import simulate_bandit_priority, load_universe, position_notional, max_dd, START_CAPITAL

DATA_DIR = "data"
RESULTS_DIR = "results"
START_DATE = pd.Timestamp("2007-01-01")
END_DATE = pd.Timestamp("2026-12-31")
N_PERMUTATIONS = 300


def simulate_random_priority(rsi2_trades, tom_trades, close_lookup, sgov_returns, calendar_start,
                              calendar_end, seed, start_capital=START_CAPITAL):
    """Та же механика допуска, что simulate_bandit_priority, но порядок
    кандидатов при конфликте — СЛУЧАЙНЫЙ (не по UCB1-скору). Даёт один
    пример из пространства "типичных" порядков допуска."""
    rng = random.Random(seed)
    all_dates = sorted(set(t["entry_date"] for t in rsi2_trades + tom_trades) |
                        set(t["exit_date"] for t in rsi2_trades + tom_trades))
    candidates_by_entry = {}
    for t in rsi2_trades:
        candidates_by_entry.setdefault(t["entry_date"], []).append(("rsi2", t))
    for t in tom_trades:
        candidates_by_entry.setdefault(t["entry_date"], []).append(("tom", t))

    full_calendar = set()
    for series in close_lookup.values():
        full_calendar.update(series.index)
    loop_dates = sorted(d for d in full_calendar if calendar_start <= d <= calendar_end)

    equity = start_capital
    open_positions = []
    curve = []

    def can_admit():
        return len(open_positions) * RISK_PER_TRADE + RISK_PER_TRADE <= MAX_OPEN_RISK + 1e-9 \
            and len(open_positions) < MAX_POSITIONS

    def close_position(pos, price, date):
        nonlocal equity
        r_mult = r_mult_of(pos, price)
        equity += r_mult * pos["risked_dollars"]
        curve.append((date, equity))

    for date in loop_dates:
        if sgov_returns is not None:
            invested = sum(p["notional"] for p in open_positions)
            cash = max(0.0, equity - invested)
            day_ret = sgov_returns.get(date, 0.0)
            if cash > 0 and not pd.isna(day_ret):
                equity += cash * day_ret

        still = []
        for pos in open_positions:
            (close_position(pos, pos["exit_price"], date) if pos["exit_date"] == date else still.append(pos))
        open_positions = still

        todays = list(candidates_by_entry.get(date, []))
        rng.shuffle(todays)
        for kind, t in todays:
            if can_admit():
                risked = RISK_PER_TRADE * equity
                open_positions.append({"type": kind, "symbol": t["symbol"], "entry_price": t["entry_price"],
                                        "stop": t["stop"], "exit_date": t["exit_date"], "exit_price": t["exit_price"],
                                        "risked_dollars": risked,
                                        "notional": position_notional(t["entry_price"], t["stop"], risked)})

        still = []
        for pos in open_positions:
            (close_position(pos, pos["exit_price"], date) if pos["exit_date"] == date else still.append(pos))
        open_positions = still

        if sgov_returns is not None:
            curve.append((date, equity))

    return curve, equity


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_symbols = [os.path.basename(p)[:-4] for p in glob.glob(os.path.join(DATA_DIR, "*.csv"))
                   if os.path.basename(p) != "SGOV.csv"]
    dfs, close_lookup = load_universe(all_symbols)
    sgov = load_sgov_returns()

    all_rsi2, all_tom = [], []
    for symbol, d in dfs.items():
        all_rsi2.extend([t for t in gen_rsi2_trades(d, symbol) if t["entry_date"] >= START_DATE])
        all_tom.extend([t for t in gen_tom_trades(d, symbol) if t["entry_date"] >= START_DATE])

    curve_b, eq_b, _, _, _ = simulate_shared(all_rsi2, all_tom, close_lookup, "leftover_only",
                                              sgov_returns=sgov, calendar_start=START_DATE, calendar_end=END_DATE)
    curve_b = [(d, v) for d, v in curve_b if d is not None]
    dd_b = max_dd(curve_b)

    curve_ban, eq_ban, _, _ = simulate_bandit_priority(all_rsi2, all_tom, close_lookup, sgov,
                                                         START_DATE, END_DATE, ucb_c=0.5)
    dd_ban = max_dd(curve_ban)

    print(f"Baseline: ${eq_b:.2f}, dd={dd_b:.1f}%")
    print(f"Bandit (c=0.5): ${eq_ban:.2f}, dd={dd_ban:.1f}%")
    print(f"Прогоняю {N_PERMUTATIONS} случайных порядков допуска...")

    rand_equities, rand_dds = [], []
    for seed in range(N_PERMUTATIONS):
        curve, eq = simulate_random_priority(all_rsi2, all_tom, close_lookup, sgov, START_DATE, END_DATE, seed)
        rand_equities.append(eq)
        rand_dds.append(max_dd(curve))

    rand_equities = np.array(rand_equities)
    rand_dds = np.array(rand_dds)

    pctile_baseline = (rand_equities < eq_b).mean() * 100
    pctile_bandit = (rand_equities < eq_ban).mean() * 100

    print(f"\nСлучайные порядки ({N_PERMUTATIONS} прогонов): "
          f"эквити mean=${rand_equities.mean():.2f} std=${rand_equities.std():.2f} "
          f"min=${rand_equities.min():.2f} max=${rand_equities.max():.2f}")
    print(f"Baseline (${eq_b:.2f}) — перцентиль {pctile_baseline:.1f}% среди случайных порядков")
    print(f"Bandit (${eq_ban:.2f}) — перцентиль {pctile_bandit:.1f}% среди случайных порядков")
    print(f"Empirical p-value (доля случайных >= бандита): {(rand_equities >= eq_ban).mean():.4f}")

    df = pd.DataFrame({"seed": range(N_PERMUTATIONS), "final_equity": rand_equities, "max_drawdown_pct": rand_dds})
    df.to_csv(os.path.join(RESULTS_DIR, "combo_bandit_priority_permutation.csv"), index=False)
    print(f"\nWrote results/combo_bandit_priority_permutation.csv")
