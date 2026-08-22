"""
Многоокновая проверка (walk-forward-стиль, по запросу пользователя,
22.08.2026) устойчивости находки §10.2 CLAUDE.md: UCB1-приоритизация
инструментов в Комбо C на пуле-25 (c=0.5) против baseline без приоритета.

Бандит — последовательный/online алгоритм (учится по ходу симуляции), не
статичный параметр вроде ADX-порога — классический train/test split здесь
неприменим напрямую (нельзя "обучить на train, применить на test" - бандит
и так уже не заглядывает вперёд по своей конструкции). Вместо этого —
ТА ЖЕ дисциплина, что уже использовалась в проекте для похожей задачи
(CLAUDE.md 5.16, сравнение Комбо C vs вечный портфель): прогнать
сравнение на НЕСКОЛЬКИХ независимых/перекрывающихся окнах и посмотреть,
не сосредоточено ли преимущество бандита в одном удачном периоде.

Каждое окно — бандит и baseline стартуют с $1000 ЗАНОВО (бандитная
статистика по инструментам тоже с нуля) на начало окна — так проверяется,
воспроизводится ли эффект при разных стартовых точках, а не просто
"глубже успевает разогнаться" в одном длинном прогоне.

Запуск: python scripts/combo_bandit_priority_walkforward.py
Выход: results/combo_bandit_priority_walkforward.csv, печатает сводку.
"""
import glob
import os

import pandas as pd

from system_ibs_rsi2 import add_extra_indicators  # noqa: F401 (нужен для load_universe)
from combo_rsi2_tom import gen_rsi2_trades, gen_tom_trades, simulate_shared, load_sgov_returns
from combo_bandit_priority import simulate_bandit_priority, load_universe, max_dd, START_CAPITAL

DATA_DIR = "data"
RESULTS_DIR = "results"

WINDOWS = [
    ("2006-2016", "2006-01-01", "2016-12-31"),
    ("2008-2018", "2008-01-01", "2018-12-31"),
    ("2011-2021", "2011-01-01", "2021-12-31"),
    ("2013-2023", "2013-01-01", "2023-12-31"),
    ("2016-2026", "2016-01-01", "2026-12-31"),
    ("2007-2026", "2007-01-01", "2026-12-31"),
]


def run_window(label, start, end, dfs, close_lookup, sgov):
    start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
    all_rsi2, all_tom = [], []
    for symbol, d in dfs.items():
        all_rsi2.extend([t for t in gen_rsi2_trades(d, symbol) if start_date <= t["entry_date"] <= end_date])
        all_tom.extend([t for t in gen_tom_trades(d, symbol) if start_date <= t["entry_date"] <= end_date])

    curve_b, eq_b, taken_b, skipped_b, _ = simulate_shared(
        all_rsi2, all_tom, close_lookup, "leftover_only", sgov_returns=sgov,
        calendar_start=start_date, calendar_end=end_date)
    curve_b = [(d, v) for d, v in curve_b if d is not None]

    curve_ban, eq_ban, taken_ban, skipped_ban = simulate_bandit_priority(
        all_rsi2, all_tom, close_lookup, sgov, start_date, end_date, ucb_c=0.5)

    return {
        "window": label,
        "baseline_final": round(eq_b, 2), "baseline_ret_pct": round((eq_b / START_CAPITAL - 1) * 100, 1),
        "baseline_dd_pct": round(max_dd(curve_b), 1),
        "bandit_final": round(eq_ban, 2), "bandit_ret_pct": round((eq_ban / START_CAPITAL - 1) * 100, 1),
        "bandit_dd_pct": round(max_dd(curve_ban), 1),
        "bandit_beats_both": (eq_ban > eq_b) and (max_dd(curve_ban) > max_dd(curve_b)),
    }


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_symbols = [os.path.basename(p)[:-4] for p in glob.glob(os.path.join(DATA_DIR, "*.csv"))
                   if os.path.basename(p) != "SGOV.csv"]
    dfs, close_lookup = load_universe(all_symbols)
    sgov = load_sgov_returns()

    rows = []
    for label, start, end in WINDOWS:
        row = run_window(label, start, end, dfs, close_lookup, sgov)
        rows.append(row)
        print(f"{label}: baseline ${row['baseline_final']:.2f} ({row['baseline_ret_pct']:+.1f}%, "
              f"dd={row['baseline_dd_pct']:.1f}%)  |  bandit ${row['bandit_final']:.2f} "
              f"({row['bandit_ret_pct']:+.1f}%, dd={row['bandit_dd_pct']:.1f}%)  "
              f"{'BEATS BOTH' if row['bandit_beats_both'] else ''}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "combo_bandit_priority_walkforward.csv"), index=False)
    n_wins = df["bandit_beats_both"].sum()
    print(f"\nБандит улучшает ОБА показателя (доходность И просадка) в {n_wins}/{len(df)} окнах")
    print(f"Wrote results/combo_bandit_priority_walkforward.csv")
