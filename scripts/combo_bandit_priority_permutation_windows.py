"""
Многоокновая версия перестановочного теста (10.6) — по прямому вопросу
пользователя "как правильно проверить" (22.08.2026). §10.6 тестировал
только полную историю целиком (2007-2026); но устойчивость на 5/6 окон
(10.4) была ключевым аргументом ЗА находку до 10.6 — и не была
перепроверена тем же строгим (перестановочным) методом. Если сигнал
реален, но слабый, полная 20-летняя история могла "размыть" его в общем
шуме; по отдельным окнам сигнал мог бы быть виднее (или наоборот —
подтвердить, что и там всё в пределах шума).

Для каждого из тех же 6 окон, что и в 10.4: N=200 случайных порядков
допуска (compute budget — меньше, чем 300 в 10.6, чтобы 6 окон уложились
в разумное время; точность перцентиля почти не страдает при N=200).

Запуск: python scripts/combo_bandit_priority_permutation_windows.py
Выход: results/combo_bandit_priority_permutation_windows.csv, печатает сводку.
"""
import glob
import os

import numpy as np
import pandas as pd

from combo_rsi2_tom import gen_rsi2_trades, gen_tom_trades, simulate_shared, load_sgov_returns
from combo_bandit_priority import simulate_bandit_priority, load_universe, max_dd, START_CAPITAL
from combo_bandit_priority_permutation import simulate_random_priority

DATA_DIR = "data"
RESULTS_DIR = "results"
N_PERMUTATIONS = 200

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

    curve_b, eq_b, _, _, _ = simulate_shared(all_rsi2, all_tom, close_lookup, "leftover_only",
                                              sgov_returns=sgov, calendar_start=start_date, calendar_end=end_date)
    eq_b_val = eq_b

    curve_ban, eq_ban, _, _ = simulate_bandit_priority(all_rsi2, all_tom, close_lookup, sgov,
                                                         start_date, end_date, ucb_c=0.5)

    rand_eq = []
    for seed in range(N_PERMUTATIONS):
        _, eq = simulate_random_priority(all_rsi2, all_tom, close_lookup, sgov, start_date, end_date, seed)
        rand_eq.append(eq)
    rand_eq = np.array(rand_eq)

    pctile_baseline = (rand_eq < eq_b_val).mean() * 100
    pctile_bandit = (rand_eq < eq_ban).mean() * 100
    p_value = (rand_eq >= eq_ban).mean()

    return {"window": label, "baseline_equity": round(eq_b_val, 2), "bandit_equity": round(eq_ban, 2),
            "random_mean": round(rand_eq.mean(), 2), "random_std": round(rand_eq.std(), 2),
            "random_min": round(rand_eq.min(), 2), "random_max": round(rand_eq.max(), 2),
            "baseline_percentile": round(pctile_baseline, 1), "bandit_percentile": round(pctile_bandit, 1),
            "bandit_p_value": round(p_value, 4)}


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
        print(f"{label}: baseline=${row['baseline_equity']:.2f} (перцентиль {row['baseline_percentile']:.1f}%)  "
              f"bandit=${row['bandit_equity']:.2f} (перцентиль {row['bandit_percentile']:.1f}%, "
              f"p={row['bandit_p_value']:.3f})  "
              f"случайные: mean=${row['random_mean']:.2f} std=${row['random_std']:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "combo_bandit_priority_permutation_windows.csv"), index=False)
    print(f"\nWrote results/combo_bandit_priority_permutation_windows.csv")
    print(f"\nОкон, где бандит значим на уровне p<0.05: {(df['bandit_p_value'] < 0.05).sum()}/{len(df)}")
