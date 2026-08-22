"""
t-тест статистической значимости находки §10.2/10.4 CLAUDE.md: даёт ли
UCB1-приоритизация инструментов (c=0.5) в Комбо C на пуле-25 статистически
значимо другой средний r_mult закрытых сделок, чем baseline (без
приоритизации)? По запросу пользователя, 22.08.2026.

Baseline и бандит допускают РАЗНЫЕ (хоть и пересекающиеся) подмножества
сделок из одного пула кандидатов — при конфликте бюджета порядок допуска
разный, значит и множества закрытых сделок разные. Поэтому это
independent two-sample t-test (Welch, unequal variance) на r_mult
ЗАКРЫТЫХ сделок каждой стратегии, scipy.stats.ttest_ind(..., equal_var=False)
— НЕ ttest_1samp против нуля (та проверка уже делалась для самих систем
RSI2/TOM в 5.13/5.14, вопрос здесь другой: отличается ли КАЧЕСТВО ОТБОРА
сделок бандитом от baseline).

Дополнительно — ttest_1samp каждой стратегии отдельно против нуля, для
контекста (согласованность с методологией 5.8).

Тест сделан на ПОЛНОЙ истории (2007-2026, пул-25) — как в 10.2, не по
окнам (10.4 уже показал устойчивость по окнам отдельно; здесь цель —
максимальная выборка для мощности теста).

Запуск: python scripts/combo_bandit_priority_ttest.py
Выход: results/combo_bandit_priority_ttest.csv, печатает сводку.
"""
import glob
import os

import pandas as pd
from scipy import stats

from combo_rsi2_tom import gen_rsi2_trades, gen_tom_trades, simulate_shared, load_sgov_returns
from combo_bandit_priority import simulate_bandit_priority, load_universe

DATA_DIR = "data"
RESULTS_DIR = "results"
START_DATE = pd.Timestamp("2007-01-01")
END_DATE = pd.Timestamp("2026-12-31")


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

    baseline_log, bandit_log = [], []
    simulate_shared(all_rsi2, all_tom, close_lookup, "leftover_only", sgov_returns=sgov,
                     calendar_start=START_DATE, calendar_end=END_DATE, trade_log=baseline_log)
    simulate_bandit_priority(all_rsi2, all_tom, close_lookup, sgov, START_DATE, END_DATE,
                              ucb_c=0.5, trade_log=bandit_log)

    baseline_r = pd.Series([t["r_mult"] for t in baseline_log])
    bandit_r = pd.Series([t["r_mult"] for t in bandit_log])

    t_1samp_base, p_1samp_base = stats.ttest_1samp(baseline_r, 0)
    t_1samp_ban, p_1samp_ban = stats.ttest_1samp(bandit_r, 0)
    t_ind, p_ind = stats.ttest_ind(bandit_r, baseline_r, equal_var=False)

    rows = [
        {"test": "baseline vs 0 (ttest_1samp)", "n_a": len(baseline_r), "n_b": None,
         "mean_a": round(baseline_r.mean(), 4), "mean_b": None, "t_stat": round(t_1samp_base, 3),
         "p_value": p_1samp_base},
        {"test": "bandit vs 0 (ttest_1samp)", "n_a": len(bandit_r), "n_b": None,
         "mean_a": round(bandit_r.mean(), 4), "mean_b": None, "t_stat": round(t_1samp_ban, 3),
         "p_value": p_1samp_ban},
        {"test": "bandit vs baseline (ttest_ind, Welch)", "n_a": len(bandit_r), "n_b": len(baseline_r),
         "mean_a": round(bandit_r.mean(), 4), "mean_b": round(baseline_r.mean(), 4),
         "t_stat": round(t_ind, 3), "p_value": p_ind},
    ]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(RESULTS_DIR, "combo_bandit_priority_ttest.csv"), index=False)
    print(f"\nWrote results/combo_bandit_priority_ttest.csv")
