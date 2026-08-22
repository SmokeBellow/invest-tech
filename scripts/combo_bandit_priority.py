"""
Combo Bandit-Priority (22.08.2026) — ДИАГНОСТИЧЕСКАЯ, НЕ боевая система.
UCB1-бандит для приоритизации ИНСТРУМЕНТОВ внутри Комбо C (RSI2<10 + TOM
offset=5) на расширенном пуле из 25 инструментов — частично закрывает
открытый вопрос §7.6 CLAUDE.md ("приоритет при конфликте сигналов для
Комбо C не переопределён"), но конкретно бандитом, не статичным правилом.

Мотивация тестировать именно на пуле-25, не live-8: в диагностике этой же
сессии (см. CLAUDE.md, раздел про сравнение пул-25 vs live-8) на 25
инструментах было МАССОВОЕ превышение риск-бюджета (сотни-тысячи пропущенных
сделок из-за нехватки бюджета) — то есть есть за что бороться: при остром
конфликте кандидатов порядок допуска в позицию реально влияет на результат.
На live-8, где бюджет почти всегда хватает, бандиту нечего оптимизировать.

Механика: каждый инструмент — рычаг бандита. UCB1-скор = средний r_mult
ПРОШЛЫХ ЗАКРЫТЫХ сделок этого инструмента + бонус на неопределённость
c*sqrt(ln(N_total+1)/(n_instrument+1)) — обновляется ТОЛЬКО по сделкам,
уже закрытым СТРОГО ДО текущего дня (без lookahead). Когда в один день
конкурируют больше кандидатов (TOM+RSI2 вместе), чем позволяет бюджет —
сортируем по UCB1-скору убыв., допускаем жадно, пока бюджет не исчерпан.
Остальная риск-механика не меняется (1%/сделку, потолок 8%, макс. 10
позиций, SGOV на свободный кэш, см. combo_rsi2_tom.py).

РЕЗУЛЬТАТ (см. CLAUDE.md, раздел "Bandit-эксперименты"): на пуле-25 бандит
(c=0.5-1.0) улучшает И доходность, И просадку одновременно относительно
baseline (простой порядок допуска без приоритизации) — редкий случай не
размена, а честного улучшения. На live-8 эффекта нет/чуть хуже — подтверждает
гипотезу, что бандит помогает именно там, где бюджет реально дефицитен.
НЕ проверено walk-forward (train/test) и не проверено на статистическую
значимость отличия от baseline — единственный прогон на всей истории,
c выбран НЕ по результату (сетка {0.5, 1.0, 2.0} показана целиком). Прежде
чем считать находку боевой, нужна честная out-of-sample проверка.

Запуск: python scripts/combo_bandit_priority.py
Выход: results/combo_bandit_priority.csv, печатает сводку.
"""
import glob
import os

import numpy as np
import pandas as pd

import backtest as bt
from system_ibs_rsi2 import add_extra_indicators
from combo_rsi2_tom import (gen_rsi2_trades, gen_tom_trades, r_mult_of, simulate_shared,
                             load_sgov_returns, RISK_PER_TRADE, MAX_OPEN_RISK, MAX_POSITIONS)

DATA_DIR = "data"
RESULTS_DIR = "results"
START_CAPITAL = 1000.0
START_DATE = pd.Timestamp("2007-01-01")
END_DATE = pd.Timestamp("2026-12-31")


def load_universe(symbols):
    dfs, close_lookup = {}, {}
    for symbol in sorted(symbols):
        path = os.path.join(DATA_DIR, f"{symbol}.csv")
        if not os.path.exists(path):
            continue
        raw = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
        d = bt.add_indicators(raw)
        d = add_extra_indicators(d)
        d = d.iloc[60:].reset_index(drop=True)
        dfs[symbol] = d
        close_lookup[symbol] = raw.set_index("date")["close"].sort_index()
    return dfs, close_lookup


def position_notional(entry_price, stop, risked_dollars):
    frac = (entry_price - stop) / entry_price
    return risked_dollars / frac if frac > 1e-9 else risked_dollars


def simulate_bandit_priority(rsi2_trades, tom_trades, close_lookup, sgov_returns, calendar_start,
                              calendar_end, ucb_c=1.0, start_capital=START_CAPITAL):
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
    range_start = calendar_start if calendar_start is not None else all_dates[0]
    range_end = calendar_end if calendar_end is not None else all_dates[-1]
    loop_dates = sorted(d for d in full_calendar if range_start <= d <= range_end)

    equity = start_capital
    open_positions = []
    curve = []
    taken = {"rsi2": 0, "tom": 0}
    skipped = {"rsi2": 0, "tom": 0}
    n_trades, sum_r = {}, {}
    total_closed = 0

    def can_admit():
        return len(open_positions) * RISK_PER_TRADE + RISK_PER_TRADE <= MAX_OPEN_RISK + 1e-9 \
            and len(open_positions) < MAX_POSITIONS

    def ucb_score(symbol):
        n = n_trades.get(symbol, 0)
        if n == 0:
            return float("inf")
        mean_r = sum_r[symbol] / n
        return mean_r + ucb_c * np.sqrt(np.log(total_closed + 1) / n)

    def close_position(pos, price, date):
        nonlocal equity, total_closed
        r_mult = r_mult_of(pos, price)
        equity += r_mult * pos["risked_dollars"]
        n_trades[pos["symbol"]] = n_trades.get(pos["symbol"], 0) + 1
        sum_r[pos["symbol"]] = sum_r.get(pos["symbol"], 0.0) + r_mult
        total_closed += 1
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

        todays = sorted(candidates_by_entry.get(date, []), key=lambda st: -ucb_score(st[1]["symbol"]))
        for kind, t in todays:
            if can_admit():
                risked = RISK_PER_TRADE * equity
                open_positions.append({"type": kind, "symbol": t["symbol"], "entry_price": t["entry_price"],
                                        "stop": t["stop"], "exit_date": t["exit_date"], "exit_price": t["exit_price"],
                                        "risked_dollars": risked,
                                        "notional": position_notional(t["entry_price"], t["stop"], risked)})
                taken[kind] += 1
            else:
                skipped[kind] += 1

        still = []
        for pos in open_positions:
            (close_position(pos, pos["exit_price"], date) if pos["exit_date"] == date else still.append(pos))
        open_positions = still

        if sgov_returns is not None:
            curve.append((date, equity))

    return curve, equity, taken, skipped


def max_dd(curve):
    peak = START_CAPITAL
    dd = 0.0
    for _, v in curve:
        peak = max(peak, v)
        dd = min(dd, (v - peak) / peak * 100)
    return dd


def run_universe(label, symbols, rows):
    dfs, close_lookup = load_universe(symbols)
    all_rsi2, all_tom = [], []
    for symbol, d in dfs.items():
        all_rsi2.extend([t for t in gen_rsi2_trades(d, symbol) if t["entry_date"] >= START_DATE])
        all_tom.extend([t for t in gen_tom_trades(d, symbol) if t["entry_date"] >= START_DATE])
    sgov = load_sgov_returns()

    curve_b, eq_b, taken_b, skipped_b, _ = simulate_shared(
        all_rsi2, all_tom, close_lookup, "leftover_only", sgov_returns=sgov,
        calendar_start=START_DATE, calendar_end=END_DATE)
    curve_b = [(d, v) for d, v in curve_b if d is not None]
    rows.append({"universe": label, "strategy": "baseline_no_priority", "final_equity": round(eq_b, 2),
                 "total_return_pct": round((eq_b / START_CAPITAL - 1) * 100, 1), "max_drawdown_pct": round(max_dd(curve_b), 1),
                 "taken_rsi2": taken_b["rsi2"], "taken_tom": taken_b["tom"],
                 "skipped_rsi2": skipped_b["rsi2"], "skipped_tom": skipped_b["tom"]})

    for c in [0.5, 1.0, 2.0]:
        curve, eq, taken, skipped = simulate_bandit_priority(all_rsi2, all_tom, close_lookup, sgov,
                                                               START_DATE, END_DATE, ucb_c=c)
        rows.append({"universe": label, "strategy": f"ucb1_bandit_c{c}", "final_equity": round(eq, 2),
                     "total_return_pct": round((eq / START_CAPITAL - 1) * 100, 1), "max_drawdown_pct": round(max_dd(curve), 1),
                     "taken_rsi2": taken["rsi2"], "taken_tom": taken["tom"],
                     "skipped_rsi2": skipped["rsi2"], "skipped_tom": skipped["tom"]})


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_symbols = [os.path.basename(p)[:-4] for p in glob.glob(os.path.join(DATA_DIR, "*.csv"))
                   if os.path.basename(p) != "SGOV.csv"]
    rows = []
    run_universe("live8", bt.LIVE_INSTRUMENTS, rows)
    run_universe("pool25", all_symbols, rows)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(RESULTS_DIR, "combo_bandit_priority.csv"), index=False)
    print(f"\nWrote {len(df)} rows to results/combo_bandit_priority.csv")
