"""
Combo Bandit-Quota (22.08.2026) — ДИАГНОСТИЧЕСКАЯ, НЕ боевая система.
Третий вариант bandit-механики для Комбо C, по запросу пользователя:
не порядок допуска при конфликте (10.2, закрыт как шум — см. CLAUDE.md
10.6/10.7) и не непрерывный сайзинг каждой сделки (10.3, тоже не дал
чистой картины) — а ПЕРИОДИЧЕСКИЙ лимит риск-бюджета на инструмент,
который выставляется раз в rebalance_days и держится ФИКСИРОВАННЫМ до
следующего пересчёта (а не пересчитывается на каждой закрытой сделке,
как в 10.2/10.3).

Механика: раз в rebalance_days дней — softmax по исторической эдж-оценке
(mean r_mult закрытых СТРОГО ДО этой даты сделок, без lookahead) даёт
долю каждого инструмента в общем потолке риска MAX_OPEN_RISK. Эта доля
(quota) держится фиксированной весь следующий период. Новая сделка по
инструменту допускается, ТОЛЬКО если открытый риск ИМЕННО ПО ЭТОМУ
инструменту (сумма risk_frac уже открытых по нему позиций) плюс новая
сделка не превышает его текущую quota — ДОПОЛНИТЕЛЬНО к обычным
ограничениям (общий потолок 8%, макс. 10 позиций).

Инструменты без истории (n=0 закрытых сделок) получают равную долю
(1/N) как нейтральный старт.

Grid: rebalance_days ∈ {30, 90, 180} — узкая сетка, фиксирована заранее
(не по результату), та же дисциплина, что и eta/c в 10.1-10.3.

Запуск: python scripts/combo_bandit_quota.py
Выход: results/combo_bandit_quota.csv, печатает сводку.
"""
import glob
import os

import numpy as np
import pandas as pd

import backtest as bt
from system_ibs_rsi2 import add_extra_indicators
from combo_rsi2_tom import (gen_rsi2_trades, gen_tom_trades, r_mult_of, simulate_shared,
                             load_sgov_returns, RISK_PER_TRADE, MAX_OPEN_RISK, MAX_POSITIONS)
from combo_bandit_priority import load_universe, position_notional, max_dd, START_CAPITAL

DATA_DIR = "data"
RESULTS_DIR = "results"
START_DATE = pd.Timestamp("2007-01-01")
END_DATE = pd.Timestamp("2026-12-31")


def simulate_bandit_quota(rsi2_trades, tom_trades, close_lookup, sgov_returns, calendar_start,
                           calendar_end, rebalance_days=30, eta=8.0, start_capital=START_CAPITAL,
                           trade_log=None):
    all_symbols = sorted(close_lookup.keys())
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

    # Масштаб квоты: НЕ делим MAX_OPEN_RISK поровну на N инструментов буквально
    # (при N=25 это дало бы 0.32% на инструмент — МЕНЬШЕ одной сделки в 1%,
    # т.е. вообще ничего не проходило бы). Квота — это ПОТОЛОК на инструмент,
    # не строгий раздел общего бюджета (общий потолок MAX_OPEN_RISK всё равно
    # проверяется отдельно, ниже) — умножаем на MAX_POSITIONS, чтобы
    # "нейтральный" (равновзвешенный) инструмент мог держать несколько
    # одновременных позиций, а не был обрезан по построению.
    QUOTA_SCALE = MAX_OPEN_RISK * MAX_POSITIONS

    quotas = {s: QUOTA_SCALE / len(all_symbols) for s in all_symbols}
    last_rebalance = loop_dates[0] if loop_dates else None

    def recompute_quotas():
        seen = {s: sum_r[s] / n_trades[s] for s in n_trades if n_trades[s] > 0}
        if not seen:
            return {s: QUOTA_SCALE / len(all_symbols) for s in all_symbols}
        neutral = np.mean(list(seen.values()))
        means = np.array([seen.get(s, neutral) for s in all_symbols])
        w = np.exp(eta * (means - means.max()))
        w = w / w.sum()
        return {s: QUOTA_SCALE * w[i] for i, s in enumerate(all_symbols)}

    def open_risk_by_symbol():
        d = {}
        for p in open_positions:
            d[p["symbol"]] = d.get(p["symbol"], 0.0) + p["risk_frac"]
        return d

    def can_admit(symbol, risk_frac, by_symbol):
        total_open = sum(p["risk_frac"] for p in open_positions)
        symbol_open = by_symbol.get(symbol, 0.0)
        return (total_open + risk_frac <= MAX_OPEN_RISK + 1e-9 and
                symbol_open + risk_frac <= quotas.get(symbol, 0.0) + 1e-9 and
                len(open_positions) < MAX_POSITIONS)

    def close_position(pos, price, date):
        nonlocal equity
        r_mult = r_mult_of(pos, price)
        equity += r_mult * pos["risked_dollars"]
        n_trades[pos["symbol"]] = n_trades.get(pos["symbol"], 0) + 1
        sum_r[pos["symbol"]] = sum_r.get(pos["symbol"], 0.0) + r_mult
        curve.append((date, equity))
        if trade_log is not None:
            trade_log.append({"date": date, "symbol": pos["symbol"], "r_mult": r_mult})

    for date in loop_dates:
        if (date - last_rebalance).days >= rebalance_days:
            quotas = recompute_quotas()
            last_rebalance = date

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

        by_symbol = open_risk_by_symbol()
        for kind, t in candidates_by_entry.get(date, []):
            risk_frac = RISK_PER_TRADE
            if can_admit(t["symbol"], risk_frac, by_symbol):
                risked = risk_frac * equity
                open_positions.append({"type": kind, "symbol": t["symbol"], "entry_price": t["entry_price"],
                                        "stop": t["stop"], "exit_date": t["exit_date"], "exit_price": t["exit_price"],
                                        "risked_dollars": risked, "risk_frac": risk_frac,
                                        "notional": position_notional(t["entry_price"], t["stop"], risked)})
                taken[kind] += 1
                by_symbol[t["symbol"]] = by_symbol.get(t["symbol"], 0.0) + risk_frac
            else:
                skipped[kind] += 1

        still = []
        for pos in open_positions:
            (close_position(pos, pos["exit_price"], date) if pos["exit_date"] == date else still.append(pos))
        open_positions = still

        if sgov_returns is not None:
            curve.append((date, equity))

    return curve, equity, taken, skipped


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

    curve_b, eq_b, taken_b, skipped_b, _ = simulate_shared(
        all_rsi2, all_tom, close_lookup, "leftover_only", sgov_returns=sgov,
        calendar_start=START_DATE, calendar_end=END_DATE)
    curve_b = [(d, v) for d, v in curve_b if d is not None]
    print(f"Baseline: ${eq_b:.2f} ({(eq_b/START_CAPITAL-1)*100:+.1f}%)  dd={max_dd(curve_b):.1f}%  "
          f"взято RSI2/TOM={taken_b['rsi2']}/{taken_b['tom']}  пропущено={skipped_b['rsi2']}/{skipped_b['tom']}")

    rows = [{"strategy": "baseline", "rebalance_days": None, "final_equity": round(eq_b, 2),
             "total_return_pct": round((eq_b / START_CAPITAL - 1) * 100, 1), "max_drawdown_pct": round(max_dd(curve_b), 1)}]

    for rebalance_days in [30, 90, 180]:
        curve, eq, taken, skipped = simulate_bandit_quota(
            all_rsi2, all_tom, close_lookup, sgov, START_DATE, END_DATE, rebalance_days=rebalance_days)
        dd = max_dd(curve)
        print(f"quota, rebalance={rebalance_days}d: ${eq:.2f} ({(eq/START_CAPITAL-1)*100:+.1f}%)  dd={dd:.1f}%  "
              f"взято RSI2/TOM={taken['rsi2']}/{taken['tom']}  пропущено={skipped['rsi2']}/{skipped['tom']}")
        rows.append({"strategy": f"quota_{rebalance_days}d", "rebalance_days": rebalance_days,
                     "final_equity": round(eq, 2), "total_return_pct": round((eq / START_CAPITAL - 1) * 100, 1),
                     "max_drawdown_pct": round(dd, 1)})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "combo_bandit_quota.csv"), index=False)
    print(f"\nWrote results/combo_bandit_quota.csv")
