"""
System TOM (Turn-of-the-Month) — календарная стратегия, НЕ индикаторная.
Вход не зависит от цены/индикатора вообще, только от даты: покупаем за
offset_before торговых дней до конца месяца, держим через границу месяца,
продаём на 3-й торговый день нового месяца по закрытию (конвенция
Quantpedia/литературы). Обоснование: институциональные и частные потоки
капитала (зарплаты, взносы в пенсионные счета, ребалансировка фондов)
концентрируются вокруг границы месяца — структурный, не ценовой эффект,
менее подвержен тому вымыванию арбитражем, что случилось с System IBS
(см. CLAUDE.md 5.13/5.14).

Диагностика по всем offset от 1 до 7 (15.08.2026) показала: значимость
держится ПЛАТО на 3-6 днях на ОБОИХ рынках одновременно (не в одной
изолированной точке) — offset=5 выбран как середина устойчивого плато,
"боевой" грид сузили до {4,5,6} для честного walk-forward (не {1..7} —
не переоптимизируем сетку, конвенция 5.2).

Стоп 2×ATR14 (боевая конвенция проекта B/C_long/E/RSI2), фиксированный при
входе, НЕ трейлинг — same-bar lookahead из 5.9 не применим.

Тестируется на ДВУХ независимых вселенных сразу:
1. US_DATA_DIR (data/) — наш обычный 25-инструментный пул (США, FX, крипта, бонды).
2. MOEX_DATA_DIR (data/moex_stocks/) — 35 ликвидных акций MOEX, забраны через
   публичный MOEX ISS API (см. data/moex_stocks/README, fetch_moex.py в
   scratchpad) — независимая проверка, что эффект не специфичен для рынка США.

Запуск: python scripts/system_tom.py
Выход: results/system_tom_pooled.csv (walk-forward train/test, обе вселенные,
колонка `market`).
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy import stats

import backtest as bt

US_DATA_DIR = "data"
MOEX_DATA_DIR = "data/moex_stocks"
RESULTS_DIR = "results"

OFFSET_GRID = [4, 5, 6]  # плато значимости из диагностики 1-7, не переоптимизируем дальше
ATR_MULT = 2.0
EXIT_DAY_OF_NEW_MONTH = 3  # 3-й торговый день нового месяца, конвенция литературы


def backtest_system_tom(data, offset_before):
    d = data.reset_index(drop=True)
    d["month"] = d["date"].dt.to_period("M")
    months = d["month"].unique()
    trades = []

    for i in range(len(months) - 1):
        month_rows = d.index[d["month"] == months[i]]
        if len(month_rows) < offset_before:
            continue
        last_idx = month_rows[-1]
        entry_idx = last_idx - (offset_before - 1)
        if entry_idx < 0:
            continue

        next_month_rows = d.index[d["month"] == months[i + 1]]
        if len(next_month_rows) < EXIT_DAY_OF_NEW_MONTH:
            continue
        exit_idx = next_month_rows[EXIT_DAY_OF_NEW_MONTH - 1]

        if entry_idx >= len(d) or exit_idx >= len(d) or exit_idx <= entry_idx:
            continue

        if entry_idx < 1:
            continue
        entry_row = d.iloc[entry_idx]
        entry_price = entry_row["open"]
        # ATR берём с ДНЯ ДО входа (известен до открытия дня входа), не с
        # самого дня входа — иначе стоп сайзится по TR дня, который на
        # момент открытия ещё не завершился (same-bar паттерн, см. 5.9),
        # та же конвенция, что и у RSI2/B (сигнальный день, не день входа).
        # Исправлено 17.08.2026.
        atr = d.iloc[entry_idx - 1]["atr14"]
        if np.isnan(atr) or np.isnan(entry_price):
            continue
        stop = entry_price - ATR_MULT * atr
        if stop >= entry_price:
            continue

        exit_price, reason = None, None
        for j in range(entry_idx, exit_idx + 1):
            rowj = d.iloc[j]
            if rowj["low"] <= stop:
                exit_price = rowj["open"] if rowj["open"] < stop else stop
                reason = "stop"
                break
        if reason is None:
            exit_price = d.iloc[exit_idx]["close"]
            reason = "scheduled_exit"

        ret_pct = (exit_price - entry_price) / entry_price * 100
        risk_pct = (entry_price - stop) / entry_price * 100
        r_mult = ret_pct / risk_pct if risk_pct > 0 else np.nan
        trades.append({"ret_pct": ret_pct, "r_mult": r_mult, "reason": reason})

    return pd.DataFrame(trades)


def run_walkforward_universe(data_dir, market_label, live_set=None):
    pooled = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        symbol = os.path.basename(path).replace(".csv", "")
        raw = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
        if len(raw) < 300 or "high" not in raw.columns:
            continue
        years_available = (raw.date.max() - raw.date.min()).days / 365.25
        d = bt.add_indicators(raw)
        d = d.iloc[60:].reset_index(drop=True)
        windows = bt.make_windows(d, int(round(years_available)))
        is_live = symbol in live_set if live_set else False

        for window_name, train, test in windows:
            for offset in OFFSET_GRID:
                for period_name, data in [("train", train), ("test", test)]:
                    trades = backtest_system_tom(data, offset)
                    key = (market_label, offset, period_name)
                    pooled.setdefault(key, []).append(trades.assign(symbol=symbol, live=is_live))

    rows = []
    for (market, offset, period), frames in sorted(pooled.items()):
        all_t = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        n = len(all_t)
        if n > 1:
            t, p = stats.ttest_1samp(all_t.r_mult.dropna(), 0)
            wr = (all_t.ret_pct > 0).mean() * 100
            mr = all_t.r_mult.mean()
            tr = all_t.r_mult.sum()
            live_t = all_t[all_t.live] if live_set else all_t.iloc[0:0]
            nl = len(live_t)
            mrl = live_t.r_mult.mean() if nl else np.nan
            rows.append({"market": market, "offset": offset, "period": period, "n": n, "win_rate": round(wr, 1),
                         "mean_r": round(mr, 3), "total_r": round(tr, 2), "p": p, "n_live": nl, "mean_r_live": mrl})
        else:
            rows.append({"market": market, "offset": offset, "period": period, "n": n})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    us_rep = run_walkforward_universe(US_DATA_DIR, "US_pool", live_set=bt.LIVE_INSTRUMENTS)
    moex_rep = run_walkforward_universe(MOEX_DATA_DIR, "MOEX")
    rep = pd.concat([us_rep, moex_rep], ignore_index=True)
    print(rep.to_string(index=False))
    rep.to_csv(os.path.join(RESULTS_DIR, "system_tom_pooled.csv"), index=False)
    print(f"\nWrote {len(rep)} pooled rows to results/system_tom_pooled.csv")
