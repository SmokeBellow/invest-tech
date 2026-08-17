"""
System IBS и System RSI2 — две mean-reversion стратегии, добавленные в
проект 15.08.2026 по результатам исследования технических стратегий на
американском рынке (QuantifiedStrategies.com, Alvarez Quant Trading).
Первые системы этой сессии, показавшие статистически значимый край и на
train, и на test, и отдельно на live-8 (см. CLAUDE.md 5.13) — в отличие от
A/B/A_ensemble/C/D/F, которые все дали p>0.05 хотя бы в одном из этих трёх
срезов.

Дисциплина та же, что и во всех системах проекта: сигнал по закрытому бару,
исполнение по открытию следующей сессии, стоп ФИКСИРУЕТСЯ один раз при
входе (не трейлинг — same-bar lookahead из 5.9 сюда не относится, стоп не
обновляется в цикле).

System IBS (Internal Bar Strength): IBS = (close-low)/(high-low) — где
находится закрытие внутри дневного диапазона. Вход на переходе IBS ниже
порога (0.1 или 0.2 — единственное варьируемое измерение, конвенция
QuantifiedStrategies), выход на переходе IBS выше 0.8 (симметричный порог,
не перебирается). Фильтр тренда close>EMA200 — как и в System B, чтобы не
ловить падающий нож. Стоп 2×ATR14 (та же "боевая" конвенция, что и
B/C_long/E, НЕ перебирается заново), тайм-стоп 10 сессий (конвенция B).

System RSI2 (Ларри Коннорс): RSI(2) вместо RSI(14). Вход на переходе RSI2
ниже порога (5 или 10 — единственное измерение, по Коннорсу порог 5 даёт
более высокий результат, чем 10). Фильтр тренда close>EMA200 (Коннорс:
"только в направлении доминирующего тренда"). Выход — классическое правило
Коннорса: close пересекает SMA5 снизу вверх (НЕ RSI-порог на выходе — это
искажало бы сравнение с оригинальной методологией). Стоп 2×ATR14, тайм-стоп
10 сессий — те же защитные конвенции, что и везде (Коннорс не описывает
защитный стоп явно, но проект без него не работает).

Запуск: python scripts/system_ibs_rsi2.py
Вход: data/*.csv (созданы fetch_data.py)
Выход: results/system_ibs_rsi2_pooled.csv (walk-forward пул по всем 25
инструментам/окнам, та же методология, что и backtest.py)
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy import stats

import backtest as bt

DATA_DIR = "data"
RESULTS_DIR = "results"

IBS_ENTRY_GRID = [0.1, 0.2]
IBS_EXIT = 0.8
RSI2_ENTRY_GRID = [5, 10]
ATR_MULT = 2.0  # боевая конвенция проекта (B/C_long/E), не перебирается
TIME_STOP = 10


def add_extra_indicators(df):
    d = df.copy()
    d["ibs"] = (d.close - d.low) / (d.high - d.low)
    d["rsi2"] = bt.rsi(d.close, 2)
    d["sma5"] = d.close.rolling(5).mean()
    return d


def backtest_system_ibs(data, entry_threshold):
    d = data.reset_index(drop=True)
    trades = []
    position = None
    for i in range(1, len(d)):
        row, prev = d.iloc[i], d.iloc[i - 1]
        if position is None:
            cond_now = (row.close > row.ema200) and (row.ibs < entry_threshold)
            cond_prev = (prev.close > prev.ema200) and (prev.ibs < entry_threshold)
            if cond_now and not cond_prev and i + 1 < len(d):
                entry_price = d.iloc[i + 1].open
                stop = entry_price - ATR_MULT * row.atr14
                if not np.isnan(stop) and stop < entry_price:
                    position = {"entry_price": entry_price, "stop": stop, "bars": 0}
        else:
            rowj = d.iloc[i]
            position["bars"] += 1
            exit_price, reason = None, None
            if rowj.low <= position["stop"]:
                exit_price = rowj.open if rowj.open < position["stop"] else position["stop"]
                reason = "stop"
            elif rowj.ibs > IBS_EXIT:
                exit_price, reason = rowj.close, "ibs_exit"
            elif position["bars"] >= TIME_STOP:
                exit_price, reason = rowj.close, "time_stop"
            if reason:
                ret_pct = (exit_price - position["entry_price"]) / position["entry_price"] * 100
                risk_pct = (position["entry_price"] - position["stop"]) / position["entry_price"] * 100
                r_mult = ret_pct / risk_pct if risk_pct > 0 else np.nan
                trades.append({"ret_pct": ret_pct, "r_mult": r_mult, "reason": reason})
                position = None
    return pd.DataFrame(trades)


def backtest_system_rsi2(data, entry_threshold):
    d = data.reset_index(drop=True)
    trades = []
    position = None
    for i in range(1, len(d)):
        row, prev = d.iloc[i], d.iloc[i - 1]
        if position is None:
            cond_now = (row.close > row.ema200) and (row.rsi2 < entry_threshold)
            cond_prev = (prev.close > prev.ema200) and (prev.rsi2 < entry_threshold)
            if cond_now and not cond_prev and i + 1 < len(d):
                entry_price = d.iloc[i + 1].open
                stop = entry_price - ATR_MULT * row.atr14
                if not np.isnan(stop) and stop < entry_price:
                    position = {"entry_price": entry_price, "stop": stop, "bars": 0}
        else:
            rowj = d.iloc[i]
            position["bars"] += 1
            exit_price, reason = None, None
            if rowj.low <= position["stop"]:
                exit_price = rowj.open if rowj.open < position["stop"] else position["stop"]
                reason = "stop"
            elif prev.close <= prev.sma5 and rowj.close > rowj.sma5:
                exit_price, reason = rowj.close, "sma5_cross"
            elif position["bars"] >= TIME_STOP:
                exit_price, reason = rowj.close, "time_stop"
            if reason:
                ret_pct = (exit_price - position["entry_price"]) / position["entry_price"] * 100
                risk_pct = (position["entry_price"] - position["stop"]) / position["entry_price"] * 100
                r_mult = ret_pct / risk_pct if risk_pct > 0 else np.nan
                trades.append({"ret_pct": ret_pct, "r_mult": r_mult, "reason": reason})
                position = None
    return pd.DataFrame(trades)


def run_pooled_walkforward():
    pooled = {}  # (system, param, period) -> list[df]
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.csv"))):
        symbol = os.path.basename(path).replace(".csv", "")
        raw = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
        years_available = (raw.date.max() - raw.date.min()).days / 365.25
        d = bt.add_indicators(raw)
        d = add_extra_indicators(d)
        d = d.iloc[60:].reset_index(drop=True)
        windows = bt.make_windows(d, int(round(years_available)))
        is_live = symbol in bt.LIVE_INSTRUMENTS

        for window_name, train, test in windows:
            for th in IBS_ENTRY_GRID:
                for period_name, data in [("train", train), ("test", test)]:
                    trades = backtest_system_ibs(data, th)
                    key = ("IBS", f"<{th}", period_name)
                    pooled.setdefault(key, []).append(trades.assign(symbol=symbol, live=is_live))
            for th in RSI2_ENTRY_GRID:
                for period_name, data in [("train", train), ("test", test)]:
                    trades = backtest_system_rsi2(data, th)
                    key = ("RSI2", f"<{th}", period_name)
                    pooled.setdefault(key, []).append(trades.assign(symbol=symbol, live=is_live))

    rows = []
    for (system, param, period), frames in sorted(pooled.items()):
        all_t = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        n = len(all_t)
        if n > 1:
            t, p = stats.ttest_1samp(all_t.r_mult.dropna(), 0)
            wr = (all_t.ret_pct > 0).mean() * 100
            mr = all_t.r_mult.mean()
            tr = all_t.r_mult.sum()
            live_t = all_t[all_t.live]
            nl = len(live_t)
            mrl = live_t.r_mult.mean() if nl else np.nan
            rows.append({"system": system, "param": param, "period": period, "n": n, "win_rate": round(wr, 1),
                         "mean_r": round(mr, 3), "total_r": round(tr, 2), "p": p, "n_live": nl, "mean_r_live": mrl})
        else:
            rows.append({"system": system, "param": param, "period": period, "n": n})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rep = run_pooled_walkforward()
    print(rep.to_string(index=False))
    rep.to_csv(os.path.join(RESULTS_DIR, "system_ibs_rsi2_pooled.csv"), index=False)
    print(f"\nWrote {len(rep)} pooled rows to results/system_ibs_rsi2_pooled.csv")
