"""
backtest.py — прогоняет системы A (TREND) и B (MR) с сеткой параметров
по всем инструментам в data/*.csv, используя anchored walk-forward
(для длинной истории) или простой 70/30 split (для короткой, напр. крипта).

Не выбирает "победителя" — только формирует полную таблицу результатов
с флагом insufficient_data (<20 сделок в окне). Решение о том, что делать
с результатами, остаётся за человеком, не за скриптом.

Запуск: python backtest.py
Вход: data/*.csv (созданы fetch_data.py)
Выход: results/backtest_report.md, results/backtest_report.csv
"""

import os
import glob
import numpy as np
import pandas as pd

DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MIN_TRADES = 20  # порог содержательности — ПРЕДВАРИТЕЛЬНЫЙ, подлежит пересмотру
ADX_GRID = [20, 25, 30]
RSI_GRID = [25, 30, 35]
ATR_STOP_GRID = [1.0, 1.5, 2.0]  # множитель ATR14 для стопа — испытательные варианты B
AGREE_GRID = [2, 3]  # A_ensemble: сколько из 3 горизонтов момента должны согласиться (см. ниже)
EMA_TOUCH_GRID = [20, 50]  # C: какая EMA служит линией входа (см. backtest_system_c)

LIVE_INSTRUMENTS = {"SPY", "QQQ", "EEM", "TLT", "XLE", "EURUSD", "USDJPY", "BTCUSD"}


# ---------- индикаторы ----------

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def true_range(df):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


def adx(df, period=14):
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = true_range(df)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean(), atr


def cci(df, period=14):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
    return (tp - sma) / (0.015 * mad)


def stochastic(df, k_period=14, k_smooth=3, d_period=3):
    """Полный (slow) стохастик (14,3,3) — конвенция, не параметр для перебора."""
    low_n = df["low"].rolling(k_period).min()
    high_n = df["high"].rolling(k_period).max()
    raw_k = 100 * (df["close"] - low_n) / (high_n - low_n)
    k = raw_k.rolling(k_smooth).mean()
    d = k.rolling(d_period).mean()
    return k, d


def add_indicators(df):
    df = df.copy()
    close = df["close"]
    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    df["ema200"] = close.ewm(span=200, adjust=False).mean()
    df["rsi14"] = rsi(close, 14)
    df["adx14"], df["atr14"] = adx(df, 14)
    df["cci14"] = cci(df, 14)
    df["macd"] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    # Bollinger Bands(20, 2sigma) — конвенция, не параметр для перебора
    df["bb_mid"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std
    # моментум на трёх горизонтах (~1, ~3, ~6 месяцев торговых дней) — конвенция
    # time-series momentum (Moskowitz/Ooi/Pedersen), периоды не перебираются,
    # как и периоды EMA/Bollinger. Используется только системой A_ensemble.
    df["mom21"] = close.pct_change(21)
    df["mom63"] = close.pct_change(63)
    df["mom126"] = close.pct_change(126)
    # Стохастик (14,3,3) — используется системой C
    df["stoch_k"], df["stoch_d"] = stochastic(df)
    return df


# ---------- бэктест-движки (та же логика, что согласована в диалоге) ----------

def backtest_system_a(data, adx_threshold):
    trades = []
    position = None
    d = data.reset_index(drop=True)
    for i in range(1, len(d)):
        row, prev = d.iloc[i], d.iloc[i - 1]
        cond_now = (row.ema20 > row.ema50 > row.ema200) and (row.adx14 >= adx_threshold) and (row.macd > 0)
        cond_prev = (prev.ema20 > prev.ema50 > prev.ema200) and (prev.adx14 >= adx_threshold) and (prev.macd > 0)
        if position is None and cond_now and not cond_prev:
            if i + 1 < len(d):
                init_stop = row.ema50 * 0.99
                position = {"entry_i": i + 1, "entry_price": d.iloc[i + 1].open,
                            "stop": init_stop, "orig_stop": init_stop}
        elif position is not None:
            rowj = d.iloc[i]
            position["stop"] = max(position["stop"], rowj.ema50 * 0.99)
            exit_price, reason = None, None
            if rowj.low <= position["stop"]:
                exit_price = rowj.open if rowj.open < position["stop"] else position["stop"]
                reason = "stop"
            elif rowj.close < rowj.ema50:
                exit_price, reason = rowj.close, "ema50_close_below"
            if reason:
                # R считается относительно СТОПА НА МОМЕНТ ВХОДА (та же дистанция,
                # что определила размер позиции), а не текущего трейлинг-стопа —
                # иначе выход по трейлинг-стопу всегда даёт ровно -1.00R, даже если
                # стоп успел подтянуться далеко в плюс.
                ret_pct = (exit_price - position["entry_price"]) / position["entry_price"] * 100
                risk_pct = (position["entry_price"] - position["orig_stop"]) / position["entry_price"] * 100
                r_mult = ret_pct / risk_pct if risk_pct > 0 else np.nan
                trades.append({"ret_pct": ret_pct, "r_mult": r_mult, "reason": reason})
                position = None
    return pd.DataFrame(trades)


def backtest_system_b(data, rsi_threshold, cci_threshold=-100):
    trades = []
    position = None
    d = data.reset_index(drop=True)
    for i in range(1, len(d)):
        row, prev = d.iloc[i], d.iloc[i - 1]
        cond_now = (row.close > row.ema200) and (row.adx14 < 20) and (row.rsi14 <= rsi_threshold or row.cci14 <= cci_threshold) and (row.close < row.ema20)
        cond_prev = (prev.close > prev.ema200) and (prev.adx14 < 20) and (prev.rsi14 <= rsi_threshold or prev.cci14 <= cci_threshold) and (prev.close < prev.ema20)
        if position is None and cond_now and not cond_prev:
            if i + 1 < len(d):
                position = {"entry_i": i + 1, "entry_price": d.iloc[i + 1].open, "stop": row.low * 0.997, "bars": 0}
        elif position is not None:
            rowj = d.iloc[i]
            position["bars"] += 1
            exit_price, reason = None, None
            if rowj.low <= position["stop"]:
                exit_price = rowj.open if rowj.open < position["stop"] else position["stop"]
                reason = "stop"
            elif rowj.rsi14 >= 55:
                exit_price, reason = rowj.close, "rsi_exit"
            elif rowj.close >= rowj.ema20:
                exit_price, reason = rowj.close, "ema20_touch"
            elif position["bars"] >= 10:
                exit_price, reason = rowj.close, "time_stop"
            if reason:
                ret_pct = (exit_price - position["entry_price"]) / position["entry_price"] * 100
                risk_pct = (position["entry_price"] - position["stop"]) / position["entry_price"] * 100
                r_mult = ret_pct / risk_pct if risk_pct > 0 else np.nan
                trades.append({"ret_pct": ret_pct, "r_mult": r_mult, "reason": reason})
                position = None
    return pd.DataFrame(trades)


def backtest_system_b_atr_stop(data, atr_mult, rsi_threshold=30, cci_threshold=-100):
    """Вариант B: тот же вход/выход по RSI/CCI, что и оригинал, но стоп на
    entry - atr_mult*ATR14 вместо фиксированных 0.3% от минимума дня —
    проверка гипотезы, что именно узкий стоп (не индикатор входа) топит B."""
    trades = []
    position = None
    d = data.reset_index(drop=True)
    for i in range(1, len(d)):
        row, prev = d.iloc[i], d.iloc[i - 1]
        cond_now = (row.close > row.ema200) and (row.adx14 < 20) and (row.rsi14 <= rsi_threshold or row.cci14 <= cci_threshold) and (row.close < row.ema20)
        cond_prev = (prev.close > prev.ema200) and (prev.adx14 < 20) and (prev.rsi14 <= rsi_threshold or prev.cci14 <= cci_threshold) and (prev.close < prev.ema20)
        if position is None and cond_now and not cond_prev:
            if i + 1 < len(d):
                entry_price = d.iloc[i + 1].open
                position = {"entry_i": i + 1, "entry_price": entry_price,
                            "stop": entry_price - atr_mult * row.atr14, "bars": 0}
        elif position is not None:
            rowj = d.iloc[i]
            position["bars"] += 1
            exit_price, reason = None, None
            if rowj.low <= position["stop"]:
                exit_price = rowj.open if rowj.open < position["stop"] else position["stop"]
                reason = "stop"
            elif rowj.rsi14 >= 55:
                exit_price, reason = rowj.close, "rsi_exit"
            elif rowj.close >= rowj.ema20:
                exit_price, reason = rowj.close, "ema20_touch"
            elif position["bars"] >= 10:
                exit_price, reason = rowj.close, "time_stop"
            if reason:
                ret_pct = (exit_price - position["entry_price"]) / position["entry_price"] * 100
                risk_pct = (position["entry_price"] - position["stop"]) / position["entry_price"] * 100
                r_mult = ret_pct / risk_pct if risk_pct > 0 else np.nan
                trades.append({"ret_pct": ret_pct, "r_mult": r_mult, "reason": reason})
                position = None
    return pd.DataFrame(trades)


def backtest_system_b_bollinger(data, atr_mult):
    """Вариант B: вход на закрытии ниже нижней полосы Боллинджера (в
    восходящем тренде, вне сильного тренда), выход на возврате к средней
    полосе, стоп на ATR. Черновик, согласованный в диалоге как замена
    RSI/CCI-версии — сравнивается напрямую с ней и с оригиналом B."""
    trades = []
    position = None
    d = data.reset_index(drop=True)
    for i in range(1, len(d)):
        row, prev = d.iloc[i], d.iloc[i - 1]
        cond_now = (row.close > row.ema200) and (row.adx14 < 20) and (row.close < row.bb_lower)
        cond_prev = (prev.close > prev.ema200) and (prev.adx14 < 20) and (prev.close < prev.bb_lower)
        if position is None and cond_now and not cond_prev:
            if i + 1 < len(d):
                entry_price = d.iloc[i + 1].open
                position = {"entry_i": i + 1, "entry_price": entry_price,
                            "stop": entry_price - atr_mult * row.atr14, "bars": 0}
        elif position is not None:
            rowj = d.iloc[i]
            position["bars"] += 1
            exit_price, reason = None, None
            if rowj.low <= position["stop"]:
                exit_price = rowj.open if rowj.open < position["stop"] else position["stop"]
                reason = "stop"
            elif rowj.close >= rowj.bb_mid:
                exit_price, reason = rowj.close, "bb_mid_touch"
            elif position["bars"] >= 10:
                exit_price, reason = rowj.close, "time_stop"
            if reason:
                ret_pct = (exit_price - position["entry_price"]) / position["entry_price"] * 100
                risk_pct = (position["entry_price"] - position["stop"]) / position["entry_price"] * 100
                r_mult = ret_pct / risk_pct if risk_pct > 0 else np.nan
                trades.append({"ret_pct": ret_pct, "r_mult": r_mult, "reason": reason})
                position = None
    return pd.DataFrame(trades)


def backtest_system_a_relaxed(data, adx_threshold=25):
    """Диагностика: система A БЕЗ условия MACD>0 — проверка гипотезы, что
    конъюнкция из трёх условий (EMA-стек + ADX + MACD) сама по себе душит
    частоту сигналов, и MACD>0 в основном избыточен, раз EMA-стек уже
    подразумевает восходящий тренд. Фиксированный ADX=25 (боевой порог),
    не пересобирается заново — меняется только само условие входа, одна
    переменная за раз."""
    trades = []
    position = None
    d = data.reset_index(drop=True)
    for i in range(1, len(d)):
        row, prev = d.iloc[i], d.iloc[i - 1]
        cond_now = (row.ema20 > row.ema50 > row.ema200) and (row.adx14 >= adx_threshold)
        cond_prev = (prev.ema20 > prev.ema50 > prev.ema200) and (prev.adx14 >= adx_threshold)
        if position is None and cond_now and not cond_prev:
            if i + 1 < len(d):
                init_stop = row.ema50 * 0.99
                position = {"entry_i": i + 1, "entry_price": d.iloc[i + 1].open,
                            "stop": init_stop, "orig_stop": init_stop}
        elif position is not None:
            rowj = d.iloc[i]
            position["stop"] = max(position["stop"], rowj.ema50 * 0.99)
            exit_price, reason = None, None
            if rowj.low <= position["stop"]:
                exit_price = rowj.open if rowj.open < position["stop"] else position["stop"]
                reason = "stop"
            elif rowj.close < rowj.ema50:
                exit_price, reason = rowj.close, "ema50_close_below"
            if reason:
                ret_pct = (exit_price - position["entry_price"]) / position["entry_price"] * 100
                risk_pct = (position["entry_price"] - position["orig_stop"]) / position["entry_price"] * 100
                r_mult = ret_pct / risk_pct if risk_pct > 0 else np.nan
                trades.append({"ret_pct": ret_pct, "r_mult": r_mult, "reason": reason})
                position = None
    return pd.DataFrame(trades)


def backtest_system_b_bollinger_relaxed(data, atr_mult=2.0):
    """Диагностика: Bollinger-версия B БЕЗ условия ADX<20 — проверка, не
    режет ли фильтр "нет сильного тренда" слишком много валидных откатов
    внутри тренда. Фиксированный ATR-множитель 2.0 (боевой), меняется
    только само условие входа."""
    trades = []
    position = None
    d = data.reset_index(drop=True)
    for i in range(1, len(d)):
        row, prev = d.iloc[i], d.iloc[i - 1]
        cond_now = (row.close > row.ema200) and (row.close < row.bb_lower)
        cond_prev = (prev.close > prev.ema200) and (prev.close < prev.bb_lower)
        if position is None and cond_now and not cond_prev:
            if i + 1 < len(d):
                entry_price = d.iloc[i + 1].open
                position = {"entry_i": i + 1, "entry_price": entry_price,
                            "stop": entry_price - atr_mult * row.atr14, "bars": 0}
        elif position is not None:
            rowj = d.iloc[i]
            position["bars"] += 1
            exit_price, reason = None, None
            if rowj.low <= position["stop"]:
                exit_price = rowj.open if rowj.open < position["stop"] else position["stop"]
                reason = "stop"
            elif rowj.close >= rowj.bb_mid:
                exit_price, reason = rowj.close, "bb_mid_touch"
            elif position["bars"] >= 10:
                exit_price, reason = rowj.close, "time_stop"
            if reason:
                ret_pct = (exit_price - position["entry_price"]) / position["entry_price"] * 100
                risk_pct = (position["entry_price"] - position["stop"]) / position["entry_price"] * 100
                r_mult = ret_pct / risk_pct if risk_pct > 0 else np.nan
                trades.append({"ret_pct": ret_pct, "r_mult": r_mult, "reason": reason})
                position = None
    return pd.DataFrame(trades)


def backtest_system_a_ensemble(data, agree_threshold):
    """A_ensemble — CTA-практика мульти-горизонтного момента вместо одного
    ADX-порога (расширенное исследование стратегий, 13.08.2026): согласие
    нескольких lookback-периодов
    (1/3/6 месяцев) вместо единственного условия EMA-стек+ADX+MACD. Не
    модификация системы A, а отдельная диагностическая система — сравнивается
    с A/A_relaxed, не заменяет их автоматически.

    Вход: не менее agree_threshold из 3 горизонтов момента (mom21/63/126)
    положительны — сессия перехода состояния (вчера условие было ложно,
    сегодня истинно), та же логика транзиции, что и везде в проекте.
    agree_threshold=2 — большинство (мягче, больше сигналов), 3 — единогласие
    (жёстче). Сетка из двух точек — одно измерение, не переоптимизация.

    Стоп и логика выхода идентичны системе A (EMA50-1%, трейлинг, R считается
    от стопа на момент входа) — чтобы поменялась именно сигнальная логика
    входа, а не риск-механика, иначе сравнение с A/A_relaxed теряет смысл."""
    trades = []
    position = None
    d = data.reset_index(drop=True)
    for i in range(1, len(d)):
        row, prev = d.iloc[i], d.iloc[i - 1]
        agree_now = sum([row.mom21 > 0, row.mom63 > 0, row.mom126 > 0])
        agree_prev = sum([prev.mom21 > 0, prev.mom63 > 0, prev.mom126 > 0])
        cond_now = agree_now >= agree_threshold
        cond_prev = agree_prev >= agree_threshold
        if position is None and cond_now and not cond_prev:
            if i + 1 < len(d):
                init_stop = row.ema50 * 0.99
                position = {"entry_i": i + 1, "entry_price": d.iloc[i + 1].open,
                            "stop": init_stop, "orig_stop": init_stop}
        elif position is not None:
            rowj = d.iloc[i]
            position["stop"] = max(position["stop"], rowj.ema50 * 0.99)
            agree_j = sum([rowj.mom21 > 0, rowj.mom63 > 0, rowj.mom126 > 0])
            exit_price, reason = None, None
            if rowj.low <= position["stop"]:
                exit_price = rowj.open if rowj.open < position["stop"] else position["stop"]
                reason = "stop"
            elif agree_j < agree_threshold:
                exit_price, reason = rowj.close, "momentum_consensus_broken"
            if reason:
                ret_pct = (exit_price - position["entry_price"]) / position["entry_price"] * 100
                risk_pct = (position["entry_price"] - position["orig_stop"]) / position["entry_price"] * 100
                r_mult = ret_pct / risk_pct if risk_pct > 0 else np.nan
                trades.append({"ret_pct": ret_pct, "r_mult": r_mult, "reason": reason})
                position = None
    return pd.DataFrame(trades)


def backtest_system_c(data, ema_period):
    """Система C — EMA-тренд + касание EMA как S/R + Стохастик, LONG И SHORT
    (первая двунаправленная система в проекте; A/B — только long). Решения
    пользователя в диалоге 13.08.2026:
    - Направление: EMA20 > EMA50 → ищем LONG, EMA20 < EMA50 → ищем SHORT.
    - Касание (grid EMA_TOUCH_GRID={20,50}, какая EMA — линия входа): в лонге
      low ≤ EMA ≤ close (цена коснулась/пробила EMA внутри дня, закрылась
      выше — отскок подтверждён закрытием); в шорте зеркально
      high ≥ EMA ≥ close.
    - Подтверждение: Стохастик (14,3,3, конвенция, не перебирается) —
      пересечение %K/%D в зоне разворота. Лонг: оба были <20 накануне,
      %K пересекает %D снизу вверх. Шорт: оба были >80 накануне, %K
      пересекает %D сверху вниз.
    - Стоп: та же EMA, что дала сигнал, ∓1% буфер, трейлинг только в пользу
      сделки (как система A). Выход: стоп ИЛИ close ушёл против тренда за
      EMA. R — от стопа на момент входа (защита от бага 5.7)."""
    trades = []
    position = None
    d = data.reset_index(drop=True)
    for i in range(1, len(d)):
        row, prev = d.iloc[i], d.iloc[i - 1]
        if position is None:
            ema_col = f"ema{ema_period}"
            trend_up = row.ema20 > row.ema50
            trend_down = row.ema20 < row.ema50
            touch_long = row.low <= row[ema_col] <= row.close
            touch_short = row.high >= row[ema_col] >= row.close
            stoch_cross_up = (prev.stoch_k <= prev.stoch_d and row.stoch_k > row.stoch_d
                               and prev.stoch_k < 20 and prev.stoch_d < 20)
            stoch_cross_down = (prev.stoch_k >= prev.stoch_d and row.stoch_k < row.stoch_d
                                 and prev.stoch_k > 80 and prev.stoch_d > 80)
            if i + 1 < len(d) and trend_up and touch_long and stoch_cross_up:
                init_stop = row[ema_col] * 0.99
                position = {"side": "long", "entry_price": d.iloc[i + 1].open,
                            "stop": init_stop, "orig_stop": init_stop, "ema_col": ema_col}
            elif i + 1 < len(d) and trend_down and touch_short and stoch_cross_down:
                init_stop = row[ema_col] * 1.01
                position = {"side": "short", "entry_price": d.iloc[i + 1].open,
                             "stop": init_stop, "orig_stop": init_stop, "ema_col": ema_col}
        else:
            rowj = d.iloc[i]
            ema_col = position["ema_col"]
            exit_price, reason = None, None
            if position["side"] == "long":
                position["stop"] = max(position["stop"], rowj[ema_col] * 0.99)
                if rowj.low <= position["stop"]:
                    exit_price = rowj.open if rowj.open < position["stop"] else position["stop"]
                    reason = "stop"
                elif rowj.close < rowj[ema_col]:
                    exit_price, reason = rowj.close, "ema_close_below"
                if reason:
                    ret_pct = (exit_price - position["entry_price"]) / position["entry_price"] * 100
                    risk_pct = (position["entry_price"] - position["orig_stop"]) / position["entry_price"] * 100
            else:
                position["stop"] = min(position["stop"], rowj[ema_col] * 1.01)
                if rowj.high >= position["stop"]:
                    exit_price = rowj.open if rowj.open > position["stop"] else position["stop"]
                    reason = "stop"
                elif rowj.close > rowj[ema_col]:
                    exit_price, reason = rowj.close, "ema_close_above"
                if reason:
                    ret_pct = (position["entry_price"] - exit_price) / position["entry_price"] * 100
                    risk_pct = (position["orig_stop"] - position["entry_price"]) / position["entry_price"] * 100
            if reason:
                r_mult = ret_pct / risk_pct if risk_pct > 0 else np.nan
                trades.append({"ret_pct": ret_pct, "r_mult": r_mult, "reason": reason, "side": position["side"]})
                position = None
    return pd.DataFrame(trades)


def backtest_system_c_long(data, adx_max=20.0, touch_depth_max=1.0, atr_mult=2.0, time_stop=20):
    """C_long_v2 — по итогам диагностики системы C (13.08.2026, диалог с
    пользователем, только LONG-сторона). Диагностика показала: 62% сделок
    исходной версии закрывались за 1-7 баров с win rate 4-8% (whipsaw от
    EMA-трейлинга), тогда как сделки, дожившие до 16+ баров, — win rate 89%.
    Четыре правки по этой диагностике, каждая — явное решение пользователя:

    1. Только EMA20 как линия входа (EMA50 эмпирически хуже: mean R −0.13
       против +0.16 у EMA20 на исходной версии).
    2в. ATR-стоп (2.0×ATR14 — тот же множитель, что «боевая» B, не подгонка
        под эту диагностику) вместо EMA-трейлинга — фиксируется на входе, не
        подтягивается. Смягчает выход, чтобы позиция не выбивалась
        однодневным шумом.
    3в. Фильтр ADX<20 на входе — вопреки интуиции «сильный тренд — хорошо»,
        диагностика показала обратное (ADX 11-16: win rate 45%; ADX 24-43:
        win rate 19%) — высокий ADX на откате чаще ловил истощение движения,
        а не паузу. Порог 20 — существующая конвенция проекта (нижняя
        граница ADX_GRID), а не подогнанное под квартили значение.
    4. Фильтр глубины касания <1.0% от цены — глубокий пробой ниже EMA20
       (Q4 диагностики) был хуже мелкого отскока (Q1). Порог округлён до
       1.0%, НЕ взят точно по границе квартиля (медиана диагностической
       выборки была 0.63%) — иначе это переобучение на тех же сделках,
       на которых сделан вывод, а не независимая проверка.

    Выход: стоп, ИЛИ close < EMA50 (медленная EMA, НЕ та, что дала сигнал —
    менее чувствительна к однодневному шуму, чем выход по EMA20 в исходной
    версии), ИЛИ тайм-стоп (не менялся у B, здесь применён по аналогии).

    Ослабление входа (13.08.2026, диалог) — первая попытка (оба ниже 20 в
    ДЕНЬ КАСАНИЯ) дала ещё меньше сделок (28), чем исходная версия с точным
    пересечением (65): день отскока от EMA обычно сам толкает стохастик
    вверх, так что «оба ещё ниже 20» в момент отскока — редкое совпадение,
    более жёсткое, чем требование пересечения. Исправлено: перепроданность
    проверяется на ВЧЕРАШНЕМ дне (до отскока), а не на дне касания —
    убрано только требование ТОЧНОГО пересечения %K/%D, привязка к
    "вчера" сохранена как в исходной версии. Транзишн-логика (сессия
    ПЕРЕХОДА состояния) сохранена на всей конъюнкции целиком — та же
    дисциплина, что и везде в проекте."""
    trades = []
    position = None
    d = data.reset_index(drop=True)

    def entry_cond(j):
        if j < 1:
            return False
        r, p = d.iloc[j], d.iloc[j - 1]
        trend_up = r.ema20 > r.ema50
        touch_long = r.low <= r.ema20 <= r.close
        touch_depth = (r.ema20 - r.low) / r.close * 100 if touch_long else np.nan
        oversold_prev = p.stoch_k < 20 and p.stoch_d < 20
        return trend_up and touch_long and oversold_prev and r.adx14 < adx_max and touch_depth < touch_depth_max

    for i in range(2, len(d)):
        row = d.iloc[i]
        if position is None:
            if i + 1 < len(d) and entry_cond(i) and not entry_cond(i - 1):
                entry_price = d.iloc[i + 1].open
                stop = entry_price - atr_mult * row.atr14
                position = {"entry_price": entry_price, "stop": stop, "bars": 0}
        else:
            rowj = d.iloc[i]
            position["bars"] += 1
            exit_price, reason = None, None
            if rowj.low <= position["stop"]:
                exit_price = rowj.open if rowj.open < position["stop"] else position["stop"]
                reason = "stop"
            elif rowj.close < rowj.ema50:
                exit_price, reason = rowj.close, "ema50_close_below"
            elif position["bars"] >= time_stop:
                exit_price, reason = rowj.close, "time_stop"
            if reason:
                ret_pct = (exit_price - position["entry_price"]) / position["entry_price"] * 100
                risk_pct = (position["entry_price"] - position["stop"]) / position["entry_price"] * 100
                r_mult = ret_pct / risk_pct if risk_pct > 0 else np.nan
                trades.append({"ret_pct": ret_pct, "r_mult": r_mult, "reason": reason})
                position = None
    return pd.DataFrame(trades)


def summarize(trades):
    n = len(trades)
    if n == 0:
        return {"n_trades": 0, "win_rate": None, "avg_r": None, "total_r": None, "max_dd_r": None, "insufficient_data": True}
    win_rate = round((trades.ret_pct > 0).mean() * 100, 1)
    avg_r = round(trades.r_mult.mean(), 2)
    total_r = round(trades.r_mult.sum(), 2)
    cum = trades.r_mult.cumsum()
    max_dd = round((cum - cum.cummax()).min(), 2)
    return {"n_trades": n, "win_rate": win_rate, "avg_r": avg_r, "total_r": total_r,
            "max_dd_r": max_dd, "insufficient_data": n < MIN_TRADES}


# ---------- разбиение на train/test ----------

def make_windows(df, years_available):
    """Anchored walk-forward для длинной истории (>=6 лет), иначе простой 70/30 split.

    Раньше брались только последние 3 окна (ближе к концу истории) — первые
    ~2 года данных вообще никогда не попадали в test. Теперь берём ВСЕ
    доступные анкорные окна начиная с 1 года train (индикаторы уже посчитаны
    на непрерывном ряду до нарезки на окна, 60-строчный разгонный период уже
    отброшен раньше — короткий train не портит их). Это даёт больше
    независимых out-of-sample проверок и показывает, размазан ли результат
    по всей истории или сосредоточен в одном-двух окнах (напр. только в
    самом последнем годе)."""
    df = df.reset_index(drop=True)
    n = len(df)
    windows = []
    if years_available >= 6:
        # anchored: train растёт, test — следующий год
        bars_per_year = n / years_available
        for k in range(1, years_available):
            train_end = int(bars_per_year * k)
            test_end = int(bars_per_year * (k + 1))
            if train_end < 60 or test_end > n:
                continue
            windows.append((f"wf_{k}", df.iloc[:train_end], df.iloc[train_end:test_end]))
    else:
        split = int(n * 0.7)
        windows.append(("split_70_30", df.iloc[:split], df.iloc[split:]))
    return windows


# ---------- основной цикл ----------

def main():
    rows = []
    # пул сделок по всем инструментам сразу (все 12 CSV в data/, не только live) —
    # per-symbol/per-window срез слишком мелко режет данные (см. обсуждение с
    # пользователем: даже суммарно по live-инструментам в одном окне почти никогда
    # не набирается 20 сделок), пул даёт статистике реальную мощность за счёт
    # сложения истории по всем инструментам и всем test-окнам сразу
    pooled_trades = {}  # (system, param, period) -> list[DataFrame]
    pooled_symbols = {}  # (system, param, period) -> set(symbol) с >=1 сделкой

    def add_to_pool(key, symbol, trades):
        pooled_trades.setdefault(key, []).append(trades)
        if len(trades) > 0:
            pooled_symbols.setdefault(key, set()).add(symbol)

    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    for path in csv_files:
        symbol = os.path.basename(path).replace(".csv", "")
        df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
        years_available = (df.date.max() - df.date.min()).days / 365.25
        df = add_indicators(df)
        df = df.iloc[60:].reset_index(drop=True)  # отбросить разгонный период индикаторов
        windows = make_windows(df, int(round(years_available)))
        is_live = symbol in LIVE_INSTRUMENTS

        for window_name, train, test in windows:
            for adx_th in ADX_GRID:
                param = f"ADX>={adx_th}"
                for period_name, data in [("train", train), ("test", test)]:
                    trades = backtest_system_a(data, adx_th)
                    stats = summarize(trades)
                    rows.append({"symbol": symbol, "live_list": is_live, "system": "A",
                                 "param": param, "window": window_name,
                                 "period": period_name, **stats})
                    add_to_pool(("A", param, period_name), symbol, trades)
            for rsi_th in RSI_GRID:
                param = f"RSI<={rsi_th}"
                for period_name, data in [("train", train), ("test", test)]:
                    trades = backtest_system_b(data, rsi_th)
                    stats = summarize(trades)
                    rows.append({"symbol": symbol, "live_list": is_live, "system": "B",
                                 "param": param, "window": window_name,
                                 "period": period_name, **stats})
                    add_to_pool(("B", param, period_name), symbol, trades)
            for atr_mult in ATR_STOP_GRID:
                param = f"ATRx{atr_mult}"
                for period_name, data in [("train", train), ("test", test)]:
                    trades = backtest_system_b_atr_stop(data, atr_mult)
                    stats = summarize(trades)
                    rows.append({"symbol": symbol, "live_list": is_live, "system": "B_atr_stop",
                                 "param": param, "window": window_name,
                                 "period": period_name, **stats})
                    add_to_pool(("B_atr_stop", param, period_name), symbol, trades)
            for atr_mult in ATR_STOP_GRID:
                param = f"ATRx{atr_mult}"
                for period_name, data in [("train", train), ("test", test)]:
                    trades = backtest_system_b_bollinger(data, atr_mult)
                    stats = summarize(trades)
                    rows.append({"symbol": symbol, "live_list": is_live, "system": "B_bollinger",
                                 "param": param, "window": window_name,
                                 "period": period_name, **stats})
                    add_to_pool(("B_bollinger", param, period_name), symbol, trades)
            param = "no_macd_filter"
            for period_name, data in [("train", train), ("test", test)]:
                trades = backtest_system_a_relaxed(data)
                stats = summarize(trades)
                rows.append({"symbol": symbol, "live_list": is_live, "system": "A_relaxed",
                             "param": param, "window": window_name,
                             "period": period_name, **stats})
                add_to_pool(("A_relaxed", param, period_name), symbol, trades)
            param = "no_adx_filter"
            for period_name, data in [("train", train), ("test", test)]:
                trades = backtest_system_b_bollinger_relaxed(data)
                stats = summarize(trades)
                rows.append({"symbol": symbol, "live_list": is_live, "system": "B_bollinger_relaxed",
                             "param": param, "window": window_name,
                             "period": period_name, **stats})
                add_to_pool(("B_bollinger_relaxed", param, period_name), symbol, trades)
            for agree_th in AGREE_GRID:
                param = f"agree>={agree_th}/3"
                for period_name, data in [("train", train), ("test", test)]:
                    trades = backtest_system_a_ensemble(data, agree_th)
                    stats = summarize(trades)
                    rows.append({"symbol": symbol, "live_list": is_live, "system": "A_ensemble",
                                 "param": param, "window": window_name,
                                 "period": period_name, **stats})
                    add_to_pool(("A_ensemble", param, period_name), symbol, trades)
            for ema_period in EMA_TOUCH_GRID:
                param = f"ema{ema_period}_touch"
                for period_name, data in [("train", train), ("test", test)]:
                    trades = backtest_system_c(data, ema_period)
                    stats = summarize(trades)
                    rows.append({"symbol": symbol, "live_list": is_live, "system": "C",
                                 "param": param, "window": window_name,
                                 "period": period_name, **stats})
                    add_to_pool(("C", param, period_name), symbol, trades)

    report = pd.DataFrame(rows)
    report.to_csv(os.path.join(RESULTS_DIR, "backtest_report.csv"), index=False)

    pooled_rows = []
    for (system, param, period_name), trade_frames in pooled_trades.items():
        all_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
        stats = summarize(all_trades)
        stats["n_symbols"] = len(pooled_symbols.get((system, param, period_name), set()))
        pooled_rows.append({"system": system, "param": param, "period": period_name, **stats})
    pooled_report = pd.DataFrame(pooled_rows).sort_values(["system", "param", "period"])
    pooled_report.to_csv(os.path.join(RESULTS_DIR, "backtest_report_pooled.csv"), index=False)

    with open(os.path.join(RESULTS_DIR, "backtest_report.md"), "w") as f:
        f.write("# Backtest report (raw results, no auto-selected winner)\n\n")
        f.write(f"MIN_TRADES threshold for insufficient_data flag: {MIN_TRADES} "
                f"(preliminary, subject to revision)\n\n")
        f.write("## Per-symbol / per-window grid (diagnostic detail)\n\n")
        f.write(report.to_markdown(index=False))
        f.write("\n\n## Pooled across all instruments and all test/train windows\n\n")
        f.write("Trades from every CSV in data/ (live + research instruments) and every "
                "walk-forward/split window are pooled per (system, param, period) — the "
                "per-symbol/per-window cells above are individually too thin (max ~7 trades "
                "on any single test window) to judge MIN_TRADES against; pooling gives the "
                "statistic real sample size. `n_symbols` is how many instruments actually "
                "contributed at least one trade.\n\n")
        f.write(pooled_report.to_markdown(index=False))

    print(f"Wrote {len(report)} rows to results/backtest_report.csv, "
          f"{len(pooled_report)} pooled rows to results/backtest_report_pooled.csv, "
          f"and both to results/backtest_report.md")


if __name__ == "__main__":
    main()
