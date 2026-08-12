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
                position = {"entry_i": i + 1, "entry_price": d.iloc[i + 1].open, "stop": row.ema50 * 0.99}
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
                risk_pct = (position["entry_price"] - position["stop"]) / position["entry_price"] * 100
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
    """Anchored walk-forward для длинной истории (>=6 лет), иначе простой 70/30 split."""
    df = df.reset_index(drop=True)
    n = len(df)
    windows = []
    if years_available >= 6:
        # anchored: train растёт, test — следующий год
        bars_per_year = n / years_available
        for k in range(years_available - 4, years_available - 1):  # 3 окна ближе к концу истории
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
                for period_name, data in [("train", train), ("test", test)]:
                    trades = backtest_system_a(data, adx_th)
                    stats = summarize(trades)
                    rows.append({"symbol": symbol, "live_list": is_live, "system": "A",
                                 "param": f"ADX>={adx_th}", "window": window_name,
                                 "period": period_name, **stats})
            for rsi_th in RSI_GRID:
                for period_name, data in [("train", train), ("test", test)]:
                    trades = backtest_system_b(data, rsi_th)
                    stats = summarize(trades)
                    rows.append({"symbol": symbol, "live_list": is_live, "system": "B",
                                 "param": f"RSI<={rsi_th}", "window": window_name,
                                 "period": period_name, **stats})

    report = pd.DataFrame(rows)
    report.to_csv(os.path.join(RESULTS_DIR, "backtest_report.csv"), index=False)

    with open(os.path.join(RESULTS_DIR, "backtest_report.md"), "w") as f:
        f.write("# Backtest report (raw results, no auto-selected winner)\n\n")
        f.write(f"MIN_TRADES threshold for insufficient_data flag: {MIN_TRADES} "
                f"(preliminary, subject to revision)\n\n")
        f.write(report.to_markdown(index=False))

    print(f"Wrote {len(report)} rows to results/backtest_report.csv and .md")


if __name__ == "__main__":
    main()
