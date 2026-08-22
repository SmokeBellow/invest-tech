"""
System Bandit-Permanent (22.08.2026) — ДИАГНОСТИЧЕСКАЯ, НЕ боевая система.
Пробует многорукого бандита для перераспределения весов ВНУТРИ вечного
портфеля (SPY/TLT/SHY/GLD) вместо фиксированных 25/25/25/25 с ежегодным
ребалансом. Запрошено пользователем как эксперимент после обсуждения, где
именно бандиты могли бы быть уместны в проекте.

Уточнение по постановке задачи: классический "многорукий бандит" — это
ЧАСТИЧНАЯ обратная связь (видишь награду только выбранного рычага). Портфель
из 4 активов, где держим ВСЕ 4 сразу — это full-information задача (Hedge /
Exponentially Weighted experts, тот же класс алгоритмов, что и в online
portfolio selection, напр. Cover's Universal Portfolio). Реализовано именно
так — доли ПРОПОРЦИОНАЛЬНЫ экспоненциально взвешенным прошлым доходностям
(Hedge), не all-in на "победителя" (это был бы дискретный bandit-выбор, но
убил бы диверсификацию вечного портфеля вообще).

Механика: ежемесячный ребаланс (не ежегодный, как в боевом вечном
портфеле — иначе бандит успевает сделать только ~19 шагов за 20 лет).
Награда за месяц = реальная доходность актива за месяц. Обновление весов:
w_i *= exp(eta * r_i), нормализация. eta фиксируется ЗАРАНЕЕ, узкая сетка
{4, 8, 16} для проверки чувствительности (не для выбора "лучшего" — та же
дисциплина, что и везде в проекте, см. CLAUDE.md 5.2).

РЕЗУЛЬТАТ (см. CLAUDE.md, раздел "Bandit-эксперименты"): бандит даёт БОЛЬШЕ
$-доходности, но заметно ХУЖЕ по риск-скорректированной метрике (Calmar) —
он постепенно перевешивает в трендовые активы (GLD/SPY), теряя как раз ту
диверсификацию, ради которой вечный портфель и был выбран как 60%-доля
боевого блендинга. НЕ рекомендуется как замена, оставлен как диагностика.

Запуск: python scripts/system_bandit_permanent.py
Выход: results/bandit_permanent_pooled.csv, печатает сводку.
"""
import os

import numpy as np
import pandas as pd

from benchmark_permanent_ofz import permanent_portfolio_daily

ASSETS = ["SPY", "TLT", "SHY", "GLD"]
DATA_DIR = "data"
RESULTS_DIR = "results"
START_CAPITAL = 1000.0
START_DATE = pd.Timestamp("2007-01-01")


def load_prices(start_date):
    prices = {}
    for a in ASSETS:
        df = pd.read_csv(os.path.join(DATA_DIR, f"{a}.csv"), parse_dates=["date"]).sort_values("date")
        df = df[df.date >= start_date].reset_index(drop=True)
        prices[a] = df.set_index("date")["close"]
    common = pd.DatetimeIndex(sorted(set.intersection(*[set(p.index) for p in prices.values()])))
    return pd.DataFrame({a: prices[a].reindex(common) for a in ASSETS})


def max_drawdown_pct(series):
    peak = series.iloc[0]
    max_dd = 0.0
    for v in series:
        peak = max(peak, v)
        max_dd = min(max_dd, (v - peak) / peak * 100)
    return max_dd


def cagr(series):
    years = (series.index[-1] - series.index[0]).days / 365.25
    return ((series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1) * 100


def hedge_bandit(price_df, eta, rebalance="M"):
    dates = price_df.index
    periods = pd.Series(dates).dt.to_period(rebalance)
    weights = np.array([1 / len(ASSETS)] * len(ASSETS))
    period_start_prices = price_df.iloc[0].values.astype(float)
    capital = START_CAPITAL
    shares = capital * weights / period_start_prices
    curve = []
    n = len(dates)
    for i in range(n):
        value = float(np.sum(shares * price_df.iloc[i].values))
        curve.append(value)
        is_last_of_period = (i == n - 1) or (periods.iloc[i + 1] != periods.iloc[i])
        if is_last_of_period and i != n - 1:
            period_returns = price_df.iloc[i].values / period_start_prices - 1.0
            weights = weights * np.exp(eta * period_returns)
            weights = weights / weights.sum()
            capital = value
            period_start_prices = price_df.iloc[i].values.astype(float)
            shares = capital * weights / period_start_prices
    return pd.Series(curve, index=dates)


def fixed_rebalance(price_df, target_weights, rebalance="M"):
    dates = price_df.index
    periods = pd.Series(dates).dt.to_period(rebalance)
    period_start_prices = price_df.iloc[0].values.astype(float)
    capital = START_CAPITAL
    shares = capital * np.array(target_weights) / period_start_prices
    curve = []
    n = len(dates)
    for i in range(n):
        value = float(np.sum(shares * price_df.iloc[i].values))
        curve.append(value)
        is_last_of_period = (i == n - 1) or (periods.iloc[i + 1] != periods.iloc[i])
        if is_last_of_period and i != n - 1:
            capital = value
            period_start_prices = price_df.iloc[i].values.astype(float)
            shares = capital * np.array(target_weights) / period_start_prices
    return pd.Series(curve, index=dates)


def buy_hold(price_df, asset):
    p = price_df[asset]
    return START_CAPITAL * p / p.iloc[0]


def summarize(name, series):
    dd = max_drawdown_pct(series)
    c = cagr(series)
    calmar = -c / dd if dd != 0 else float("nan")
    return {"strategy": name, "final_equity": round(series.iloc[-1], 2),
            "total_return_pct": round((series.iloc[-1] / START_CAPITAL - 1) * 100, 1),
            "cagr_pct": round(c, 2), "max_drawdown_pct": round(dd, 1), "calmar": round(calmar, 2)}


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    price_df = load_prices(START_DATE)
    rows = []

    rows.append(summarize("fixed_25_25_25_25_annual", permanent_portfolio_daily(START_DATE)))
    rows.append(summarize("fixed_25_25_25_25_monthly", fixed_rebalance(price_df, [0.25] * 4, "M")))
    for a in ASSETS:
        rows.append(summarize(f"buy_hold_{a}", buy_hold(price_df, a)))
    for eta in [4, 8, 16]:
        rows.append(summarize(f"bandit_hedge_eta{eta}", hedge_bandit(price_df, eta, "M")))

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(RESULTS_DIR, "bandit_permanent_pooled.csv"), index=False)
    print(f"\nWrote {len(df)} rows to results/bandit_permanent_pooled.csv")
