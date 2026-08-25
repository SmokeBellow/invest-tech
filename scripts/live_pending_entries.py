"""
live_pending_entries.py — "утренний" проход live paper-trading счёта (см.
CLAUDE.md, раздел 9). Решает конкретную проблему: обычный ночной прогон
(live_review.py, 04:00 UTC) видит цену открытия дня сделки только через
сутки ПОСЛЕ того, как она уже фактически исполнилась (EOD-бар появляется
только когда день уже закрылся) — узнаём о собственных сделках с
задержкой в лишний день.

КРИТИЧЕСКИ ВАЖНАЯ ДИСЦИПЛИНА (прямое требование пользователя, та же
причина, по которой весь проект защищается от same-bar lookahead, см.
5.9 CLAUDE.md): триггер и цена входа берутся из ДВУХ независимых
источников, никогда не смешиваются:

  1. ТРИГГЕР — считается ИСКЛЮЧИТЕЛЬНО по уже ЗАКРЫТЫМ дням в data/*.csv.
     С 22.08.2026 (см. CLAUDE.md §9.3) сам этот скрипт больше НЕ полагается
     только на то, что успел зафиксировать ночной прогон (06:00 EKT) — шаг
     "Re-fetch daily bars" в live_morning_check.yml догоняет данные прямо
     перед запуском этого скрипта, на случай если бар не успел появиться у
     FMP/Tiingo к 06:00. Это НЕ нарушает дисциплину two-source: повторный
     фетч тянет ТОЛЬКО уже закрытые дни (сегодняшняя сессия к моменту
     запуска, 15:00 UTC, ещё не закрылась) — сигнал по-прежнему полностью
     определён ДО открытия сегодняшней сессии, просто на более свежих (но
     всё ещё исторических) данных.
  2. ЦЕНА ВХОДА — отдельный, независимый запрос FMP `quote` (реал-тайм),
     сделанный СЕГОДНЯ УТРОМ, уже после открытия рынка, ТОЛЬКО для
     инструментов, где сработал триггер. Это открытие ТЕКУЩЕЙ сессии —
     то самое значение, которое было бы реально доступно для исполнения
     ордера на настоящем счёте в момент, когда триггер уже известен.

Логика триггера — точная копия entry-условий gen_rsi2_trades/gen_tom_trades
(combo_rsi2_tom.py), но с явным трекингом "уже открыта ли позиция сейчас"
(эти функции возвращают только ЗАВЕРШЁННЫЕ сделки и не говорят, открыта ли
позиция ПРЯМО СЕЙЧАС — для утренней проверки это нужно знать явно, чтобы не
задвоить вход по уже открытой позиции).

НЕ резолвит выходы (стоп/sma5-cross/time-stop) — они по-прежнему
определяются только вечерним EOD-прогоном (см. ограничения v1 в
live_review.py). Гэп-стопы (когда open уже хуже уровня стопа) сюда тоже
пока не входят — открытый вопрос на будущее.

Запуск: python scripts/live_pending_entries.py (утром, после открытия NYSE,
отдельным шагом расписания в .github/workflows/live_morning_check.yml)
Требует FMP_API_KEY (тот же секрет, что и остальные live-скрипты).
Выход: дописывает "provisional"-строки в data/live/journal.csv,
обновляет data/live/state.json::pending_confirmed_today.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(__file__))
import backtest as bt
from system_ibs_rsi2 import add_extra_indicators
from combo_rsi2_tom import tom_entry_calendar_day

DATA_DIR = "data"
LIVE_DIR = "data/live"
API_KEY = os.environ["FMP_API_KEY"]
QUOTE_URL = "https://financialmodelingprep.com/stable/quote"

FMP_SYMBOL = {  # маппинг наших тикеров на символы FMP quote-эндпоинта
    "EURUSD": "EURUSD", "USDJPY": "USDJPY", "BTCUSD": "BTCUSD",
}


def rsi2_open_position_and_pending(d, symbol):
    """Реплика стейтфул-цикла gen_rsi2_trades (combo_rsi2_tom.py), но
    вместо списка завершённых сделок возвращает: открыта ли позиция ПРЯМО
    СЕЙЧАС (после последнего доступного дня), и сработал ли триггер на
    вход НА ПОСЛЕДНЕМ доступном дне (если позиции нет)."""
    position = None
    n = len(d)
    for i in range(1, n):
        row, prev = d.iloc[i], d.iloc[i - 1]
        if position is None:
            cond_now = (row.close > row.ema200) and (row.rsi2 < 10)
            cond_prev = (prev.close > prev.ema200) and (prev.rsi2 < 10)
            if cond_now and not cond_prev and i + 1 < n:
                entry_price = d.iloc[i + 1].open
                stop = entry_price - 2.0 * row.atr14
                if not np.isnan(stop) and stop < entry_price:
                    position = {"symbol": symbol, "entry_date": d.iloc[i + 1].date,
                                "entry_price": entry_price, "stop": stop, "bars": 0}
        else:
            rowj = d.iloc[i]
            position["bars"] += 1
            exit_price, reason = None, None
            if rowj.low <= position["stop"]:
                exit_price = rowj.open if rowj.open < position["stop"] else position["stop"]
                reason = "stop"
            elif prev.close <= prev.sma5 and rowj.close > rowj.sma5:
                exit_price, reason = rowj.close, "sma5_cross"
            elif position["bars"] >= 10:
                exit_price, reason = rowj.close, "time_stop"
            if reason:
                position = None

    if position is not None:
        return position, None  # уже в позиции — новый вход не нужен

    # позиции нет — проверяем, не сработал ли триггер ИМЕННО на последнем дне
    row, prev = d.iloc[-1], d.iloc[-2]
    cond_now = (row.close > row.ema200) and (row.rsi2 < 10)
    cond_prev = (prev.close > prev.ema200) and (prev.rsi2 < 10)
    if cond_now and not cond_prev:
        stop_hint = row.close - 2.0 * row.atr14  # финальный стоп пересчитается от реальной цены входа
        return None, {"symbol": symbol, "system": "RSI2", "reason": "rsi2<10+ema200", "stop_hint": stop_hint}
    return None, None


def tom_pending_entry(d, symbol, today):
    """Вход TOM — чисто календарный (не индикаторный), известен заранее:
    entry_idx = 5-й с конца торговый день месяца.

    ИСПРАВЛЕНО 25.08.2026 (см. CLAUDE.md §11 и докстринг
    tom_entry_calendar_day в combo_rsi2_tom.py): старая версия сравнивала
    "последний доступный день" (т.е. вчера) с самим собой минус 4 —
    математически никогда не совпадало (разница всегда ровно TOM_OFFSET-1,
    не 0), поэтому эта функция НИКОГДА не детектировала вход в реальном
    времени, ни разу с момента создания скрипта (17.08.2026) — TOM-сделки
    ловились только постфактум через ночной пересчёт `gen_tom_trades`.
    Правильная проверка — является ли СЕГОДНЯШНИЙ (ещё не закрытый)
    календарный день ожидаемым днём входа, через tom_entry_calendar_day
    (считает от конца календарного месяца, известного заранее, а не от
    последнего ДОСТУПНОГО дня данных)."""
    if not tom_entry_calendar_day(today):
        return None
    last_row = d.iloc[-1]
    if np.isnan(last_row["atr14"]):
        return None
    stop_hint = last_row["close"] - 2.0 * last_row["atr14"]
    return {"symbol": symbol, "system": "TOM", "reason": "turn_of_month_offset5", "stop_hint": stop_hint}


def fetch_open_price(symbol):
    """Возвращает None (никогда не бросает исключение) при любой проблеме с
    запросом — HTTP-ошибка здесь НЕ должна ронять весь скрипт: другие
    инструменты в этом же прогоне (и state.json/morning_check_date) должны
    сохраниться независимо от того, что один конкретный тикер недоступен.
    Обнаружено 25.08.2026 (см. CLAUDE.md раздел 11): `/quote`-эндпоинт FMP на
    free-плане возвращает 402 Payment Required для тех же equity-тикеров,
    что и EOD-эндпоинт (QQQ/EEM/TLT/XLE — весь живой список кроме SPY, см.
    CLAUDE.md §4) — раньше это приводило к необработанному краху всего
    прогона (`raise_for_status()`), из-за чего 19.08.2026 была пропущена
    provisional-запись по реальному входу QQQ (сделка позже корректно
    восстановилась постфактум через ночной прогон, но без реальной
    утренней котировки). Для этих тикеров провижнл-механизм системно не
    работает — Tiingo free-план не даёт intraday (см. CLAUDE.md 5.6), чистой
    альтернативы нет; такие сделки просто ловятся штатным ночным прогоном,
    как и было задокументировано как "известное упрощение v1" в
    live_review.py."""
    fmp_symbol = FMP_SYMBOL.get(symbol, symbol)
    try:
        resp = requests.get(QUOTE_URL, params={"symbol": fmp_symbol, "apikey": API_KEY}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"    ошибка запроса котировки для {symbol}: {e}")
        return None
    if not isinstance(data, list) or not data or "open" not in data[0] or data[0]["open"] is None:
        return None
    return float(data[0]["open"])


def load_state():
    path = os.path.join(LIVE_DIR, "state.json")
    with open(path) as f:
        return json.load(f)


def save_state(state):
    with open(os.path.join(LIVE_DIR, "state.json"), "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def append_journal(rows):
    if not rows:
        return
    path = os.path.join(LIVE_DIR, "journal.csv")
    df = pd.DataFrame(rows)
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


def main():
    if not os.path.exists(os.path.join(LIVE_DIR, "state.json")):
        print("data/live/state.json не найден — сначала должен отработать ночной live_review.py")
        return
    state = load_state()
    today = pd.Timestamp.today().normalize()
    already_done_today = state.get("morning_check_date") == today.strftime("%Y-%m-%d")
    if already_done_today:
        print(f"Утренняя проверка на {today.date()} уже проводилась сегодня, пропускаем (идемпотентность)")
        return

    pending = []  # список кандидатов на вход, определённых ТОЛЬКО по вчерашним закрытым данным
    for symbol in sorted(bt.LIVE_INSTRUMENTS):
        raw = pd.read_csv(os.path.join(DATA_DIR, f"{symbol}.csv"), parse_dates=["date"]).sort_values("date")
        d = bt.add_indicators(raw)
        d = add_extra_indicators(d)
        d = d.iloc[60:].reset_index(drop=True)
        if len(d) < 3:
            continue

        _, rsi2_pending = rsi2_open_position_and_pending(d, symbol)
        if rsi2_pending:
            pending.append(rsi2_pending)

        tom_pending = tom_pending_entry(d, symbol, today)
        if tom_pending:
            pending.append(tom_pending)

    print(f"Триггеров на вход (по вчерашнему закрытию, ДО открытия сегодняшней сессии): {len(pending)}")
    if not pending:
        state["morning_check_date"] = today.strftime("%Y-%m-%d")
        save_state(state)
        print("Нечего проверять сегодня утром.")
        return

    new_rows = []
    logged = set(state.get("logged_events", []))
    for p in pending:
        print(f"  {p['system']} {p['symbol']} ({p['reason']}) — запрашиваю сегодняшнее открытие...")
        open_price = fetch_open_price(p["symbol"])
        if open_price is None:
            print(f"    не удалось получить котировку для {p['symbol']}, пропуск (доберёт ночной прогон)")
            continue
        # size_usd/risk_usd/stop оставляем пустыми здесь: точный стоп = open_price - 2*ATR14
        # (ATR берётся с вчерашнего дня, известен уже сейчас), но окончательные
        # $-цифры вычисляет ночной прогон (live_review.py) при подтверждении —
        # не дублируем формулу риск-сайзинга в двух местах.
        key = f"{p['system']}|{p['symbol']}|{today.date()}|entry"
        if key in logged:
            continue
        new_rows.append({"date": today.strftime("%Y-%m-%d"), "system": p["system"], "symbol": p["symbol"],
                          "action": "entry", "price": round(open_price, 4), "size_usd": "",
                          "stop": "", "risk_usd": "", "risk_pct": "",
                          "reason": p["reason"] + " (provisional, подтвердит ночной прогон)"})
        logged.add(key)
        print(f"    вход по ${open_price:.4f} (сегодняшнее открытие, provisional)")

    append_journal(new_rows)
    state["logged_events"] = sorted(logged)
    state["morning_check_date"] = today.strftime("%Y-%m-%d")
    save_state(state)
    print(f"\nЗаписано предварительных входов: {len(new_rows)}")


if __name__ == "__main__":
    main()
