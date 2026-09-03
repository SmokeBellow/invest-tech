"""
Четыре варианта совместной работы System RSI2 (<10) и System TOM (offset=5)
на live-8, последние ~10 лет (фикс. старт 2016, без переноса капитала с
более ранних лет):

A) "force_all"     — при каждом новом сигнале TOM ПРИНУДИТЕЛЬНО закрываются
                      ВСЕ открытые позиции RSI2 (реализуем текущий P&L как
                      есть, в плюс или в минус), чтобы TOM могла зайти всем
                      освободившимся капиталом.
B) "force_profit"  — закрываем открытые позиции RSI2 перед входом TOM,
                      ТОЛЬКО если они СЕЙЧАС в плюсе (незафиксированный P&L
                      положителен на день ДО сигнала TOM); убыточные не
                      трогаем, TOM входит только на то, что реально
                      освободилось.
C) "leftover_only" — TOM НЕ форсирует ничего, конкурирует за уже свободный
                      риск-бюджет наравне с RSI2 (простой общий пул).
D) "split_50_50"   — раздельные портфели: $500 отдельно System RSI2,
                      $500 отдельно System TOM, независимо друг от друга,
                      суммируются в конце. Контрольная группа — показывает,
                      что объединённый риск-бюджет (A/B/C) даёт больше, чем
                      простое разделение капитала пополам.

Для варианта B нужен day-by-day mark-to-market открытых позиций RSI2 —
отдельный, более детальный симулятор, чем в equity_ibs_rsi2.py (там нет
трекинга незафиксированного P&L).

Риск-механика проекта не меняется: 1% риска на сделку от ТЕКУЩЕГО капитала,
потолок открытого риска 4%, макс. 5 позиций (на практике 4, см. CLAUDE.md
про недогруженный бюджет). Приоритет в день конфликта (A/B/C): TOM забирает
освободившееся место первой (это и есть цель принудительного закрытия),
затем RSI2 занимает оставшееся.

Запуск: python scripts/combo_rsi2_tom.py [--start-year YYYY]
Выход: results/combo_rsi2_tom_summary.md, results/combo_{mode}.csv
"""
import argparse
import heapq
import os

import numpy as np
import pandas as pd

import backtest as bt
from system_ibs_rsi2 import add_extra_indicators
from system_tom import backtest_system_tom  # noqa: F401 (для справки/консистентности)

DATA_DIR = "data"
RESULTS_DIR = "results"

START_CAPITAL = 1000.0
RISK_PER_TRADE = 0.01
MAX_OPEN_RISK = 0.08  # пересмотрено 15.08.2026: сетка 2-20% показала, что 4% сдерживало
                       # доходность именно в дни кластеризации сигналов TOM (конец месяца),
                       # 8% — точка, где эффект в основном насыщается (см. CLAUDE.md)
MAX_POSITIONS = 10    # поднято вместе с MAX_OPEN_RISK, чтобы не стать связывающим раньше него
RSI2_THRESHOLD = 10
TOM_OFFSET = 5


def load_sgov_returns():
    """Дневная доходность SGOV (парковка свободного кэша, см. §9 CLAUDE.md).
    История с 28.05.2020 — до этой даты фонда не существовало, дни без
    покрытия просто не приносят доходности (см. simulate_shared)."""
    path = os.path.join(DATA_DIR, "SGOV.csv")
    raw = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    return raw.set_index("date")["close"].pct_change()


def gen_rsi2_trades(d, symbol):
    """Та же логика, что backtest_system_rsi2 (system_ibs_rsi2.py), но с
    entry_price/stop/датами — нужно для mark-to-market открытых позиций."""
    trades = []
    position = None
    for i in range(1, len(d)):
        row, prev = d.iloc[i], d.iloc[i - 1]
        if position is None:
            cond_now = (row.close > row.ema200) and (row.rsi2 < RSI2_THRESHOLD)
            cond_prev = (prev.close > prev.ema200) and (prev.rsi2 < RSI2_THRESHOLD)
            if cond_now and not cond_prev and i + 1 < len(d):
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
                position["exit_date"] = rowj.date
                position["exit_price"] = exit_price
                position["reason"] = reason
                trades.append(position)
                position = None
    return trades


def gen_tom_trades(d, symbol):
    """offset=5 (боевой параметр TOM), с деталями входа/выхода."""
    d = d.reset_index(drop=True)
    d["month"] = d["date"].dt.to_period("M")
    months = d["month"].unique()
    trades = []
    for i in range(len(months) - 1):
        month_rows = d.index[d["month"] == months[i]]
        if len(month_rows) < TOM_OFFSET:
            continue
        last_idx = month_rows[-1]
        entry_idx = last_idx - (TOM_OFFSET - 1)
        if entry_idx < 0:
            continue
        next_month_rows = d.index[d["month"] == months[i + 1]]
        if len(next_month_rows) < 3:
            continue
        exit_idx = next_month_rows[2]
        if entry_idx >= len(d) or exit_idx >= len(d) or exit_idx <= entry_idx:
            continue
        if entry_idx < 1:
            continue
        entry_row = d.iloc[entry_idx]
        entry_price = entry_row["open"]
        # ATR с ДНЯ ДО входа, не с самого дня входа — см. system_tom.py,
        # исправлено 17.08.2026 (та же конвенция, что RSI2/B в этом проекте).
        atr = d.iloc[entry_idx - 1]["atr14"]
        if np.isnan(atr) or np.isnan(entry_price):
            continue
        stop = entry_price - 2.0 * atr
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
        trades.append({"symbol": symbol, "entry_date": entry_row["date"], "entry_price": entry_price,
                        "stop": stop, "exit_date": d.iloc[exit_idx]["date"], "exit_price": exit_price,
                        "reason": reason})
    return trades


def r_mult_of(pos, price):
    return (price - pos["entry_price"]) / (pos["entry_price"] - pos["stop"])


def rsi2_open_position(d, symbol):
    """Добавлено 25.08.2026 (см. CLAUDE.md §11 — по запросу пользователя
    показывать открытые позиции). Реплика позиционного стейт-машина
    gen_rsi2_trades, но вместо накопления только ЗАКРЫТЫХ сделок возвращает
    ОТКРЫТУЮ позицию на конец доступных данных (или None, если её нет)."""
    position = None
    for i in range(1, len(d)):
        row, prev = d.iloc[i], d.iloc[i - 1]
        if position is None:
            cond_now = (row.close > row.ema200) and (row.rsi2 < RSI2_THRESHOLD)
            cond_prev = (prev.close > prev.ema200) and (prev.rsi2 < RSI2_THRESHOLD)
            if cond_now and not cond_prev and i + 1 < len(d):
                entry_price = d.iloc[i + 1].open
                stop = entry_price - 2.0 * row.atr14
                if not np.isnan(stop) and stop < entry_price:
                    position = {"system": "RSI2", "symbol": symbol, "entry_date": d.iloc[i + 1].date,
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
    return position


def tom_entry_calendar_day(date):
    """ИСПРАВЛЕНО 25.08.2026 (нашёл баг при реализации rsi2_open_position/
    tom_open_position, см. CLAUDE.md §11): предыдущая попытка определить
    "5-й с конца торговый день месяца" ДО того, как месяц закончился,
    сравнивала month_rows[-1] (последний ДОСТУПНЫЙ день, т.е. просто
    "вчера") САМ С СОБОЙ минус 4 — это НИКОГДА не выполняется (разница
    всегда ровно TOM_OFFSET-1, не 0), поэтому вход по TOM в реальном
    времени НЕ детектировался никогда, ни разу с момента создания
    live_pending_entries.py (only обнаруживался постфактум, когда
    начинался следующий месяц и gen_tom_trades мог штатно посчитать
    exit_idx). Правильный способ — считать от КОНЦА КАЛЕНДАРНОГО месяца
    (известен заранее, не зависит от будущих цен), в рабочих днях
    (pandas bdate_range — недельные выходные учтены, биржевые праздники
    НЕТ, поэтому возможен редкий сдвиг на 1 день в месяцы с праздником в
    последнюю неделю — тот же класс приближения, что и everywhere в
    live-скриптах, где точная сверка всё равно происходит через
    gen_tom_trades постфактум).
    Возвращает True, если `date` — ожидаемый (по календарю) день входа TOM
    в своём месяце."""
    month_end = date + pd.offsets.MonthEnd(0)
    if date > month_end:
        return False
    remaining_bdays = len(pd.bdate_range(date + pd.Timedelta(days=1), month_end))
    return remaining_bdays == TOM_OFFSET - 1


def tom_open_position(d, symbol):
    """Добавлено 25.08.2026, см. rsi2_open_position выше — TOM-аналог.
    gen_tom_trades сознательно пропускает ТЕКУЩИЙ (ещё не завершённый)
    месяц целиком (не знает, где будет exit_idx, пока не появятся первые 3
    дня следующего месяца) — здесь явно проверяем именно этот случай, но
    используя календарный день входа (tom_entry_calendar_day), а НЕ
    "последний доступный день минус 4" (см. исправленный баг в докстринге
    tom_entry_calendar_day). Если стоп уже задет внутри данных, но
    exit_idx ещё не вычислим (мало данных следующего месяца) — возвращает
    None: такая сделка технически уже закрыта, но безопаснее подождать,
    пока gen_tom_trades сможет обработать её штатно.

    ИСПРАВЛЕНО 02.09.2026 (см. CLAUDE.md — баг обнаружен при живом
    ежедневном пересмотре: все открытые TOM-позиции, вход в которые
    состоялся в ПРЕДЫДУЩЕМ месяце, разом пропадали из open_positions.csv в
    первый же день нового месяца, хотя выход по ним ещё не наступил и не
    залогирован в journal.csv). Причина — поиск дня входа ограничивался
    ТОЛЬКО месяцем `last_date` (месяц последнего доступного бара). Пока
    last_date оставался в исходном месяце — это работало. Как только
    начинался новый календарный месяц, функция искала календарный день
    входа TOM внутри НОВОГО месяца (он наступит только ближе к его концу,
    ещё не наступил) — не находила ничего и возвращала None, хотя позиция,
    открытая в конце ПРЕДЫДУЩЕГО месяца, физически ещё держится (exit —
    только на 3-й торговый день нового месяца, см. gen_tom_trades). Теперь,
    если в текущем месяце входа ещё не было И в нём накопилось <3 торговых
    дней (то есть gen_tom_trades ещё физически не может вычислить exit_idx
    для позиции из прошлого месяца), ищем день входа в ПРЕДЫДУЩЕМ месяце —
    ровно то окно, в котором такая позиция может быть ещё не резолвлена
    штатно.

    ИСПРАВЛЕНО 03.09.2026: для ПРЕДЫДУЩЕГО (уже завершившегося) месяца день
    входа ищем ТЕМ ЖЕ способом, что и gen_tom_trades (последняя строка
    месяца минус TOM_OFFSET-1, по факту имеющихся в данных строк), а НЕ
    через tom_entry_calendar_day (bdate_range, только будние дни). Для
    инструментов, торгующих по выходным (BTCUSD — каждый день; EURUSD/
    USDJPY — с воскресного вечера), число строк за месяц в данных БОЛЬШЕ,
    чем число будних дней, поэтому "5-я с конца строка данных" и "5-й с
    конца будний день" — РАЗНЫЕ даты (на практике расхождение доходило до
    2 дней). Обнаружено по симптому: BTCUSD/USDJPY/EURUSD пропадали из
    open_positions.csv на 2-3 дня в начале месяца (calendar-приближение
    находило "входа нет", хотя gen_tom_trades уже был готов посчитать
    вход на другую дату), а в journal.csv по ним закреплялись ДВЕ строки
    входа с разными датами — provisional (по calendar-приближению, из
    live_pending_entries.py) и обычная (по gen_tom_trades, из ночного
    прогона). Точный расчёт возможен именно для завершённого месяца — все
    его строки уже есть в данных, в отличие от текущего месяца (там
    calendar-приближение остаётся единственным вариантом — не знаем, сколько
    ещё строк будет до конца месяца)."""
    d = d.reset_index(drop=True)
    n = len(d)
    last_date = d.iloc[-1]["date"]
    last_period = last_date.to_period("M")
    month_idx = d.index[d["date"].dt.to_period("M") == last_period]
    entry_idx = None
    for i in month_idx:
        if tom_entry_calendar_day(d.iloc[i]["date"]):
            entry_idx = i
            break
    if entry_idx is None and len(month_idx) < 3:
        prev_period = last_period - 1
        prev_month_idx = d.index[d["date"].dt.to_period("M") == prev_period]
        if len(prev_month_idx) >= TOM_OFFSET:
            entry_idx = prev_month_idx[-1] - (TOM_OFFSET - 1)
    if entry_idx is None or entry_idx < 1 or entry_idx >= n:
        return None
    entry_row = d.iloc[entry_idx]
    entry_price = entry_row["open"]
    atr = d.iloc[entry_idx - 1]["atr14"]
    if np.isnan(atr) or np.isnan(entry_price):
        return None
    stop = entry_price - 2.0 * atr
    if stop >= entry_price:
        return None
    for j in range(entry_idx, n):
        if d.iloc[j]["low"] <= stop:
            return None  # уже задет стоп, но exit_idx ещё не определить штатно — подождём gen_tom_trades
    return {"system": "TOM", "symbol": symbol, "entry_date": entry_row["date"],
            "entry_price": entry_price, "stop": stop, "bars": n - 1 - entry_idx}


def simulate_shared(rsi2_trades, tom_trades, close_lookup, mode, start_capital=START_CAPITAL,
                     sgov_returns=None, calendar_start=None, calendar_end=None, trade_log=None):
    """mode: 'force_all' | 'force_profit' | 'leftover_only' — один общий
    капитал и риск-бюджет на обе системы.

    sgov_returns (добавлено 20.08.2026, см. CLAUDE.md §9): дневная доходность
    SGOV (close.pct_change()), индексированная по дате. Если задана — капитал,
    НЕ занятый под открытые позиции ("свободный кэш"), каждый день зарабатывает
    эту доходность (аналог парковки в TMON для рублёвого портфеля). Занятый
    капитал = сумма $-номинала открытых позиций, где номинал считается из уже
    существующей формулы риск-сайзинга проекта (3.4 CLAUDE.md): номинал =
    risked_dollars / дистанция_до_стопа_в_% — т.е. позиция с более тесным
    стопом обходится дороже в $, с более широким — дешевле, ровно как и должно
    быть при risk-based sizing. Кэш floor'ится на 0 (маржа/шорт не моделируются,
    тот же известный пробел, что и остальные $-упрощения проекта, см. 3.4).
    Если sgov_returns=None — поведение не меняется (для обратной совместимости
    со старыми вызовами/результатами без парковки кэша).

    calendar_start/calendar_end (добавлено 21.08.2026 — фикс бага): границы
    календаря дневного accrual'а кэша. По умолчанию (None) выводятся из дат
    сделок (как раньше) — этого достаточно для многолетнего бэктеста, где
    сделок всегда много. НО на live-счёте в первые недели (когда сделок ещё
    НИ ОДНОЙ) даты сделок пусты, и без явных границ цикл вообще не запускался
    бы — кэш не копил бы доходность SGOV, а кривая эквити молчала бы до первой
    сделки. combo_c_curve() передаёт сюда live_start_date/as_of_date явно."""
    all_dates = sorted(set(t["entry_date"] for t in rsi2_trades + tom_trades) |
                        set(t["exit_date"] for t in rsi2_trades + tom_trades))
    rsi2_by_entry = {}
    for t in rsi2_trades:
        rsi2_by_entry.setdefault(t["entry_date"], []).append(t)
    tom_by_entry = {}
    for t in tom_trades:
        tom_by_entry.setdefault(t["entry_date"], []).append(t)

    equity = start_capital
    open_positions = []
    curve = []
    taken = {"rsi2": 0, "tom": 0}
    skipped = {"rsi2": 0, "tom": 0}
    forced_closes = 0

    def n_open():
        return len(open_positions)

    def open_risk():
        return len(open_positions) * RISK_PER_TRADE

    def can_admit():
        return open_risk() + RISK_PER_TRADE <= MAX_OPEN_RISK + 1e-9 and n_open() < MAX_POSITIONS

    def position_notional(entry_price, stop, risked_dollars):
        stop_dist_frac = (entry_price - stop) / entry_price
        return risked_dollars / stop_dist_frac if stop_dist_frac > 1e-9 else risked_dollars

    def close_position(pos, price, date):
        nonlocal equity
        r_mult = r_mult_of(pos, price)
        equity += r_mult * pos["risked_dollars"]
        curve.append((date, equity))
        if trade_log is not None:
            trade_log.append({"date": date, "symbol": pos["symbol"], "type": pos["type"], "r_mult": r_mult})

    range_start = calendar_start if calendar_start is not None else (all_dates[0] if all_dates else None)
    range_end = calendar_end if calendar_end is not None else (all_dates[-1] if all_dates else None)

    if sgov_returns is not None and range_start is not None:
        # Календарь торговых дней — объединение дат ВСЕХ инструментов
        # (close_lookup), не только sgov_returns.index: история SGOV
        # начинается только с 28.05.2020, а торговля живёт с более раннего
        # start_date (по умолчанию 2016) — если бы календарь брался из
        # sgov_returns, дни/сделки ДО запуска SGOV тихо выпали бы из цикла.
        # До 2020 кэш просто не зарабатывает ничего (sgov_returns.get даёт 0
        # для отсутствующих дат) — это и есть корректное поведение, фонда
        # тогда не существовало. Границы диапазона — из calendar_start/end,
        # если заданы явно (нужно для live-счёта без единой сделки в окне,
        # см. докстринг), иначе — из дат сделок, как раньше.
        full_calendar = set()
        for series in close_lookup.values():
            full_calendar.update(series.index)
        loop_dates = sorted(d for d in full_calendar if range_start <= d <= range_end)
    else:
        loop_dates = all_dates

    for date in loop_dates:
        # 0) доходность SGOV на кэш, НЕ занятый под открытые с вчера позиции
        # (номинал считается по цене/стопу на момент входа — не переоценивается
        # день в день, тот же принцип, что и risked_dollars в остальном движке)
        if sgov_returns is not None:
            invested_notional = sum(pos["notional"] for pos in open_positions)
            cash = max(0.0, equity - invested_notional)
            day_ret = sgov_returns.get(date, 0.0)
            if cash > 0 and not pd.isna(day_ret):
                equity += cash * day_ret

        # 1) натуральные закрытия на сегодня
        still_open = []
        for pos in open_positions:
            if pos["exit_date"] == date:
                close_position(pos, pos["exit_price"], date)
            else:
                still_open.append(pos)
        open_positions = still_open

        # 2) если сегодня TOM хочет войти — освобождаем капитал по правилу mode
        todays_tom = tom_by_entry.get(date, [])
        if todays_tom and mode in ("force_all", "force_profit"):
            still_open = []
            for pos in open_positions:
                if pos["type"] != "rsi2":
                    still_open.append(pos)
                    continue
                mtm_price = close_lookup[pos["symbol"]].asof(date - pd.Timedelta(days=1))
                if pd.isna(mtm_price):
                    still_open.append(pos)
                    continue
                mtm_r = r_mult_of(pos, mtm_price)
                should_force = (mode == "force_all") or (mode == "force_profit" and mtm_r > 0)
                if should_force:
                    close_position(pos, mtm_price, date)
                    forced_closes += 1
                else:
                    still_open.append(pos)
            open_positions = still_open

        # 3) впускаем TOM-сделки на сегодня (приоритет — освобождали именно под них)
        for t in todays_tom:
            if can_admit():
                risked = RISK_PER_TRADE * equity
                open_positions.append({"type": "tom", "symbol": t["symbol"], "entry_price": t["entry_price"],
                                        "stop": t["stop"], "exit_date": t["exit_date"], "exit_price": t["exit_price"],
                                        "risked_dollars": risked,
                                        "notional": position_notional(t["entry_price"], t["stop"], risked)})
                taken["tom"] += 1
            else:
                skipped["tom"] += 1

        # 4) впускаем RSI2-сделки на сегодня (из оставшегося бюджета)
        for t in rsi2_by_entry.get(date, []):
            if can_admit():
                risked = RISK_PER_TRADE * equity
                open_positions.append({"type": "rsi2", "symbol": t["symbol"], "entry_price": t["entry_price"],
                                        "stop": t["stop"], "exit_date": t["exit_date"], "exit_price": t["exit_price"],
                                        "risked_dollars": risked,
                                        "notional": position_notional(t["entry_price"], t["stop"], risked)})
                taken["rsi2"] += 1
            else:
                skipped["rsi2"] += 1

        # 5) редкий случай: сделка открылась и закрылась В ТОТ ЖЕ ДЕНЬ (напр.
        # SMA5-пересечение уже на дне входа) — шаг 1 её не увидел, т.к. её ещё
        # не существовало на момент обработки закрытий; закрываем сразу же,
        # иначе позиция "зависнет" открытой навсегда.
        still_open = []
        for pos in open_positions:
            if pos["exit_date"] == date:
                close_position(pos, pos["exit_price"], date)
            else:
                still_open.append(pos)
        open_positions = still_open

        # 6) с SGOV-доходностью на кэш эквити меняется КАЖДЫЙ день, не только
        # в дни закрытия сделок (те попадают в curve через close_position) —
        # без этой строки кривая "застревала" на последней сделке, и дни без
        # сделок (в т.ч. весь период, пока сделок ещё не было ни одной — как
        # на live-счёте в первые недели) не отражали накопленную доходность
        # кэша вообще. drop_duplicates(keep="last") у потребителей curve
        # (напр. combo_c_curve) корректно оставит именно это, "закрывающее
        # день" значение, даже если сегодня был ещё и close_position().
        if sgov_returns is not None:
            curve.append((date, equity))

    return curve, equity, taken, skipped, forced_closes


def simulate_independent(trades, start_capital):
    """Один независимый портфель (для варианта D — раздельные $500/$500)."""
    trades = sorted(trades, key=lambda t: t["entry_date"])
    equity = start_capital
    open_heap = []
    open_risk = 0.0
    curve = [(trades[0]["entry_date"] if trades else None, equity)]
    taken, skipped = 0, 0

    def flush_until(cutoff_date):
        nonlocal equity, open_risk
        while open_heap and open_heap[0][0] <= cutoff_date:
            exit_date, risked, r_mult = heapq.heappop(open_heap)
            equity += r_mult * risked
            open_risk -= RISK_PER_TRADE
            curve.append((exit_date, equity))

    for t in trades:
        flush_until(t["entry_date"])
        if open_risk + RISK_PER_TRADE > MAX_OPEN_RISK + 1e-9 or len(open_heap) >= MAX_POSITIONS:
            skipped += 1
            continue
        r_mult = r_mult_of(t, t["exit_price"])
        risked = RISK_PER_TRADE * equity
        heapq.heappush(open_heap, (t["exit_date"], risked, r_mult))
        open_risk += RISK_PER_TRADE
        taken += 1
    while open_heap:
        exit_date, risked, r_mult = heapq.heappop(open_heap)
        equity += r_mult * risked
        curve.append((exit_date, equity))
    return curve, equity, taken, skipped


def annual_returns(curve, start_capital=START_CAPITAL):
    if not curve:
        return pd.DataFrame()
    df = pd.DataFrame([c for c in curve if c[0] is not None], columns=["date", "equity"])
    if df.empty:
        return pd.DataFrame()
    df["year"] = df.date.apply(lambda d: d.year)
    yearly = df.groupby("year")["equity"].last().reset_index()
    yearly["equity_prev"] = yearly["equity"].shift(1).fillna(start_capital)
    yearly["return_pct"] = (yearly["equity"] / yearly["equity_prev"] - 1) * 100
    return yearly


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2016)
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    start_date = pd.Timestamp(f"{args.start_year}-01-01")

    dfs, close_lookup = {}, {}
    for symbol in sorted(bt.LIVE_INSTRUMENTS):
        raw = pd.read_csv(os.path.join(DATA_DIR, f"{symbol}.csv"), parse_dates=["date"]).sort_values("date")
        d = bt.add_indicators(raw)
        d = add_extra_indicators(d)
        d = d.iloc[60:].reset_index(drop=True)
        dfs[symbol] = d
        close_lookup[symbol] = raw.set_index("date")["close"].sort_index()

    all_rsi2, all_tom = [], []
    for symbol, d in dfs.items():
        all_rsi2.extend([t for t in gen_rsi2_trades(d, symbol) if t["entry_date"] >= start_date])
        all_tom.extend([t for t in gen_tom_trades(d, symbol) if t["entry_date"] >= start_date])

    sgov_returns = load_sgov_returns()

    summary = [f"# RSI2 + TOM combined portfolio variants ({args.start_year}-2026, live-8, $1000 старт)\n",
               "Свободный кэш (не занятый под открытые позиции) парковка в SGOV, "
               "см. CLAUDE.md §9 (добавлено 20.08.2026).\n"]
    for mode, label in [("force_all", "A: TOM закрывает ВСЕ открытые RSI2"),
                         ("force_profit", "B: TOM закрывает RSI2 только В ПЛЮСЕ"),
                         ("leftover_only", "C: TOM входит только на остаток бюджета")]:
        curve, final_equity, taken, skipped, forced = simulate_shared(all_rsi2, all_tom, close_lookup, mode,
                                                                        sgov_returns=sgov_returns)
        yearly = annual_returns(curve)
        yearly.to_csv(os.path.join(RESULTS_DIR, f"combo_{mode}.csv"), index=False)
        summary.append(f"## {label}\n\nИтог: ${final_equity:.2f} ({(final_equity/START_CAPITAL-1)*100:+.1f}%), "
                        f"взято RSI2/TOM={taken['rsi2']}/{taken['tom']}, "
                        f"пропущено RSI2/TOM={skipped['rsi2']}/{skipped['tom']}, "
                        f"принудительных закрытий={forced}\n")
        summary.append(yearly.to_markdown(index=False) + "\n")

    # D: раздельные портфели $500/$500
    half = START_CAPITAL / 2
    curve_rsi2, eq_rsi2, taken_rsi2, skipped_rsi2 = simulate_independent(all_rsi2, half)
    curve_tom, eq_tom, taken_tom, skipped_tom = simulate_independent(all_tom, half)
    yearly_rsi2 = annual_returns(curve_rsi2, half)
    yearly_tom = annual_returns(curve_tom, half)
    combined = yearly_rsi2.set_index("year")["equity"].add(yearly_tom.set_index("year")["equity"], fill_value=None)
    combined_df = combined.reset_index()
    combined_df.columns = ["year", "equity"]
    combined_df["equity_prev"] = combined_df["equity"].shift(1).fillna(START_CAPITAL)
    combined_df["return_pct"] = (combined_df["equity"] / combined_df["equity_prev"] - 1) * 100
    combined_df.to_csv(os.path.join(RESULTS_DIR, "combo_split_50_50.csv"), index=False)
    final_d = eq_rsi2 + eq_tom
    summary.append(f"## D: раздельные портфели 50/50 (RSI2 ${half:.0f} + TOM ${half:.0f}, независимо)\n\n"
                    f"Итог: ${final_d:.2f} ({(final_d/START_CAPITAL-1)*100:+.1f}%), "
                    f"RSI2 взято/пропущено={taken_rsi2}/{skipped_rsi2}, "
                    f"TOM взято/пропущено={taken_tom}/{skipped_tom}\n")
    summary.append(combined_df.to_markdown(index=False) + "\n")

    with open(os.path.join(RESULTS_DIR, "combo_rsi2_tom_summary.md"), "w") as f:
        f.write("\n".join(summary))
    print(f"Wrote results/combo_rsi2_tom_summary.md and results/combo_{{mode}}.csv")


if __name__ == "__main__":
    main()
