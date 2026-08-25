"""
live_review.py — ежедневный МЕХАНИЧЕСКИЙ пересмотр live paper-trading счёта
(см. CLAUDE.md, раздел 9). Никакого участия LLM не требует — все решения
(вход/выход RSI2/TOM, ежегодный ребаланс вечного портфеля) уже полностью
формализованы в blend_combo_permanent.py/combo_rsi2_tom.py/
benchmark_permanent_ofz.py и переиспользуются здесь как есть, БЕЗ отдельной
"боевой" копии логики — чтобы не могло возникнуть расхождения между
бэктестом и живым счётом.

Идея: боевая стратегия (40% Комбо C / 60% вечный портфель, ежегодный
ребаланс — CLAUDE.md 5.17/5.18) — это чистая функция от истории цен.
Каждый день скрипт просто пересчитывает `blend()` заново на данных,
дополненных свежим баром (см. live_fetch_incremental.py), с фиксированной
датой старта live-счёта (data/live/state.json::live_start_date). Последняя
точка этой кривой — это и есть текущая стоимость счёта. Никакого отдельного
"движка исполнения" не нужно — та же функция, что использовалась для всего
бэктеста, теперь считается на растущем в реальном времени наборе данных.

Отдельно: сравнивает список сделок RSI2/TOM (entry_date/exit_date) с уже
залогированными в data/live/state.json, чтобы понять, что произошло НОВОГО
с прошлого запуска, и дописывает эти события в data/live/journal.csv
(формат CLAUDE.md 3.7). Ежегодный ребаланс вечного портфеля обрабатывается
внутри permanent_portfolio_daily автоматически, отдельной записью в журнал
пока не логируется (см. ограничения v1 ниже).

Известные упрощения v1 (открыто на будущее, не додумывать молча):
  - `gen_rsi2_trades`/`gen_tom_trades` (переиспользуются как есть) возвращают
    только ЗАВЕРШЁННЫЕ сделки — значит, строка "entry" в журнале появляется
    не в момент фактического открытия позиции, а ПОСТФАКТУМ, одновременно
    со строкой "exit", как только сделка закрылась (дата в колонке `date`
    при этом честная — исторический день входа, просто сама запись в
    журнал делается с задержкой). На расчёт эквити это не влияет — она
    тоже обновляется только на закрытии сделки, как и во всём бэктесте
    (открытая позиция нигде в проекте не оценивается по рынку день в день).
    ЧАСТИЧНО закрыто отдельным утренним проходом — см.
    `live_pending_entries.py` (детектирует триггер по вчерашнему закрытию
    ДО открытия сегодняшней сессии, затем берёт реальную цену открытия
    отдельным запросом `quote` — триггер и цена входа НИКОГДА не смешаны в
    одном источнике). Пишет "provisional"-строку в журнал сразу же, в день
    фактического входа. Эта provisional-строка НЕ дозаполняется точным
    stop/size_usd/risk_usd, когда сделка позже закрывается штатным ночным
    прогоном (ключ уже "залогирован", `maybe_log` его пропускает) —
    сознательный компромисс, чтобы не городить логику редактирования уже
    записанных строк CSV; цена и дата входа при этом всегда точные.
  - Ребаланс вечного портфеля не логируется как отдельная строка журнала.
  - size_usd/risk_usd в журнале — оценка (1% от стоимости доли Комбо C на
    момент детектирования, не точный след исполнения ордера).
  - Не проверяются корреляционные потолки/мин.кэш/макс.25% на позицию —
    те же упрощения, что и в equity_curve.py (см. CLAUDE.md 3.4).

Запуск: python scripts/live_review.py (после live_fetch_incremental.py)
Выход: data/live/{journal.csv, equity.csv, state.json}, печатает сводку
для дальнейшей ретрансляции в чат.
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import backtest as bt
from system_ibs_rsi2 import add_extra_indicators
from combo_rsi2_tom import gen_rsi2_trades, gen_tom_trades, rsi2_open_position, tom_open_position, r_mult_of
from blend_combo_permanent import combo_c_curve, blend, START_CAPITAL
from benchmark_permanent_ofz import permanent_portfolio_daily

DATA_DIR = "data"
LIVE_DIR = "data/live"
WEIGHT = 0.4  # доля Комбо C в блендинге, боевое решение CLAUDE.md 5.17
RISK_PER_TRADE = 0.01


def load_state(default_start_date):
    """default_start_date используется ТОЛЬКО при первом запуске (state.json
    ещё не существует) — берём последнюю дату, реально доступную в данных
    (не wall-clock "сегодня"), иначе live_start_date мог бы оказаться позже
    as_of_date (например, если инкрементальный fetch ещё не подтянул
    сегодняшний бар) и blend()/permanent_portfolio_daily упадут на пустом
    диапазоне дат."""
    path = os.path.join(LIVE_DIR, "state.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"live_start_date": default_start_date.strftime("%Y-%m-%d"), "logged_events": [],
            "capital": START_CAPITAL, "peak_equity": START_CAPITAL}


def save_state(state):
    os.makedirs(LIVE_DIR, exist_ok=True)
    with open(os.path.join(LIVE_DIR, "state.json"), "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def append_journal(rows):
    if not rows:
        return
    path = os.path.join(LIVE_DIR, "journal.csv")
    df = pd.DataFrame(rows)
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


def write_open_positions(dfs, close_lookup, combo_capital_now, as_of_date):
    """Добавлено 25.08.2026 (см. CLAUDE.md §11, по запросу пользователя):
    открытые (ещё не закрытые) позиции RSI2/TOM нигде раньше не были видны —
    gen_rsi2_trades/gen_tom_trades возвращают только ЗАВЕРШЁННЫЕ сделки,
    поэтому реально открытая позиция (напр. QQQ RSI2 с 19.08.2026) была
    невидима до своего закрытия. Здесь — переоценка по рынку (mark-to-
    market) на текущий момент, отдельным файлом, не смешивается с journal.csv
    (тот остаётся строго "закрытые события", как и раньше)."""
    rows = []
    for symbol, d in dfs.items():
        for pos in (rsi2_open_position(d, symbol), tom_open_position(d, symbol)):
            if pos is None:
                continue
            current_price = close_lookup[symbol].asof(as_of_date)
            r_mult_now = r_mult_of(pos, current_price)
            risk_usd = RISK_PER_TRADE * combo_capital_now
            rows.append({"system": pos["system"], "symbol": pos["symbol"],
                         "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                         "entry_price": round(pos["entry_price"], 4), "stop": round(pos["stop"], 4),
                         "current_price": round(float(current_price), 4), "bars_held": pos["bars"],
                         "unrealized_r_mult": round(r_mult_now, 3),
                         "unrealized_usd": round(r_mult_now * risk_usd, 2)})
    path = os.path.join(LIVE_DIR, "open_positions.csv")
    pd.DataFrame(rows, columns=["system", "symbol", "entry_date", "entry_price", "stop", "current_price",
                                 "bars_held", "unrealized_r_mult", "unrealized_usd"]).to_csv(path, index=False)
    return rows


def append_equity(date_str, equity, combo_value, permanent_value):
    path = os.path.join(LIVE_DIR, "equity.csv")
    row = pd.DataFrame([{"date": date_str, "equity": round(equity, 2),
                          "combo_c_value": round(combo_value, 2),
                          "permanent_value": round(permanent_value, 2)}])
    if os.path.exists(path):
        existing = pd.read_csv(path)
        existing = existing[existing["date"] != date_str]  # идемпотентность при повторном запуске в тот же день
        row = pd.concat([existing, row], ignore_index=True)
    row.to_csv(path, index=False)


def main():
    os.makedirs(LIVE_DIR, exist_ok=True)

    dfs, close_lookup = {}, {}
    for symbol in sorted(bt.LIVE_INSTRUMENTS):
        raw = pd.read_csv(os.path.join(DATA_DIR, f"{symbol}.csv"), parse_dates=["date"]).sort_values("date")
        d = bt.add_indicators(raw)
        d2 = add_extra_indicators(d)
        d2 = d2.iloc[60:].reset_index(drop=True)
        dfs[symbol] = d2
        close_lookup[symbol] = raw.set_index("date")["close"].sort_index()

    # min, не max: форекс/крипта торгуют и в выходные, акции (в т.ч. SPY/TLT/
    # SHY/GLD, нужные вечному портфелю) — нет. Если взять max, as_of_date
    # может оказаться позже последней доступной даты акций, и
    # permanent_portfolio_daily получит пустой диапазон дат и упадёт.
    as_of_date = min(df["date"].max() for df in dfs.values())
    state = load_state(default_start_date=as_of_date)
    live_start_date = pd.Timestamp(state["live_start_date"])
    print(f"Данные актуальны на: {as_of_date.date()}  (старт live-счёта: {live_start_date.date()})")

    if as_of_date < live_start_date:
        print("Свежих данных ещё нет (as_of_date < live_start_date) — пропускаем пересчёт, "
              "повторить после live_fetch_incremental.py")
        return

    all_rsi2, all_tom = [], []
    for symbol, d in dfs.items():
        all_rsi2.extend(gen_rsi2_trades(d, symbol))
        all_tom.extend(gen_tom_trades(d, symbol))

    # --- детектируем новые события с прошлого запуска (по entry_date/exit_date) ---
    logged = set(state.get("logged_events", []))
    new_rows = []
    combo_capital_now = START_CAPITAL * WEIGHT  # приближение для risk_usd, см. докстринг

    def maybe_log(trade, system, action, date_field, price_field, reason):
        d = trade[date_field]
        if d < live_start_date or d > as_of_date:
            return
        key = f"{system}|{trade['symbol']}|{trade['entry_date'].date()}|{action}"
        if key in logged:
            return
        risk_usd = RISK_PER_TRADE * combo_capital_now
        new_rows.append({"date": d.strftime("%Y-%m-%d"), "system": system, "symbol": trade["symbol"],
                          "action": action, "price": round(trade[price_field], 4),
                          "size_usd": round(risk_usd / max(abs(trade["entry_price"] - trade["stop"]), 1e-9)
                                             * trade["entry_price"], 2) if action == "entry" else "",
                          "stop": round(trade["stop"], 4), "risk_usd": round(risk_usd, 2),
                          "risk_pct": RISK_PER_TRADE * 100, "reason": reason})
        logged.add(key)

    for t in all_rsi2:
        maybe_log(t, "RSI2", "entry", "entry_date", "entry_price", "rsi2<10+ema200")
        if "exit_date" in t:
            maybe_log(t, "RSI2", "exit", "exit_date", "exit_price", t.get("reason", "exit"))
    for t in all_tom:
        maybe_log(t, "TOM", "entry", "entry_date", "entry_price", "turn_of_month_offset5")
        if "exit_date" in t:
            maybe_log(t, "TOM", "exit", "exit_date", "exit_price", t.get("reason", "exit"))

    append_journal(new_rows)
    state["logged_events"] = sorted(logged)

    open_rows = write_open_positions(dfs, close_lookup, combo_capital_now, as_of_date)

    # --- сегодняшняя стоимость счёта (та же функция, что и весь бэктест) ---
    blended = blend(WEIGHT, live_start_date, as_of_date)
    equity_now = blended.iloc[-1]

    # combo_s/perm_s из combo_c_curve()/permanent_portfolio_daily() масштабо-
    # инвариантны (каждая — свои $1000 старта, см. blend_combo_permanent.py) —
    # для ОТЧЁТА нужен фактический $-разрез внутри блендинга (доли дрейфуют
    # в течение года до следующего ежегодного ребаланса), поэтому повторяем
    # ту же трекинг-логику весов, что и внутри blend(), только для чтения.
    combo_s = combo_c_curve(live_start_date, as_of_date)
    perm_s = permanent_portfolio_daily(live_start_date)
    all_days = pd.DatetimeIndex(sorted(set(combo_s.index) | set(perm_s.index)))
    combo_mult = (combo_s.reindex(all_days).ffill().bfill())
    combo_mult = combo_mult / combo_mult.iloc[0]
    perm_mult = (perm_s.reindex(all_days).ffill().bfill())
    perm_mult = perm_mult / perm_mult.iloc[0]
    years_track = pd.Series(all_days).dt.year
    combo_dollars, perm_dollars = START_CAPITAL * WEIGHT, START_CAPITAL * (1 - WEIGHT)
    combo_base, perm_base = combo_mult.iloc[0], perm_mult.iloc[0]
    cur_year = years_track.iloc[0]
    for i, d in enumerate(all_days):
        y = years_track.iloc[i]
        if y != cur_year:
            total = combo_dollars * (combo_mult.loc[d]/combo_base) + perm_dollars * (perm_mult.loc[d]/perm_base)
            combo_dollars, perm_dollars = total * WEIGHT, total * (1 - WEIGHT)
            combo_base, perm_base = combo_mult.loc[d], perm_mult.loc[d]
            cur_year = y
    combo_now = combo_dollars * (combo_mult.iloc[-1]/combo_base)
    perm_now = perm_dollars * (perm_mult.iloc[-1]/perm_base)
    peak = max(state.get("peak_equity", START_CAPITAL), equity_now)
    dd_pct = (equity_now - peak) / peak * 100

    append_equity(as_of_date.strftime("%Y-%m-%d"), equity_now, combo_now, perm_now)
    state["peak_equity"] = peak
    state["capital"] = equity_now
    save_state(state)

    print(f"\n=== Live-счёт на {as_of_date.date()} ===")
    print(f"Эквити: ${equity_now:,.2f} ({(equity_now/START_CAPITAL-1)*100:+.2f}% с начала live-трекинга)")
    print(f"  из них Комбо C (40%): ${combo_now:,.2f}, вечный портфель (60%): ${perm_now:,.2f}")
    print(f"Просадка от пика: {dd_pct:.2f}%")
    if new_rows:
        print(f"\nНовые события сегодня ({len(new_rows)}):")
        for r in new_rows:
            print(f"  {r['date']} {r['system']} {r['symbol']} {r['action']} @ {r['price']} ({r['reason']})")
    else:
        print("\nНовых сделок сегодня нет.")

    if open_rows:
        print(f"\nОткрытые позиции ({len(open_rows)}):")
        for r in open_rows:
            print(f"  {r['system']} {r['symbol']}: вход {r['entry_date']} @ {r['entry_price']}, "
                  f"стоп {r['stop']}, сейчас {r['current_price']} ({r['bars_held']} дн.), "
                  f"unrealized {r['unrealized_r_mult']:+.2f}R (${r['unrealized_usd']:+.2f})")
    else:
        print("\nОткрытых позиций нет.")


if __name__ == "__main__":
    main()
