"""Виртуальный (paper) live-счёт стратегии Polymarket "FAVORITE-FADE".

Механика (см. CLAUDE.md, раздел про Polymarket, и scripts/pm_retro_check.py
для истории пивота): сканируем ОТКРЫТЫЕ рынки в шортлисте категорий
(pm_api.CATEGORY_TAGS). Если цена одной стороны бинарного Yes/No рынка ниже
ENTRY_THRESHOLD — рынок уже сигналит "эта сторона почти наверняка не
случится" — покупаем ПРОТИВОПОЛОЖНУЮ, дорогую сторону ("фаворита") по её
текущей котировке из Gamma API (реальная цена на момент скана, не
приближение 1-p, в отличие от ретро-чека на истории). Держим до разрешения
рынка (closed=true) — выход только по факту исхода, без стопа (стоп здесь
структурно не нужен: макс. убыток по конструкции бинарного рынка = размер
ставки, риск заранее ограничен самим инструментом).

Изолировано от остального проекта: свои данные (data/polymarket/), свой
журнал/эквити, отдельный workflow. НЕ трогает A/B/Комбо C/вечный портфель.

Идемпотентность повторных запусков в один день: `state.json::scanned_market_ids`
хранит id уже проверенных на этот скан открытых рынков (не даёт открыть
вторую позицию по тому же рынку повторным запуском в тот же день); закрытые
позиции резолвятся по market id независимо от даты скана.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from pm_api import CATEGORY_TAGS, fetch_markets_paginated

START_CAPITAL = 1000.0
ENTRY_THRESHOLD = 0.08          # боевой порог — середина диапазона 5-10%, см. retro-check
STAKE_PCT = 0.01                # ставка = 1% от текущего капитала (компаундинг), фикс. риск = ставке
MIN_STAKE_USD = 2.0
MAX_STAKE_PCT_OF_EQUITY = 0.05  # потолок на одну ставку (защита от узкой равой доли, аналог "не более 25%")
MAX_OPEN_POSITIONS = 50
MAX_CATEGORY_EXPOSURE_PCT = 0.25
MIN_CASH_RESERVE_PCT = 0.30     # выше, чем у A/B (0.10) — позиции здесь неликвидны до разрешения рынка
MIN_VOLUME_USD = 2000
OPEN_MARKETS_PER_CATEGORY = 100  # сколько открытых рынков на категорию сканировать (по объёму)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "polymarket"
JOURNAL_PATH = DATA_DIR / "journal.csv"
EQUITY_PATH = DATA_DIR / "equity.csv"
STATE_PATH = DATA_DIR / "state.json"
OPEN_POSITIONS_PATH = DATA_DIR / "open_positions.csv"

JOURNAL_FIELDS = [
    "date", "action", "market_id", "category", "question", "side",
    "price", "stake_usd", "payout_usd", "reason",
]


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {
        "equity": START_CAPITAL,
        "live_start_date": datetime.now(timezone.utc).date().isoformat(),
        "open_positions": [],   # list of dict: market_id, category, question, side, price, stake_usd, opened_date
        "scanned_market_ids": [],
    }


def save_state(state: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def append_journal(rows: list):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not JOURNAL_PATH.exists()
    with open(JOURNAL_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def append_equity(date: str, equity: float, open_stake: float, n_positions: int):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not EQUITY_PATH.exists()
    with open(EQUITY_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "equity", "open_stake_usd", "n_open_positions"])
        if is_new:
            writer.writeheader()
        writer.writerow({
            "date": date, "equity": round(equity, 2),
            "open_stake_usd": round(open_stake, 2), "n_open_positions": n_positions,
        })


def parse_binary_market(m: dict):
    try:
        outcomes = json.loads(m.get("outcomes", "[]"))
        outcome_prices = json.loads(m.get("outcomePrices", "[]"))
    except (json.JSONDecodeError, TypeError):
        return None
    if [o.lower() for o in outcomes] != ["yes", "no"] or len(outcome_prices) != 2:
        return None
    try:
        prices = [float(p) for p in outcome_prices]
    except (TypeError, ValueError):
        return None
    return outcomes, prices


def main():
    today = datetime.now(timezone.utc).date().isoformat()
    state = load_state()
    equity = state["equity"]

    # --- 1. Резолв уже открытых позиций: тянем их рынки по id напрямую ---
    from pm_api import _get, GAMMA_BASE  # локальный импорт, чтобы не грузить лишнее в остальных скриптах

    still_open = []
    journal_rows = []
    realized_pnl_today = 0.0
    for pos in state["open_positions"]:
        try:
            data = _get(f"{GAMMA_BASE}/markets/{pos['market_id']}", {})
        except RuntimeError as exc:
            print(f"  warn: couldn't refresh market {pos['market_id']}: {exc}")
            still_open.append(pos)
            continue
        parsed = parse_binary_market(data)
        if data.get("closed") and parsed:
            outcomes, prices = parsed
            side_idx = outcomes.index(pos["side"]) if pos["side"] in outcomes else 0
            won = prices[side_idx] >= 0.5
            payout = (pos["stake_usd"] / pos["price"]) if won else 0.0
            equity += payout - pos["stake_usd"]
            realized_pnl_today += payout - pos["stake_usd"]
            journal_rows.append({
                "date": today, "action": "exit", "market_id": pos["market_id"],
                "category": pos["category"], "question": pos["question"], "side": pos["side"],
                "price": pos["price"], "stake_usd": round(pos["stake_usd"], 2),
                "payout_usd": round(payout, 2),
                "reason": "resolved_win" if won else "resolved_loss",
            })
        else:
            still_open.append(pos)
    state["open_positions"] = still_open

    # --- 2. Скан открытых рынков на новые кандидаты ---
    open_ids = {p["market_id"] for p in state["open_positions"]}
    category_exposure = {}
    for p in state["open_positions"]:
        category_exposure[p["category"]] = category_exposure.get(p["category"], 0.0) + p["stake_usd"]

    candidates = []
    for category, tag_id in CATEGORY_TAGS.items():
        markets = fetch_markets_paginated(tag_id, closed=False, max_markets=OPEN_MARKETS_PER_CATEGORY)
        for m in markets:
            if m["id"] in open_ids:
                continue
            parsed = parse_binary_market(m)
            if not parsed:
                continue
            outcomes, prices = parsed
            volume = float(m.get("volumeNum") or 0)
            if volume < MIN_VOLUME_USD:
                continue
            for idx, price in enumerate(prices):
                if price < ENTRY_THRESHOLD:
                    fav_idx = 1 - idx
                    candidates.append({
                        "market_id": m["id"], "category": category,
                        "question": m.get("question", "")[:120],
                        "side": outcomes[fav_idx], "price": prices[fav_idx],
                        "cheap_side": outcomes[idx], "cheap_price": price,
                    })

    # Приоритет при конфликте бюджета НЕ определён по силе сигнала (ретро-чек не проверял,
    # даёт ли более экстремальная cheap_price лучший favorite_r_mult — сортировка по ней была
    # бы неподтверждённой догадкой). Как и в Комбо C проекта (§7.6/5.13 CLAUDE.md — там тоже
    # решили не сортировать без данных), берём стабильный, детерминированный порядок —
    # по market_id, не по set()/произвольному порядку скана (см. баг 5.15 про недетерминизм set()).
    candidates.sort(key=lambda c: c["market_id"])

    open_stake_total = sum(p["stake_usd"] for p in state["open_positions"])
    for cand in candidates:
        if len(state["open_positions"]) >= MAX_OPEN_POSITIONS:
            break
        stake = max(MIN_STAKE_USD, equity * STAKE_PCT)
        stake = min(stake, equity * MAX_STAKE_PCT_OF_EQUITY)
        cat_exposure = category_exposure.get(cand["category"], 0.0)
        if cat_exposure + stake > equity * MAX_CATEGORY_EXPOSURE_PCT:
            continue
        if open_stake_total + stake > equity * (1.0 - MIN_CASH_RESERVE_PCT):
            continue
        if cand["price"] <= 0 or cand["price"] >= 1:
            continue

        state["open_positions"].append({
            "market_id": cand["market_id"], "category": cand["category"],
            "question": cand["question"], "side": cand["side"],
            "price": cand["price"], "stake_usd": stake, "opened_date": today,
        })
        category_exposure[cand["category"]] = cat_exposure + stake
        open_stake_total += stake
        journal_rows.append({
            "date": today, "action": "entry", "market_id": cand["market_id"],
            "category": cand["category"], "question": cand["question"], "side": cand["side"],
            "price": cand["price"], "stake_usd": round(stake, 2), "payout_usd": "",
            "reason": f"favorite_fade cheap_side={cand['cheap_side']}@{cand['cheap_price']:.3f}",
        })

    # --- 3. Сохранение состояния/журнала/эквити ---
    state["equity"] = equity
    save_state(state)
    if journal_rows:
        append_journal(journal_rows)
    open_stake_total = sum(p["stake_usd"] for p in state["open_positions"])
    append_equity(today, equity, open_stake_total, len(state["open_positions"]))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OPEN_POSITIONS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "market_id", "category", "question", "side", "price", "stake_usd", "opened_date",
        ])
        writer.writeheader()
        writer.writerows(state["open_positions"])

    print(f"{today}: equity=${equity:.2f}, realized_pnl_today=${realized_pnl_today:.2f}, "
          f"open_positions={len(state['open_positions'])}, new_entries={sum(1 for r in journal_rows if r['action']=='entry')}, "
          f"exits={sum(1 for r in journal_rows if r['action']=='exit')}")


if __name__ == "__main__":
    main()
