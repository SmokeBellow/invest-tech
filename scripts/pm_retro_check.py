"""Лёгкая ретро-проверка стратегии Polymarket на уже РАЗРЕШЁННЫХ
(closed=true) рынках.

НЕ полноценный walk-forward бэктест в духе scripts/backtest.py (там есть
годовая история дневных баров и train/test-окна; здесь рынки Polymarket
живут месяцы/годы каждый, разрешённых рынков в нужных категориях меньше, и
формального train/test по времени пока не строим — только пул по всем
рынкам сразу). Задача — эмпирически проверить, недооценивает или
переоценивает рынок редкие "хвостовые" события (favorite-longshot bias),
прежде чем открывать виртуальный live-счёт.

**Первый прогон (см. CLAUDE.md, раздел про Polymarket) проверял исходную
гипотезу "покупать дешёвую (<10%) сторону, ждать редкого выигрыша с
большой выплатой" — она провалилась: mean_r от -1.00 до -0.76, p<0.001,
почти 0% побед. По итогам явного решения пользователя механика
ПЕРЕВЁРНУТА: покупаем ПРОТИВОПОЛОЖНУЮ, дорогую ("фаворит") сторону именно
тогда, когда другая сторона того же рынка дешёвая (<порога) — то есть
рынок такой ценой уже сигналит "почти наверняка НЕ произойдёт". Экономика
другая (не лотерейный билет, а сбор небольшой премии за почти
гарантированный исход) — этот скрипт считает ОБЕ интерпретации по одним и
тем же точкам входа: `loser_*` (исходная, отвергнутая гипотеза, оставлена
для сравнения/истории) и `favorite_*` (боевая механика).**

Логика на каждый (закрытый рынок × его "дешёвая" сторона):
- Берём дневную историю цены дешёвого outcome-токена (CLOB /prices-history).
- Ищем ПЕРВЫЙ день, когда цена этой стороны опустилась ниже порога (сетка
  {0.05, 0.08, 0.10} — узкая, как ADX/ATR-сетки проекта, не по результату).
- Требуем, чтобы это произошло НЕ в последние MIN_DAYS_BEFORE_END дней
  истории токена — иначе это не "вход с ожиданием", а точка, снятая почти
  в момент разрешения (нет времени, чтобы paper-сделка реально была открыта).
- `favorite_price = 1 - entry_price` (приближение: цены двух сторон
  бинарного Yes/No рынка на CLOB в норме почти точно суммируются в 1 за
  счёт арбитража; в живом сканере вместо этого приближения используется
  РЕАЛЬНАЯ котировка второй стороны из Gamma API, см. pm_live_scan.py) —
  favorite выигрывает ровно тогда, когда дешёвая сторона проигрывает
  (комплементарный бинарный исход).
- r_mult = (1/price - 1) при выигрыше, иначе -1 — та же семантика
  "выигрыш/проигрыш от ставки", что и r_mult остальных систем проекта,
  просто без стопа (стоп здесь не нужен: макс. убыток по конструкции
  бинарного рынка равен размеру ставки).

Отчёт — ПОЛНАЯ таблица по (категория × порог), без выбора "победителя"
скриптом (принцип проекта, см. CLAUDE.md 5.5) — решение о готовности к live
остаётся за человеком.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from scipy import stats

from pm_api import CATEGORY_TAGS, fetch_markets_paginated, fetch_price_history

MARKETS_PER_CATEGORY = 15          # top-N по объёму на категорию (контроль числа запросов к CLOB), режим "вся история"
MARKETS_PER_CATEGORY_YEAR = 40     # то же, но при фильтре по конкретному году (--year) — пул уже сам по себе уже
PRICE_THRESHOLDS = [0.05, 0.08, 0.10]
MIN_DAYS_BEFORE_END = 2            # порог должен пробиться не позже, чем за N дней до конца истории токена
MIN_VOLUME_USD = 2000              # отсеиваем совсем мёртвые/нерепрезентативные рынки
STAKE_PCT = 0.01                   # та же формула сайзинга, что в pm_live_scan.py
MAX_STAKE_PCT_OF_EQUITY = 0.05
MIN_STAKE_USD = 2.0
START_CAPITAL = 1000.0

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def fetch_markets_paginated_in_year(tag_id: int, year: int, max_markets: int) -> list:
    """Закрытые рынки, чей ПЛАНОВЫЙ endDate попадает в календарный год (Gamma API
    end_date_min/end_date_max) — прокси для "рынок был активен/должен был
    разрешиться в этом году", не гарантия фактической даты разрешения (рынок
    мог разрешиться и раньше endDate — см. resolution_date ниже, который
    берётся из реальной истории цены, не из endDate)."""
    out = []
    offset = 0
    page_size = 100
    while len(out) < max_markets:
        params = {
            "tag_id": tag_id, "closed": "true", "active": "false",
            "limit": page_size, "offset": offset, "order": "volume", "ascending": "false",
            "end_date_min": f"{year}-01-01", "end_date_max": f"{year}-12-31",
        }
        from pm_api import _get, GAMMA_BASE
        page = _get(f"{GAMMA_BASE}/markets", params)
        if not page:
            break
        out.extend(page)
        offset += page_size
        if len(page) < page_size:
            break
    return out[:max_markets]


def find_crossing(history: list, threshold: float, min_days_before_end: int):
    """Первый день, когда p < threshold, не позже последних min_days_before_end точек."""
    if len(history) <= min_days_before_end:
        return None
    usable = history[: len(history) - min_days_before_end]
    for point in usable:
        if point["p"] < threshold:
            return point
    return None


def collect_trades(year: int | None = None) -> list:
    trades = []
    for category, tag_id in CATEGORY_TAGS.items():
        if year is not None:
            markets = fetch_markets_paginated_in_year(tag_id, year, MARKETS_PER_CATEGORY_YEAR)
        else:
            markets = fetch_markets_paginated(tag_id, closed=True, max_markets=MARKETS_PER_CATEGORY)
        print(f"[{category}] closed markets fetched: {len(markets)}")
        for m in markets:
            try:
                outcomes = json.loads(m.get("outcomes", "[]"))
                outcome_prices = json.loads(m.get("outcomePrices", "[]"))
                token_ids = json.loads(m.get("clobTokenIds", "[]"))
            except (json.JSONDecodeError, TypeError):
                continue
            if [o.lower() for o in outcomes] != ["yes", "no"]:
                continue
            if len(token_ids) != 2 or len(outcome_prices) != 2:
                continue
            volume = float(m.get("volumeNum") or 0)
            if volume < MIN_VOLUME_USD:
                continue

            for side_idx, side_name in enumerate(outcomes):
                try:
                    final_price = float(outcome_prices[side_idx])
                except (TypeError, ValueError):
                    continue
                won = final_price >= 0.5  # разрешённый рынок: цена стороны -> 1 или 0
                try:
                    history = fetch_price_history(token_ids[side_idx])
                except RuntimeError as exc:
                    print(f"  skip token (history failed): {exc}")
                    continue
                if len(history) < MIN_DAYS_BEFORE_END + 3:
                    continue
                # Дата фактического разрешения ≈ последняя точка истории цены токена
                # (последний торговый день перед resolve) — НАДЁЖНЕЕ, чем endDate
                # рынка (тот часто остаётся плановой датой даже при досрочном разрешении).
                resolution_date = datetime.fromtimestamp(
                    history[-1]["t"], tz=timezone.utc
                ).date().isoformat()
                for threshold in PRICE_THRESHOLDS:
                    cross = find_crossing(history, threshold, MIN_DAYS_BEFORE_END)
                    if cross is None:
                        continue
                    entry_price = cross["p"]
                    entry_date = datetime.fromtimestamp(
                        cross["t"], tz=timezone.utc
                    ).date().isoformat()
                    loser_won = won
                    loser_r = (1.0 / entry_price - 1.0) if loser_won else -1.0
                    favorite_price = max(1.0 - entry_price, 0.01)
                    favorite_won = not won
                    favorite_r = (1.0 / favorite_price - 1.0) if favorite_won else -1.0
                    trades.append({
                        "category": category,
                        "question": m.get("question", "")[:80],
                        "cheap_side": side_name,
                        "threshold": threshold,
                        "entry_date": entry_date,
                        "resolution_date": resolution_date,
                        "loser_entry_price": entry_price,
                        "loser_won": int(loser_won),
                        "loser_r_mult": loser_r,
                        "favorite_entry_price": round(favorite_price, 4),
                        "favorite_won": int(favorite_won),
                        "favorite_r_mult": round(favorite_r, 4),
                        "volume": volume,
                    })
    return trades


def build_equity_curve(trades: list, threshold: float) -> list:
    """Хронологическая эквити-кривая боевой (favorite) механики на одном
    пороге — сделки сортируются по РЕАЛЬНОЙ дате разрешения (resolution_date),
    та же формула сайзинга (1% капитала, потолок 5%), что в pm_live_scan.py.
    Позиция открывается на entry_date, но капитал в симуляции меняется на
    resolution_date (упрощение: параллельные по времени сделки не увеличивают
    факт. занятый капитал — тот же пробел, что и в equity_curve.py A/B, §3.4)."""
    chosen = [t for t in trades if t["threshold"] == threshold]
    chosen.sort(key=lambda t: t["resolution_date"])
    equity = START_CAPITAL
    curve = []
    for t in chosen:
        stake = max(MIN_STAKE_USD, min(equity * STAKE_PCT, equity * MAX_STAKE_PCT_OF_EQUITY))
        equity += stake * t["favorite_r_mult"]
        curve.append({
            "resolution_date": t["resolution_date"], "category": t["category"],
            "question": t["question"], "stake_usd": round(stake, 2),
            "r_mult": t["favorite_r_mult"], "equity": round(equity, 2),
        })
    return curve


def yearly_breakdown(curve: list) -> list:
    if not curve:
        return []
    rows = []
    by_year = {}
    for point in curve:
        by_year.setdefault(point["resolution_date"][:4], []).append(point)
    prev_equity = START_CAPITAL
    for year in sorted(by_year):
        points = by_year[year]
        start_eq = prev_equity
        end_eq = points[-1]["equity"]
        rows.append({
            "year": year, "n_trades": len(points),
            "start_equity": round(start_eq, 2), "end_equity": round(end_eq, 2),
            "return_pct": round((end_eq / start_eq - 1) * 100, 2),
        })
        prev_equity = end_eq
    return rows


def _summary_row(category, threshold, items, won_key, r_key):
    n = len(items)
    r = [x[r_key] for x in items]
    win_rate = sum(x[won_key] for x in items) / n
    mean_r = sum(r) / n
    if n >= 2:
        t_stat, p_val = stats.ttest_1samp(r, 0.0)
    else:
        t_stat, p_val = float("nan"), float("nan")
    return {
        "category": category, "threshold": threshold, "n": n,
        "win_rate": round(win_rate, 4), "mean_r": round(mean_r, 4),
        "t_stat": round(t_stat, 3) if n >= 2 else "", "p_value": round(p_val, 4) if n >= 2 else "",
    }


def summarize(trades: list, won_key: str, r_key: str) -> list:
    rows = []
    groups = {}
    for t in trades:
        key = (t["category"], t["threshold"])
        groups.setdefault(key, []).append(t)
    for (category, threshold), items in sorted(groups.items()):
        rows.append(_summary_row(category, threshold, items, won_key, r_key))

    pooled_by_threshold = {}
    for t in trades:
        pooled_by_threshold.setdefault(t["threshold"], []).append(t)
    pooled_rows = [
        _summary_row("ALL (pooled)", threshold, items, won_key, r_key)
        for threshold, items in sorted(pooled_by_threshold.items())
    ]
    return rows + pooled_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=None,
                         help="Ограничить выборку закрытыми рынками с плановым endDate в этом "
                              "календарном году (Gamma API end_date_min/max) — для оценки "
                              "доходности за конкретный период, а не 'весь пул без хронологии'.")
    args = parser.parse_args()

    trades = collect_trades(year=args.year)
    print(f"Total trade-observations collected: {len(trades)}")

    suffix = f"_{args.year}" if args.year else ""
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"polymarket_retro_trades{suffix}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "category", "question", "cheap_side", "threshold", "entry_date", "resolution_date",
            "loser_entry_price", "loser_won", "loser_r_mult",
            "favorite_entry_price", "favorite_won", "favorite_r_mult", "volume",
        ])
        writer.writeheader()
        writer.writerows(trades)

    loser_summary = summarize(trades, "loser_won", "loser_r_mult")
    favorite_summary = summarize(trades, "favorite_won", "favorite_r_mult")
    with open(RESULTS_DIR / f"polymarket_retro_check{suffix}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mechanic", "category", "threshold", "n", "win_rate", "mean_r", "t_stat", "p_value",
        ])
        writer.writeheader()
        for row in loser_summary:
            writer.writerow({"mechanic": "loser (отвергнута)", **row})
        for row in favorite_summary:
            writer.writerow({"mechanic": "favorite (боевая)", **row})

    # Хронологическая эквити-кривая боевой механики на боевом пороге 0.08
    boevoy_threshold = 0.08
    curve = build_equity_curve(trades, boevoy_threshold)
    with open(RESULTS_DIR / f"polymarket_retro_equity{suffix}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "resolution_date", "category", "question", "stake_usd", "r_mult", "equity",
        ])
        writer.writeheader()
        writer.writerows(curve)
    yearly = yearly_breakdown(curve)

    def _write_table(f, title, summary):
        f.write(f"## {title}\n\n")
        f.write("| Категория | Порог | n | win_rate | mean_r | t | p |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for row in summary:
            f.write(
                f"| {row['category']} | {row['threshold']} | {row['n']} | "
                f"{row['win_rate']} | {row['mean_r']} | {row['t_stat']} | {row['p_value']} |\n"
            )
        f.write("\n")

    with open(RESULTS_DIR / f"polymarket_retro_check{suffix}.md", "w") as f:
        f.write("# Polymarket — ретро-проверка калибровки (закрытые рынки)\n\n")
        if args.year:
            f.write(
                f"**Фильтр по году: {args.year}** (плановый endDate рынка в этом году, "
                "Gamma API end_date_min/max — фактическая дата разрешения может быть раньше, "
                "см. `resolution_date` в CSV сделок и эквити-кривую ниже).\n\n"
            )
        f.write(
            f"Категорий: {len(CATEGORY_TAGS)}, до "
            f"{MARKETS_PER_CATEGORY_YEAR if args.year else MARKETS_PER_CATEGORY} закрытых рынков на "
            f"категорию по объёму, пороги входа {PRICE_THRESHOLDS}, "
            f"мин. объём рынка ${MIN_VOLUME_USD}. Полные таблицы по всем "
            "категориям и порогам — без выбора \"победителя\" скриптом.\n\n"
        )
        _write_table(
            f,
            "Механика LOSER (исходная гипотеза, отвергнута) — покупка дешёвой стороны",
            loser_summary,
        )
        _write_table(
            f,
            "Механика FAVORITE (боевая) — покупка дорогой стороны, когда противоположная дешевле порога",
            favorite_summary,
        )
        f.write(
            f"## Хронологическая эквити-кривая (боевой порог {boevoy_threshold}, "
            f"сортировка по реальной дате разрешения {START_CAPITAL:.0f}$ старт)\n\n"
        )
        if yearly:
            f.write("| Год | Сделок | Капитал на начало | Капитал на конец | Доходность за год |\n")
            f.write("|---|---|---|---|---|\n")
            for row in yearly:
                f.write(
                    f"| {row['year']} | {row['n_trades']} | ${row['start_equity']} | "
                    f"${row['end_equity']} | {row['return_pct']:+.2f}% |\n"
                )
        else:
            f.write("Нет сделок на этом пороге в выбранной выборке.\n")
        f.write("\n")

    print(f"Written: results/polymarket_retro_check{suffix}.{{csv,md}}, "
          f"results/polymarket_retro_trades{suffix}.csv, results/polymarket_retro_equity{suffix}.csv")


if __name__ == "__main__":
    main()
