"""Пилот: сравнение цены Polymarket с независимой статистической моделью
для рынков вида "Will there be N earthquakes of M+ magnitude worldwide?"
(категории earthquake/natural_disaster).

Идея пользователя (29-30.08.2026, см. диалог): не покупать любую дешёвую
сторону вслепую (это уже проверено и отвергнуто — LOSER-механика, CLAUDE.md
14.2), а входить ТОЛЬКО там, где независимый источник (здесь — статистическая
модель на исторической частоте землетрясений USGS) даёт СУЩЕСТВЕННО более
высокую вероятность, чем цена рынка.

Модель: глобальная частота землетрясений магнитудой ≥M — процесс Пуассона
со ставкой λ, оценённой по 20-летней истории USGS (2005-2025, полный
глобальный каталог, надёжен для M≥5.5). Для рынка с окном W дней:
λ_window = λ_per_day(M) × W. Дальше по типу вопроса:
- "ровно N" → P(X=N) = Poisson.pmf(N, λ_window)
- "более N" → P(X>N) = Poisson.sf(N, λ_window)
- "не более N" / "≤N" → P(X≤N) = Poisson.cdf(N, λ_window)
- "хотя бы 1" (Megaquake) → P(X≥1) = 1-Poisson.cdf(0, λ_window)

ВАЖНОЕ ОГРАНИЧЕНИЕ МОДЕЛИ: реальная сейсмичность НЕ строго пуассоновский
процесс — афтершоковые последовательности (закон Омори) нарушают
независимость событий, особенно для более низких порогов магнитуды и
коротких окон. Пуассон — стандартная бейзлайн-модель в сейсмологии
(используется USGS для долгосрочных прогнозов), не точная физическая
модель — расхождение с рынком может отражать как реальную неэффективность
цены, так и слепое пятно самой модели (например, если открытие рынка
пришлось на период сейсмического роя/aftershock-кластера).

Парсинг вопроса и описания — по регулярным выражениям под конкретный,
наблюдаемый на практике формат вопросов Polymarket (см. QUESTION_PATTERNS).
Рынки, не подошедшие ни под один паттерн (например, "highest-magnitude
earthquake in 2026 be X-Y" — распределение МАКСИМУМА, не счёта, нужна
другая модель) — пропускаются, не в скоупе этого пилота.
"""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import requests
from scipy.stats import poisson

from pm_api import _get, GAMMA_BASE, fetch_price_history

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
TAG_IDS = {"earthquake": 100184, "natural_disaster": 101998}

USGS_COUNT_URL = "https://earthquake.usgs.gov/fdsnws/event/1/count"
RATE_LOOKBACK_START = "2005-01-01"
RATE_LOOKBACK_END = "2025-01-01"
RATE_LOOKBACK_DAYS = (datetime(2025, 1, 1) - datetime(2005, 1, 1)).days

MIN_PRICE = 0.03
MAX_PRICE = 0.25
GAP_THRESHOLDS = [0.03, 0.05, 0.08]  # узкая сетка, не по результату — тот же принцип, что ADX/ATR-сетки проекта

MAG_PHRASE = r'of (?:magnitude\s+)?([\d.]+) or (?:higher|above)(?:\s+magnitude)? worldwide'
QUESTION_PATTERNS = [
    (re.compile(r'Will there be more than (\d+) earthquakes? ' + MAG_PHRASE, re.I), "more_than"),
    (re.compile(r'Will there be (\d+) or fewer earthquakes? ' + MAG_PHRASE, re.I), "at_most"),
    (re.compile(r'Will there be (?:≤|at most )(\d+) earthquakes? ' + MAG_PHRASE, re.I), "at_most"),
    (re.compile(r'Will there be (?:exactly\s+)?(\d+) earthquakes? ' + MAG_PHRASE, re.I), "exactly"),
]
# ", 11:59 PM ET, and June 21, ..." — запятая перед "and" опциональна (формат менялся в течение года)
DESC_WINDOW_PATTERN = re.compile(
    r'between ([A-Za-z]+ \d{1,2}, \d{4}),\s*[\d:]+\s*(?:AM|PM)?\s*ET,?\s*and\s*([A-Za-z]+ \d{1,2}, \d{4})'
)

# Более старый (2023-2025) формат вопросов — "хотя бы 1 землетрясение M+ по/в/до ДАТА",
# без диапазона в тексте — окно берём из полей самого рынка (startDate/createdAt -> endDate),
# не из description (там формат дат ещё менее единообразен). Глобальные версии ТОЛЬКО —
# если после "in"/"by"/"before" не название месяца (а, например, регион вроде "Mediterranean"/
# "LA"), считаем рынок региональным и ПРОПУСКАЕМ (модель считает только глобальную ставку).
_MONTHS = ("January|February|March|April|May|June|July|August|September|"
           "October|November|December")
GLOBAL_AT_LEAST_PATTERNS = [
    (re.compile(rf'^Megaquake (?:by|in|before) ({_MONTHS})', re.I), None),  # magnitude фиксирована = 8.0
    (re.compile(rf'^(?:Another )?[Ee]arthquake ([\d.]+)\+? or (?:above|higher) (?:by|in|before) ({_MONTHS})', re.I), "group1"),
]

_rate_cache: dict = {}


def get_daily_rate(magnitude: float) -> float:
    if magnitude in _rate_cache:
        return _rate_cache[magnitude]
    r = requests.get(USGS_COUNT_URL, params={
        "starttime": RATE_LOOKBACK_START, "endtime": RATE_LOOKBACK_END, "minmagnitude": magnitude,
    }, timeout=30)
    n = int(r.text.strip())
    rate = n / RATE_LOOKBACK_DAYS
    _rate_cache[magnitude] = rate
    return rate


def parse_market(m: dict):
    q = (m.get("question") or "").strip()
    desc = m.get("description") or ""

    for pat, kind in QUESTION_PATTERNS:
        mm = pat.search(q)
        if mm:
            n = int(mm.group(1))
            mag = float(mm.group(2))
            dm = DESC_WINDOW_PATTERN.search(desc)
            if not dm:
                return None
            try:
                start = datetime.strptime(dm.group(1), "%B %d, %Y")
                end = datetime.strptime(dm.group(2), "%B %d, %Y")
            except ValueError:
                return None
            window_days = (end - start).days + 1
            return {"kind": kind, "n": n, "magnitude": mag, "window_days": window_days}

    for pat, mag_group in GLOBAL_AT_LEAST_PATTERNS:
        mm = pat.search(q)
        if not mm:
            continue
        magnitude = 8.0 if mag_group is None else float(mm.group(1))
        try:
            start_raw = m.get("startDate") or m.get("createdAt")
            start = datetime.fromisoformat(start_raw.replace("Z", "+00:00")).replace(tzinfo=None)
            end = datetime.fromisoformat(m["endDate"].replace("Z", "+00:00")).replace(tzinfo=None)
        except (ValueError, KeyError, AttributeError):
            return None
        window_days = max((end - start).total_seconds() / 86400, 0.1)
        return {"kind": "at_least", "n": 1, "magnitude": magnitude, "window_days": window_days}

    return None


def model_prob_yes(parsed: dict) -> float:
    lam = get_daily_rate(parsed["magnitude"]) * parsed["window_days"]
    n = parsed["n"]
    if parsed["kind"] == "exactly":
        return float(poisson.pmf(n, lam))
    if parsed["kind"] == "more_than":
        return float(poisson.sf(n, lam))
    if parsed["kind"] == "at_most":
        return float(poisson.cdf(n, lam))
    if parsed["kind"] == "at_least":
        return float(1 - poisson.cdf(n - 1, lam))
    raise ValueError(parsed["kind"])


def fetch_closed_earthquake_markets(year: int, max_per_tag: int = 300) -> list:
    seen_ids = set()
    markets = []
    for tag_name, tag_id in TAG_IDS.items():
        offset = 0
        page_size = 100
        fetched = 0
        while fetched < max_per_tag:
            params = {
                "tag_id": tag_id, "closed": "true", "active": "false",
                "limit": page_size, "offset": offset, "order": "volume", "ascending": "false",
                "end_date_min": f"{year}-01-01", "end_date_max": f"{year}-12-31",
            }
            page = _get(f"{GAMMA_BASE}/markets", params)
            if not page:
                break
            for m in page:
                if m["id"] not in seen_ids:
                    seen_ids.add(m["id"])
                    markets.append(m)
            fetched += len(page)
            offset += page_size
            if len(page) < page_size:
                break
        print(f"[{tag_name}] fetched (dedup total so far {len(markets)})")
    return markets


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--max-per-tag", type=int, default=300)
    args = ap.parse_args()

    markets = fetch_closed_earthquake_markets(args.year, args.max_per_tag)
    print(f"Total unique closed earthquake/natural_disaster markets ({args.year}): {len(markets)}")

    rows = []
    n_parsed = 0
    n_skipped = 0
    for m in markets:
        parsed = parse_market(m)
        if parsed is None:
            n_skipped += 1
            continue
        n_parsed += 1
        try:
            outcomes = json.loads(m.get("outcomes", "[]"))
            outcome_prices = json.loads(m.get("outcomePrices", "[]"))
            token_ids = json.loads(m.get("clobTokenIds", "[]"))
        except (json.JSONDecodeError, TypeError):
            continue
        if [o.lower() for o in outcomes] != ["yes", "no"] or len(token_ids) != 2:
            continue

        try:
            p_model_yes = model_prob_yes(parsed)
        except Exception as exc:
            print(f"  model failed for {m.get('question')}: {exc}")
            continue
        p_model = {"Yes": p_model_yes, "No": 1 - p_model_yes}
        won = {"Yes": float(outcome_prices[0]) >= 0.5, "No": float(outcome_prices[1]) >= 0.5}

        for side_idx, side_name in enumerate(outcomes):
            try:
                history = fetch_price_history(token_ids[side_idx])
            except RuntimeError:
                continue
            if not history:
                continue
            entry_price = history[0]["p"]  # первая доступная котировка — рынок живёт считаные дни
            if not (MIN_PRICE <= entry_price <= MAX_PRICE):
                continue
            gap = p_model[side_name] - entry_price
            r_mult = (1.0 / entry_price - 1.0) if won[side_name] else -1.0
            rows.append({
                "question": m.get("question", "")[:100],
                "side": side_name, "magnitude": parsed["magnitude"], "kind": parsed["kind"],
                "n": parsed["n"], "window_days": round(parsed["window_days"], 2),
                "entry_price": round(entry_price, 4), "model_prob": round(p_model[side_name], 4),
                "gap": round(gap, 4), "won": int(won[side_name]), "r_mult": round(r_mult, 4),
                "volume": float(m.get("volumeNum") or 0),
            })

    print(f"Markets parsed by regex: {n_parsed}, skipped (unknown format): {n_skipped}")
    print(f"Price observations in [{MIN_PRICE},{MAX_PRICE}] range: {len(rows)}")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"pm_earthquake_model_check_{args.year}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "question", "side", "magnitude", "kind", "n", "window_days",
            "entry_price", "model_prob", "gap", "won", "r_mult", "volume",
        ])
        writer.writeheader()
        writer.writerows(rows)

    # Отчёт по сетке порогов gap — сколько сделок прошло бы фильтр "модель существенно выше рынка"
    with open(RESULTS_DIR / f"pm_earthquake_model_check_{args.year}.md", "w") as f:
        f.write(f"# Earthquake model vs market — {args.year}\n\n")
        f.write(
            f"Разобрано по регулярным выражениям: {n_parsed} рынков, пропущено (не подошёл формат "
            f"вопроса): {n_skipped}. Всего цен в диапазоне [{MIN_PRICE},{MAX_PRICE}]: {len(rows)}.\n\n"
        )
        f.write("## Без фильтра по gap (все наблюдения, обе стороны любого рынка)\n\n")
        n = len(rows)
        if n:
            win_rate = sum(r["won"] for r in rows) / n
            mean_r = sum(r["r_mult"] for r in rows) / n
            f.write(f"n={n}, win_rate={win_rate:.4f}, mean_r={mean_r:+.4f}\n\n")
        f.write("## С фильтром gap ≥ порог (модель существенно выше цены рынка)\n\n")
        f.write("| gap ≥ | n | win_rate | mean_r | сумма r (\\$1/сделку) |\n")
        f.write("|---|---|---|---|---|\n")
        for gap_thr in GAP_THRESHOLDS:
            sub = [r for r in rows if r["gap"] >= gap_thr]
            ns = len(sub)
            if ns == 0:
                f.write(f"| {gap_thr} | 0 | — | — | — |\n")
                continue
            wr = sum(r["won"] for r in sub) / ns
            mr = sum(r["r_mult"] for r in sub) / ns
            total = sum(r["r_mult"] for r in sub)
            f.write(f"| {gap_thr} | {ns} | {wr:.4f} | {mr:+.4f} | \\${total:+.2f} |\n")

    print(f"Written: results/pm_earthquake_model_check_{args.year}.{{csv,md}}")


if __name__ == "__main__":
    main()
