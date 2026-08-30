"""Объединяет годовые ретро-прогоны (`pm_retro_check.py --year YYYY`, уже
записанные в results/polymarket_retro_trades_{year}.csv) в один пул — больше
статистической мощности, чем в одном годе, и одна непрерывная хронологическая
эквити-кривая через границы лет (в отличие от `--year`, где каждый год
считается от $1000 заново).

НЕ делает новых запросов к API — читает уже сохранённые CSV. Год считается
доступным, если файл `results/polymarket_retro_trades_{year}.csv` существует
(нужно сначала прогнать `python pm_retro_check.py --year YYYY` за каждый год).
"""
from __future__ import annotations

import csv
from pathlib import Path

from scipy import stats

from pm_retro_check import (
    PRICE_THRESHOLDS, START_CAPITAL, STAKE_PCT, MAX_STAKE_PCT_OF_EQUITY,
    MIN_STAKE_USD, summarize, yearly_breakdown,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
YEARS = [2023, 2024, 2025, 2026]  # доступные годы с достаточным покрытием — см. CLAUDE.md


def load_pooled_trades() -> list:
    seen = set()
    trades = []
    for year in YEARS:
        path = RESULTS_DIR / f"polymarket_retro_trades_{year}.csv"
        if not path.exists():
            print(f"skip {year}: {path} not found (run pm_retro_check.py --year {year} first)")
            continue
        for row in csv.DictReader(open(path)):
            key = (row["category"], row["question"], row["cheap_side"], row["threshold"], row["entry_date"])
            if key in seen:
                continue
            seen.add(key)
            row["threshold"] = float(row["threshold"])
            row["favorite_r_mult"] = float(row["favorite_r_mult"])
            row["favorite_won"] = int(row["favorite_won"])
            row["loser_r_mult"] = float(row["loser_r_mult"])
            row["loser_won"] = int(row["loser_won"])
            trades.append(row)
    return trades


def build_equity_curve(trades: list, threshold: float) -> list:
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


def main():
    trades = load_pooled_trades()
    print(f"Pooled unique trade-observations across {YEARS}: {len(trades)}")

    RESULTS_DIR.mkdir(exist_ok=True)
    loser_summary = summarize(trades, "loser_won", "loser_r_mult")
    favorite_summary = summarize(trades, "favorite_won", "favorite_r_mult")

    boevoy_threshold = 0.08
    curve = build_equity_curve(trades, boevoy_threshold)
    with open(RESULTS_DIR / "polymarket_retro_equity_pooled.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "resolution_date", "category", "question", "stake_usd", "r_mult", "equity",
        ])
        writer.writeheader()
        writer.writerows(curve)
    yearly = yearly_breakdown(curve)

    with open(RESULTS_DIR / "polymarket_retro_check_pooled.md", "w") as f:
        f.write(f"# Polymarket — пул нескольких календарных лет ({YEARS[0]}-{YEARS[-1]})\n\n")
        f.write(
            f"Объединены отдельные прогоны `pm_retro_check.py --year YYYY` за {YEARS} "
            f"(дедупликация по market/side/threshold/entry_date). Непрерывная хронология "
            "по реальной дате разрешения (`resolution_date`), не пересечение случайного "
            "порядка в CSV.\n\n"
        )
        f.write("## Механика FAVORITE (боевая) — пул по годам\n\n")
        f.write("| Категория | Порог | n | win_rate | mean_r | t | p |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for row in favorite_summary:
            f.write(
                f"| {row['category']} | {row['threshold']} | {row['n']} | "
                f"{row['win_rate']} | {row['mean_r']} | {row['t_stat']} | {row['p_value']} |\n"
            )
        f.write(f"\n## Непрерывная эквити-кривая (порог {boevoy_threshold}, $1000 старт)\n\n")
        f.write("| Год | Сделок | Капитал на начало | Капитал на конец | Доходность за год |\n")
        f.write("|---|---|---|---|---|\n")
        for row in yearly:
            f.write(
                f"| {row['year']} | {row['n_trades']} | ${row['start_equity']} | "
                f"${row['end_equity']} | {row['return_pct']:+.2f}% |\n"
            )
        if curve:
            f.write(
                f"\nВесь период: {curve[0]['resolution_date']} → {curve[-1]['resolution_date']}, "
                f"итог ${curve[-1]['equity']} ({(curve[-1]['equity']/START_CAPITAL-1)*100:+.2f}%).\n"
            )

    print("Written: results/polymarket_retro_check_pooled.md, results/polymarket_retro_equity_pooled.csv")


if __name__ == "__main__":
    main()
