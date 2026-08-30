"""Объединяет годовые прогоны `pm_earthquake_model_check.py --year YYYY`
(уже записанные в results/pm_earthquake_model_check_{year}.csv) в один пул —
та же дисциплина, что `pm_retro_pool_years.py` для LOSER/FAVORITE-механик:
один год легко может оказаться удачным/неудачным случайно, пул по
нескольким независимым годам честнее.

НЕ делает новых запросов к API — читает уже сохранённые CSV.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
YEARS = [2023, 2024, 2025, 2026]
GAP_THRESHOLDS = [0.03, 0.05, 0.08]


def load_pooled_rows(suffix: str) -> list:
    rows = []
    for year in YEARS:
        path = RESULTS_DIR / f"pm_earthquake_model_check_{year}{suffix}.csv"
        if not path.exists():
            print(f"skip {year}: {path} not found")
            continue
        for row in csv.DictReader(open(path)):
            row["year"] = year
            row["gap"] = float(row["gap"])
            row["r_mult"] = float(row["r_mult"])
            row["won"] = int(row["won"])
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="", help='Например "_allprices" — объединить вариант без ценового фильтра')
    args = ap.parse_args()

    rows = load_pooled_rows(args.suffix)
    print(f"Pooled observations across {YEARS}: {len(rows)}")

    price_note = "все цены (без фильтра 3-25¢)" if args.suffix else "диапазон цены 3-25¢"
    out_name = f"pm_earthquake_model_check_pooled{args.suffix}.md"
    with open(RESULTS_DIR / out_name, "w") as f:
        f.write(f"# Earthquake model vs market — пул {YEARS[0]}-{YEARS[-1]} ({price_note})\n\n")
        f.write(f"Всего наблюдений (все года, {price_note}): {len(rows)}\n\n")

        f.write("## По годам, без фильтра по gap\n\n")
        f.write("| Год | n | win_rate | mean_r |\n|---|---|---|---|\n")
        for year in YEARS:
            sub = [r for r in rows if r["year"] == year]
            if not sub:
                f.write(f"| {year} | 0 | — | — |\n")
                continue
            n = len(sub)
            wr = sum(r["won"] for r in sub) / n
            mr = sum(r["r_mult"] for r in sub) / n
            f.write(f"| {year} | {n} | {wr:.4f} | {mr:+.4f} |\n")

        f.write("\n## Пул всех лет, по сетке gap-порогов\n\n")
        f.write("| gap ≥ | n | win_rate | mean_r | сумма r (\\$1/сделку) |\n")
        f.write("|---|---|---|---|---|\n")
        for gap_thr in [0.0] + GAP_THRESHOLDS:
            sub = [r for r in rows if r["gap"] >= gap_thr]
            ns = len(sub)
            if ns == 0:
                f.write(f"| {gap_thr} | 0 | — | — | — |\n")
                continue
            wr = sum(r["won"] for r in sub) / ns
            mr = sum(r["r_mult"] for r in sub) / ns
            total = sum(r["r_mult"] for r in sub)
            f.write(f"| {gap_thr} | {ns} | {wr:.4f} | {mr:+.4f} | \\${total:+.2f} |\n")

        # t-тест на пуле, если хватает наблюдений
        try:
            from scipy import stats
            f.write("\n## Статзначимость (t-test vs 0), пул всех лет\n\n")
            f.write("| gap ≥ | n | t | p |\n|---|---|---|---|\n")
            for gap_thr in [0.0] + GAP_THRESHOLDS:
                sub = [r["r_mult"] for r in rows if r["gap"] >= gap_thr]
                if len(sub) < 2:
                    f.write(f"| {gap_thr} | {len(sub)} | — | — |\n")
                    continue
                t, p = stats.ttest_1samp(sub, 0.0)
                f.write(f"| {gap_thr} | {len(sub)} | {t:.3f} | {p:.4g} |\n")
        except ImportError:
            pass

    print(f"Written: results/{out_name}")


if __name__ == "__main__":
    main()
