"""Проверка независимости наблюдений в pm_earthquake_model_check (--allprices).

Обнаружена проблема (30.08.2026, см. диалог): без ценового фильтра 3-25¢
эффект (gap ≥ порог) выглядит статистически значимым на уровне ОТДЕЛЬНЫХ
бакетов ("exactly 0/1/2/.../N earthquakes"), но эти бакеты НЕ независимы —
несколько бакетов ссылаются на ОДНУ И ТУ ЖЕ неделю/магнитуду. Реальный
недельный счёт землетрясений один — если он равен, скажем, 3, то ставки
"No" ("не ровно 0", "не ровно 1", "не ровно 2", "не ровно 4"...) ВСЕ
выигрывают ОДНОВРЕМЕННО чисто механически, не по независимой удаче. Это
псевдо-репликация: наивный t-test на уровне бакетов завышает значимость.

Скрипт группирует наблюдения по независимому событию (магнитуда + текст
периода, извлечённый из вопроса) и считает t-test на суммарном P&L СОБЫТИЯ
(не бакета) — честная оценка, сколько на самом деле независимых испытаний.
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

from scipy import stats

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
YEARS = [2024, 2025, 2026]
GAP_THRESHOLDS = [0.0, 0.03, 0.05, 0.08]


def date_tail(question: str) -> str:
    """Извлекает текст периода из вопроса — используется как часть ключа
    независимого события вместе с магнитудой."""
    idx = question.lower().find("worldwide")
    if idx != -1:
        return question[idx + len("worldwide"):].strip(" ?")
    m = re.search(r"\b(by|in|before)\s+(.+?)\??$", question, re.I)
    return m.group(2) if m else question


def load_rows() -> list:
    rows = []
    for year in YEARS:
        path = RESULTS_DIR / f"pm_earthquake_model_check_{year}_allprices.csv"
        if not path.exists():
            continue
        for row in csv.DictReader(open(path)):
            row["gap"] = float(row["gap"])
            row["r_mult"] = float(row["r_mult"])
            rows.append(row)
    return rows


def main():
    rows = load_rows()
    print(f"Loaded {len(rows)} bucket-level observations (all-prices variant)")

    with open(RESULTS_DIR / "pm_earthquake_decluster_check.md", "w") as f:
        f.write("# Earthquake model — проверка независимости наблюдений (declustering)\n\n")
        f.write(
            "Наивный t-test по отдельным бакетам ('exactly N earthquakes') завышает "
            "значимость — несколько бакетов часто описывают ОДНУ неделю/магнитуду, их "
            "исходы механически коррелированы (см. докстринг). Группируем по "
            "(magnitude, период) и считаем P&L НА СОБЫТИЕ, не на бакет.\n\n"
        )
        f.write("| gap ≥ | n бакетов (наивно) | p (наивный) | n событий (декластерено) | p (честный) |\n")
        f.write("|---|---|---|---|---|\n")
        for gap_thr in GAP_THRESHOLDS:
            sub = [r for r in rows if r["gap"] >= gap_thr]
            n_buckets = len(sub)
            if n_buckets < 2:
                f.write(f"| {gap_thr} | {n_buckets} | — | — | — |\n")
                continue
            r_mults = [r["r_mult"] for r in sub]
            _, p_naive = stats.ttest_1samp(r_mults, 0.0)

            groups = defaultdict(list)
            for r in sub:
                key = (r["magnitude"], date_tail(r["question"]))
                groups[key].append(r["r_mult"])
            event_pnls = [sum(v) for v in groups.values()]
            n_events = len(event_pnls)
            if n_events >= 2:
                _, p_honest = stats.ttest_1samp(event_pnls, 0.0)
                p_honest_str = f"{p_honest:.4f}"
            else:
                p_honest_str = "—"
            f.write(f"| {gap_thr} | {n_buckets} | {p_naive:.4f} | {n_events} | {p_honest_str} |\n")

    print("Written: results/pm_earthquake_decluster_check.md")


if __name__ == "__main__":
    main()
