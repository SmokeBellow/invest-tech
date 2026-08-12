"""
equity_curve.py — событийная симуляция долларовой эквити портфелей A и B
поверх уже собранных данных (data/*.csv) и логики сигналов из backtest.py.

Считается по ВСЕЙ доступной истории хронологически (не train/test —
эквити по своей природе последовательна), с параметрами, которые
согласованы с пользователем как "боевые", а не значениями из сетки
калибровки backtest.py:
  - Система A: ADX>=25 (документированный порог, раздел 3.1 CLAUDE.md).
  - Система B: Bollinger Bands (20, 2sigma) + ATR-стоп x2.0 — заменила
    исходную RSI/CCI-версию после того, как узкий фиксированный стоп
    (0.3%) и сам вход по RSI/CCI были признаны нерабочими (см. обсуждение
    в истории проекта). ATR-множитель 2.0 выбран пользователем по
    результатам сравнения на пуле (см. results/backtest_report_pooled.csv).

Портфельная симуляция (событийная, по датам входа/выхода):
- Старт $1000 на каждый портфель, независимо.
- Риск на сделку 1% ТЕКУЩЕГО капитала портфеля на момент входа именно
  этой сделки (компаундинг — размер уже открытой позиции не
  пересчитывается задним числом).
- Потолок суммарного открытого риска 4%, макс. 5 одновременных позиций.
  Сделка, не влезающая в бюджет на день своего входа, целиком
  пропускается (раздел 3.5 CLAUDE.md) — не дробится под остаток.

УПРОЩЕНИЯ (не путать с полной боевой системой):
  - Корреляционные потолки (SPY+QQQ+EEM<=$400) не смоделированы — нужен
    трекинг $-стоимости позиций, не только риска.
  - Правило мин. 10% кэш / макс. 25% на позицию не смоделировано.
  - Комиссии и проскальзывание не учтены.

Запуск: python scripts/equity_curve.py (после backtest.py, использует
те же data/*.csv)
Выход: results/equity_curve.csv, results/equity_curve_summary.md,
       results/equity_curve.html (самодостаточный график, открывается
       в браузере без сети)
"""

import glob
import heapq
import json
import os

import numpy as np
import pandas as pd

from backtest import add_indicators, LIVE_INSTRUMENTS

DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

START_CAPITAL = 1000.0
RISK_PER_TRADE = 0.01
MAX_OPEN_RISK = 0.04
MAX_POSITIONS = 5

SYSTEM_A_ADX_THRESHOLD = 25
SYSTEM_B_ATR_MULT = 2.0


def trades_system_a(data, symbol, adx_threshold=SYSTEM_A_ADX_THRESHOLD):
    trades = []
    position = None
    d = data.reset_index(drop=True)
    for i in range(1, len(d)):
        row, prev = d.iloc[i], d.iloc[i - 1]
        cond_now = (row.ema20 > row.ema50 > row.ema200) and (row.adx14 >= adx_threshold) and (row.macd > 0)
        cond_prev = (prev.ema20 > prev.ema50 > prev.ema200) and (prev.adx14 >= adx_threshold) and (prev.macd > 0)
        if position is None and cond_now and not cond_prev:
            if i + 1 < len(d):
                init_stop = row.ema50 * 0.99
                position = {"entry_date": d.iloc[i + 1].date, "entry_price": d.iloc[i + 1].open,
                            "stop": init_stop, "orig_stop": init_stop}
        elif position is not None:
            rowj = d.iloc[i]
            position["stop"] = max(position["stop"], rowj.ema50 * 0.99)
            exit_price, reason = None, None
            if rowj.low <= position["stop"]:
                exit_price = rowj.open if rowj.open < position["stop"] else position["stop"]
                reason = "stop"
            elif rowj.close < rowj.ema50:
                exit_price, reason = rowj.close, "ema50_close_below"
            if reason:
                r_mult = (exit_price - position["entry_price"]) / (position["entry_price"] - position["orig_stop"])
                trades.append({"symbol": symbol, "entry_date": position["entry_date"],
                               "exit_date": rowj.date, "r_mult": r_mult, "reason": reason})
                position = None
    return trades


def trades_system_b_bollinger(data, symbol, atr_mult=SYSTEM_B_ATR_MULT):
    trades = []
    position = None
    d = data.reset_index(drop=True)
    for i in range(1, len(d)):
        row, prev = d.iloc[i], d.iloc[i - 1]
        cond_now = (row.close > row.ema200) and (row.adx14 < 20) and (row.close < row.bb_lower)
        cond_prev = (prev.close > prev.ema200) and (prev.adx14 < 20) and (prev.close < prev.bb_lower)
        if position is None and cond_now and not cond_prev:
            if i + 1 < len(d):
                entry_price = d.iloc[i + 1].open
                position = {"entry_date": d.iloc[i + 1].date, "entry_price": entry_price,
                            "stop": entry_price - atr_mult * row.atr14, "bars": 0}
        elif position is not None:
            rowj = d.iloc[i]
            position["bars"] += 1
            exit_price, reason = None, None
            if rowj.low <= position["stop"]:
                exit_price = rowj.open if rowj.open < position["stop"] else position["stop"]
                reason = "stop"
            elif rowj.close >= rowj.bb_mid:
                exit_price, reason = rowj.close, "bb_mid_touch"
            elif position["bars"] >= 10:
                exit_price, reason = rowj.close, "time_stop"
            if reason:
                r_mult = (exit_price - position["entry_price"]) / (position["entry_price"] - position["stop"])
                trades.append({"symbol": symbol, "entry_date": position["entry_date"],
                               "exit_date": rowj.date, "r_mult": r_mult, "reason": reason})
                position = None
    return trades


def simulate_portfolio(trades, start_capital=START_CAPITAL):
    """Событийная симуляция общего капитала портфеля по всем инструментам сразу."""
    trades = sorted(trades, key=lambda t: t["entry_date"])
    equity = start_capital
    open_heap = []  # (exit_date, risked_dollars, r_mult)
    open_risk = 0.0
    curve = [(trades[0]["entry_date"] if trades else None, equity)]
    skipped = 0
    taken = 0

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
        risked_dollars = RISK_PER_TRADE * equity
        heapq.heappush(open_heap, (t["exit_date"], risked_dollars, t["r_mult"]))
        open_risk += RISK_PER_TRADE
        taken += 1

    while open_heap:
        exit_date, risked, r_mult = heapq.heappop(open_heap)
        equity += r_mult * risked
        curve.append((exit_date, equity))

    return curve, equity, skipped, taken


def max_drawdown(curve):
    peak = curve[0][1]
    peak_date = curve[0][0]
    max_dd_pct, max_dd_usd = 0.0, 0.0
    trough_date, worst_peak_date = curve[0][0], curve[0][0]
    for date, eq in curve:
        if eq > peak:
            peak, peak_date = eq, date
        dd_pct = (eq - peak) / peak
        if dd_pct < max_dd_pct:
            max_dd_pct, max_dd_usd = dd_pct, eq - peak
            trough_date, worst_peak_date = date, peak_date
    return max_dd_pct, max_dd_usd, worst_peak_date, trough_date


def build_html(results):
    data_json = json.dumps({
        k: {
            "curve": [(str(d.date()) if d is not None else None, round(e, 2)) for d, e in v["curve"]],
            "final_equity": round(v["final_equity"], 2),
            "trades_taken": v["trades_taken"],
            "trades_skipped_budget": v["trades_skipped"],
            "max_dd_pct": round(v["max_dd_pct"] * 100, 2),
            "max_dd_usd": round(v["max_dd_usd"], 2),
            "peak_date": str(v["peak_date"].date()) if v["peak_date"] is not None else None,
            "trough_date": str(v["trough_date"].date()) if v["trough_date"] is not None else None,
        } for k, v in results.items()
    }, ensure_ascii=False)

    return HTML_TEMPLATE.replace("__DATA__", data_json)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Кривые эквити A/B</title>
<style>
  :root {
    --bg: #eef0f1; --surface: #ffffff; --surface-2: #e4e8e9;
    --text: #16212b; --text-muted: #57636d; --text-faint: #898781;
    --border: #d7dbdc; --grid: #dde1e2;
    --series-a: #2a78d6; --series-a-wash: rgba(42, 120, 214, 0.10);
    --series-b: #eb6834; --series-b-wash: rgba(235, 104, 52, 0.10);
    --good: #0ca30c; --critical: #d03b3b;
    --shadow: 0 1px 2px rgba(22, 33, 43, 0.06), 0 4px 14px rgba(22, 33, 43, 0.05);
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #12181c; --surface: #182226; --surface-2: #1f2a2f;
      --text: #e7ecee; --text-muted: #8fa0a8; --text-faint: #6c7a80;
      --border: #2a363c; --grid: #263136;
      --series-a: #3987e5; --series-a-wash: rgba(57, 135, 229, 0.14);
      --series-b: #d95926; --series-b-wash: rgba(217, 89, 38, 0.14);
      --good: #0ca30c; --critical: #e66767;
      --shadow: 0 1px 2px rgba(0,0,0,.3), 0 4px 18px rgba(0,0,0,.35);
    }
  }
  :root[data-theme="dark"] {
    --bg: #12181c; --surface: #182226; --surface-2: #1f2a2f;
    --text: #e7ecee; --text-muted: #8fa0a8; --text-faint: #6c7a80;
    --border: #2a363c; --grid: #263136;
    --series-a: #3987e5; --series-a-wash: rgba(57, 135, 229, 0.14);
    --series-b: #d95926; --series-b-wash: rgba(217, 89, 38, 0.14);
    --good: #0ca30c; --critical: #e66767;
    --shadow: 0 1px 2px rgba(0,0,0,.3), 0 4px 18px rgba(0,0,0,.35);
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
    font-family: ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif; line-height: 1.5; }
  .wrap { max-width: 1080px; margin: 0 auto; padding: 2.25rem 1.5rem 4rem; }
  header.masthead { border-bottom: 2px solid var(--text); padding-bottom: 1.1rem; margin-bottom: 1.5rem;
    display: flex; justify-content: space-between; align-items: flex-end; gap: 1.5rem; flex-wrap: wrap; }
  .masthead-title { font-family: ui-serif, "Iowan Old Style", "Palatino Linotype", Georgia, serif;
    font-size: clamp(1.6rem, 3vw, 2.1rem); font-weight: 600; margin: 0; }
  .masthead-meta { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: .78rem; color: var(--text-muted); text-align: right; line-height: 1.6; }
  .cards { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.75rem; }
  @media (max-width: 620px) { .cards { grid-template-columns: 1fr; } }
  .pcard { background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
    padding: 1.15rem 1.3rem; box-shadow: var(--shadow); border-top: 3px solid var(--pcolor); }
  .pcard.a { --pcolor: var(--series-a); } .pcard.b { --pcolor: var(--series-b); }
  .pcard .pname { font-size: .78rem; text-transform: uppercase; letter-spacing: .06em;
    color: var(--text-muted); display: flex; align-items: center; gap: .4rem; margin-bottom: .5rem; }
  .pcard .pname .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--pcolor); display: inline-block; }
  .pcard .hero { font-size: 2.05rem; font-weight: 700; letter-spacing: -.01em; }
  .pcard .hero .delta { font-size: 1.1rem; font-weight: 600; margin-left: .4rem; }
  .delta.pos { color: var(--good); } .delta.neg { color: var(--critical); }
  .pcard .subrow { display: flex; justify-content: space-between; margin-top: .9rem; padding-top: .75rem;
    border-top: 1px solid var(--border); font-size: .83rem; }
  .pcard .stat-label { color: var(--text-muted); }
  .pcard .stat-value { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-variant-numeric: tabular-nums; font-weight: 600; }
  .pcard .stat-value.dd { color: var(--critical); }
  .pcard .metaline { margin-top: .6rem; font-size: .75rem; color: var(--text-faint); }
  h2.section-title { font-family: ui-serif, "Iowan Old Style", "Palatino Linotype", Georgia, serif;
    font-size: 1.05rem; font-weight: 600; margin: 0 0 .75rem; }
  .chart-card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
    padding: 1.25rem 1.4rem .75rem; box-shadow: var(--shadow); margin-bottom: 1.75rem; }
  .legend { display: flex; gap: 1.25rem; margin-bottom: .6rem; font-size: .82rem; color: var(--text-muted); }
  .legend .item { display: flex; align-items: center; gap: .4rem; }
  .legend .swatch { width: 16px; height: 2px; border-radius: 2px; display: inline-block; }
  .legend .swatch.a { background: var(--series-a); } .legend .swatch.b { background: var(--series-b); }
  .chart-scroll { overflow-x: auto; }
  svg.chart { display: block; width: 100%; height: auto; min-width: 640px; }
  .gridline { stroke: var(--grid); stroke-width: 1; }
  .baseline { stroke: var(--text-faint); stroke-width: 1; stroke-dasharray: 2 3; }
  .axis-label { fill: var(--text-faint); font-size: 11px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  .line-a { fill: none; stroke: var(--series-a); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
  .line-b { fill: none; stroke: var(--series-b); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
  .end-dot { stroke: var(--surface); stroke-width: 2; }
  .end-label { font-size: 12px; font-weight: 600; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  .crosshair-line { stroke: var(--text-faint); stroke-width: 1; stroke-dasharray: 3 3; opacity: 0; pointer-events: none; }
  .hover-dot { opacity: 0; pointer-events: none; stroke: var(--surface); stroke-width: 2; }
  .tooltip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; padding: .5rem .65rem; font-size: .78rem; box-shadow: var(--shadow); opacity: 0;
    transition: opacity .08s; white-space: nowrap; z-index: 5; }
  .tooltip .trow { display: flex; justify-content: space-between; gap: .8rem; }
  .tooltip .tdate { color: var(--text-muted); font-size: .72rem; margin-bottom: .3rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  .tooltip .tval { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-variant-numeric: tabular-nums; font-weight: 600; }
  .chart-wrap { position: relative; }
  footer.notes { margin-top: 1.75rem; font-size: .8rem; color: var(--text-muted);
    border-top: 1px solid var(--border); padding-top: 1rem; }
  footer.notes code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    background: var(--surface-2); padding: .05rem .3rem; border-radius: 3px; }
  footer.notes ul { margin: .4rem 0 0; padding-left: 1.1rem; }
  footer.notes li { margin-bottom: .25rem; }
</style>
</head>
<body>
<div class="wrap">
  <header class="masthead">
    <h1 class="masthead-title">Эквити портфелей A и B, $</h1>
    <div class="masthead-meta">старт $1000 каждый · вся доступная история<br>
      A: ADX&ge;25 · B: Bollinger(20,2&sigma;) + ATR&times;2.0 стоп</div>
  </header>

  <div class="cards" id="cards"></div>

  <div class="chart-card">
    <h2 class="section-title">Кривая эквити</h2>
    <div class="legend">
      <span class="item"><span class="swatch a"></span>Портфель A (тренд)</span>
      <span class="item"><span class="swatch b"></span>Портфель B (Bollinger MR)</span>
    </div>
    <div class="chart-scroll">
      <div class="chart-wrap">
        <svg class="chart" id="chart" viewBox="0 0 960 420" preserveAspectRatio="xMidYMid meet"></svg>
        <div class="tooltip" id="tooltip"></div>
      </div>
    </div>
  </div>

  <footer class="notes">
    Автоматически пересчитывается скриптом <code>scripts/equity_curve.py</code> после каждого
    прогона <code>backtest.py</code> в GitHub Actions. Симуляция событийная: сделки по 8 живым
    инструментам сведены в общий хронологический поток на каждую систему, риск 1% капитала на
    сделку (на момент входа), потолок суммарного открытого риска 4%, макс. 5 позиций.
    <ul>
      <li>Не смоделировано: корреляционные потолки (SPY+QQQ+EEM ≤ $400) и правило мин. 10% кэш / макс. 25% на позицию.</li>
      <li>Комиссии и проскальзывание не учтены.</li>
    </ul>
  </footer>
</div>

<script>
(function () {
  const DATA = __DATA__;
  const fmtUSD = v => "$" + v.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 });
  const fmtUSD2 = v => (v < 0 ? "-$" : "$") + Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const fmtDate = s => new Date(s + "T00:00:00").toLocaleDateString("ru-RU", { day: "2-digit", month: "short", year: "numeric" });

  function buildCards() {
    const holder = document.getElementById("cards");
    const specs = [
      { key: "A", label: "Портфель A — тренд", cls: "a" },
      { key: "B", label: "Портфель B — Bollinger MR", cls: "b" },
    ];
    holder.innerHTML = specs.map(s => {
      const d = DATA[s.key];
      const totalPct = (d.final_equity / 1000 - 1) * 100;
      const deltaCls = totalPct >= 0 ? "pos" : "neg";
      const sign = totalPct >= 0 ? "+" : "";
      return `<div class="pcard ${s.cls}">
          <div class="pname"><span class="dot"></span>${s.label}</div>
          <div class="hero">${fmtUSD(d.final_equity)}<span class="delta ${deltaCls}">${sign}${totalPct.toFixed(1)}%</span></div>
          <div class="subrow"><span class="stat-label">Макс. просадка от пика</span>
            <span class="stat-value dd">${d.max_dd_pct.toFixed(1)}% (${fmtUSD2(d.max_dd_usd)})</span></div>
          <div class="subrow"><span class="stat-label">Сделок взято / пропущено (бюджет риска)</span>
            <span class="stat-value">${d.trades_taken} / ${d.trades_skipped_budget}</span></div>
          <div class="metaline">просадка: пик ${fmtDate(d.peak_date)} &rarr; дно ${fmtDate(d.trough_date)}</div>
        </div>`;
    }).join("");
  }

  const W = 960, H = 420, M = { top: 20, right: 64, bottom: 32, left: 56 };
  const plotW = W - M.left - M.right, plotH = H - M.top - M.bottom;
  const allPoints = [...DATA.A.curve, ...DATA.B.curve];
  const allDates = allPoints.map(p => new Date(p[0] + "T00:00:00").getTime());
  const allVals = allPoints.map(p => p[1]);
  const xMin = Math.min(...allDates), xMax = Math.max(...allDates);
  const yMin = Math.min(...allVals) * 0.95, yMax = Math.max(...allVals) * 1.05;

  function xScale(t) { return M.left + ((t - xMin) / (xMax - xMin)) * plotW; }
  function yScale(v) { return M.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH; }

  function niceTicks(min, max, count) {
    const range = max - min, rawStep = range / count;
    const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
    const norm = rawStep / mag;
    const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
    const start = Math.ceil(min / step) * step;
    const ticks = [];
    for (let v = start; v <= max; v += step) ticks.push(Math.round(v));
    return ticks;
  }

  function pathFor(curve) {
    return curve.map((p, i) => {
      const x = xScale(new Date(p[0] + "T00:00:00").getTime());
      const y = yScale(p[1]);
      return (i === 0 ? "M" : "L") + x.toFixed(1) + "," + y.toFixed(1);
    }).join(" ");
  }

  function render() {
    const svg = document.getElementById("chart");
    const yTicks = niceTicks(yMin, yMax, 5);
    const yearTicks = [];
    for (let y = new Date(xMin).getFullYear(); y <= new Date(xMax).getFullYear(); y++) {
      const t = new Date(y + "-01-01T00:00:00").getTime();
      if (t >= xMin && t <= xMax) yearTicks.push({ y, t });
    }
    let parts = [];
    yTicks.forEach(v => {
      const y = yScale(v);
      parts.push(`<line class="gridline" x1="${M.left}" y1="${y}" x2="${W - M.right}" y2="${y}" />`);
      parts.push(`<text class="axis-label" x="${M.left - 8}" y="${y + 3}" text-anchor="end">${fmtUSD(v)}</text>`);
    });
    const baseY = yScale(1000);
    parts.push(`<line class="baseline" x1="${M.left}" y1="${baseY}" x2="${W - M.right}" y2="${baseY}" />`);
    parts.push(`<text class="axis-label" x="${W - M.right + 6}" y="${baseY + 3}">старт</text>`);
    yearTicks.forEach(({ y, t }) => {
      const x = xScale(t);
      parts.push(`<text class="axis-label" x="${x}" y="${H - M.bottom + 18}" text-anchor="middle">${y}</text>`);
    });
    parts.push(`<path class="line-a" d="${pathFor(DATA.A.curve)}" />`);
    parts.push(`<path class="line-b" d="${pathFor(DATA.B.curve)}" />`);
    function endMark(key, colorVar) {
      const curve = DATA[key].curve;
      const last = curve[curve.length - 1];
      const x = xScale(new Date(last[0] + "T00:00:00").getTime());
      const y = yScale(last[1]);
      parts.push(`<circle class="end-dot" cx="${x}" cy="${y}" r="4.5" fill="var(${colorVar})" />`);
      parts.push(`<text class="end-label" x="${x + 8}" y="${y + 4}" fill="var(${colorVar})">${fmtUSD(last[1])}</text>`);
    }
    endMark("A", "--series-a"); endMark("B", "--series-b");
    parts.push(`<line id="crosshair" class="crosshair-line" x1="0" y1="${M.top}" x2="0" y2="${M.top + plotH}" />`);
    parts.push(`<circle id="hoverA" class="hover-dot" r="5" fill="var(--series-a)" />`);
    parts.push(`<circle id="hoverB" class="hover-dot" r="5" fill="var(--series-b)" />`);
    parts.push(`<rect id="hitrect" x="${M.left}" y="${M.top}" width="${plotW}" height="${plotH}" fill="transparent" />`);
    svg.innerHTML = parts.join("\n");
    setupHover();
  }

  function nearestPoint(curve, targetT) {
    let best = curve[0], bestDist = Infinity;
    for (const p of curve) {
      const t = new Date(p[0] + "T00:00:00").getTime();
      const dist = Math.abs(t - targetT);
      if (dist < bestDist) { bestDist = dist; best = p; }
    }
    return best;
  }

  function setupHover() {
    const svg = document.getElementById("chart");
    const hitrect = document.getElementById("hitrect");
    const crosshair = document.getElementById("crosshair");
    const hoverA = document.getElementById("hoverA");
    const hoverB = document.getElementById("hoverB");
    const tooltip = document.getElementById("tooltip");
    const wrap = document.querySelector(".chart-wrap");

    function onMove(evt) {
      const rect = svg.getBoundingClientRect();
      const scaleX = W / rect.width;
      const svgX = (evt.clientX - rect.left) * scaleX;
      const t = xMin + ((svgX - M.left) / plotW) * (xMax - xMin);
      if (t < xMin || t > xMax) return;
      const pa = nearestPoint(DATA.A.curve, t), pb = nearestPoint(DATA.B.curve, t);
      const xa = xScale(new Date(pa[0] + "T00:00:00").getTime()), ya = yScale(pa[1]);
      const xb = xScale(new Date(pb[0] + "T00:00:00").getTime()), yb = yScale(pb[1]);
      crosshair.setAttribute("x1", xa); crosshair.setAttribute("x2", xa); crosshair.style.opacity = 1;
      hoverA.setAttribute("cx", xa); hoverA.setAttribute("cy", ya); hoverA.style.opacity = 1;
      hoverB.setAttribute("cx", xb); hoverB.setAttribute("cy", yb); hoverB.style.opacity = 1;
      tooltip.innerHTML = `<div class="tdate">${fmtDate(pa[0])}</div>
        <div class="trow"><span style="color:var(--series-a)">&#9679; A</span> <span class="tval">${fmtUSD2(pa[1])}</span></div>
        <div class="trow"><span style="color:var(--series-b)">&#9679; B</span> <span class="tval">${fmtUSD2(pb[1])}</span></div>`;
      tooltip.style.opacity = 1;
      const wrapRect = wrap.getBoundingClientRect();
      const localX = evt.clientX - wrapRect.left, localY = evt.clientY - wrapRect.top;
      const tw = tooltip.offsetWidth || 120;
      let left = localX + 14;
      if (left + tw > wrapRect.width) left = localX - tw - 14;
      tooltip.style.left = left + "px";
      tooltip.style.top = Math.max(0, localY - 46) + "px";
    }
    function onLeave() { crosshair.style.opacity = 0; hoverA.style.opacity = 0; hoverB.style.opacity = 0; tooltip.style.opacity = 0; }
    hitrect.addEventListener("mousemove", onMove);
    hitrect.addEventListener("mouseleave", onLeave);
    hitrect.addEventListener("touchmove", e => { onMove(e.touches[0]); e.preventDefault(); }, { passive: false });
    hitrect.addEventListener("touchend", onLeave);
  }

  buildCards();
  render();
  window.addEventListener("resize", render);
})();
</script>
</body>
</html>
"""


def main():
    dfs = {}
    for symbol in LIVE_INSTRUMENTS:
        path = os.path.join(DATA_DIR, f"{symbol}.csv")
        if not os.path.exists(path):
            print(f"  WARNING: {path} missing, skipping {symbol} in equity curve")
            continue
        df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
        df = add_indicators(df)
        df = df.iloc[60:].reset_index(drop=True)
        dfs[symbol] = df

    results = {}
    for system_name, trade_fn in [("A", trades_system_a), ("B", trades_system_b_bollinger)]:
        all_trades = []
        for symbol, df in dfs.items():
            all_trades.extend(trade_fn(df, symbol))
        curve, final_equity, skipped, taken = simulate_portfolio(all_trades)
        dd_pct, dd_usd, peak_date, trough_date = max_drawdown(curve)
        results[system_name] = {
            "curve": curve, "final_equity": final_equity, "trades_taken": taken,
            "trades_skipped": skipped, "max_dd_pct": dd_pct, "max_dd_usd": dd_usd,
            "peak_date": peak_date, "trough_date": trough_date,
        }
        print(f"Portfolio {system_name}: final=${final_equity:.2f} "
              f"({(final_equity/START_CAPITAL-1)*100:+.1f}%), "
              f"max_dd={dd_pct*100:.1f}% (${dd_usd:.2f}), "
              f"trades taken={taken} skipped={skipped}")

    # results/equity_curve.csv — плоский формат для дальнейшего анализа
    rows = []
    for system_name, r in results.items():
        for date, equity in r["curve"]:
            rows.append({"system": system_name, "date": date, "equity": round(equity, 2)})
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "equity_curve.csv"), index=False)

    with open(os.path.join(RESULTS_DIR, "equity_curve_summary.md"), "w") as f:
        f.write("# Equity curve summary (live portfolio simulation)\n\n")
        f.write(f"Start capital: ${START_CAPITAL:.0f} each, independent portfolios.\n\n")
        f.write("System A: ADX>=25 (documented threshold). "
                "System B: Bollinger(20,2sigma) + ATR14 x2.0 stop.\n\n")
        f.write("| Portfolio | Final equity | Return | Max drawdown | Trades taken / skipped |\n")
        f.write("|---|---|---|---|---|\n")
        for name, r in results.items():
            ret_pct = (r["final_equity"] / START_CAPITAL - 1) * 100
            f.write(f"| {name} | ${r['final_equity']:.2f} | {ret_pct:+.1f}% | "
                     f"{r['max_dd_pct']*100:.1f}% (${r['max_dd_usd']:.2f}) | "
                     f"{r['trades_taken']} / {r['trades_skipped']} |\n")
        f.write("\nSee `equity_curve.html` for the interactive chart and "
                "`equity_curve.csv` for raw data.\n")

    with open(os.path.join(RESULTS_DIR, "equity_curve.html"), "w") as f:
        f.write(build_html(results))

    print("Wrote results/equity_curve.csv, equity_curve_summary.md, equity_curve.html")


if __name__ == "__main__":
    main()
