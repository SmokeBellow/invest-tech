"""
render_live_chart.py — собирает самодостаточную HTML-страницу с графиком
live paper-trading счёта (data/live/equity.csv, journal.csv, state.json) —
для публикации как Artifact (см. CLAUDE.md, раздел 9). Не запускается из
GitHub Actions — вызывается сессией Claude при еженедельном/ежедневном
пересмотре, которая затем публикует получившийся файл через Artifact-тул.

Запуск: python scripts/render_live_chart.py [--out PATH]
"""
import argparse
import json
import os

import pandas as pd

LIVE_DIR = "data/live"


def load_data():
    equity_path = os.path.join(LIVE_DIR, "equity.csv")
    if not os.path.exists(equity_path):
        raise SystemExit("data/live/equity.csv не найден — сначала запустите live_review.py")
    equity = pd.read_csv(equity_path, parse_dates=["date"]).sort_values("date")

    journal_path = os.path.join(LIVE_DIR, "journal.csv")
    journal = pd.read_csv(journal_path, parse_dates=["date"]) if os.path.exists(journal_path) else pd.DataFrame()

    state_path = os.path.join(LIVE_DIR, "state.json")
    with open(state_path) as f:
        state = json.load(f)

    return equity, journal, state


def build_html(equity, journal, state):
    start_capital = 1000.0
    final = equity.iloc[-1]["equity"]
    start_date = pd.Timestamp(state["live_start_date"])
    days_live = (equity.iloc[-1]["date"] - start_date).days
    total_ret = (final / start_capital - 1) * 100

    peak = equity["equity"].cummax()
    dd = (equity["equity"] - peak) / peak * 100
    max_dd = dd.min()

    curve = [(d.strftime("%Y-%m-%d"), round(e, 2), round(c, 2), round(p, 2))
             for d, e, c, p in zip(equity["date"], equity["equity"], equity["combo_c_value"], equity["permanent_value"])]

    recent_events = []
    if not journal.empty:
        j = journal.sort_values("date", ascending=False).head(15)
        for _, r in j.iterrows():
            recent_events.append({
                "date": r["date"].strftime("%Y-%m-%d"), "system": r["system"], "symbol": r["symbol"],
                "action": r["action"], "price": r["price"], "reason": r["reason"],
            })

    data_json = json.dumps({
        "curve": curve, "final": round(final, 2), "total_ret": round(total_ret, 2),
        "max_dd": round(max_dd, 2), "days_live": days_live, "start_date": start_date.strftime("%Y-%m-%d"),
        "combo_now": round(equity.iloc[-1]["combo_c_value"], 2),
        "perm_now": round(equity.iloc[-1]["permanent_value"], 2),
        "events": recent_events,
    }, ensure_ascii=False)

    return HTML_TEMPLATE.replace("__DATA__", data_json)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Live-счёт: Комбо C / Вечный портфель</title>
<style>
  :root {
    --bg: #f4efe4; --surface: #fffdf8; --surface-2: #ece4d2;
    --ink: #241f16; --ink-muted: #6b6152; --ink-faint: #9c9282;
    --border: #ddd2ba; --grid: #e6dcc6;
    --accent: #a9781f; --accent-strong: #7d5714;
    --good: #2f7a4f; --bad: #b8482f;
    --shadow: 0 1px 2px rgba(36,31,22,.06), 0 6px 20px rgba(36,31,22,.06);
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #14110b; --surface: #1c1811; --surface-2: #241f16;
      --ink: #ece4d2; --ink-muted: #a89c85; --ink-faint: #766c5a;
      --border: #332c1f; --grid: #2a2418;
      --accent: #d6a84a; --accent-strong: #eec06a;
      --good: #57b57e; --bad: #e0785e;
      --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 26px rgba(0,0,0,.4);
    }
  }
  :root[data-theme="dark"] {
    --bg: #14110b; --surface: #1c1811; --surface-2: #241f16;
    --ink: #ece4d2; --ink-muted: #a89c85; --ink-faint: #766c5a;
    --border: #332c1f; --grid: #2a2418;
    --accent: #d6a84a; --accent-strong: #eec06a;
    --good: #57b57e; --bad: #e0785e;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 26px rgba(0,0,0,.4);
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
    font-family: ui-sans-serif, "Segoe UI", Roboto, sans-serif; line-height: 1.5; }
  .wrap { max-width: 980px; margin: 0 auto; padding: 2.4rem 1.5rem 4rem; }
  .eyebrow { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; color: var(--accent-strong);
    margin: 0 0 .5rem; }
  h1 { font-family: ui-serif, "Iowan Old Style", "Palatino Linotype", Georgia, serif;
    font-size: clamp(1.7rem, 3.4vw, 2.3rem); font-weight: 600; margin: 0 0 .3rem; text-wrap: balance; }
  .sub { font-size: .88rem; color: var(--ink-muted); margin: 0 0 1.8rem; }
  .hero { display: flex; align-items: baseline; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.8rem; }
  .hero .value { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: clamp(2.4rem, 6vw, 3.4rem); font-weight: 600; font-variant-numeric: tabular-nums; }
  .hero .delta { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 1.25rem; font-weight: 600; }
  .delta.pos { color: var(--good); } .delta.neg { color: var(--bad); }
  .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: var(--border);
    border: 1px solid var(--border); border-radius: 10px; overflow: hidden; margin-bottom: 1.8rem; }
  @media (max-width: 640px) { .stats { grid-template-columns: repeat(2, 1fr); } }
  .stat { background: var(--surface); padding: .9rem 1rem; }
  .stat .label { font-size: .68rem; text-transform: uppercase; letter-spacing: .08em; color: var(--ink-faint); }
  .stat .val { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 1.1rem; font-weight: 600; font-variant-numeric: tabular-nums; margin-top: .2rem; }
  .stat .val.bad { color: var(--bad); }
  .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
    padding: 1.3rem 1.4rem 1rem; box-shadow: var(--shadow); margin-bottom: 1.6rem; }
  .panel h2 { font-family: ui-serif, "Iowan Old Style", "Palatino Linotype", Georgia, serif;
    font-size: 1rem; font-weight: 600; margin: 0 0 .8rem; }
  .legend { display: flex; gap: 1.2rem; margin-bottom: .5rem; font-size: .78rem; color: var(--ink-muted); }
  .legend .item { display: flex; align-items: center; gap: .4rem; }
  .legend .sw { width: 14px; height: 2px; display: inline-block; }
  .sw.total { background: var(--accent); } .sw.combo { background: var(--good); } .sw.perm { background: var(--ink-faint); }
  .chart-scroll { overflow-x: auto; }
  svg.chart { display: block; width: 100%; height: auto; min-width: 560px; }
  .gridline { stroke: var(--grid); stroke-width: 1; }
  .baseline { stroke: var(--ink-faint); stroke-width: 1; stroke-dasharray: 2 3; }
  .axis-label { fill: var(--ink-faint); font-size: 10.5px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  .line-total { fill: none; stroke: var(--accent); stroke-width: 2.25; stroke-linejoin: round; stroke-linecap: round; }
  .end-dot { stroke: var(--surface); stroke-width: 2; }
  table.events { width: 100%; border-collapse: collapse; font-size: .82rem; }
  table.events th { text-align: left; font-size: .68rem; text-transform: uppercase; letter-spacing: .06em;
    color: var(--ink-faint); font-weight: 600; padding: .4rem .5rem; border-bottom: 1px solid var(--border); }
  table.events td { padding: .45rem .5rem; border-bottom: 1px solid var(--grid);
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-variant-numeric: tabular-nums; }
  table.events tr:last-child td { border-bottom: none; }
  .pill { display: inline-block; padding: .1rem .5rem; border-radius: 999px; font-size: .7rem;
    font-family: ui-sans-serif, sans-serif; font-weight: 600; }
  .pill.entry { background: color-mix(in srgb, var(--good) 18%, transparent); color: var(--good); }
  .pill.exit { background: color-mix(in srgb, var(--ink-faint) 22%, transparent); color: var(--ink-muted); }
  .empty { color: var(--ink-faint); font-size: .85rem; padding: .6rem 0; }
  footer { font-size: .76rem; color: var(--ink-muted); border-top: 1px solid var(--border);
    padding-top: 1rem; margin-top: .5rem; }
  footer code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    background: var(--surface-2); padding: .05rem .3rem; border-radius: 3px; }
</style>
</head>
<body>
<div class="wrap">
  <p class="eyebrow">Виртуальный счёт · invest-tech</p>
  <h1>Комбо C / Вечный портфель</h1>
  <p class="sub" id="subline"></p>

  <div class="hero">
    <span class="value" id="heroValue"></span>
    <span class="delta" id="heroDelta"></span>
  </div>

  <div class="stats" id="stats"></div>

  <div class="panel">
    <h2>Кривая эквити</h2>
    <div class="legend">
      <span class="item"><span class="sw total"></span>Итого (40/60)</span>
    </div>
    <div class="chart-scroll">
      <svg class="chart" id="chart" viewBox="0 0 900 320" preserveAspectRatio="xMidYMid meet"></svg>
    </div>
  </div>

  <div class="panel">
    <h2>Последние события журнала</h2>
    <div id="eventsHolder"></div>
  </div>

  <footer>
    Боевая конфигурация: 40% Комбо C (System RSI2&lt;10 + System TOM offset=5, leftover_only, потолок риска 8%)
    / 60% вечный портфель (25% SPY/TLT/SHY/GLD, ежегодный ребаланс между долями). Данные обновляются ежедневно
    через <code>live_fetch_incremental.py</code> + <code>live_review.py</code> (чисто механические правила, без
    участия LLM). Виртуальный счёт, реальных сделок нет. См. CLAUDE.md, раздел 9.
  </footer>
</div>

<script>
(function () {
  const DATA = __DATA__;
  const fmtUSD = v => "$" + v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const fmtUSD0 = v => "$" + v.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 });

  document.getElementById("subline").textContent =
    `Старт ${DATA.start_date} · ${DATA.days_live} дн. в трекинге · виртуальный счёт, $1000 на старте`;
  document.getElementById("heroValue").textContent = fmtUSD(DATA.final);
  const deltaEl = document.getElementById("heroDelta");
  deltaEl.textContent = (DATA.total_ret >= 0 ? "+" : "") + DATA.total_ret.toFixed(2) + "%";
  deltaEl.className = "delta " + (DATA.total_ret >= 0 ? "pos" : "neg");

  const stats = [
    { label: "Комбо C (40%)", val: fmtUSD0(DATA.combo_now) },
    { label: "Вечный портфель (60%)", val: fmtUSD0(DATA.perm_now) },
    { label: "Макс. просадка", val: DATA.max_dd.toFixed(2) + "%", bad: true },
    { label: "Дней в трекинге", val: DATA.days_live },
  ];
  document.getElementById("stats").innerHTML = stats.map(s =>
    `<div class="stat"><div class="label">${s.label}</div><div class="val ${s.bad ? 'bad' : ''}">${s.val}</div></div>`
  ).join("");

  const eventsHolder = document.getElementById("eventsHolder");
  if (!DATA.events.length) {
    eventsHolder.innerHTML = '<div class="empty">Пока сделок не было.</div>';
  } else {
    eventsHolder.innerHTML = `<table class="events"><thead><tr>
        <th>Дата</th><th>Система</th><th>Инструмент</th><th>Действие</th><th>Цена</th><th>Причина</th>
      </tr></thead><tbody>${DATA.events.map(e => `<tr>
        <td>${e.date}</td><td>${e.system}</td><td>${e.symbol}</td>
        <td><span class="pill ${e.action}">${e.action}</span></td>
        <td>${e.price}</td><td>${e.reason}</td>
      </tr>`).join("")}</tbody></table>`;
  }

  const W = 900, H = 320, M = { top: 16, right: 20, bottom: 30, left: 60 };
  const plotW = W - M.left - M.right, plotH = H - M.top - M.bottom;
  const curve = DATA.curve; // [date, equity, combo, perm]
  const vals = curve.map(p => p[1]);
  const dates = curve.map(p => new Date(p[0] + "T00:00:00").getTime());
  const xMin = Math.min(...dates), xMax = Math.max(...dates);
  const yMin = Math.min(1000, ...vals) * 0.98, yMax = Math.max(1000, ...vals) * 1.02;

  function xScale(t) { return curve.length > 1 ? M.left + ((t - xMin) / (xMax - xMin || 1)) * plotW : M.left; }
  function yScale(v) { return M.top + plotH - ((v - yMin) / (yMax - yMin || 1)) * plotH; }

  function niceTicks(min, max, count) {
    const range = max - min || 1, rawStep = range / count;
    const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
    const norm = rawStep / mag;
    const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
    const start = Math.ceil(min / step) * step;
    const ticks = [];
    for (let v = start; v <= max; v += step) ticks.push(Math.round(v));
    return ticks;
  }

  const svg = document.getElementById("chart");
  let parts = [];
  niceTicks(yMin, yMax, 5).forEach(v => {
    const y = yScale(v);
    parts.push(`<line class="gridline" x1="${M.left}" y1="${y}" x2="${W - M.right}" y2="${y}" />`);
    parts.push(`<text class="axis-label" x="${M.left - 8}" y="${y + 3}" text-anchor="end">${fmtUSD0(v)}</text>`);
  });
  const baseY = yScale(1000);
  parts.push(`<line class="baseline" x1="${M.left}" y1="${baseY}" x2="${W - M.right}" y2="${baseY}" />`);
  parts.push(`<text class="axis-label" x="${W - M.right + 4}" y="${baseY + 3}">старт</text>`);

  if (curve.length > 1) {
    const path = curve.map((p, i) => {
      const x = xScale(new Date(p[0] + "T00:00:00").getTime()), y = yScale(p[1]);
      return (i === 0 ? "M" : "L") + x.toFixed(1) + "," + y.toFixed(1);
    }).join(" ");
    parts.push(`<path class="line-total" d="${path}" />`);
    const last = curve[curve.length - 1];
    const lx = xScale(new Date(last[0] + "T00:00:00").getTime()), ly = yScale(last[1]);
    parts.push(`<circle class="end-dot" cx="${lx}" cy="${ly}" r="4" fill="var(--accent)" />`);
  } else if (curve.length === 1) {
    const p = curve[0];
    const x = xScale(new Date(p[0] + "T00:00:00").getTime()), y = yScale(p[1]);
    parts.push(`<circle class="end-dot" cx="${x}" cy="${y}" r="4" fill="var(--accent)" />`);
    parts.push(`<text class="axis-label" x="${x}" y="${y - 12}" text-anchor="middle">старт</text>`);
  }
  svg.innerHTML = parts.join("\n");
})();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/live_equity_chart.html")
    args = parser.parse_args()
    equity, journal, state = load_data()
    html = build_html(equity, journal, state)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
