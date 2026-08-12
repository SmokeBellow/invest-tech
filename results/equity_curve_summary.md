# Equity curve summary (live portfolio simulation)

Start capital: $1000 each, independent portfolios.

System A: ADX>=25 (documented threshold). System B: Bollinger(20,2sigma) + ATR14 x2.0 stop.

| Portfolio | Final equity | Return | Max drawdown | Trades taken / skipped |
|---|---|---|---|---|
| A | $1203.97 | +20.4% | -9.5% ($-118.37) | 89 / 3 |
| B | $1091.64 | +9.2% | -3.0% ($-31.95) | 65 / 0 |

See `equity_curve.html` for the interactive chart and `equity_curve.csv` for raw data.
