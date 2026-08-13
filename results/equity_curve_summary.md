# Equity curve summary (live portfolio simulation)

Start capital: $1000 each, independent portfolios.

System A: ADX>=25 (documented threshold). System B: Bollinger(20,2sigma) + ATR14 x2.0 stop.

| Portfolio | Final equity | Return | Max drawdown | Trades taken / skipped |
|---|---|---|---|---|
| A | $1532.85 | +53.3% | -9.5% ($-150.70) | 204 / 9 |
| B | $960.48 | -4.0% | -21.6% ($-216.15) | 192 / 0 |

See `equity_curve.html` for the interactive chart and `equity_curve.csv` for raw data.
