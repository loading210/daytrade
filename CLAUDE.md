# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip3 install -r requirements.txt --break-system-packages

# Run the live trading bot
python3 -m bot.main

# Run the 6-month backtest (fetches data from Alpaca, saves backtest_results.png + backtest_summary.json)
python3 -m bot.backtest

# Start the web dashboard (http://localhost:8000)
./run_dashboard.sh
# or directly:
python3 -m uvicorn bot.app:app --host 0.0.0.0 --port 8000 --reload

# Syntax-check all modules after editing
python3 -m py_compile config.py bot/strategies/mean_reversion.py bot/strategies/momentum_breakout.py bot/strategies/trend_following.py bot/risk_manager.py bot/portfolio.py bot/main.py bot/backtest.py bot/app.py

# Kill the dashboard server
lsof -ti:8000 | xargs kill -9
```

Alpaca credentials live in `.env` (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_BASE_URL`). The default base URL is the paper-trading endpoint. Equity bar data requires `feed="iex"` — the default SIP feed needs a paid subscription.

## Architecture

`config.py` is the single source of truth for every strategy parameter, symbol group membership, risk limits, and check intervals. All other modules import from it directly. Changing a parameter in `config.py` propagates everywhere automatically — including `MIN_WARMUP` calculations in `backtest.py`, which are expressed as arithmetic over config values.

### Strategy dispatch pattern

Each strategy is a stateless module in `bot/strategies/` that exposes one function: `generate_signal(symbol, bars_df, current_position, ...)`. It receives a slice of OHLCV bars and returns a dict with keys `signal` (`"long"`, `"short"`, `"exit"`, or `None`), `price`, `atr`, and strategy-specific extras (e.g. `trailing_stop`, `sma`, `rsi`, `slope`). No state is stored inside strategy modules.

All caller state (position direction, trailing stop level, cooldown counters) is managed by the caller — either `TradingBot` in `bot/main.py` for live trading, or the simulation loops in `bot/backtest.py`.

### Instrument → strategy mapping

| Instruments | Strategy module | Timeframe | Special notes |
|---|---|---|---|
| SPY, QQQ | `mean_reversion` | 15-min | SPY also has a trend-regime filter (`MR_TREND_FILTER_SYMBOLS`); QQQ does not |
| BTC/USD | `momentum_breakout` | 1-hr | Crypto — no market-hours guard; uses `get_crypto_bars` not `get_bars` |
| GLD | `trend_following` | 4-hr | Uses global `TF_FAST_EMA` / `TF_SLOW_EMA` (20/50) |
| USO | `trend_following` | 4-hr | Uses `USO_TF_*` overrides (12/26 EMA, 4.5× trailing stop) |

### Risk management invariant

Position size and hard-stop distance are always computed together via `HARD_STOP_ATR_MULT` so that exactly 1% of equity is risked per trade:

```
size      = (equity × RISK_PER_TRADE) / (HARD_STOP_ATR_MULT[strategy] × ATR)
hard_stop = entry ± HARD_STOP_ATR_MULT[strategy] × ATR
```

Both the sizing denominator and the stop distance must use the **same multiplier** or the 1%-risk invariant breaks. This is enforced by `_hs_mult(symbol)` in `backtest.py`. The live bot submits an Alpaca stop-loss order; the backtester checks the stop intrabar using the bar's high/low.

### Backtest simulation mechanics

`simulate_instrument` and `simulate_portfolio` in `bot/backtest.py` share the same logic:

1. **Intrabar stop check first** — uses bar's `high`/`low` to detect if hard stop or trailing stop was breached. On a hit, sets MR cooldown (for `MEAN_REVERSION_SYMBOLS`) or BTC cooldown (`MB_COOLDOWN_BARS`) before `continue`-ing to next bar.
2. **Signal generation** — calls `_get_signal()` which dispatches to the correct strategy module, passing cooldown state for MR and BTC.
3. **Entry/exit** — applies 0.05% slippage to the fill price on both sides.

`simulate_portfolio` additionally:
- Maintains `last_prices` for mark-to-market of all open positions
- Applies the correlation filter (blocks BTC/USD longs when both SPY and QQQ are long)
- Sizes positions from total portfolio equity (realised + unrealised)

At the end of `run_backtest()`, metrics are written to `backtest_summary.json` for the dashboard to read.

### Live bot loop (`bot/main.py`)

`TradingBot.run()` polls every 60 seconds. Each strategy checks its own interval via `_due(key, seconds)`. Mean reversion and trend following skip when the equity market is closed (`_market_is_open()`); momentum (BTC) runs 24/7. `_sync_positions()` diffs live Alpaca positions against the in-memory `Portfolio` every cycle to catch externally-triggered stop fills.

### Dashboard (`bot/app.py` + `static/index.html`)

FastAPI serves the single-page HTML. The `/api/backtest/run` endpoint spawns `python3 -m bot.backtest` as a subprocess and streams its stdout/stderr as Server-Sent Events. The frontend consumes these via XHR (not `EventSource`, since the run is triggered by POST). Config is read/written by regex-replacing scalar assignments in `config.py` directly — only the fields listed in `CONFIG_FIELDS` are exposed.

### Logging and CSV output

The live bot writes to `trading_bot.log` (rotating, human-readable) and appends to `trades.csv` and `daily_pnl.csv` on every position close and daily rollover. The backtest only writes `backtest_results.png` and `backtest_summary.json` — it does not touch the live CSV files.
