import os
from dotenv import load_dotenv

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Instrument groups
MEAN_REVERSION_SYMBOLS = ["SPY", "QQQ"]
MOMENTUM_SYMBOLS = ["BTC/USD"]
TREND_FOLLOWING_SYMBOLS = ["GLD", "USO"]
EQUITY_SYMBOLS = MEAN_REVERSION_SYMBOLS + TREND_FOLLOWING_SYMBOLS
CRYPTO_SYMBOLS = MOMENTUM_SYMBOLS
ALL_SYMBOLS = EQUITY_SYMBOLS + CRYPTO_SYMBOLS

# Strategy 1: Mean Reversion (SPY, QQQ)
MR_LOOKBACK = 20
MR_STD_THRESHOLD = {"SPY": 4.0, "QQQ": 3.2}

# Once a MR trade is in profit by this many ATRs, activate a trailing stop
# that locks in gains and prevents giving them back to the SMA exit.
# Set to 0 to disable (use pure SMA exit).
MR_PROFIT_LOCK_ATR    = 1.5    # activate trailing lock once up 1.5 ATR
MR_PROFIT_TRAIL_ATR   = 1.0    # trail 1.0 ATR behind the high-water mark
MR_PROFIT_LOCK_SYMBOLS = ["QQQ", "SPY"]

MR_RSI_PERIOD = 14
MR_RSI_LONG_THRESHOLD  = {"SPY": 25, "QQQ": 35}  # per-symbol RSI gate for longs
MR_RSI_SHORT_THRESHOLD = {"SPY": 75, "QQQ": 65}  # per-symbol RSI gate for shorts

# Trend-regime filter for mean reversion
# Only take longs when short-term trend is up, shorts when trend is down.
MR_TREND_FILTER_BARS   = 50   # SMA period for trend direction (bars)
MR_TREND_COMPARE_BARS  = 20   # how many bars ago to compare SMA against
MR_TREND_FLAT_THRESH   = 0.0003  # 0.03% — slope smaller than this = "flat" (allow both directions)
MR_TREND_FILTER_SYMBOLS = ["SPY"]   # only these symbols use the trend-regime gate

# Strategy 2: Momentum Breakout (BTC/USD)
MB_LOOKBACK = 60
MB_VOLUME_MULTIPLIER = 3.0
MB_ATR_TRAILING_STOP_MULT = 5.5
MB_COOLDOWN_BARS = 8

# Strategy 3: Trend Following (GLD, USO)
TF_FAST_EMA = 20
TF_SLOW_EMA = 50
TF_ATR_TRAILING_STOP_MULT = 3.0

# USO-specific overrides (faster EMA pair for oil's shorter swing cycles)
USO_TF_FAST_EMA = 12
USO_TF_SLOW_EMA = 26
USO_TF_ATR_TRAILING_STOP_MULT = 4.5

# Risk management
ATR_PERIOD = 14
RISK_PER_TRADE = 0.01  # 1% of equity per trade

# Hard-stop distance in ATR multiples, per strategy group.
# Position sizing denominator uses the same multiplier so risk stays at 1%.
HARD_STOP_ATR_MULT = {
    "mean_reversion":    2.5,
    "momentum_breakout": 2.5,
    "trend_following":   2.5,
}

# Per-symbol overrides for hard-stop ATR multiplier.
# These allow larger positions (same 1% equity risk) where the R ratio supports it.
# SPY is deliberately excluded — it stays at the strategy-level 2.5x default.
SYMBOL_HARD_STOP_ATR_MULT = {}   # no per-symbol overrides — use HARD_STOP_ATR_MULT defaults

# Log files
TRADES_LOG = "trades.csv"
DAILY_PNL_LOG = "daily_pnl.csv"

# Check intervals in seconds
MEAN_REVERSION_INTERVAL = 15 * 60
MOMENTUM_INTERVAL = 60 * 60
TREND_FOLLOWING_INTERVAL = 4 * 60 * 60
