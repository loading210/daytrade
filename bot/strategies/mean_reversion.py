"""
Strategy 1 - Mean Reversion for SPY and QQQ.

Timeframe: 15-minute candles
Logic:
  - Compute 20-period SMA and standard deviation
  - Long when price < SMA - threshold*std  (SPY: 1.5, QQQ: 1.8)
  - Short when price > SMA + threshold*std
  - Exit when price crosses back to SMA
"""

import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import MR_LOOKBACK, MR_STD_THRESHOLD, ATR_PERIOD, MR_RSI_PERIOD, MR_RSI_LONG_THRESHOLD, MR_RSI_SHORT_THRESHOLD, MR_TREND_FILTER_BARS, MR_TREND_COMPARE_BARS, MR_TREND_FLAT_THRESH, MR_TREND_FILTER_SYMBOLS


def calculate_rsi(series: pd.Series, period: int) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("inf"))
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def trend_slope(closes: pd.Series, sma_bars: int, compare_bars: int) -> float:
    """
    Returns the fractional slope of the slow SMA.
    Positive = uptrend, negative = downtrend, near-zero = flat.
    Uses all available history — caller must ensure len(closes) > sma_bars + compare_bars.
    """
    if len(closes) < sma_bars + compare_bars:
        return 0.0
    sma = closes.rolling(sma_bars).mean()
    current = sma.iloc[-1]
    past    = sma.iloc[-(compare_bars + 1)]
    if past == 0:
        return 0.0
    return float((current - past) / past)


def calculate_atr(bars: pd.DataFrame, period: int) -> float:
    high = bars["high"]
    low = bars["low"]
    prev_close = bars["close"].shift(1)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    return float(tr.rolling(period).mean().iloc[-1])


def generate_signal(
    symbol: str,
    bars: pd.DataFrame,
    current_position: str | None,
    cooldown_band: float | None = None,
    cooldown_direction: str | None = None,
) -> dict:
    """
    Returns dict: {signal, price, atr, sma, upper_band, lower_band}
    signal: 'long' | 'short' | 'exit' | None

    cooldown_band / cooldown_direction: set by the caller after a hard stop
    fires. New entries are blocked until price crosses back through the band
    that triggered the original entry, preventing immediate re-entry into a
    position that was just stopped out.
      cooldown_direction='long'  → blocked until price > cooldown_band
      cooldown_direction='short' → blocked until price < cooldown_band
    """
    empty = {"signal": None, "price": None, "atr": None,
             "upper_band": None, "lower_band": None, "sma": None}

    if len(bars) < MR_LOOKBACK + ATR_PERIOD:
        return empty

    closes = bars["close"]
    current_price = float(closes.iloc[-1])

    # Compute indicators on all bars except the current one to avoid look-ahead
    window = closes.iloc[-(MR_LOOKBACK + 1):-1]
    sma = float(window.mean())
    std = float(window.std(ddof=1))

    if std == 0:
        return empty

    threshold  = MR_STD_THRESHOLD.get(symbol, 1.5)
    upper_band = sma + threshold * std
    lower_band = sma - threshold * std
    atr        = calculate_atr(bars, ATR_PERIOD)
    rsi        = calculate_rsi(closes.iloc[:-1], MR_RSI_PERIOD)

    rsi_long_thresh  = MR_RSI_LONG_THRESHOLD.get(symbol, 40)  if isinstance(MR_RSI_LONG_THRESHOLD, dict)  else MR_RSI_LONG_THRESHOLD
    rsi_short_thresh = MR_RSI_SHORT_THRESHOLD.get(symbol, 60) if isinstance(MR_RSI_SHORT_THRESHOLD, dict) else MR_RSI_SHORT_THRESHOLD

    # Trend-regime filter (applied to selected symbols only)
    if symbol in MR_TREND_FILTER_SYMBOLS:
        slope = trend_slope(closes.iloc[:-1], MR_TREND_FILTER_BARS, MR_TREND_COMPARE_BARS)
        trend_up   = slope >  MR_TREND_FLAT_THRESH
        trend_down = slope < -MR_TREND_FLAT_THRESH
    else:
        slope = 0.0
        trend_up = trend_down = False   # flat → allow both directions

    long_allowed  = not trend_down   # allow long if not in confirmed downtrend
    short_allowed = not trend_up     # allow short if not in confirmed uptrend

    # Resolve cooldown: block re-entry until price crosses back through the band
    in_cooldown = False
    if cooldown_band is not None and cooldown_direction is not None:
        if cooldown_direction == "long" and current_price <= cooldown_band:
            in_cooldown = True   # price hasn't recovered above lower band yet
        elif cooldown_direction == "short" and current_price >= cooldown_band:
            in_cooldown = True   # price hasn't recovered below upper band yet

    signal = None

    if current_position in (None, "flat"):
        if not in_cooldown and current_price < lower_band and rsi < rsi_long_thresh and long_allowed:
            signal = "long"
        elif not in_cooldown and current_price > upper_band and rsi > rsi_short_thresh and short_allowed:
            signal = "short"
    elif current_position == "long":
        if current_price >= sma:
            signal = "exit"
    elif current_position == "short":
        if current_price <= sma:
            signal = "exit"

    return {
        "signal": signal,
        "price": current_price,
        "atr": atr,
        "sma": sma,
        "upper_band": upper_band,
        "lower_band": lower_band,
        "rsi": rsi,
        "slope": slope if symbol in MR_TREND_FILTER_SYMBOLS else 0.0,
    }
