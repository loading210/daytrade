"""
Strategy 2 - Momentum Breakout for BTC/USD.

Timeframe: 1-hour candles
Logic:
  - Track 40-period high and 40-period low
  - Long when price breaks above 40-period high with volume >= 2.5x avg volume
  - Short when price breaks below 40-period low with volume >= 2.5x avg volume
  - Exit when trailing stop (3.5x ATR) is hit; trailing stop is the sole exit mechanism
"""

import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import MB_LOOKBACK, MB_VOLUME_MULTIPLIER, MB_ATR_TRAILING_STOP_MULT, ATR_PERIOD, MB_COOLDOWN_BARS


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
    trailing_stop: float | None = None,
    cooldown_remaining: int = 0,
) -> dict:
    """
    Returns dict: {signal, price, atr, trailing_stop, period_high, period_low}
    signal: 'long' | 'short' | 'exit' | None
    """
    empty = {"signal": None, "price": None, "atr": None, "trailing_stop": trailing_stop}

    if len(bars) < MB_LOOKBACK + ATR_PERIOD:
        return empty

    current_bar = bars.iloc[-1]
    current_price = float(current_bar["close"])
    current_volume = float(current_bar["volume"])

    # Lookback window excludes the current bar
    lookback = bars.iloc[-(MB_LOOKBACK + 1):-1]
    period_high = float(lookback["high"].max())
    period_low = float(lookback["low"].min())
    avg_volume = float(lookback["volume"].mean())

    atr = calculate_atr(bars, ATR_PERIOD)

    signal = None
    new_trailing_stop = trailing_stop

    if current_position in (None, "flat"):
        if cooldown_remaining <= 0:
            volume_confirmed = avg_volume > 0 and current_volume >= avg_volume * MB_VOLUME_MULTIPLIER
            if current_price > period_high and volume_confirmed:
                signal = "long"
                new_trailing_stop = current_price - MB_ATR_TRAILING_STOP_MULT * atr
            elif current_price < period_low and volume_confirmed:
                signal = "short"
                new_trailing_stop = current_price + MB_ATR_TRAILING_STOP_MULT * atr

    elif current_position == "long":
        # Ratchet stop upward only
        candidate = current_price - MB_ATR_TRAILING_STOP_MULT * atr
        new_trailing_stop = max(trailing_stop or 0.0, candidate)
        if current_price <= new_trailing_stop:
            signal = "exit"

    elif current_position == "short":
        # Ratchet stop downward only
        candidate = current_price + MB_ATR_TRAILING_STOP_MULT * atr
        new_trailing_stop = min(trailing_stop or float("inf"), candidate)
        if current_price >= new_trailing_stop:
            signal = "exit"

    return {
        "signal": signal,
        "price": current_price,
        "atr": atr,
        "trailing_stop": new_trailing_stop,
        "period_high": period_high,
        "period_low": period_low,
        "cooldown_remaining": cooldown_remaining,
    }
