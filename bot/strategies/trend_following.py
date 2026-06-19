"""
Strategy 3 - Trend Following for GLD and USO.

Timeframe: 4-hour candles
Logic:
  - Calculate 20-period and 50-period EMAs
  - Long on golden cross (20 EMA crosses above 50 EMA)
  - Exit / short on death cross (20 EMA crosses below 50 EMA)
  - Trailing stop of 3x ATR
"""

import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import TF_FAST_EMA, TF_SLOW_EMA, TF_ATR_TRAILING_STOP_MULT, ATR_PERIOD, USO_TF_FAST_EMA, USO_TF_SLOW_EMA, USO_TF_ATR_TRAILING_STOP_MULT


def calculate_atr(bars: pd.DataFrame, period: int) -> float:
    high = bars["high"]
    low = bars["low"]
    prev_close = bars["close"].shift(1)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    return float(tr.rolling(period).mean().iloc[-1])


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def generate_signal(
    symbol: str,
    bars: pd.DataFrame,
    current_position: str | None,
    trailing_stop: float | None = None,
) -> dict:
    """
    Returns dict: {signal, price, atr, trailing_stop, fast_ema, slow_ema}
    signal: 'long' | 'short' | 'exit' | None
    """
    empty = {"signal": None, "price": None, "atr": None, "trailing_stop": trailing_stop}

    # Use instrument-specific EMA parameters if defined
    if symbol == "USO":
        fast_period = USO_TF_FAST_EMA
        slow_period = USO_TF_SLOW_EMA
        trail_mult  = USO_TF_ATR_TRAILING_STOP_MULT
    else:
        fast_period = TF_FAST_EMA
        slow_period = TF_SLOW_EMA
        trail_mult  = TF_ATR_TRAILING_STOP_MULT

    # Need enough bars to compute the slow EMA reliably
    if len(bars) < slow_period + ATR_PERIOD + 2:
        return empty

    closes = bars["close"]
    current_price = float(closes.iloc[-1])

    fast_ema = calculate_ema(closes, fast_period)
    slow_ema = calculate_ema(closes, slow_period)

    fast_now = float(fast_ema.iloc[-1])
    fast_prev = float(fast_ema.iloc[-2])
    slow_now = float(slow_ema.iloc[-1])
    slow_prev = float(slow_ema.iloc[-2])

    atr = calculate_atr(bars, ATR_PERIOD)

    # Crossover detection
    golden_cross = fast_prev <= slow_prev and fast_now > slow_now
    death_cross = fast_prev >= slow_prev and fast_now < slow_now

    signal = None
    new_trailing_stop = trailing_stop

    if current_position in (None, "flat"):
        if golden_cross:
            signal = "long"
            new_trailing_stop = current_price - trail_mult * atr
        elif death_cross:
            signal = "short"
            new_trailing_stop = current_price + trail_mult * atr

    elif current_position == "long":
        candidate = current_price - trail_mult * atr
        new_trailing_stop = max(trailing_stop or 0.0, candidate)
        if death_cross or current_price <= new_trailing_stop:
            signal = "exit"

    elif current_position == "short":
        candidate = current_price + trail_mult * atr
        new_trailing_stop = min(trailing_stop or float("inf"), candidate)
        if golden_cross or current_price >= new_trailing_stop:
            signal = "exit"

    return {
        "signal": signal,
        "price": current_price,
        "atr": atr,
        "trailing_stop": new_trailing_stop,
        "fast_ema": fast_now,
        "slow_ema": slow_now,
    }
