"""
bot/backtest.py – Historical backtester for all 3 trading strategies.

Usage (from project root):
    python -m bot.backtest

Outputs:
    backtest_results.png   – equity curves (one per instrument + combined portfolio)
    Console summary table  – metrics per instrument + portfolio
    ⚠  Warning flags for any strategy with Sharpe < 0
"""

import json
import os
import sys
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from bot.strategies import mean_reversion, momentum_breakout, trend_following

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("backtest")

SLIPPAGE       = 0.0005         # 0.05% per side (round-trip ≈ 0.10%)
COMMISSION     = 0.0            # Alpaca is commission-free
INITIAL_EQUITY = 100_000.0
BACKTEST_DAYS  = 182            # ≈ 6 months

TIMEFRAMES: dict[str, TimeFrame] = {
    "SPY":    TimeFrame(15, TimeFrameUnit.Minute),
    "QQQ":    TimeFrame(15, TimeFrameUnit.Minute),
    "BTC/USD": TimeFrame(1, TimeFrameUnit.Hour),
    "GLD":    TimeFrame(4, TimeFrameUnit.Hour),
    "USO":    TimeFrame(4, TimeFrameUnit.Hour),
}

IS_CRYPTO: dict[str, bool] = {s: s in config.CRYPTO_SYMBOLS for s in config.ALL_SYMBOLS}

# Minimum bars needed before the first signal can fire.
MIN_WARMUP: dict[str, int] = {
    "SPY":    config.MR_LOOKBACK + config.ATR_PERIOD + config.MR_TREND_FILTER_BARS + config.MR_TREND_COMPARE_BARS + 2,  # 20+14+50+20+2 = 106
    "QQQ":    config.MR_LOOKBACK + config.ATR_PERIOD + 2,   # 20 + 14 + 2 = 36  (no trend filter)
    "BTC/USD": config.MB_LOOKBACK + config.ATR_PERIOD + 2,  # 60 + 14 + 2 = 76
    "GLD":    config.TF_SLOW_EMA + config.ATR_PERIOD + 4,   # 50 + 14 + 4 = 68
    "USO":    config.USO_TF_SLOW_EMA + config.ATR_PERIOD + 4,  # 26 + 14 + 4 = 44
}

CHART_LABELS: dict[str, str] = {
    "SPY":       "SPY – Mean Reversion (15 min)",
    "QQQ":       "QQQ – Mean Reversion (15 min)",
    "BTC/USD":   "BTC/USD – Momentum Breakout (1 hr)",
    "GLD":       "GLD – Trend Following (4 hr)",
    "USO":       "USO – Trend Following (4 hr)",
    "PORTFOLIO": "Combined Portfolio (all strategies)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    symbol:        str
    direction:     str               # 'long' | 'short'
    entry_time:    datetime
    exit_time:     Optional[datetime]
    entry_price:   float
    exit_price:    float
    position_size: float             # shares or BTC
    pnl:           float = 0.0       # realised P&L in dollars


# ─────────────────────────────────────────────────────────────────────────────
# Data fetching
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Consistent lowercase columns, single-level UTC datetime index."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if df.index.nlevels > 1:
        df = df.droplevel(0)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


def fetch_all_bars(api) -> dict[str, pd.DataFrame]:
    """Pull 6 months of OHLCV bars for every instrument from Alpaca."""
    end   = datetime.utcnow()
    start = end - timedelta(days=BACKTEST_DAYS)
    s_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    e_str = end.strftime("%Y-%m-%dT%H:%M:%SZ")

    all_bars: dict[str, pd.DataFrame] = {}

    for symbol in config.ALL_SYMBOLS:
        tf = TIMEFRAMES[symbol]
        logger.info(f"Fetching {symbol} ({BACKTEST_DAYS}d @ {tf})…")

        for attempt in range(3):
            try:
                if symbol in config.CRYPTO_SYMBOLS:
                    raw = api.get_crypto_bars(symbol, tf, start=s_str, end=e_str).df
                else:
                    # Use IEX feed — available on all account tiers including paper.
                    # SIP (default) requires a paid data subscription.
                    raw = api.get_bars(
                        symbol, tf,
                        start=s_str, end=e_str,
                        adjustment="raw",
                        feed="iex",
                    ).df
                bars = _normalise_bars(raw)
                logger.info(f"  → {len(bars):,} bars")
                all_bars[symbol] = bars
                break
            except Exception as exc:
                logger.warning(f"  Attempt {attempt+1} failed: {exc}")
                time.sleep(2 ** attempt)
        else:
            logger.error(f"  Giving up on {symbol}")
            all_bars[symbol] = pd.DataFrame()

        time.sleep(0.35)   # stay within Alpaca rate limits

    return all_bars


# ─────────────────────────────────────────────────────────────────────────────
# Strategy signal dispatch
# ─────────────────────────────────────────────────────────────────────────────

def _get_signal(
    symbol: str,
    history: pd.DataFrame,
    direction: Optional[str],
    trailing: Optional[float],
    cooldown_band: Optional[float] = None,
    cooldown_dir: Optional[str] = None,
    cooldown_remaining: int = 0,
) -> dict:
    if symbol in config.MEAN_REVERSION_SYMBOLS:
        return mean_reversion.generate_signal(
            symbol, history, direction, cooldown_band, cooldown_dir
        )
    elif symbol in config.MOMENTUM_SYMBOLS:
        return momentum_breakout.generate_signal(
            symbol, history, direction, trailing, cooldown_remaining
        )
    else:
        return trend_following.generate_signal(symbol, history, direction, trailing)


def _hs_mult(symbol: str) -> float:
    """Return the ATR multiplier for hard-stop distance and position sizing."""
    # Per-symbol override takes precedence over strategy-level default
    if symbol in config.SYMBOL_HARD_STOP_ATR_MULT:
        return config.SYMBOL_HARD_STOP_ATR_MULT[symbol]
    if symbol in config.MEAN_REVERSION_SYMBOLS:
        return config.HARD_STOP_ATR_MULT["mean_reversion"]
    elif symbol in config.MOMENTUM_SYMBOLS:
        return config.HARD_STOP_ATR_MULT["momentum_breakout"]
    else:
        return config.HARD_STOP_ATR_MULT["trend_following"]


# ─────────────────────────────────────────────────────────────────────────────
# Single-instrument simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_instrument(
    symbol: str,
    bars: pd.DataFrame,
) -> tuple[list[Trade], pd.Series]:
    """
    Bar-by-bar simulation of one instrument with $INITIAL_EQUITY.
    Equity curve is mark-to-market at every bar close.
    Returns (trades_list, equity_curve_series).
    """
    warmup    = MIN_WARMUP[symbol]
    is_crypto = IS_CRYPTO[symbol]

    if len(bars) <= warmup:
        logger.warning(
            f"{symbol}: only {len(bars)} bars available, need >{warmup}. "
            "No signals will fire."
        )
        return [], pd.Series(dtype=float)

    realised_pnl: float = 0.0
    position: Optional[dict] = None   # {direction, size, entry_price, entry_time, hard_stop}
    trailing: Optional[float] = None
    trades: list[Trade] = []
    eq_idx: list = []
    eq_val: list[float] = []

    # Post-stop cooldown for mean reversion: block re-entry until price
    # recovers through the band that originally triggered the entry.
    mr_cooldown_band: Optional[float] = None
    mr_cooldown_dir:  Optional[str]   = None

    btc_cooldown: int = 0  # bars remaining before BTC can re-enter after a stop

    profit_lock_trail: Optional[float] = None  # high-water trailing stop for MR profit lock

    def _mtm(px: float) -> float:
        if position is None:
            return INITIAL_EQUITY + realised_pnl
        unreal = (px - position["entry_price"]) * position["size"] \
                 if position["direction"] == "long" \
                 else (position["entry_price"] - px) * position["size"]
        return INITIAL_EQUITY + realised_pnl + unreal

    def _close(ts, fill: float) -> None:
        nonlocal realised_pnl, position, trailing, profit_lock_trail
        if position is None:
            return
        d   = position["direction"]
        pnl = (fill - position["entry_price"]) * position["size"] \
              if d == "long" else (position["entry_price"] - fill) * position["size"]
        realised_pnl += pnl
        trades.append(Trade(
            symbol=symbol, direction=d,
            entry_time=position["entry_time"], exit_time=ts,
            entry_price=position["entry_price"], exit_price=fill,
            position_size=position["size"], pnl=pnl,
        ))
        position = None
        trailing = None
        profit_lock_trail = None

    for i in range(warmup, len(bars)):
        bar   = bars.iloc[i]
        ts    = bars.index[i]
        close = float(bar["close"])
        hi    = float(bar["high"])
        lo    = float(bar["low"])

        if btc_cooldown > 0:
            btc_cooldown -= 1

        # ── Intrabar stop check: hard stop AND trailing stop ────────────────
        # Hard stop (1 ATR from entry = 1% equity) is always active.
        # Trailing stop supplements it; whichever is more protective fires first.
        if position is not None:
            d          = position["direction"]
            hard_stop  = position.get("hard_stop")
            candidates = [s for s in [hard_stop, trailing] if s is not None]

            if candidates:
                if d == "long":
                    effective_stop = max(candidates)   # highest = most protective
                    if lo < effective_stop:
                        fill = effective_stop * (1 - SLIPPAGE)
                        lb = position.get("lower_band")   # capture before _close nulls position
                        _close(ts, fill)
                        if symbol in config.MEAN_REVERSION_SYMBOLS:
                            mr_cooldown_band = lb
                            mr_cooldown_dir  = "long"
                        if symbol in config.MOMENTUM_SYMBOLS:
                            btc_cooldown = config.MB_COOLDOWN_BARS
                        eq_idx.append(ts); eq_val.append(INITIAL_EQUITY + realised_pnl)
                        continue
                else:
                    effective_stop = min(candidates)   # lowest = most protective
                    if hi > effective_stop:
                        fill = effective_stop * (1 + SLIPPAGE)
                        ub = position.get("upper_band")   # capture before _close nulls position
                        _close(ts, fill)
                        if symbol in config.MEAN_REVERSION_SYMBOLS:
                            mr_cooldown_band = ub
                            mr_cooldown_dir  = "short"
                        if symbol in config.MOMENTUM_SYMBOLS:
                            btc_cooldown = config.MB_COOLDOWN_BARS
                        eq_idx.append(ts); eq_val.append(INITIAL_EQUITY + realised_pnl)
                        continue

        # ── Profit-lock trailing stop (QQQ MR only) ─────────────────────────
        if (position is not None
                and symbol in config.MR_PROFIT_LOCK_SYMBOLS
                and profit_lock_trail is not None):
            d = position["direction"]
            if d == "long" and lo < profit_lock_trail:
                fill = profit_lock_trail * (1 - SLIPPAGE)
                lb = position.get("lower_band")
                _close(ts, fill)
                if symbol in config.MEAN_REVERSION_SYMBOLS:
                    mr_cooldown_band = lb
                    mr_cooldown_dir  = "long"
                profit_lock_trail = None
                eq_idx.append(ts); eq_val.append(INITIAL_EQUITY + realised_pnl)
                continue
            elif d == "short" and hi > profit_lock_trail:
                fill = profit_lock_trail * (1 + SLIPPAGE)
                ub = position.get("upper_band")
                _close(ts, fill)
                if symbol in config.MEAN_REVERSION_SYMBOLS:
                    mr_cooldown_band = ub
                    mr_cooldown_dir  = "short"
                profit_lock_trail = None
                eq_idx.append(ts); eq_val.append(INITIAL_EQUITY + realised_pnl)
                continue

        # ── Strategy signal ──────────────────────────────────────────────────
        cur_dir = position["direction"] if position else None
        result  = _get_signal(
            symbol, bars.iloc[:i+1], cur_dir, trailing,
            mr_cooldown_band, mr_cooldown_dir,
            cooldown_remaining=btc_cooldown,
        )
        # Clear cooldown once price has crossed back through the band
        if mr_cooldown_band is not None and mr_cooldown_dir is not None:
            if mr_cooldown_dir == "long"  and close > mr_cooldown_band:
                mr_cooldown_band = mr_cooldown_dir = None
            elif mr_cooldown_dir == "short" and close < mr_cooldown_band:
                mr_cooldown_band = mr_cooldown_dir = None

        # Ratchet trailing stop (never let it move adversely)
        new_tr = result.get("trailing_stop")
        if new_tr is not None and position is not None:
            d = position["direction"]
            if d == "long":
                trailing = max(trailing or 0.0, new_tr)
            else:
                trailing = min(trailing or float("inf"), new_tr)

        signal = result.get("signal")
        atr    = result.get("atr")

        # ── Update profit-lock trail (MR symbols only) ───────────────────────
        if (position is not None
                and symbol in config.MR_PROFIT_LOCK_SYMBOLS
                and atr and atr > 0):
            d   = position["direction"]
            ep  = position["entry_price"]
            lock_threshold = config.MR_PROFIT_LOCK_ATR * atr
            trail_distance = config.MR_PROFIT_TRAIL_ATR * atr

            if d == "long":
                unrealised = close - ep
                if unrealised >= lock_threshold:
                    candidate = close - trail_distance
                    profit_lock_trail = max(profit_lock_trail or 0.0, candidate)
            else:
                unrealised = ep - close
                if unrealised >= lock_threshold:
                    candidate = close + trail_distance
                    profit_lock_trail = min(profit_lock_trail or float("inf"), candidate)

        if signal == "exit":
            if position is not None:
                d    = position["direction"]
                fill = close * (1 - SLIPPAGE if d == "long" else 1 + SLIPPAGE)
                _close(ts, fill)

        elif signal in ("long", "short"):
            # Reverse if currently in the opposite direction
            if position is not None and position["direction"] != signal:
                d    = position["direction"]
                fill = close * (1 - SLIPPAGE if d == "long" else 1 + SLIPPAGE)
                _close(ts, fill)

            # Open new position
            if position is None and atr and atr > 0:
                eq_now = INITIAL_EQUITY + realised_pnl
                size   = (eq_now * config.RISK_PER_TRADE) / (_hs_mult(symbol) * atr)
                size   = round(size, 6) if is_crypto else int(size)
                min_sz = 0.0001 if is_crypto else 1

                if size >= min_sz:
                    entry     = close * (1 + SLIPPAGE if signal == "long" else 1 - SLIPPAGE)
                    hard_stop = entry - _hs_mult(symbol) * atr if signal == "long" else entry + _hs_mult(symbol) * atr
                    trailing  = result.get("trailing_stop")
                    position  = {
                        "direction":   signal,
                        "size":        size,
                        "entry_price": entry,
                        "entry_time":  ts,
                        "hard_stop":   hard_stop,
                        "lower_band":  result.get("lower_band"),  # for MR cooldown
                        "upper_band":  result.get("upper_band"),
                    }
                    mr_cooldown_band = mr_cooldown_dir = None   # clear on fresh entry

        eq_idx.append(ts)
        eq_val.append(_mtm(close))

    # Close any position still open at the end of the test period
    if position:
        ts_last = bars.index[-1]
        close   = float(bars.iloc[-1]["close"])
        d       = position["direction"]
        _close(ts_last, close * (1 - SLIPPAGE if d == "long" else 1 + SLIPPAGE))
        eq_idx.append(ts_last)
        eq_val.append(INITIAL_EQUITY + realised_pnl)

    equity_curve = pd.Series(eq_val, index=eq_idx)
    equity_curve = equity_curve[~equity_curve.index.duplicated(keep="last")]
    return trades, equity_curve.sort_index()


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio simulation (shared equity + correlation filter)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_portfolio(
    all_bars: dict[str, pd.DataFrame],
) -> tuple[list[Trade], pd.Series]:
    """
    Run all strategies simultaneously with a single $INITIAL_EQUITY pool.

    Sizing at each entry uses current total equity (realised + unrealised).
    Correlation filter: BTC/USD longs blocked when SPY & QQQ are both long.
    """
    # Build a chronological event queue: (timestamp, symbol, bar_index)
    events: list[tuple] = []
    for sym, bars in all_bars.items():
        if bars.empty:
            continue
        warmup = MIN_WARMUP[sym]
        for i in range(warmup, len(bars)):
            events.append((bars.index[i], sym, i))
    events.sort(key=lambda x: (x[0], x[1]))

    if not events:
        return [], pd.Series(dtype=float)

    realised_pnl: float = 0.0
    positions:    dict[str, dict]            = {}
    trailings:    dict[str, Optional[float]] = {s: None for s in config.ALL_SYMBOLS}
    last_prices:  dict[str, float]           = {}
    all_trades:   list[Trade]                = []
    eq_idx:       list                       = []
    eq_val:       list[float]                = []

    # Per-symbol MR cooldown state (portfolio version)
    mr_cb:  dict[str, Optional[float]] = {s: None for s in config.ALL_SYMBOLS}
    mr_cd:  dict[str, Optional[str]]   = {s: None for s in config.ALL_SYMBOLS}

    # Per-symbol profit-lock trailing stop (portfolio version)
    profit_lock_trails: dict[str, Optional[float]] = {s: None for s in config.ALL_SYMBOLS}

    btc_cooldown: int = 0  # bars remaining before BTC can re-enter after a stop

    def _unreal() -> float:
        total = 0.0
        for sym, pos in positions.items():
            px = last_prices.get(sym, pos["entry_price"])
            total += (px - pos["entry_price"]) * pos["size"] \
                     if pos["direction"] == "long" \
                     else (pos["entry_price"] - px) * pos["size"]
        return total

    def _total_equity() -> float:
        return INITIAL_EQUITY + realised_pnl + _unreal()

    def _close_pos(sym: str, ts, fill: float) -> None:
        nonlocal realised_pnl
        pos = positions.pop(sym, None)
        if pos is None:
            return
        d   = pos["direction"]
        pnl = (fill - pos["entry_price"]) * pos["size"] \
              if d == "long" else (pos["entry_price"] - fill) * pos["size"]
        realised_pnl += pnl
        all_trades.append(Trade(
            symbol=sym, direction=d,
            entry_time=pos["entry_time"], exit_time=ts,
            entry_price=pos["entry_price"], exit_price=fill,
            position_size=pos["size"], pnl=pnl,
        ))
        trailings[sym] = None
        profit_lock_trails[sym] = None

    for ts, sym, i in events:
        bars  = all_bars[sym]
        bar   = bars.iloc[i]
        close = float(bar["close"])
        hi    = float(bar["high"])
        lo    = float(bar["low"])
        last_prices[sym] = close

        if sym in config.MOMENTUM_SYMBOLS and btc_cooldown > 0:
            btc_cooldown -= 1

        # ── Intrabar stop check: hard stop AND trailing stop ────────────────
        pos = positions.get(sym)
        tr  = trailings.get(sym)
        if pos is not None:
            d          = pos["direction"]
            hard_stop  = pos.get("hard_stop")
            candidates = [s for s in [hard_stop, tr] if s is not None]

            if candidates:
                if d == "long":
                    effective_stop = max(candidates)
                    if lo < effective_stop:
                        stopped_pos = positions.get(sym, {})
                        _close_pos(sym, ts, effective_stop * (1 - SLIPPAGE))
                        if sym in config.MEAN_REVERSION_SYMBOLS:
                            mr_cb[sym] = stopped_pos.get("lower_band")
                            mr_cd[sym] = "long"
                        if sym in config.MOMENTUM_SYMBOLS:
                            btc_cooldown = config.MB_COOLDOWN_BARS
                        eq_idx.append(ts); eq_val.append(_total_equity())
                        continue
                else:
                    effective_stop = min(candidates)
                    if hi > effective_stop:
                        stopped_pos = positions.get(sym, {})
                        _close_pos(sym, ts, effective_stop * (1 + SLIPPAGE))
                        if sym in config.MEAN_REVERSION_SYMBOLS:
                            mr_cb[sym] = stopped_pos.get("upper_band")
                            mr_cd[sym] = "short"
                        if sym in config.MOMENTUM_SYMBOLS:
                            btc_cooldown = config.MB_COOLDOWN_BARS
                        eq_idx.append(ts); eq_val.append(_total_equity())
                        continue

        # ── Profit-lock trailing stop (QQQ MR only) ─────────────────────────
        pos_pl = positions.get(sym)
        if (pos_pl is not None
                and sym in config.MR_PROFIT_LOCK_SYMBOLS
                and profit_lock_trails[sym] is not None):
            d_pl = pos_pl["direction"]
            if d_pl == "long" and lo < profit_lock_trails[sym]:
                lb_pl = pos_pl.get("lower_band")
                _close_pos(sym, ts, profit_lock_trails[sym] * (1 - SLIPPAGE))
                if sym in config.MEAN_REVERSION_SYMBOLS:
                    mr_cb[sym] = lb_pl
                    mr_cd[sym] = "long"
                profit_lock_trails[sym] = None
                eq_idx.append(ts); eq_val.append(_total_equity())
                continue
            elif d_pl == "short" and hi > profit_lock_trails[sym]:
                ub_pl = pos_pl.get("upper_band")
                _close_pos(sym, ts, profit_lock_trails[sym] * (1 + SLIPPAGE))
                if sym in config.MEAN_REVERSION_SYMBOLS:
                    mr_cb[sym] = ub_pl
                    mr_cd[sym] = "short"
                profit_lock_trails[sym] = None
                eq_idx.append(ts); eq_val.append(_total_equity())
                continue

        # ── Strategy signal ──────────────────────────────────────────────────
        cur_dir = positions[sym]["direction"] if sym in positions else None
        result  = _get_signal(
            sym, bars.iloc[:i+1], cur_dir, trailings.get(sym),
            mr_cb.get(sym), mr_cd.get(sym),
            cooldown_remaining=btc_cooldown,
        )
        # Clear MR cooldown once price recovers through the band
        if mr_cb.get(sym) is not None:
            if mr_cd[sym] == "long"  and close > mr_cb[sym]:
                mr_cb[sym] = mr_cd[sym] = None
            elif mr_cd[sym] == "short" and close < mr_cb[sym]:
                mr_cb[sym] = mr_cd[sym] = None

        # Update trailing stop (ratchet only)
        new_tr = result.get("trailing_stop")
        if new_tr is not None and sym in positions:
            d = positions[sym]["direction"]
            trailings[sym] = max(trailings[sym] or 0.0, new_tr) if d == "long" \
                             else min(trailings[sym] or float("inf"), new_tr)

        signal = result.get("signal")
        atr    = result.get("atr")

        # ── Update profit-lock trail (MR symbols only) ───────────────────────
        if (sym in positions
                and sym in config.MR_PROFIT_LOCK_SYMBOLS
                and atr and atr > 0):
            d_pf  = positions[sym]["direction"]
            ep_pf = positions[sym]["entry_price"]
            lock_threshold = config.MR_PROFIT_LOCK_ATR * atr
            trail_distance = config.MR_PROFIT_TRAIL_ATR * atr

            if d_pf == "long":
                unrealised = close - ep_pf
                if unrealised >= lock_threshold:
                    candidate = close - trail_distance
                    profit_lock_trails[sym] = max(profit_lock_trails[sym] or 0.0, candidate)
            else:
                unrealised = ep_pf - close
                if unrealised >= lock_threshold:
                    candidate = close + trail_distance
                    profit_lock_trails[sym] = min(profit_lock_trails[sym] or float("inf"), candidate)

        if signal == "exit" and sym in positions:
            d    = positions[sym]["direction"]
            fill = close * (1 - SLIPPAGE if d == "long" else 1 + SLIPPAGE)
            _close_pos(sym, ts, fill)

        elif signal in ("long", "short"):
            # Reverse if in opposite direction
            if sym in positions and positions[sym]["direction"] != signal:
                d    = positions[sym]["direction"]
                fill = close * (1 - SLIPPAGE if d == "long" else 1 + SLIPPAGE)
                _close_pos(sym, ts, fill)

            # ── Correlation filter: block BTC/USD long ───────────────────────
            if sym in config.MOMENTUM_SYMBOLS and signal == "long":
                spy_d = positions.get("SPY",  {}).get("direction")
                qqq_d = positions.get("QQQ",  {}).get("direction")
                if spy_d == "long" and qqq_d == "long":
                    logger.debug(f"Corr-filter blocked BTC/USD long at {ts}")
                    eq_idx.append(ts); eq_val.append(_total_equity())
                    continue

            # Enter new position
            if sym not in positions and atr and atr > 0:
                eq_now  = _total_equity()
                size    = (eq_now * config.RISK_PER_TRADE) / (_hs_mult(sym) * atr)
                is_cryp = IS_CRYPTO[sym]
                size    = round(size, 6) if is_cryp else int(size)
                min_sz  = 0.0001 if is_cryp else 1

                if size >= min_sz:
                    entry     = close * (1 + SLIPPAGE if signal == "long" else 1 - SLIPPAGE)
                    hard_stop = entry - _hs_mult(sym) * atr if signal == "long" else entry + _hs_mult(sym) * atr
                    positions[sym] = {
                        "direction":   signal,
                        "size":        size,
                        "entry_price": entry,
                        "entry_time":  ts,
                        "hard_stop":   hard_stop,
                        "lower_band":  result.get("lower_band"),
                        "upper_band":  result.get("upper_band"),
                    }
                    trailings[sym] = result.get("trailing_stop")
                    mr_cb[sym] = mr_cd[sym] = None   # clear cooldown on fresh entry

        eq_idx.append(ts)
        eq_val.append(_total_equity())

    # Close all remaining positions at final bar of each instrument
    for sym in list(positions.keys()):
        bars_sym = all_bars.get(sym)
        if bars_sym is None or bars_sym.empty:
            continue
        close_final = float(bars_sym.iloc[-1]["close"])
        ts_final    = bars_sym.index[-1]
        d = positions[sym]["direction"]
        _close_pos(sym, ts_final, close_final * (1 - SLIPPAGE if d == "long" else 1 + SLIPPAGE))

    if not eq_val:
        return all_trades, pd.Series(dtype=float)

    equity_curve = pd.Series(eq_val, index=eq_idx)
    equity_curve = equity_curve[~equity_curve.index.duplicated(keep="last")]
    return all_trades, equity_curve.sort_index()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def calculate_metrics(trades: list[Trade], equity_curve: pd.Series) -> dict:
    _empty = dict(
        total_trades=0, win_rate=0.0,
        avg_win=0.0, avg_loss=0.0, profit_factor=0.0,
        max_drawdown=0.0, sharpe=0.0, total_return=0.0,
        final_equity=INITIAL_EQUITY,
    )

    if equity_curve.empty:
        return _empty

    pnls   = [t.pnl for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    n          = len(pnls)
    win_rate   = len(wins) / n if n else 0.0
    avg_win    = float(np.mean(wins))   if wins   else 0.0
    avg_loss   = float(np.mean(losses)) if losses else 0.0
    gross_p    = sum(wins)
    gross_l    = abs(sum(losses))
    pf         = (gross_p / gross_l) if gross_l > 0 else (float("inf") if gross_p > 0 else 0.0)

    # Maximum drawdown (peak-to-trough %)
    peak       = equity_curve.cummax()
    mdd        = float(((equity_curve - peak) / peak).min())

    # Annualised Sharpe from daily equity
    try:
        daily_eq = equity_curve.resample("D").last().dropna().ffill()
        dr       = daily_eq.pct_change().dropna()
        sharpe   = float(dr.mean() / dr.std() * np.sqrt(252)) if len(dr) > 1 and dr.std() > 0 else 0.0
    except Exception:
        sharpe = 0.0

    total_ret = float((equity_curve.iloc[-1] - INITIAL_EQUITY) / INITIAL_EQUITY)

    return dict(
        total_trades=n,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=pf,
        max_drawdown=mdd,
        sharpe=sharpe,
        total_return=total_ret,
        final_equity=float(equity_curve.iloc[-1]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(
    equity_curves: dict[str, pd.Series],
    metrics_map:   dict[str, dict],
    output_path:   str = "backtest_results.png",
) -> None:
    # Try a nicer style; fall back gracefully.
    # Fix 3: track whether any seaborn style loaded so we can apply a manual
    # fallback that ensures the drawdown fill is always visible.
    style_loaded = False
    for style in ("seaborn-v0_8-darkgrid", "seaborn-darkgrid", "ggplot"):
        try:
            plt.style.use(style)
            style_loaded = True
            break
        except OSError:
            continue

    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    axes_flat = axes.flatten()

    # Fix 3: explicit light-gray background + grid when no seaborn style was
    # available, preventing the drawdown fill from being invisible on white.
    if not style_loaded:
        fig.patch.set_facecolor("#f8f9fa")
        for ax in axes_flat:
            ax.set_facecolor("#ffffff")
            ax.grid(True, alpha=0.4, linewidth=0.5)

    # Fix 6: smaller suptitle font (11 pt); top margin adjusted in tight_layout.
    fig.suptitle(
        f"Strategy Backtest — {BACKTEST_DAYS}-Day Window  |  "
        f"Initial Equity ${INITIAL_EQUITY:,.0f}  |  Slippage {SLIPPAGE*100:.2f}% per side",
        fontsize=11, fontweight="bold", y=0.995,
    )

    row_order = ["SPY", "QQQ", "BTC/USD", "GLD", "USO", "PORTFOLIO"]

    for ax, sym in zip(axes_flat, row_order):
        ec = equity_curves.get(sym, pd.Series(dtype=float))
        m  = metrics_map.get(sym, {})

        sharpe = m.get("sharpe",       0.0)
        ret    = m.get("total_return", 0.0)
        mdd    = m.get("max_drawdown", 0.0)
        ntrade = m.get("total_trades", 0)

        # Title colour: green / amber / red by Sharpe
        tc = "#27ae60" if sharpe >= 1.0 else "#e67e22" if sharpe >= 0 else "#c0392b"

        # Fix 2: smaller title font (8 pt) to prevent two-line titles from
        # being clipped by tight_layout.
        ax.set_title(
            f"{CHART_LABELS.get(sym, sym)}\n"
            f"Return: {ret*100:+.1f}%   Sharpe: {sharpe:.2f}   "
            f"MaxDD: {mdd*100:.1f}%   Trades: {ntrade}",
            fontsize=8, color=tc, fontweight="bold", pad=6,
        )

        if ec.empty:
            ax.text(0.5, 0.5, "Insufficient data\n(warm-up exceeds available history)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="#7f8c8d")
            ax.set_ylabel("Equity ($)")
            continue

        # Fix 5: thicker line + higher z-order so the curve is visible even
        # when the equity series is nearly flat (few trades, e.g. GLD/USO).
        ax.plot(ec.index, ec.values, linewidth=2.0, color="#2980b9", label="Equity", zorder=5)

        # Drawdown shading
        peak = ec.cummax()
        ax.fill_between(ec.index, ec.values, peak.values,
                        alpha=0.28, color="#e74c3c", label="Drawdown")

        # Initial-equity baseline
        ax.axhline(INITIAL_EQUITY, color="#95a5a6",
                   linewidth=0.9, linestyle="--", alpha=0.8)

        # Fix 4: annotate the baseline at the right edge with "$100k".
        ax.annotate(
            "$100k",
            xy=(1, INITIAL_EQUITY),
            xycoords=("axes fraction", "data"),
            fontsize=6, color="#95a5a6", va="center",
            xytext=(3, 0), textcoords="offset points",
        )

        ax.set_ylabel("Equity ($)")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
        )

        # Fix 1: limit x-axis to 4-7 clean date labels regardless of how many
        # bars are plotted (SPY/QQQ have ~4,000 15-min bars).
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=7)

        ax.legend(fontsize=7, loc="upper left")

    # Fix 2 + 6: extra vertical padding between rows; top rect gives suptitle
    # enough room to avoid being clipped.
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.5, w_pad=1.5)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Chart saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(metrics_map: dict[str, dict]) -> None:
    order = ["SPY", "QQQ", "BTC/USD", "GLD", "USO", "PORTFOLIO"]

    rows = []
    for sym in order:
        m = metrics_map.get(sym, {})
        pf = m.get("profit_factor", 0.0)
        pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"
        rows.append({
            "Instrument":    sym,
            "Trades":        m.get("total_trades", 0),
            "Win %":         f"{m.get('win_rate', 0)*100:.1f}%",
            "Avg Win ($)":   f"{m.get('avg_win',  0):>8,.0f}",
            "Avg Loss ($)":  f"{m.get('avg_loss', 0):>8,.0f}",
            "Prof. Factor":  pf_str,
            "Max DD":        f"{m.get('max_drawdown', 0)*100:.2f}%",
            "Sharpe":        f"{m.get('sharpe', 0):.2f}",
            "Total Return":  f"{m.get('total_return', 0)*100:+.2f}%",
            "Final Equity":  f"${m.get('final_equity', INITIAL_EQUITY):>10,.0f}",
        })

    df = pd.DataFrame(rows).set_index("Instrument")

    sep = "─" * 100
    print(f"\n{sep}")
    print(
        f"  BACKTEST SUMMARY  │  {BACKTEST_DAYS} calendar days  │  "
        f"${INITIAL_EQUITY:,.0f} starting equity  │  "
        f"slippage {SLIPPAGE*100:.2f}% per side  │  commission $0"
    )
    print(sep)
    print(df.to_string())
    print(sep)

    # ── Negative Sharpe warnings ──────────────────────────────────────────────
    bad = [(s, metrics_map[s]["sharpe"]) for s in order[:-1]
           if metrics_map.get(s, {}).get("sharpe", 0) < 0]

    if bad:
        print("\n  ⚠  NEGATIVE SHARPE — consider adjusting these strategy parameters:")
        hints = {
            "SPY":    "  Try loosening the std-dev threshold (currently 1.5σ) "
                      "or increasing the lookback period.",
            "QQQ":    "  Try loosening the std-dev threshold (currently 1.8σ) "
                      "or using a shorter lookback.",
            "BTC/USD":"  Try tightening the volume multiplier (currently 1.5×) "
                      "or reducing the trailing-stop multiplier (currently 2×ATR).",
            "GLD":    "  200-period EMA may need more history; try 50/100 EMA pair, "
                      "or widen the trailing stop.",
            "USO":    "  200-period EMA may need more history; try 50/100 EMA pair, "
                      "or widen the trailing stop.",
        }
        for sym, s in bad:
            print(f"\n     {sym} (Sharpe = {s:.2f})")
            print(f"    {hints.get(sym, '')}")
    else:
        print("\n  ✓  All individual strategies have non-negative Sharpe ratios.")

    print(f"{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest() -> None:
    api = tradeapi.REST(
        config.ALPACA_API_KEY,
        config.ALPACA_SECRET_KEY,
        config.ALPACA_BASE_URL,
        api_version="v2",
    )

    # ── Fetch data ────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"Starting {BACKTEST_DAYS}-day backtest  "
                f"(slippage={SLIPPAGE*100:.2f}%, commission=$0)")
    logger.info("=" * 60)
    all_bars = fetch_all_bars(api)

    equity_curves: dict[str, pd.Series] = {}
    metrics_map:   dict[str, dict]      = {}

    # ── Individual strategy simulations ──────────────────────────────────────
    logger.info("\nRunning individual instrument simulations…")
    for sym in config.ALL_SYMBOLS:
        bars = all_bars.get(sym, pd.DataFrame())
        logger.info(f"  {sym}…")
        trades, ec = simulate_instrument(sym, bars)
        equity_curves[sym] = ec
        metrics_map[sym]   = calculate_metrics(trades, ec)
        m = metrics_map[sym]
        logger.info(
            f"    {m['total_trades']} trades  |  "
            f"return {m['total_return']*100:+.1f}%  |  "
            f"Sharpe {m['sharpe']:.2f}  |  "
            f"MaxDD {m['max_drawdown']*100:.1f}%"
        )

    # ── Portfolio simulation ──────────────────────────────────────────────────
    logger.info("\nRunning combined portfolio simulation (with correlation filter)…")
    port_trades, port_ec = simulate_portfolio(all_bars)
    equity_curves["PORTFOLIO"] = port_ec
    metrics_map["PORTFOLIO"]   = calculate_metrics(port_trades, port_ec)
    m = metrics_map["PORTFOLIO"]
    logger.info(
        f"  Portfolio: {m['total_trades']} trades  |  "
        f"return {m['total_return']*100:+.1f}%  |  "
        f"Sharpe {m['sharpe']:.2f}  |  "
        f"MaxDD {m['max_drawdown']*100:.1f}%"
    )

    # ── Output ────────────────────────────────────────────────────────────────
    print_summary(metrics_map)
    plot_results(equity_curves, metrics_map)

    # ── Save summary JSON for dashboard ──────────────────────────────────────
    summary_path = Path(__file__).parent.parent / "backtest_summary.json"
    summary: dict[str, dict] = {}
    for sym in ["SPY", "QQQ", "BTC/USD", "GLD", "USO", "PORTFOLIO"]:
        summary[sym] = metrics_map.get(sym, {})
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, default=str, indent=2)
    logger.info(f"Summary saved → {summary_path}")

    logger.info("Backtest complete.")


if __name__ == "__main__":
    run_backtest()
