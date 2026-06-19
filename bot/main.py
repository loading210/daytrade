"""
Main trading bot entry point.

Runs a continuous loop that fires each strategy at its own cadence:
  - Mean Reversion  (SPY, QQQ)   → every 15 minutes, equity hours only
  - Momentum Breakout (BTC/USD)  → every 1 hour,   24/7
  - Trend Following  (GLD, USO)  → every 4 hours,  equity hours only
"""

import logging
import sys
import time
import os
from datetime import datetime, date

import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from bot.portfolio import Portfolio
from bot.risk_manager import RiskManager
from bot.strategies import mean_reversion, momentum_breakout, trend_following

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Timeframe constants
# ---------------------------------------------------------------------------

TF_15MIN = TimeFrame(15, TimeFrameUnit.Minute)
TF_1HOUR = TimeFrame(1, TimeFrameUnit.Hour)
TF_4HOUR = TimeFrame(4, TimeFrameUnit.Hour)

# Bars to request (generous buffer so indicators are well-warmed-up)
BARS_MR = config.MR_LOOKBACK + config.ATR_PERIOD + 20          # ~54
BARS_MB = config.MB_LOOKBACK + config.ATR_PERIOD + 20          # ~54
BARS_TF = config.TF_SLOW_EMA + config.ATR_PERIOD + 20          # ~234


class TradingBot:
    def __init__(self):
        self.api = tradeapi.REST(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY,
            config.ALPACA_BASE_URL,
            api_version="v2",
        )
        self.portfolio = Portfolio()
        self.risk_manager = RiskManager(self.api, self.portfolio)

        # Timestamps of last strategy run
        self._last: dict[str, datetime | None] = {
            "mean_reversion": None,
            "momentum": None,
            "trend_following": None,
        }

        # Trailing stop state (managed here, not on Alpaca)
        self._trailing_stops: dict[str, float | None] = {
            sym: None for sym in config.ALL_SYMBOLS
        }

        # Daily P&L tracking
        self._today: date = date.today()
        self._day_start_equity: float | None = None
        self._cumulative_pnl: float = 0.0

    # -----------------------------------------------------------------------
    # Market-hours guard
    # -----------------------------------------------------------------------

    def _market_is_open(self) -> bool:
        try:
            return self.api.get_clock().is_open
        except Exception as exc:
            logger.error(f"Clock check failed: {exc}")
            return False

    # -----------------------------------------------------------------------
    # Data fetching
    # -----------------------------------------------------------------------

    def _get_bars(self, symbol: str, timeframe: TimeFrame, limit: int):
        """Return a DataFrame of OHLCV bars, or an empty DataFrame on error."""
        try:
            is_crypto = symbol in config.CRYPTO_SYMBOLS
            if is_crypto:
                bars = self.api.get_crypto_bars(symbol, timeframe, limit=limit).df
            else:
                bars = self.api.get_bars(
                    symbol, timeframe, limit=limit, adjustment="raw", feed="iex"
                ).df

            if bars is None or bars.empty:
                logger.warning(f"No bar data returned for {symbol}")
                return bars.__class__()

            bars.columns = [c.lower() for c in bars.columns]
            # Drop multi-index if present (crypto returns symbol-indexed df)
            if bars.index.nlevels > 1:
                bars = bars.droplevel(0)

            return bars

        except Exception as exc:
            logger.error(f"get_bars failed for {symbol}: {exc}")
            import pandas as pd
            return pd.DataFrame()

    # -----------------------------------------------------------------------
    # Order execution
    # -----------------------------------------------------------------------

    def _trading_symbol(self, symbol: str) -> str:
        """Alpaca trading uses BTCUSD (no slash) for crypto."""
        return symbol.replace("/", "")

    def _submit_entry(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
    ) -> bool:
        """Submit a market entry order plus a hard stop-loss order."""
        ts = self._trading_symbol(symbol)
        is_crypto = symbol in config.CRYPTO_SYMBOLS
        tif = "gtc" if is_crypto else "day"

        try:
            order = self.api.submit_order(
                symbol=ts,
                qty=qty,
                side=side,
                type="market",
                time_in_force=tif,
            )
            logger.info(f"Entry order submitted: {side} {qty} {ts} (id={order.id})")
        except Exception as exc:
            logger.error(f"Entry order failed for {ts}: {exc}")
            return False

        # Hard stop-loss
        stop_side = "sell" if side == "buy" else "buy"
        try:
            self.api.submit_order(
                symbol=ts,
                qty=qty,
                side=stop_side,
                type="stop",
                stop_price=round(stop_price, 2),
                time_in_force="gtc",
            )
            logger.info(f"Stop-loss order submitted at {stop_price:.4f} for {ts}")
        except Exception as exc:
            logger.warning(f"Stop-loss order failed for {ts}: {exc}")
            # Entry already placed – we'll manage stop manually via trailing logic

        return True

    def _close_position(self, symbol: str) -> bool:
        """Close the full position on Alpaca."""
        ts = self._trading_symbol(symbol)
        try:
            self.api.close_position(ts)
            logger.info(f"Close-position request sent for {ts}")
            return True
        except Exception as exc:
            logger.error(f"close_position failed for {ts}: {exc}")
            return False

    # -----------------------------------------------------------------------
    # Position synchronisation with Alpaca
    # -----------------------------------------------------------------------

    def _sync_positions(self):
        """
        If Alpaca no longer holds a position (stop hit, manual close, etc.)
        update the local portfolio so we don't hold stale state.
        """
        try:
            live = {p.symbol for p in self.api.list_positions()}
        except Exception as exc:
            logger.error(f"list_positions failed: {exc}")
            return

        for sym in list(self.portfolio.positions.keys()):
            if self._trading_symbol(sym) not in live:
                logger.info(f"{sym} position closed externally – removing from portfolio")
                # Attempt to get a last price for P&L
                try:
                    last_trade = self.api.get_latest_trade(self._trading_symbol(sym))
                    exit_px = float(last_trade.price)
                except Exception:
                    exit_px = self.portfolio.positions[sym]["entry_price"]

                pnl = self.portfolio.close_position(sym, exit_px)
                if pnl is not None:
                    self._cumulative_pnl += pnl
                self._trailing_stops[sym] = None

    # -----------------------------------------------------------------------
    # Daily P&L bookkeeping
    # -----------------------------------------------------------------------

    def _check_daily_reset(self):
        today = date.today()
        if today == self._today:
            return

        # Roll over: log yesterday's result
        try:
            equity = float(self.api.get_account().equity)
            if self._day_start_equity is not None:
                daily_pnl = equity - self._day_start_equity
                self._cumulative_pnl += daily_pnl
                self.portfolio.log_daily_pnl(daily_pnl, self._cumulative_pnl)
                logger.info(f"Daily P&L: {daily_pnl:+.2f}  Cumulative: {self._cumulative_pnl:+.2f}")
            self._day_start_equity = equity
        except Exception as exc:
            logger.error(f"Daily reset failed: {exc}")

        self._today = today

    # -----------------------------------------------------------------------
    # Strategy runners
    # -----------------------------------------------------------------------

    def _due(self, key: str, interval_seconds: int) -> bool:
        last = self._last[key]
        if last is None:
            return True
        return (datetime.now() - last).total_seconds() >= interval_seconds

    def _run_mean_reversion(self):
        if not self._due("mean_reversion", config.MEAN_REVERSION_INTERVAL):
            return
        if not self._market_is_open():
            return

        self._last["mean_reversion"] = datetime.now()
        logger.info("Running Mean Reversion check…")

        for symbol in config.MEAN_REVERSION_SYMBOLS:
            try:
                bars = self._get_bars(symbol, TF_15MIN, BARS_MR)
                if bars.empty:
                    continue

                current_pos = self.portfolio.get_position(symbol)
                result = mean_reversion.generate_signal(symbol, bars, current_pos)
                self._handle_signal(symbol, result, is_crypto=False)

            except Exception as exc:
                logger.error(f"Mean reversion error for {symbol}: {exc}", exc_info=True)

    def _run_momentum_breakout(self):
        if not self._due("momentum", config.MOMENTUM_INTERVAL):
            return

        # Crypto trades 24/7 – no market-hours guard
        self._last["momentum"] = datetime.now()
        logger.info("Running Momentum Breakout check…")

        for symbol in config.MOMENTUM_SYMBOLS:
            try:
                bars = self._get_bars(symbol, TF_1HOUR, BARS_MB)
                if bars.empty:
                    continue

                current_pos = self.portfolio.get_position(symbol)
                trailing = self._trailing_stops.get(symbol)
                result = momentum_breakout.generate_signal(symbol, bars, current_pos, trailing)

                # Persist updated trailing stop
                if result.get("trailing_stop") is not None:
                    self._trailing_stops[symbol] = result["trailing_stop"]
                    self.portfolio.update_trailing_stop(symbol, result["trailing_stop"])

                self._handle_signal(symbol, result, is_crypto=True)

            except Exception as exc:
                logger.error(f"Momentum breakout error for {symbol}: {exc}", exc_info=True)

    def _run_trend_following(self):
        if not self._due("trend_following", config.TREND_FOLLOWING_INTERVAL):
            return
        if not self._market_is_open():
            return

        self._last["trend_following"] = datetime.now()
        logger.info("Running Trend Following check…")

        for symbol in config.TREND_FOLLOWING_SYMBOLS:
            try:
                bars = self._get_bars(symbol, TF_4HOUR, BARS_TF)
                if bars.empty:
                    continue

                current_pos = self.portfolio.get_position(symbol)
                trailing = self._trailing_stops.get(symbol)
                result = trend_following.generate_signal(symbol, bars, current_pos, trailing)

                if result.get("trailing_stop") is not None:
                    self._trailing_stops[symbol] = result["trailing_stop"]
                    self.portfolio.update_trailing_stop(symbol, result["trailing_stop"])

                self._handle_signal(symbol, result, is_crypto=False)

            except Exception as exc:
                logger.error(f"Trend following error for {symbol}: {exc}", exc_info=True)

    # -----------------------------------------------------------------------
    # Unified signal handler
    # -----------------------------------------------------------------------

    def _handle_signal(self, symbol: str, result: dict, is_crypto: bool):
        signal = result.get("signal")
        price = result.get("price")
        atr = result.get("atr")

        if signal is None or price is None:
            return

        current_pos = self.portfolio.get_position(symbol)

        # ----- EXIT -----
        if signal == "exit":
            if current_pos is not None:
                if self._close_position(symbol):
                    pnl = self.portfolio.close_position(symbol, price)
                    if pnl is not None:
                        self._cumulative_pnl += pnl
                    self._trailing_stops[symbol] = None
            return

        # ----- ENTRY -----
        if signal not in ("long", "short"):
            return

        # Don't re-enter if already in this direction
        if current_pos == signal:
            return

        # Close opposite position first
        if current_pos is not None and current_pos != signal:
            if self._close_position(symbol):
                pnl = self.portfolio.close_position(symbol, price)
                if pnl is not None:
                    self._cumulative_pnl += pnl
                self._trailing_stops[symbol] = None

        validation = self.risk_manager.validate_trade(symbol, signal, atr)
        if not validation["allowed"]:
            logger.info(f"Trade blocked for {symbol}: {validation.get('reason')}")
            return

        raw_size = validation["position_size"]

        # Equities: whole shares; crypto: fractional (6 decimal places)
        if is_crypto:
            qty = round(raw_size, 6)
            min_qty = 0.0001
        else:
            qty = int(raw_size)
            min_qty = 1

        if qty < min_qty:
            logger.info(f"Skipping {symbol}: computed qty {qty} below minimum {min_qty}")
            return

        stop_px = self.risk_manager.calculate_stop_price(price, signal, atr, symbol)
        side = "buy" if signal == "long" else "sell"

        if self._submit_entry(symbol, side, qty, stop_px):
            self.portfolio.open_position(
                symbol=symbol,
                direction=signal,
                entry_price=price,
                position_size=qty,
                stop_price=stop_px,
                atr=atr,
                trailing_stop=result.get("trailing_stop"),
            )

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def run(self):
        logger.info("=" * 60)
        logger.info("Trading bot starting up")
        logger.info("=" * 60)

        try:
            self._day_start_equity = float(self.api.get_account().equity)
            logger.info(f"Account equity: ${self._day_start_equity:,.2f}")
        except Exception as exc:
            logger.error(f"Could not fetch starting equity: {exc}")

        while True:
            try:
                self._check_daily_reset()
                self._sync_positions()

                self._run_mean_reversion()
                self._run_momentum_breakout()
                self._run_trend_following()

            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as exc:
                logger.error(f"Unhandled error in main loop: {exc}", exc_info=True)

            time.sleep(60)  # Poll every minute; strategies gate on their own intervals


if __name__ == "__main__":
    TradingBot().run()
