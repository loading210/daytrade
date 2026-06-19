"""
Risk management:
  - ATR-based position sizing: 1 ATR move = 1% of equity
  - Hard stop at 1% of account equity
  - Correlation filter: block BTC/USD longs when SPY and QQQ are both long
"""

import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RISK_PER_TRADE, MOMENTUM_SYMBOLS, SYMBOL_HARD_STOP_ATR_MULT, HARD_STOP_ATR_MULT, MEAN_REVERSION_SYMBOLS

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, api, portfolio):
        self.api = api
        self.portfolio = portfolio

    # ------------------------------------------------------------------
    # Account helpers
    # ------------------------------------------------------------------

    def get_account_equity(self) -> float:
        return float(self.api.get_account().equity)

    # ------------------------------------------------------------------
    # Per-symbol ATR multiplier
    # ------------------------------------------------------------------

    def get_hs_mult(self, symbol: str) -> float:
        """Per-symbol hard-stop ATR multiplier — mirrors backtest._hs_mult."""
        if symbol in SYMBOL_HARD_STOP_ATR_MULT:
            return SYMBOL_HARD_STOP_ATR_MULT[symbol]
        if symbol in MEAN_REVERSION_SYMBOLS:
            return HARD_STOP_ATR_MULT["mean_reversion"]
        elif symbol in MOMENTUM_SYMBOLS:
            return HARD_STOP_ATR_MULT["momentum_breakout"]
        else:
            return HARD_STOP_ATR_MULT["trend_following"]

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def calculate_position_size(self, atr: float, equity: float, symbol: str = "") -> float:
        """
        Size so that a 1-ATR adverse move costs exactly RISK_PER_TRADE of equity.
        For equities the result is shares; for crypto it is the base-currency qty.
        """
        mult = self.get_hs_mult(symbol) if symbol else 1.0
        if atr <= 0:
            return 0.0
        return (equity * RISK_PER_TRADE) / (mult * atr)

    # ------------------------------------------------------------------
    # Stop price
    # ------------------------------------------------------------------

    def calculate_stop_price(self, entry_price: float, direction: str, atr: float, symbol: str = "") -> float:
        """Hard stop at mult*ATR from entry, consistent with the position-sizing formula."""
        mult = self.get_hs_mult(symbol) if symbol else 1.0
        if direction == "long":
            return round(entry_price - mult * atr, 6)
        return round(entry_price + mult * atr, 6)

    # ------------------------------------------------------------------
    # Correlation filter
    # ------------------------------------------------------------------

    def check_correlation_filter(self, symbol: str, signal: str) -> bool:
        """
        Return False (blocked) if the trade would pile on risk-on exposure:
        BTC/USD long is blocked when both SPY and QQQ are already long.
        """
        if symbol not in MOMENTUM_SYMBOLS or signal != "long":
            return True

        spy_pos = self.portfolio.get_position("SPY")
        qqq_pos = self.portfolio.get_position("QQQ")

        if spy_pos == "long" and qqq_pos == "long":
            logger.info(
                "Correlation filter active: blocking BTC/USD long "
                "(SPY and QQQ are both long)"
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Trade validation (sizing + correlation)
    # ------------------------------------------------------------------

    def validate_trade(self, symbol: str, signal: str, atr: float) -> dict:
        """
        Returns {"allowed": bool, "reason": str, "equity": float,
                 "position_size": float, "atr": float}
        """
        if not self.check_correlation_filter(symbol, signal):
            return {"allowed": False, "reason": "correlation_filter"}

        if atr is None or atr <= 0:
            return {"allowed": False, "reason": "invalid_atr"}

        try:
            equity = self.get_account_equity()
        except Exception as exc:
            logger.error(f"Cannot fetch account equity: {exc}")
            return {"allowed": False, "reason": "api_error"}

        position_size = self.calculate_position_size(atr, equity, symbol)

        return {
            "allowed": True,
            "equity": equity,
            "position_size": position_size,
            "atr": atr,
        }
