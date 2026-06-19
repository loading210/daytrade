"""
In-memory position tracker and CSV trade logger.
"""

import csv
import logging
import os
from datetime import datetime
from typing import Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import TRADES_LOG, DAILY_PNL_LOG

logger = logging.getLogger(__name__)


class Portfolio:
    def __init__(self):
        # symbol -> {"direction", "entry_price", "position_size",
        #            "stop_price", "atr", "trailing_stop", "open_time"}
        self.positions: dict[str, dict] = {}
        self._init_csv_files()

    # ------------------------------------------------------------------
    # CSV setup
    # ------------------------------------------------------------------

    def _init_csv_files(self):
        if not os.path.exists(TRADES_LOG):
            with open(TRADES_LOG, "w", newline="") as fh:
                csv.writer(fh).writerow(
                    ["timestamp", "instrument", "direction",
                     "entry_price", "exit_price", "profit_loss", "position_size"]
                )

        if not os.path.exists(DAILY_PNL_LOG):
            with open(DAILY_PNL_LOG, "w", newline="") as fh:
                csv.writer(fh).writerow(
                    ["date", "daily_pnl", "cumulative_pnl", "open_positions"]
                )

    # ------------------------------------------------------------------
    # Position queries
    # ------------------------------------------------------------------

    def get_position(self, symbol: str) -> Optional[str]:
        """Return 'long', 'short', or None."""
        pos = self.positions.get(symbol)
        return pos["direction"] if pos else None

    def get_trailing_stop(self, symbol: str) -> Optional[float]:
        pos = self.positions.get(symbol)
        return pos.get("trailing_stop") if pos else None

    # ------------------------------------------------------------------
    # Position lifecycle
    # ------------------------------------------------------------------

    def open_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        position_size: float,
        stop_price: float,
        atr: float,
        trailing_stop: Optional[float] = None,
    ):
        self.positions[symbol] = {
            "direction": direction,
            "entry_price": entry_price,
            "position_size": position_size,
            "stop_price": stop_price,
            "atr": atr,
            "trailing_stop": trailing_stop,
            "open_time": datetime.now(),
        }
        logger.info(
            f"OPEN  {direction:5s} {symbol:8s}  "
            f"entry={entry_price:.4f}  size={position_size:.6f}  stop={stop_price:.4f}"
        )

    def close_position(self, symbol: str, exit_price: float) -> Optional[float]:
        """Close position, log to CSV, return realised P&L."""
        if symbol not in self.positions:
            logger.warning(f"close_position called for {symbol} but no open position found")
            return None

        pos = self.positions.pop(symbol)
        direction = pos["direction"]
        entry = pos["entry_price"]
        size = pos["position_size"]

        pnl = (exit_price - entry) * size if direction == "long" else (entry - exit_price) * size

        self._log_trade(symbol, direction, entry, exit_price, pnl, size)
        logger.info(
            f"CLOSE {direction:5s} {symbol:8s}  "
            f"entry={entry:.4f}  exit={exit_price:.4f}  pnl={pnl:+.2f}"
        )
        return pnl

    def update_trailing_stop(self, symbol: str, new_stop: float):
        if symbol in self.positions:
            self.positions[symbol]["trailing_stop"] = new_stop

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_trade(
        self,
        instrument: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        profit_loss: float,
        position_size: float,
    ):
        with open(TRADES_LOG, "a", newline="") as fh:
            csv.writer(fh).writerow(
                [
                    datetime.now().isoformat(),
                    instrument,
                    direction,
                    f"{entry_price:.6f}",
                    f"{exit_price:.6f}",
                    f"{profit_loss:.2f}",
                    f"{position_size:.6f}",
                ]
            )

    def log_daily_pnl(self, daily_pnl: float, cumulative_pnl: float):
        open_positions = "|".join(
            f"{sym}:{pos['direction']}" for sym, pos in self.positions.items()
        )
        with open(DAILY_PNL_LOG, "a", newline="") as fh:
            csv.writer(fh).writerow(
                [
                    datetime.now().date().isoformat(),
                    f"{daily_pnl:.2f}",
                    f"{cumulative_pnl:.2f}",
                    open_positions,
                ]
            )
