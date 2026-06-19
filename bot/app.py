"""
bot/app.py — FastAPI dashboard backend for DayTrade Bot.

Run from project root:
    python3 -m uvicorn bot.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import csv
import json
import re
import sys
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import (
    HTMLResponse,
    StreamingResponse,
    FileResponse,
    JSONResponse,
)
from fastapi.staticfiles import StaticFiles

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent.parent           # /Users/jacobalegre/dev/daytrade
CONFIG_FILE = BASE_DIR / "config.py"
STATIC_DIR  = BASE_DIR / "static"
CHART_FILE  = BASE_DIR / "backtest_results.png"
SUMMARY_FILE= BASE_DIR / "backtest_summary.json"
TRADES_FILE = BASE_DIR / "trades.csv"
PNL_FILE    = BASE_DIR / "daily_pnl.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Config field definitions (name → Python type)
# ─────────────────────────────────────────────────────────────────────────────

CONFIG_FIELDS: dict[str, type] = {
    "MR_LOOKBACK":               int,
    "MB_LOOKBACK":               int,
    "MB_VOLUME_MULTIPLIER":      float,
    "MB_ATR_TRAILING_STOP_MULT": float,
    "MB_COOLDOWN_BARS":          int,
    "TF_FAST_EMA":               int,
    "TF_SLOW_EMA":               int,
    "RISK_PER_TRADE":            float,
}

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="DayTrade Bot Dashboard")

# Mount /static only after the directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_config_text() -> str:
    return CONFIG_FILE.read_text(encoding="utf-8")


def _parse_config() -> dict[str, Any]:
    """Extract CONFIG_FIELDS values from config.py using regex."""
    text   = _read_config_text()
    result = {}
    for field, cast in CONFIG_FIELDS.items():
        # Match:  FIELD_NAME = <number>   (ignores inline comments)
        m = re.search(
            rf"^\s*{re.escape(field)}\s*=\s*([0-9]+\.?[0-9]*)",
            text,
            re.MULTILINE,
        )
        if m:
            try:
                result[field] = cast(m.group(1))
            except (ValueError, TypeError):
                result[field] = m.group(1)
    return result


def _update_config(updates: dict[str, Any]) -> None:
    """Write updated values back to config.py via regex substitution."""
    text = _read_config_text()
    for field, value in updates.items():
        if field not in CONFIG_FIELDS:
            continue
        cast = CONFIG_FIELDS[field]
        new_val = str(cast(value))
        # Replace the assignment line, preserving any trailing comment
        text = re.sub(
            rf"(^\s*{re.escape(field)}\s*=\s*)[0-9]+\.?[0-9]*",
            rf"\g<1>{new_val}",
            text,
            flags=re.MULTILINE,
        )
    CONFIG_FILE.write_text(text, encoding="utf-8")


def _csv_to_json(path: Path) -> list[dict]:
    """Read a CSV file and return a list of row dicts."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the single-page dashboard."""
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ── Config ────────────────────────────────────────────────────────────────────

@app.get("/api/config")
async def get_config() -> JSONResponse:
    """Return current strategy config values as JSON."""
    try:
        return JSONResponse(_parse_config())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/config")
async def post_config(payload: dict = Body(...)) -> JSONResponse:
    """
    Update config.py with supplied key/value pairs.
    Only keys present in CONFIG_FIELDS are accepted.
    """
    try:
        _update_config(payload)
        return JSONResponse({"status": "ok", "updated": list(payload.keys())})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Backtest ──────────────────────────────────────────────────────────────────

@app.post("/api/backtest/run")
async def run_backtest() -> StreamingResponse:
    """
    Launch the backtest subprocess and stream stdout/stderr as
    Server-Sent Events (SSE).  Each event carries a JSON object:
      { "log": "<line>" }   — while running
      { "done": true, "returncode": <int> }  — when finished
    """
    async def event_stream():
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "bot.backtest",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(BASE_DIR),
        )
        async for line in proc.stdout:          # type: ignore[union-attr]
            text = line.decode(errors="replace").rstrip()
            yield f"data: {json.dumps({'log': text})}\n\n"
        await proc.wait()
        yield f"data: {json.dumps({'done': True, 'returncode': proc.returncode})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/backtest/chart")
async def backtest_chart() -> FileResponse:
    """Serve backtest_results.png. Returns 404 JSON if the file doesn't exist yet."""
    if not CHART_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Chart not found. Run the backtest first.",
        )
    return FileResponse(str(CHART_FILE), media_type="image/png")


@app.get("/api/backtest/results")
async def backtest_results() -> JSONResponse:
    """Return the last backtest summary metrics from backtest_summary.json."""
    if not SUMMARY_FILE.exists():
        return JSONResponse({})
    try:
        data = json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
        return JSONResponse(data)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Malformed summary JSON: {exc}")


# ── Trade & P&L logs ──────────────────────────────────────────────────────────

@app.get("/api/trades")
async def get_trades() -> JSONResponse:
    """Return trades.csv as a JSON array (all rows)."""
    return JSONResponse(_csv_to_json(TRADES_FILE))


@app.get("/api/pnl")
async def get_pnl() -> JSONResponse:
    """Return daily_pnl.csv as a JSON array."""
    return JSONResponse(_csv_to_json(PNL_FILE))


# ── Live positions ────────────────────────────────────────────────────────────

@app.get("/api/positions")
async def get_positions() -> JSONResponse:
    """
    Fetch current open positions from Alpaca.
    Returns an empty list if credentials are missing or the call fails.
    """
    try:
        # Import lazily so the app starts even without alpaca installed
        import alpaca_trade_api as tradeapi  # type: ignore
        import config as cfg                  # type: ignore

        if not cfg.ALPACA_API_KEY or not cfg.ALPACA_SECRET_KEY:
            return JSONResponse([])

        api  = tradeapi.REST(
            cfg.ALPACA_API_KEY,
            cfg.ALPACA_SECRET_KEY,
            cfg.ALPACA_BASE_URL,
            api_version="v2",
        )
        positions = api.list_positions()
        return JSONResponse([
            {
                "symbol":        p.symbol,
                "qty":           p.qty,
                "side":          p.side,
                "avg_entry_price": p.avg_entry_price,
                "current_price": p.current_price,
                "unrealized_pl": p.unrealized_pl,
                "unrealized_plpc": p.unrealized_plpc,
                "market_value":  p.market_value,
            }
            for p in positions
        ])
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=200)
