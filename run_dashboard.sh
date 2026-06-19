#!/bin/bash
# Start the DayTrade Bot dashboard.
# Usage: ./run_dashboard.sh
cd "$(dirname "$0")"
python3 -m uvicorn bot.app:app --host 0.0.0.0 --port 8000 --reload
