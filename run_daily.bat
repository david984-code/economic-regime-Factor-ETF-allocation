@echo off
REM Daily Portfolio Update Script
REM Run this via Windows Task Scheduler at 8:30 AM and 4:30 PM ET

cd /d "%~dp0"
python run_daily_update.py

REM Optional: Keep window open to see results (remove for automated runs)
REM pause
