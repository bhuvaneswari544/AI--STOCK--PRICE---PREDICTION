@echo off
title AI Stock Intelligence Dashboard
cd /d "%~dp0"
echo ========================================================
echo   Launching AI Stock Intelligence Dashboard (Rupees Suite)
echo ========================================================
echo.
start "" "http://127.0.0.1:5000"
python app.py
pause