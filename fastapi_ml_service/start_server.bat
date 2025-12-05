@echo off
echo Starting Zigbee Radio ML Service...
cd /d "%~dp0"
py -m uvicorn app.main:app --reload
pause

