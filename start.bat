@echo off
setlocal enabledelayedexpansion
title MewHelp Customer Support
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo [ERROR] .venv not found. Run README setup first.
    pause
    exit /b 1
)
echo ============================================
echo   MewHelp starting...
echo   Chat page : http://127.0.0.1:8000
echo   API docs  : http://127.0.0.1:8000/docs
echo   Stop: close this window
echo ============================================

rem 1. Start the server in the background of this window
start /b "" .venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000

rem 2. Poll until the server really answers, then open the browser once
set "READY=0"
for /l %%i in (1,1,30) do (
    if "!READY!"=="0" (
        powershell -NoProfile -Command "try{Invoke-WebRequest -Uri 'http://127.0.0.1:8000/' -UseBasicParsing -TimeoutSec 1|Out-Null;exit 0}catch{exit 1}" >nul 2>&1
        if not errorlevel 1 set "READY=1"
        if "!READY!"=="0" timeout /t 1 /nobreak >nul
    )
)
start "" http://127.0.0.1:8000
echo   Server is ready. Browser opened. Close this window to stop.

rem 3. Keep the window alive while the server runs
:keep
timeout /t 3600 /nobreak >nul
goto keep
