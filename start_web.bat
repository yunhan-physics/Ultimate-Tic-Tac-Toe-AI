@echo off
setlocal
cd /d "%~dp0"

set "CHESS_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%CHESS_PYTHON%" set "CHESS_PYTHON=python"

"%CHESS_PYTHON%" -B start.py
if errorlevel 1 (
  echo.
  echo The website could not start. See logs\web_server.stderr.log for details.
  pause
) else (
  echo.
  echo The game has opened in your browser. Press any key to close this window.
  echo The game server will keep running in the background.
  pause >nul
)
