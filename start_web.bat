@echo off
setlocal
cd /d "%~dp0"

set "CHESS_PYTHON=%USERPROFILE%\miniconda3\envs\d2l\python.exe"
if not exist "%CHESS_PYTHON%" set "CHESS_PYTHON=python"

"%CHESS_PYTHON%" -B launch_web.py --lan
if errorlevel 1 (
  echo.
  echo The website could not start. See logs\web_server.stderr.log for details.
  pause
) else (
  echo.
  echo The phone address is shown above. Press any key to close this window.
  echo The game server will keep running in the background.
  pause >nul
)
