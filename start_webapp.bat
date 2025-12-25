@echo off
echo ======================================================================
echo ML STOCK TRADING PLATFORM - Web Application
echo ======================================================================
echo.
echo Starting Flask server...
echo.
echo Access the application at: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo ======================================================================
echo.

cd /d "%~dp0"
python webapp.py

pause
