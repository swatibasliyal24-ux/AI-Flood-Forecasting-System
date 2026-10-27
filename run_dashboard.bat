@echo off
title Almatti Reservoir Flood Forecasting Dashboard
color 0B

echo =======================================================================
echo   AI-Based Flood Forecasting and Early Warning System
echo   Target: Almatti Reservoir (Krishna River Basin)
echo =======================================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.10 or higher from https://www.python.org
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [1/2] Verifying and installing required packages...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [WARNING] Some dependencies failed to install silently. Retrying with verbose output...
    pip install -r requirements.txt
)

echo.
echo [2/2] Launching Streamlit Dashboard...
echo Starting server at http://localhost:8501 ...
echo.
python -m streamlit run app.py

pause
