#!/usr/bin/env bash
# AI-Based Flood Forecasting Dashboard Launcher (macOS / Linux)

echo "======================================================================="
echo "  AI-Based Flood Forecasting and Early Warning System"
echo "  Target: Almatti Reservoir (Krishna River Basin)"
echo "======================================================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed. Please install Python 3.10+."
    exit 1
fi

echo "[1/2] Verifying and installing required packages..."
pip3 install -r requirements.txt --quiet

echo ""
echo "[2/2] Launching Streamlit Dashboard..."
streamlit run app.py
