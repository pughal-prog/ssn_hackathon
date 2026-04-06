@echo off
echo ============================================
echo   GLOF Early Warning System - Setup & Run
echo ============================================

echo [1/3] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

echo [2/3] Installing dependencies...
pip install -r requirements.txt

echo [3/3] Training ML model...
python -c "from modules.risk_model import train; train()"

echo.
echo ✅ Setup complete! Launching dashboard...
streamlit run app.py
