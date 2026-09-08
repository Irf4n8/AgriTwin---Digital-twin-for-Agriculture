@echo off
echo ===================================================
echo 🌾 Starting Agri-Twin Digital Twin Dashboard...
echo ===================================================

:: 1. Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH!
    echo Please install Python 3.10+ and try again.
    pause
    exit /b
)

:: 2. Set directory (optional but safer)
cd /d "%~dp0"

:: 2.5 Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    echo 🔄 Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo ⚠️ Virtual environment not found. Please run 'python -m venv venv' and install requirements.
)


:: 3. Open Browser securely (waits 3s for server to warm up)
echo 🌍 Launching Dashboard in Browser (waiting 5s for server)...
timeout /t 5 >nul
start "" "http://127.0.0.1:8000"

:: 4. Run Backend Server
echo 📡 Starting Telemetry Generator...
start "Telemetry Generator" cmd /k python backend/unified_telemetry_generator.py

echo 🤖 Starting Telegram Bot...
start "Telegram Bot" cmd /k python backend/telegram_bot.py

echo 🚀 Starting Backend Server...
python backend/main.py

pause
