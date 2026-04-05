@echo off
echo ============================================================
echo   AI Trading Agent - Setup Script
echo   Platform: Smarkets (UK-legal)
echo ============================================================
echo.

REM Check Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Download from: https://www.python.org/downloads/
    echo Make sure to tick "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo [OK] Python found
python --version
echo.

REM Install dependencies
echo [1/4] Installing Python dependencies...
pip install -r requirements.txt --break-system-packages 2>nul || pip install -r requirements.txt
echo.

REM Check for config
if not exist config.yaml (
    echo [2/4] Creating config.yaml from template...
    copy config.example.yaml config.yaml
    echo.
    echo ============================================================
    echo   IMPORTANT: You need to edit config.yaml with your keys!
    echo ============================================================
    echo.
    echo   Open config.yaml in VS Code and fill in:
    echo.
    echo   1. smarkets.username  - Your Smarkets email
    echo   2. smarkets.password  - Your Smarkets password
    echo   3. anthropic.api_key  - From console.anthropic.com
    echo.
    echo   Then run this script again, or use:
    echo   python -m agent.main --swarm --mode paper
    echo.
    pause
    exit /b 0
) else (
    echo [2/4] config.yaml already exists - OK
)

REM Create data directory
if not exist data mkdir data
echo [3/4] Data directory ready
echo.

echo [4/4] Setup complete!
echo.
echo ============================================================
echo   Ready to launch!
echo ============================================================
echo.
echo   Paper trading (recommended first):
echo     python -m agent.main --swarm --mode paper
echo.
echo   Live trading (real money - be careful!):
echo     python -m agent.main --swarm --mode live
echo.
echo   View status only:
echo     python -m agent.main --status
echo.
echo ============================================================
echo.

set /p LAUNCH="Launch paper trading now? (y/n): "
if /i "%LAUNCH%"=="y" (
    python -m agent.main --swarm --mode paper
)

pause
