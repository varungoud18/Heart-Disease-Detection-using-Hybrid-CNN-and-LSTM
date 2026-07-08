@echo off
echo === Heart Disease Detection Project Environment Setup (Windows) ===

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not added to your system PATH.
    echo Please install Python 3.10+ and check 'Add Python to environment variables'.
    pause
    exit /b %errorlevel%
)

:: Create virtual environment
if not exist ".venv" (
    echo Creating virtual environment '.venv'...
    python -m venv .venv
) else (
    echo Virtual environment '.venv' already exists.
)

:: Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install requirements
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo.
echo ==========================================================
echo Setup completed successfully!
echo To activate the virtual environment in your command prompt, run:
echo     .venv\Scripts\activate.bat
echo ==========================================================
pause
