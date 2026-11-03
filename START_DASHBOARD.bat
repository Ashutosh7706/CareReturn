@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo       CareReturn Dashboard Launcher
echo ==========================================

where py >nul 2>nul
if %errorlevel%==0 (
    set "PY=py -3"
) else (
    set "PY=python"
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/5] Creating virtual environment...
    %PY% -m venv .venv
    if errorlevel 1 goto :error
) else (
    echo [1/5] Virtual environment already exists.
)

echo [2/5] Installing project dependencies, including Streamlit...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip install -e ".[dev]"
if errorlevel 1 goto :error

if not exist "data\raw\diabetes_130_us_hospitals.csv" (
    echo [3/5] Downloading the UCI dataset...
    ".venv\Scripts\python.exe" scripts\fetch_dataset.py
    if errorlevel 1 goto :error
) else (
    echo [3/5] Dataset already present.
)

if not exist "reports\evaluation.json" (
    echo [4/5] Training and evaluating the models...
    ".venv\Scripts\python.exe" scripts\run_analysis.py
    if errorlevel 1 goto :error
) else (
    echo [4/5] Evaluation report already exists. Skipping training.
)

echo [5/5] Starting Streamlit dashboard...
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 goto :error
goto :eof

:error
echo.
echo Something went wrong. Read the error above.
echo Check your internet connection and Python installation, then try again.
pause
exit /b 1
