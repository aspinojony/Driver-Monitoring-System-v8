@echo off
REM DMS 一键启动脚本 (Windows)
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM 1) 虚拟环境检测
if exist venv\Scripts\activate.bat (
    set "VENV_DIR=venv"
) else if exist .venv\Scripts\activate.bat (
    set "VENV_DIR=.venv"
) else (
    echo [setup] Creating venv...
    where python3.11 >nul 2>nul || (
        echo [error] Python 3.11 not found
        exit /b 1
    )
    python3.11 -m venv venv
    set "VENV_DIR=venv"
)
call "%VENV_DIR%\Scripts\activate.bat"

REM 2) 依赖检测
python -c "import flask_socketio, ultralytics, mediapipe, PyQt6, pyqtgraph" 2>nul
if errorlevel 1 (
    echo [setup] Installing dependencies...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
)

REM 3) 模式选择
set "MODE=%~1"
if "%MODE%"=="" (
    echo Select mode:
    echo   [1] Web Dashboard
    echo   [2] PyQt Desktop
    echo   [3] Record training data
    echo   [4] Train new model
    set /p CHOICE="Input 1-4: "
    if "!CHOICE!"=="1" set "MODE=web"
    if "!CHOICE!"=="2" set "MODE=desktop"
    if "!CHOICE!"=="3" set "MODE=record"
    if "!CHOICE!"=="4" set "MODE=train"
)

if "%MODE%"=="web" (
    echo [run] Web Dashboard at http://127.0.0.1:5050
    python web_app.py
) else if "%MODE%"=="desktop" (
    python main.py
) else if "%MODE%"=="record" (
    python scripts\record_my_domain_v2.py
) else if "%MODE%"=="train" (
    if not exist data\dms_v2_cls (
        python scripts\build_v2_dataset.py
    )
    python scripts\train_dms_v2.py --stage all
) else (
    echo Unknown mode: %MODE%
    exit /b 1
)
