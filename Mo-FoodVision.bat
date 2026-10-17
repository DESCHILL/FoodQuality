@echo off
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\pythonw.exe" (
    echo Chua co moi truong Python .venv. Xem HUONG_DAN_HAI_MODEL.md
    pause
    exit /b 1
)
start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0launch_user_app.py"
