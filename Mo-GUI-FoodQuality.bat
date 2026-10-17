@echo off
cd /d "%~dp0"

set "FQ_PYTHON=%~dp0.venv\Scripts\pythonw.exe"
if exist "%FQ_PYTHON%" (
    start "" "%FQ_PYTHON%" "%~dp0launch_gui.py"
    exit /b
)
set "FQ_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%FQ_PYTHON%" set "FQ_PYTHON=python"
"%FQ_PYTHON%" "%~dp0launch_gui.py"
if errorlevel 1 pause
