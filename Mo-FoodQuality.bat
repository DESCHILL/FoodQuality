@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" chay_web.py
if errorlevel 1 pause
