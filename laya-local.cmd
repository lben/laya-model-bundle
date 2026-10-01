@echo off
setlocal
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo ERROR: Python environment is missing. Run test_windows.ps1 -Install first. 1>&2
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0laya_local.py" %*
exit /b %errorlevel%
