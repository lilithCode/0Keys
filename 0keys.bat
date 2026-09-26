@echo off
set "DIR=%~dp0"
if not exist "%DIR%.venv\Scripts\python.exe" (
    echo 0Keys is not installed yet. Run install.ps1 first.
    exit /b 1
)
if "%~1"=="" (
    start "" "%DIR%.venv\Scripts\pythonw.exe" "%DIR%app.py"
) else (
    "%DIR%.venv\Scripts\python.exe" "%DIR%app.py" %*
)
