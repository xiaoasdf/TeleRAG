@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"

if exist "%PYTHON_EXE%" goto start_app

where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_EXE=py"
    goto start_app
)

where python >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_EXE=python"
    goto start_app
)

echo Python not found.
echo Install Python or create the virtual environment at .venv\Scripts\python.exe
pause
exit /b 1

:start_app
echo Starting TeleRAG...
"%PYTHON_EXE%" -m streamlit run app.py
if errorlevel 1 (
    echo.
    echo Failed to start Streamlit. Make sure dependencies are installed:
    echo   %PYTHON_EXE% -m pip install -r requirements.txt
    pause
    exit /b 1
)

endlocal
