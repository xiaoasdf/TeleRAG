@echo off
setlocal

cd /d "%~dp0"

set "CONDA_ENV_PREFIX=D:\conda_envs\telerag"
set "CONDA_ENV_PYTHON=D:\conda_envs\telerag\python.exe"
set "PYTHON_EXE="
set "USE_CONDA_RUN="
set "DASHSCOPE_API_KEY=sk-7cd212bb3a944270ad231b89293126ab"

if defined CONDA_PREFIX (
    if exist "%CONDA_PREFIX%\python.exe" (
        set "PYTHON_EXE=%CONDA_PREFIX%\python.exe"
        goto start_app
    )
)

if exist "%CONDA_ENV_PYTHON%" (
    set "PYTHON_EXE=%CONDA_ENV_PYTHON%"
    goto start_app
)

if defined CONDA_EXE (
    set "USE_CONDA_RUN=1"
    goto start_app
)

where conda >nul 2>nul
if %errorlevel%==0 (
    set "USE_CONDA_RUN=1"
    goto start_app
)

where python >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_EXE=python"
    goto start_app
)

echo Python or Conda not found.
echo Create the Conda environment first:
echo   setup_conda.bat
echo Or manually:
echo   conda create --prefix %CONDA_ENV_PREFIX% python=3.10 -y
echo   conda activate %CONDA_ENV_PREFIX%
pause
exit /b 1

:start_app
if defined PYTHON_EXE (
    echo Using Python: %PYTHON_EXE%
) else (
    echo Using Conda prefix: %CONDA_ENV_PREFIX%
)
if not defined DASHSCOPE_API_KEY (
    echo DASHSCOPE_API_KEY is not set. DashScope mode will fail until you set it.
)
echo Starting TeleRAG...
if defined PYTHON_EXE (
    "%PYTHON_EXE%" -m streamlit run app.py
) else (
    call conda run --prefix "%CONDA_ENV_PREFIX%" python -m streamlit run app.py
)
if errorlevel 1 (
    echo.
    echo Failed to start Streamlit.
    echo If the Conda environment does not exist yet, run:
    echo   setup_conda.bat
    echo If dependencies are missing, run:
    if defined PYTHON_EXE (
        echo   %PYTHON_EXE% -m pip install -r requirements-managed.txt
    ) else (
        echo   conda run --prefix "%CONDA_ENV_PREFIX%" python -m pip install -r requirements-managed.txt
    )
    pause
    exit /b 1
)

endlocal
