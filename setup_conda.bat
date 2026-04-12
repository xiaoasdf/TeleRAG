@echo off
setlocal

cd /d "%~dp0"

set "CONDA_ENV_PREFIX=D:\conda_envs\telerag"
set "REQUIREMENTS_FILE=requirements-managed.txt"

where conda >nul 2>nul
if errorlevel 1 (
    echo Conda not found in PATH.
    echo Install Miniconda or Anaconda first, then reopen the terminal and rerun this script.
    pause
    exit /b 1
)

if not exist "D:\conda_envs" (
    mkdir "D:\conda_envs"
)

if exist "%CONDA_ENV_PREFIX%\python.exe" (
    echo Conda environment already exists at %CONDA_ENV_PREFIX%
) else (
    echo Creating Conda environment at %CONDA_ENV_PREFIX% ...
    call conda create --prefix "%CONDA_ENV_PREFIX%" python=3.10 -y
    if errorlevel 1 (
        echo Failed to create the Conda environment.
        pause
        exit /b 1
    )
)

echo Installing dependencies from %REQUIREMENTS_FILE% ...
call conda run --prefix "%CONDA_ENV_PREFIX%" python -m pip install -r "%REQUIREMENTS_FILE%"
if errorlevel 1 (
    echo Failed to install dependencies into %CONDA_ENV_PREFIX%.
    pause
    exit /b 1
)

echo.
echo Conda environment is ready:
echo   %CONDA_ENV_PREFIX%
echo.
echo Next steps:
echo   conda activate %CONDA_ENV_PREFIX%
echo   set DASHSCOPE_API_KEY=your_key
echo   run_app.bat
echo.
pause
endlocal
