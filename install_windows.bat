@echo off
setlocal enabledelayedexpansion

rem install_windows.bat - Braingraph Pipeline Environment Setup
rem Author: Braingraph Pipeline Team
rem Description: Creates virtual environment and installs all required packages for the braingraph pipeline

echo.
echo ================================================================================
echo.
echo    BRAINGRAPH PIPELINE ENVIRONMENT SETUP
echo.
echo    Setting up Python environment and installing all required packages
echo.
echo ================================================================================
echo.

rem Check prerequisites
echo Checking prerequisites...

rem Check if uv is installed
uv --version >nul 2>&1
if errorlevel 1 (
    echo uv was not found on PATH. Attempting to install via pip...
    py -m pip install --upgrade pip >nul 2>&1
    py -m pip install uv >nul 2>&1
    if errorlevel 1 (
        echo Failed to install uv automatically.
        echo Please install uv manually from https://github.com/astral-sh/uv and rerun this script.
        pause
        exit /b 1
    ) else (
        rem Ensure the newly installed uv is on PATH for this session
        for /f "delims=" %%I in ('py -m uv --version 2^>nul') do set UV_VERSION=%%I
    )
) else (
    echo uv is already installed
)

rem Remove existing virtual environment if it exists
if exist "braingraph_pipeline" (
    echo Removing existing virtual environment...
    rmdir /s /q braingraph_pipeline
)

rem Create virtual environment
echo Creating virtual environment "braingraph_pipeline"...
uv venv braingraph_pipeline --python 3.10

rem Activate the virtual environment
echo Activating virtual environment...
call braingraph_pipeline\Scripts\activate.bat

echo Installing OptiConn and dependencies (editable, with dev extras)...
uv pip install -e ".[dev]"

echo.
echo Package installation completed successfully!
echo.
echo Environment Summary:
echo • Virtual environment: braingraph_pipeline\
echo • Python version: 3.10
echo • OptiConn installed in editable mode with dev extras
echo.
echo To activate the environment:
echo   braingraph_pipeline\Scripts\activate.bat
echo.
echo To deactivate the environment:
echo   deactivate
echo.
echo Environment ready for braingraph pipeline!
echo.
pause