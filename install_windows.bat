@echo off
setlocal enabledelayedexpansion

rem install_windows.bat - Braingraph Pipeline Environment Setup
rem Author: Braingraph Pipeline Team
rem Description: Creates virtual environment and installs all required packages for the braingraph pipeline

echo.
echo ================================================================================
echo.
echo    🧠 BRAINGRAPH PIPELINE ENVIRONMENT SETUP
echo.
echo    Setting up Python environment and installing all required packages
echo.
echo ================================================================================
echo.

rem Check prerequisites
echo 🔍 Checking prerequisites...

rem Check if uv is installed
uv --version >nul 2>&1
if errorlevel 1 (
    echo ❌ uv is not installed. Please install uv first:
    echo    Visit: https://github.com/astral-sh/uv
    echo    Or run: pip install uv
    pause
    exit /b 1
) else (
    echo ✅ uv is already installed
)

rem Remove existing virtual environment if it exists
if exist "braingraph_pipeline" (
    echo 🗑️  Removing existing virtual environment...
    rmdir /s /q braingraph_pipeline
)

rem Create virtual environment
echo 📦 Creating virtual environment 'braingraph_pipeline'...
uv venv braingraph_pipeline --python 3.10

rem Activate the virtual environment
echo 🔧 Activating virtual environment...
call braingraph_pipeline\Scripts\activate.bat

rem Install core Python packages
echo 📚 Installing core Python packages...
uv pip install ^
    "numpy>=1.20.0,<2.0.0" ^
    "pandas>=1.5.0" ^
    "matplotlib>=3.5.0" ^
    "seaborn>=0.11.0" ^
    "scipy>=1.8.0" ^
    "scikit-learn>=1.2.0" ^
    "tqdm>=4.60.0" ^
    "jsonschema>=4.0.0" ^
    "pathlib2>=2.3.0" ^
    "typing-extensions>=4.0.0"

rem Install statistical analysis packages
echo 📊 Installing statistical analysis packages...
uv pip install ^
    "statsmodels>=0.13.0" ^
    "pingouin>=0.5.0" ^
    "scikit-posthocs>=0.7.0"

rem Install graph/network analysis packages
echo 🕸️ Installing graph and network analysis packages...
uv pip install ^
    "networkx>=2.8" ^
    "igraph>=0.10.0" ^
    "bctpy>=0.5.2" ^
    "python-louvain>=0.16"

rem Install neuroimaging packages
echo 🧠 Installing neuroimaging packages...
uv pip install ^
    "nilearn>=0.10.0" ^
    "nibabel>=4.0.0" ^
    "dipy>=1.5.0"

rem Install additional analysis packages
echo 📈 Installing additional analysis packages...
uv pip install ^
    "plotly>=5.0.0" ^
    "dash>=2.0.0" ^
    "jupyter>=1.0.0" ^
    "ipywidgets>=8.0.0" ^
    "openpyxl>=3.0.0" ^
    "xlsxwriter>=3.0.0" ^
    "h5py>=3.0.0"

echo.
echo ✅ Package installation completed successfully!
echo.
echo 🎯 Environment Summary:
echo • Virtual environment: braingraph_pipeline\
echo • Python version: 3.10
echo • All required packages installed
echo.
echo 📋 To activate the environment:
echo   braingraph_pipeline\Scripts\activate.bat
echo.
echo 📋 To deactivate the environment:
echo   deactivate
echo.
echo 🚀 Environment ready for braingraph pipeline!
echo.
pause