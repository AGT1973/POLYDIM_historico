@echo off
echo =======================================================================
echo  POLYDIM-CLA: Empirical Validation dependency Installer
echo =======================================================================
echo.
echo [1/3] Checking Python installation...
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found. Please install Python 3.10+ from: https://python.org
    echo Make sure to check the option "Add Python to PATH" during installation.
    pause
    exit /b 1
)
echo [OK] Python is installed.

echo.
echo [2/3] Installing Python libraries (numpy, tensorly, jax, jaxlib)...
python -m pip install --upgrade pip
python -m pip install numpy tensorly jax jaxlib
if %errorlevel% neq 0 (
    echo [WARNING] Some python dependencies failed to install.
    echo Please make sure you are connected to the Internet and run this script again.
) else (
    echo [OK] Python dependencies installed successfully.
)

echo.
echo [3/3] Checking Rust and Cargo compiler toolchain...
where cargo >nul 2>nul
if %errorlevel% neq 0 (
    echo [NOTE] 'cargo' was not found on your PATH.
    echo To build and execute the compilable Rust prototype, please install Rust from:
    echo https://rustup.rs
) else (
    echo [OK] 'cargo' detected. Rust tests are ready to be executed.
)

echo.
echo =======================================================================
echo  Setup Completed! Run 'run_all.bat' to execute the validation suite.
echo =======================================================================
pause
