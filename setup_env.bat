@echo off
echo ==============================================
echo  Creating conda env: pybullet_env (Python 3.10)
echo ==============================================

call conda create -n pybullet_env python=3.10 -y
if %errorlevel% neq 0 (
    echo [ERROR] conda create failed. Is conda installed?
    pause & exit /b 1
)

call conda activate pybullet_env
if %errorlevel% neq 0 (
    echo [ERROR] conda activate failed.
    pause & exit /b 1
)

pip install pybullet numpy
if %errorlevel% neq 0 (
    echo [ERROR] pip install failed.
    pause & exit /b 1
)

echo.
echo ==============================================
echo  Done! To run the simulation:
echo    conda activate pybullet_env
echo    python biped_sim.py
echo ==============================================
pause
