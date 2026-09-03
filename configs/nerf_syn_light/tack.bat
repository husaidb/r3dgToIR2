@echo off
title Generate Camera Trajectory

:: =================================================================================
:: 配置区域: 如果您的Conda环境名称不是 "3dgs"，请在此处修改
set CONDA_ENV_NAME=3dgs
:: =================================================================================

echo Activating conda environment: %CONDA_ENV_NAME%...
call conda activate %CONDA_ENV_NAME%

IF %ERRORLEVEL% NEQ 0 (
    echo Failed to activate conda environment. Please check the environment name.
    pause
    exit /b
)

echo.
echo Running trajectory generation script...

python generate_trajectory.py ^
    --output configs/chair/trajectory.json ^
    --num_frames 300 ^
    --radius 4.0 ^
    --height 2.5 ^
    --target 0 0 0

echo.
echo Trajectory generation complete.
echo The file 'configs/chair/trajectory.json' has been updated.
echo.
echo Press any key to exit.
pause > nul