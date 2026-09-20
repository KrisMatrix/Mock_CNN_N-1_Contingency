@echo off
setlocal enabledelayedexpansion

echo ======================================================================
echo   N-1 CONTINGENCY RECOMMENDATION SYSTEM: FULL PIPELINE (CNN ^& GNN)
echo ======================================================================
echo.

rem Detect Python executable in virtual environment or fallback to system python
if exist env\Scripts\python.exe (
    set PYTHON_BIN=env\Scripts\python.exe
    echo Using Virtual Environment Python: !PYTHON_BIN!
) else (
    set PYTHON_BIN=python
    echo Using System Python: !PYTHON_BIN!
)

echo.
echo ======================================================================
echo [STEP 1/7] Generating 35-Bus Power System Grid ^& AUX Model File...
echo ======================================================================
%PYTHON_BIN% grid_generator.py
if %errorlevel% neq 0 (
    echo [ERROR] Step 1 failed with exit code %errorlevel%.
    exit /b %errorlevel%
)

echo.
echo ======================================================================
echo [STEP 2/7] Generating Shared 5,000-Sample Dynamic Topology Dataset...
echo ======================================================================
%PYTHON_BIN% data_generator.py
if %errorlevel% neq 0 (
    echo [ERROR] Step 2 failed with exit code %errorlevel%.
    exit /b %errorlevel%
)

echo.
echo ======================================================================
echo [STEP 3/7] Training 2D Convolutional Neural Network (CNN) Model...
echo ======================================================================
%PYTHON_BIN% train.py --epochs 25
if %errorlevel% neq 0 (
    echo [ERROR] Step 3 failed with exit code %errorlevel%.
    exit /b %errorlevel%
)

echo.
echo ======================================================================
echo [STEP 4/7] Evaluating ^& Visualizing 2D CNN Model...
echo ======================================================================
%PYTHON_BIN% evaluate.py
%PYTHON_BIN% visualize.py
if %errorlevel% neq 0 (
    echo [ERROR] Step 4 failed with exit code %errorlevel%.
    exit /b %errorlevel%
)

echo.
echo ======================================================================
echo [STEP 5/7] Training Graph Neural Network (GNN) Model...
echo ======================================================================
%PYTHON_BIN% train_gnn.py --epochs 25
if %errorlevel% neq 0 (
    echo [ERROR] Step 5 failed with exit code %errorlevel%.
    exit /b %errorlevel%
)

echo.
echo ======================================================================
echo [STEP 6/7] Evaluating ^& Visualizing Graph Neural Network (GNN) Model...
echo ======================================================================
%PYTHON_BIN% evaluate_gnn.py
%PYTHON_BIN% visualize_gnn.py
if %errorlevel% neq 0 (
    echo [ERROR] Step 6 failed with exit code %errorlevel%.
    exit /b %errorlevel%
)

echo.
echo ======================================================================
echo [STEP 7/7] Running Comparative Benchmarking (2D CNN vs GNN)...
echo ======================================================================
%PYTHON_BIN% compare_models.py
if %errorlevel% neq 0 (
    echo [ERROR] Step 7 failed with exit code %errorlevel%.
    exit /b %errorlevel%
)

echo.
echo ======================================================================
echo   FULL PIPELINE EXECUTED SUCCESSFULLY!
echo   Shared Dataset: data/dataset.npz
echo   CNN Metrics:    data/test_evaluation.json
echo   GNN Metrics:    data/gnn_test_evaluation.json
echo   All Plots:      plots/
echo ======================================================================
echo.
pause
