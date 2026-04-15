@echo off
echo ============================================
echo AI 照片选片系统 - 滑动窗口优化版
echo ============================================
echo.
echo 正在启动应用...
echo.

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

REM 检查依赖
echo 检查依赖项...
python -c "import flask, torch, cv2, numpy, PIL, sqlite3" >nul 2>&1
if errorlevel 1 (
    echo 正在安装依赖项...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo 依赖项安装失败
        pause
        exit /b 1
    )
)

REM 启动应用
echo.
echo 启动应用...
echo 访问地址: http://localhost:5000
echo 按 Ctrl+C 停止应用
echo.
python app.py

pause