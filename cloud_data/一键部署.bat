@echo off
REM DeepCode-Analyst 0.5B学生模型一键部署脚本

echo ================================================
echo  DeepCode-Analyst 0.5B 学生模型一键部署
echo ================================================
echo.

REM 获取当前目录
set SCRIPT_DIR=%~dp0

echo 🎯 目标: 部署知识蒸馏的0.5B学生模型到本地
echo 📁 部署目录: %SCRIPT_DIR%
echo.

echo 🔍 检查文件结构...
if exist "%SCRIPT_DIR%models\student_0.5b\lora-micro-distilled\adapter_config.json" (
    echo ✅ LoRA适配器文件存在
) else (
    echo ❌ LoRA适配器文件不存在
    echo 💡 请先运行下载脚本获取模型文件
    echo.
    echo 下载命令:
    echo scp -P 18812 -r root@connect.bjb1.seetacloud.com:/root/autodl-tmp/models/lora-micro-distilled/ "%SCRIPT_DIR%models\student_0.5b\"
    echo.
    pause
    exit /b 1
)

echo.
echo 🚀 选择部署方式:
echo 1. 🔧 安装依赖 + 环境检查
echo 2. 💻 命令行部署 (测试和交互)
echo 3. 🌐 Web界面部署 (推荐日常使用)
echo 4. 📖 查看完整部署指南
echo 0. ❌ 退出
echo.

set /p choice="请选择 (0-4): "

if "%choice%"=="1" (
    echo.
    echo 🔧 开始安装依赖...
    python "%SCRIPT_DIR%install_dependencies.py"
    echo.
    echo ✅ 依赖安装完成!
    echo 💡 现在可以选择运行方式 2 或 3
    echo.
    pause
    goto :start

) else if "%choice%"=="2" (
    echo.
    echo 💻 启动命令行部署...
    python "%SCRIPT_DIR%scripts\部署学生模型.py"
    echo.
    pause

) else if "%choice%"=="3" (
    echo.
    echo 🌐 启动Web界面...
    echo 📍 访问地址: http://localhost:7860
    echo 🔄 正在启动，请稍候...
    python "%SCRIPT_DIR%scripts\web_interface.py"
    echo.
    pause

) else if "%choice%"=="4" (
    echo.
    echo 📖 打开部署指南...
    start "" "%SCRIPT_DIR%本地部署完整指南.md"
    echo.
    pause

) else if "%choice%"=="0" (
    echo.
    echo 👋 再见!
    exit /b 0

) else (
    echo.
    echo ❌ 无效选择，请重试
    echo.
    pause
    goto :start
)

:start
echo.
echo 🔄 返回主菜单...
goto :eof
