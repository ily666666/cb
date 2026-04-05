#!/bin/bash
# 云边端协同计算框架 Web 应用启动脚本
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================"
echo "  云边端协同计算框架 Web 应用"
echo "========================================"
echo "项目根目录: $PROJECT_ROOT"

# 安装后端依赖
echo ""
echo "[1/3] 安装后端依赖..."
pip install -r "$SCRIPT_DIR/backend/requirements.txt" -q

# 安装前端依赖 & 构建
if [ -d "$SCRIPT_DIR/frontend/node_modules" ]; then
    echo "[2/3] 前端依赖已存在，跳过安装"
else
    echo "[2/3] 安装前端依赖..."
    cd "$SCRIPT_DIR/frontend"
    npm install
    cd "$SCRIPT_DIR"
fi

# 启动后端
echo "[3/3] 启动后端服务 (端口 8000)..."
echo ""
echo "  后端 API:  http://localhost:8000/docs"
echo "  前端开发:  cd web/frontend && npm run dev"
echo ""

cd "$SCRIPT_DIR/backend"
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
