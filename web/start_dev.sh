#!/bin/bash
# 开发模式：同时启动前端和后端
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo "  云边端协同计算框架 - 开发模式"
echo "========================================"

# 安装依赖
pip install -r "$SCRIPT_DIR/backend/requirements.txt" -q

if [ ! -d "$SCRIPT_DIR/frontend/node_modules" ]; then
    echo "安装前端依赖..."
    cd "$SCRIPT_DIR/frontend" && npm install && cd "$SCRIPT_DIR"
fi

echo ""
echo "  后端 API:  http://localhost:8000"
echo "  前端页面:  http://localhost:3000"
echo "  API 文档:  http://localhost:8000/docs"
echo ""

# 后台启动后端
cd "$SCRIPT_DIR/backend"
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# 前台启动前端
cd "$SCRIPT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
