#!/usr/bin/env bash
# DMS 一键启动脚本（macOS / Linux）
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# ---------------------------------------------------------
# 1) 虚拟环境检测
# ---------------------------------------------------------
if [ -d "venv" ]; then
    VENV_DIR="venv"
elif [ -d ".venv" ]; then
    VENV_DIR=".venv"
else
    echo "[setup] 未找到虚拟环境，正在创建 venv/"
    if ! command -v python3.11 >/dev/null 2>&1; then
        echo "[error] 未找到 python3.11。请先安装 Python 3.11"
        exit 1
    fi
    python3.11 -m venv venv
    VENV_DIR="venv"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ---------------------------------------------------------
# 2) 依赖检测
# ---------------------------------------------------------
NEED_INSTALL=0
python -c "import flask_socketio, ultralytics, mediapipe, PyQt6, pyqtgraph" 2>/dev/null || NEED_INSTALL=1
if [ "$NEED_INSTALL" -eq 1 ]; then
    echo "[setup] 安装依赖（首次运行需 1-2 分钟）..."
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# ---------------------------------------------------------
# 3) 模式选择
# ---------------------------------------------------------
MODE="${1:-}"
if [ -z "$MODE" ]; then
    echo "请选择启动模式："
    echo "  [1] Web Dashboard（推荐答辩演示）→ http://127.0.0.1:5050"
    echo "  [2] PyQt 桌面端"
    echo "  [3] 录制自有训练数据"
    echo "  [4] 训练新模型（v2 两阶段）"
    read -r -p "输入 1-4: " CHOICE
    case "$CHOICE" in
        1) MODE="web" ;;
        2) MODE="desktop" ;;
        3) MODE="record" ;;
        4) MODE="train" ;;
        *) echo "无效选择"; exit 1 ;;
    esac
fi

case "$MODE" in
    web)
        echo "[run] 启动 Web Dashboard → http://127.0.0.1:5050"
        python web_app.py
        ;;
    desktop)
        echo "[run] 启动 PyQt 桌面端"
        python main.py
        ;;
    record)
        echo "[run] 启动数据采集器"
        python scripts/record_my_domain_v2.py
        ;;
    train)
        echo "[run] 启动两阶段训练（前置检查：data/dms_v2_cls/ 是否已生成）"
        if [ ! -d "data/dms_v2_cls" ]; then
            echo "[setup] 数据集未构建，先运行 build_v2_dataset.py"
            python scripts/build_v2_dataset.py
        fi
        python scripts/train_dms_v2.py --stage all
        ;;
    *)
        echo "未知模式: $MODE"
        exit 1
        ;;
esac
