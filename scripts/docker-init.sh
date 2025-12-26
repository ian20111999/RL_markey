#!/bin/bash
# Docker 初始化腳本

set -e

echo "================================"
echo "RL Market Making - Docker 初始化"
echo "================================"
echo ""

# 檢查 Docker 和 Docker Compose
if ! command -v docker &> /dev/null; then
    echo "錯誤: 找不到 docker 命令"
    echo "請先安裝 Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "錯誤: 找不到 docker-compose 命令"
    echo "請先安裝 Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# 使用 docker compose (V2) 或 docker-compose (V1)
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

echo "使用命令: $DOCKER_COMPOSE"
echo ""

# 檢查 .env 檔案
if [ ! -f .env ]; then
    echo "⚠️  找不到 .env 檔案，從 .env.example 複製..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✅ 已建立 .env 檔案"
        echo "⚠️  請編輯 .env 檔案設定資料庫密碼等參數"
        echo ""
    else
        echo "❌ 找不到 .env.example 檔案"
        exit 1
    fi
fi

# 載入環境變數
export $(grep -v '^#' .env | xargs)

echo "環境變數設定:"
echo "  - DB_TYPE: ${DB_TYPE:-sqlite}"
echo "  - POSTGRES_DB: ${POSTGRES_DB:-rl_market}"
echo "  - POSTGRES_USER: ${POSTGRES_USER:-rl_user}"
echo ""

# 建立必要的目錄
echo "建立必要的目錄..."
mkdir -p data models runs logs plots backups configs production/templates frontend/dashboards
echo "✅ 目錄已建立"
echo ""

# 停止並移除舊容器
echo "清理舊容器..."
$DOCKER_COMPOSE down -v
echo "✅ 舊容器已清理"
echo ""

# 建構映像檔
echo "建構 Docker 映像檔..."
$DOCKER_COMPOSE build
echo "✅ 映像檔建構完成"
echo ""

# 啟動 PostgreSQL
echo "啟動 PostgreSQL 資料庫..."
$DOCKER_COMPOSE up -d postgres
echo "等待 PostgreSQL 準備就緒..."
sleep 10
echo "✅ PostgreSQL 已啟動"
echo ""

# 檢查是否需要遷移資料
if [ -f "logs/metrics.db" ]; then
    echo "⚠️  發現現有的 SQLite 資料庫"
    echo "是否要將資料遷移到 PostgreSQL? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "開始資料遷移..."
        export DB_TYPE=postgresql
        python scripts/migrate_db.py
        echo "✅ 資料遷移完成"
    else
        echo "跳過資料遷移"
    fi
    echo ""
fi

# 啟動所有服務
echo "啟動所有服務..."
$DOCKER_COMPOSE up -d
echo "✅ 所有服務已啟動"
echo ""

# 顯示服務狀態
echo "服務狀態:"
$DOCKER_COMPOSE ps
echo ""

# 顯示訪問資訊
echo "================================"
echo "✅ 初始化完成！"
echo "================================"
echo ""
echo "Dashboard 訪問地址:"
echo "  - Basic Dashboard:      http://localhost:5555"
echo "  - Enhanced Dashboard:   http://localhost:5556"
echo "  - Production Dashboard: http://localhost:8080"
echo ""
echo "資料庫連接資訊:"
echo "  - Host: localhost"
echo "  - Port: ${POSTGRES_PORT:-5432}"
echo "  - Database: ${POSTGRES_DB:-rl_market}"
echo "  - User: ${POSTGRES_USER:-rl_user}"
echo ""
echo "常用命令:"
echo "  - 查看日誌: $DOCKER_COMPOSE logs -f"
echo "  - 停止服務: $DOCKER_COMPOSE down"
echo "  - 重啟服務: $DOCKER_COMPOSE restart"
echo "  - 執行訓練: $DOCKER_COMPOSE --profile training up training"
echo ""
