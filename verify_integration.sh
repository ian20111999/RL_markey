#!/bin/bash
# 整合後功能驗證腳本

echo "🔍 開始驗證整合後的功能..."
echo ""

# 檢查 Python 版本
echo "1️⃣ 檢查 Python 環境..."
python --version
echo ""

# 檢查關鍵文件存在
echo "2️⃣ 檢查關鍵文件..."
files=(
    "pipeline.py"
    "production/cli.py"
    "production/api.py"
    "production/model_registry.py"
    "production/dashboard.py"
    "README.md"
    "QUICKSTART.md"
    "CONFIG_GUIDE.md"
    "DEVELOPMENT.md"
    "PROJECT_STRUCTURE.md"
    "Dockerfile"
    "docker-compose.yml"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file 缺失！"
    fi
done
echo ""

# 檢查依賴安裝
echo "3️⃣ 檢查關鍵依賴..."
python -c "import pandas; print('✅ pandas')" 2>/dev/null || echo "❌ pandas"
python -c "import numpy; print('✅ numpy')" 2>/dev/null || echo "❌ numpy"
python -c "import gymnasium; print('✅ gymnasium')" 2>/dev/null || echo "❌ gymnasium"
python -c "import stable_baselines3; print('✅ stable-baselines3')" 2>/dev/null || echo "❌ stable-baselines3"
python -c "import fastapi; print('✅ fastapi')" 2>/dev/null || echo "❌ fastapi"
python -c "import uvicorn; print('✅ uvicorn')" 2>/dev/null || echo "❌ uvicorn"
echo ""

# 檢查 production 模組導入
echo "4️⃣ 檢查 production 模組..."
python -c "from production import cli; print('✅ production.cli')" 2>/dev/null || echo "❌ production.cli"
python -c "from production import api; print('✅ production.api')" 2>/dev/null || echo "❌ production.api"
python -c "from production import model_registry; print('✅ production.model_registry')" 2>/dev/null || echo "❌ production.model_registry"
python -c "from production import dashboard; print('✅ production.dashboard')" 2>/dev/null || echo "❌ production.dashboard"
echo ""

# 檢查 CLI 工具
echo "5️⃣ 檢查 CLI 工具..."
if python production/cli.py --help > /dev/null 2>&1; then
    echo "✅ production CLI 正常運作"
else
    echo "❌ production CLI 有問題"
fi
echo ""

# 檢查 API 配置
echo "6️⃣ 檢查 API 配置..."
python -c "from production.api import app; print('✅ FastAPI app 可以導入')" 2>/dev/null || echo "❌ FastAPI app 導入失敗"
echo ""

# 檢查 Docker 文件
echo "7️⃣ 檢查 Docker 配置..."
if [ -f "Dockerfile" ] && [ -f "docker-compose.yml" ]; then
    echo "✅ Docker 配置文件存在"
    if command -v docker &> /dev/null; then
        echo "✅ Docker 已安裝"
        if docker ps &> /dev/null; then
            echo "✅ Docker daemon 正在運行"
        else
            echo "⚠️  Docker daemon 未運行"
        fi
    else
        echo "ℹ️  Docker 未安裝（可選）"
    fi
else
    echo "❌ Docker 配置文件缺失"
fi
echo ""

# 檢查文檔完整性
echo "8️⃣ 檢查文檔完整性..."
doc_count=$(find . -name "*.md" -type f | wc -l | tr -d ' ')
echo "📄 找到 $doc_count 個 Markdown 文檔"
echo ""

# 總結
echo "========================================="
echo "✅ 驗證完成！"
echo ""
echo "🚀 快速開始："
echo "   1. 基礎訓練: python pipeline.py --symbol btc"
echo "   2. 生產訓練: python production/cli.py train --symbol btc"
echo "   3. 啟動 API: python production/cli.py serve"
echo "   4. 查看面板: python production/dashboard.py"
echo ""
echo "📚 查看文檔："
echo "   - 快速開始: cat QUICKSTART.md"
echo "   - 整合總結: cat MERGE_SUMMARY.md"
echo "   - 完整說明: cat README.md"
echo "========================================="
