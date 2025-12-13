#!/bin/bash
# 儀表板啟動腳本

echo "🚀 RL Market Making - 儀表板啟動器"
echo "=================================="
echo ""
echo "選擇要啟動的儀表板:"
echo ""
echo "  1. 基礎儀表板      (端口 5555) - 簡單監控"
echo "  2. 增強版儀表板    (端口 5556) - 進階圖表"
echo "  3. 生產級儀表板    (端口 8080) - 模型管理"
echo ""
read -p "請輸入選項 [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "✅ 啟動基礎儀表板..."
        echo "📍 訪問: http://localhost:5555"
        echo ""
        python web_dashboard.py
        ;;
    2)
        echo ""
        echo "✅ 啟動增強版儀表板..."
        echo "📍 訪問: http://localhost:5556"
        echo ""
        python monitoring_dashboard.py --server
        ;;
    3)
        echo ""
        echo "✅ 啟動生產級儀表板..."
        echo "📍 訪問: http://localhost:8080"
        echo ""
        python production/dashboard.py
        ;;
    *)
        echo "❌ 無效選項"
        exit 1
        ;;
esac
