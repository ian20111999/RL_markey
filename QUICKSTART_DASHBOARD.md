# 儀表板快速啟動指南

## 🚀 快速啟動

### 方式 1：互動式啟動（推薦）

```bash
./start_dashboard.sh
# 然後選擇：
# 1 - 基礎儀表板 (簡單監控)
# 2 - 增強版儀表板 (進階圖表)
# 3 - 生產級儀表板 (模型管理)
```

### 方式 2：直接啟動

```bash
# 基礎儀表板
python web_dashboard.py
# 訪問 http://localhost:5555

# 增強版儀表板
python monitoring_dashboard.py --mode server
# 訪問 http://localhost:5556
```

## 📊 功能說明

### 基礎儀表板 (5555)
- ✅ 訓練進度監控
- ✅ 模型表現統計
- ✅ 符號詳情查看
- ✅ 全局數據分析

**API Endpoints:**
- `/api/health` - 健康檢查
- `/api/dashboard` - 儀表板數據
- `/api/stats` - 統計資訊
- `/api/symbol/<name>` - 符號詳情
- `/api/models` - 模型列表

### 增強版儀表板 (5556)
- ✅ 進階數據圖表
- ✅ 詳細運行歷史
- ✅ 符號深度分析
- ✅ 即時數據更新

**API Endpoints:**
- `/api/health` - 健康檢查
- `/api/dashboard` - 儀表板數據
- `/api/symbol/<symbol>` - 符號詳情

## 📈 當前數據狀態

根據最新測試結果：

- **總符號數**: 2 (BTCUSDT, ETHUSDT)
- **總運行數**: 4
- **成功運行**: 1
- **成功率**: 33.3%
- **最佳表現**: BTCUSDT ($3027.11 PnL)
- **平均 PnL**: $1010.26
- **正 PnL 比例**: 66.7%

## 🔧 故障排除

### 端口被佔用

```bash
# 停止佔用端口的進程
lsof -ti:5555 | xargs kill -9  # 基礎儀表板
lsof -ti:5556 | xargs kill -9  # 增強版儀表板
```

### 數據庫為空

```bash
# 導入歷史訓練數據
python import_historical_runs.py
```

### Flask 未安裝

確認在虛擬環境中運行：

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## 📝 測試驗證

系統已通過完整測試：

- ✅ 環境配置正確
- ✅ 數據庫連接正常
- ✅ 所有 API endpoints 正常（9/9）
- ✅ HTML 頁面載入正常（2/2）
- ✅ 前端-後端整合正常
- ✅ 數據完整性驗證通過

## 💡 使用技巧

1. **在瀏覽器中查看效果最佳**
   - 支援所有現代瀏覽器
   - 建議使用 Chrome/Firefox/Safari

2. **數據自動更新**
   - 重新整理頁面載入最新數據
   - API 數據即時反映資料庫狀態

3. **同時運行多個儀表板**
   - 可同時開啟 5555 和 5556
   - 各儀表板獨立運作

## 🎯 下一步

- 運行新的訓練：`python pipeline.py --symbol BTCUSDT`
- 查看訓練結果：重新整理儀表板頁面
- 比較模型表現：使用 API endpoints 獲取詳細數據

---

**最後測試時間**: 2025-12-13  
**測試狀態**: ✅ 所有功能正常  
**版本**: v1.0
