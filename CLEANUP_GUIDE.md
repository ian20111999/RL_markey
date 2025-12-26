# 🧹 專案清理建議

## ✅ 已完成的清理

### 刪除的檔案
- ✅ `README.md.backup` - 舊的備份檔案
- ✅ `INTEGRATION_COMPLETE.txt` - 整合完成報告（454 行）
- ✅ `docs/` - 整個文件目錄（已合併到主 README）
  - `docs/archive/ARCHITECTURE.txt`
  - `docs/reports/` (空目錄)

### 優化的配置
- ✅ 更新 `.gitignore` - 排除大型檔案
- ✅ 新增 `data/.gitkeep` - 保持目錄結構
- ✅ 新增 `data/README.md` - 資料目錄說明
- ✅ 新增 `PROJECT_STRUCTURE.md` - 詳細結構文件

## 🎯 CSV 資料檔案處理建議

### 當前狀況
```
data/
├── btc_usdt_1m_2023.csv    (122 MB)
└── ethusdt_usdt_1m_2023.csv (27 MB)
```

### 選項 1: 保留 CSV（推薦）
**優點：**
- 快速開發和測試
- 無需每次從資料庫載入
- 可以快速驗證資料品質

**缺點：**
- 佔用磁碟空間
- Git 已忽略，不會上傳

**適用場景：**
- 本地開發環境
- 快速原型測試

### 選項 2: 刪除 CSV，僅用資料庫
**優點：**
- 節省磁碟空間（~150 MB）
- 統一資料來源
- 符合生產環境架構

**缺點：**
- 需要先將資料匯入 PostgreSQL
- 離線開發需要運行 Docker

**適用場景：**
- 生產環境
- 已完成開發階段

### 選項 3: 混合模式（當前狀態）
**做法：**
- 保留 CSV 用於快速測試
- 主要訓練從資料庫讀取
- 定期同步 CSV → PostgreSQL

**執行：**
```bash
# 將 CSV 資料匯入資料庫
python scripts/import_csv_to_db.py --symbol BTCUSDT
python scripts/import_csv_to_db.py --symbol ETHUSDT

# 驗證匯入
docker-compose exec postgres psql -U rl_user -d rl_market -c "
SELECT symbol, COUNT(*) as records 
FROM market_data md
JOIN symbols s ON md.symbol_id = s.symbol_id
GROUP BY symbol;"
```

## 📊 建議的清理策略

### 立即可執行
```bash
# 1. 清理 Python 快取
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# 2. 清理 pytest 快取
rm -rf .pytest_cache/

# 3. 清理空目錄
rmdir plots/ 2>/dev/null
```

### 定期執行（每週）
```bash
# 清理舊訓練記錄（保留最近 10 次）
cd runs/
ls -t | tail -n +11 | xargs -I {} rm -rf {}

# 清理舊日誌（保留 7 天內）
find logs/ -name "*.log" -mtime +7 -delete
```

### 生產環境前
```bash
# 1. 刪除開發用 CSV（已匯入資料庫）
rm data/*.csv

# 2. 清理所有訓練記錄
rm -rf runs/run_*

# 3. 僅保留最佳模型
cd models/
rm -f *.zip
# 手動保留: btc_best_model.zip, eth_best_model.zip

# 4. 清理 SQLite（改用 PostgreSQL）
rm logs/metrics.db
```

## 🗂️ 根目錄腳本整理建議

### 當前狀況（根目錄 Python 腳本）
```
根目錄:
├── pipeline.py              (基礎訓練)
├── auto_pipeline.py         (自動訓練)
├── integrated_pipeline.py   (整合訓練)
├── monitoring_dashboard.py  (Streamlit)
├── web_dashboard.py         (Flask)
├── examples.py              (使用範例)
├── start_pipeline.py        (啟動器)
├── test_dashboards.py       (測試)
└── import_historical_runs.py (匯入工具)
```

### 建議的組織方式

#### 選項 A: 保持現狀（推薦）
**理由：**
- 主要執行檔案在根目錄，方便執行
- 符合 Python 專案慣例
- `python pipeline.py` 比 `python scripts/pipeline.py` 更直觀

#### 選項 B: 移動到子目錄
```bash
# 創建 bin/ 目錄
mkdir -p bin/

# 移動主要腳本
mv pipeline.py auto_pipeline.py integrated_pipeline.py bin/
mv monitoring_dashboard.py web_dashboard.py bin/

# 移動輔助腳本到 scripts/
mv examples.py start_pipeline.py test_dashboards.py scripts/
mv import_historical_runs.py scripts/
```

**優點：**
- 根目錄更乾淨
- 腳本分類更清楚

**缺點：**
- 需要更新所有文件中的路徑
- 執行命令變長
- 可能破壞現有的 import 路徑

## 🎯 最終建議

### 立即執行（無風險）
```bash
# 清理快取和臨時檔案
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
rm -rf .pytest_cache/
```

### 可選執行（根據需求）

#### 如果磁碟空間充足
```bash
# 保留一切，僅清理日誌
find logs/ -name "*.log" -mtime +30 -delete
```

#### 如果需要節省空間
```bash
# 1. 確保資料已在 PostgreSQL
docker-compose exec postgres psql -U rl_user -d rl_market -c "\dt"

# 2. 備份 CSV（可選）
tar -czf data_backup_$(date +%Y%m%d).tar.gz data/*.csv

# 3. 刪除 CSV
rm data/*.csv

# 4. 清理舊訓練記錄
rm -rf runs/run_btc_* runs/run_eth_*
```

## 📝 總結

### 當前專案狀態
- ✅ **非常乾淨** - 只有 1 個 README.md
- ✅ **結構清晰** - 有 PROJECT_STRUCTURE.md 詳細說明
- ✅ **Docker 整合** - 包含 pgAdmin，統一管理
- ✅ **資料庫完整** - PostgreSQL Schema 2.0 運行良好

### 進一步優化空間
1. **CSV 檔案**: 根據使用場景決定保留或刪除
2. **訓練記錄**: 定期清理 `runs/` 目錄
3. **模型檔案**: 僅保留最佳模型，刪除中間版本
4. **日誌檔案**: 設定自動清理策略

### 不需要改動
- ✅ 根目錄腳本組織 - 當前結構很好
- ✅ 目錄結構 - 已經很清晰
- ✅ 文件組織 - README + PROJECT_STRUCTURE 已足夠
