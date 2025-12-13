# 🚀 RL Market Making - Integrated Training Pipeline

## 📋 概述

這是一個全自動化的強化學習市場做市訓練系統，可以：

✅ **自動判斷環境品質** - 數據質量檢查、參數自動調整  
✅ **自動重訓練** - 結果不佳時自動重試，直到獲得生產級模型  
✅ **多幣種支援** - 可同時訓練多個加密貨幣或股票  
✅ **實時監控** - Web 儀表板追蹤訓練進度  
✅ **生產就緒** - 自動驗證模型可靠性，只保存可實際部署的模型  

## 🎯 核心功能

### 1. 全自動化訓練流程
- 自動數據質量驗證
- 智能參數調整（基於價格、波動率等）
- 環境健康檢查
- 訓練結果驗證（PnL、勝率、Sharpe等）
- 生產就緒檢查（更嚴格的標準）
- 失敗自動重試

### 2. 多幣種管理
- 單一幣種訓練
- 批量多幣種訓練
- 自動發現數據並訓練

### 3. 實時監控儀表板
- 訓練進度追蹤
- 效能指標可視化
- 多幣種對比
- 可用模型管理

## 🚀 快速開始

### 安裝依賴

```bash
pip install -r requirements.txt
```

### 單幣種訓練（最簡單）

```bash
# 訓練單一幣種（例如 BTC）
python integrated_pipeline.py --symbol btc

# 自定義重試次數
python integrated_pipeline.py --symbol eth --retries 10
```

### 多幣種訓練

```bash
# 訓練多個幣種
python integrated_pipeline.py --symbols btc eth bnb sol ada

# 自動發現所有可用數據並訓練
python integrated_pipeline.py --auto-discover
```

### 啟動監控儀表板

```bash
# 終端機模式（查看當前狀態）
python monitoring_dashboard.py

# 啟動 Web API 服務器
python web_dashboard.py --host 0.0.0.0 --port 5000

# 然後在瀏覽器打開
# http://localhost:5000
# 或打開 dashboard_enhanced.html
```

## 📊 訓練流程詳解

### 自動化流程步驟

```
1. 數據驗證
   ├─ 檢查數據完整性（最少樣本數、缺失率）
   ├─ 計算統計指標（價格、波動率、收益率）
   └─ 如果不合格 → 停止並報錯

2. 智能參數調整
   ├─ 基於平均價格調整 base_spread
   ├─ 基於價格計算合理的 initial_cash
   ├─ 動態調整 reward_scale
   └─ 環境健康檢查

3. 訓練循環（最多 N 次重試）
   ├─ 創建適應性配置
   ├─ 執行訓練
   ├─ 評估結果
   ├─ 驗證是否滿足標準
   │   ├─ 基本標準：PnL > 0, 勝率 > 45%
   │   └─ 生產標準：Sharpe > 0.5, 勝率 > 48%, 利潤因子 > 1.2
   └─ 如果不合格 → 重試（不同隨機種子）

4. 保存最佳模型
   ├─ 保存到 models/ 目錄
   ├─ 記錄到數據庫
   └─ 標記為生產就緒（如果通過所有檢查）
```

### 質量標準

#### 訓練接受標準（Training Acceptance）
- ✅ PnL ≥ 0
- ✅ 勝率 ≥ 45%
- ✅ Sharpe ≥ 0
- ✅ 最大回撤 ≤ 30%
- ✅ 綜合評分 ≥ 60

#### 生產就緒標準（Production Ready）
- 🔒 Sharpe ≥ 0.5
- 🔒 勝率 ≥ 48%
- 🔒 最大回撤 ≤ 25%
- 🔒 最少交易數 ≥ 50
- 🔒 利潤因子 ≥ 1.2
- 🔒 波動率比率 ≤ 2.0

## ⚙️ 配置說明

主要配置文件：`configs/pipeline_config.yaml`

### 關鍵配置項

```yaml
pipeline:
  max_retries: 5              # 最大重試次數
  total_timesteps: 200000     # 訓練步數
  early_stopping: true        # 找到好結果後提前停止
  cleanup_failed_runs: true   # 清理失敗的訓練記錄

auto_tuning:
  enabled: true               # 啟用自動參數調整
  spread_ratio: 0.0005        # 價差比率（0.05% = 5 bps）
  cash_multiplier: 10.0       # 初始資金乘數

training_acceptance:
  min_pnl: 0
  min_win_rate: 0.45
  min_sharpe: 0.0
  
production_readiness:
  min_sharpe: 0.5
  min_win_rate: 0.48
  min_profit_factor: 1.2
```

## 📁 項目結構

```
RL_markey/
├── integrated_pipeline.py       # 🆕 整合式多幣種訓練主程序
├── auto_pipeline.py             # 自動化訓練流程（單幣種）
├── web_dashboard.py             # 🆕 Web API 服務器
├── monitoring_dashboard.py      # 終端監控工具
├── dashboard_enhanced.html      # 🆕 增強版 Web 儀表板
├── dashboard.html               # 原始 Web 儀表板
│
├── configs/
│   ├── pipeline_config.yaml    # 流程配置
│   └── default.yaml             # 環境默認配置
│
├── models/                      # 訓練好的模型（生產就緒）
│   ├── btc_best_model.zip
│   ├── btc_best_config.yaml
│   └── ...
│
├── logs/
│   ├── metrics.db              # 訓練指標數據庫
│   ├── metrics.json            # JSON 格式指標
│   └── pipeline/               # 訓練日誌
│
├── data/                        # 訓練數據
├── runs/                        # 訓練運行記錄
├── scripts/                     # 工具腳本
└── utils/                       # 工具模組
    ├── validators.py           # 質量驗證器
    ├── production_checker.py   # 生產就緒檢查器
    └── metrics_db.py           # 指標數據庫
```

## 🎮 使用場景

### 場景 1：首次訓練新幣種

```bash
# 1. 準備數據（如果還沒有）
python scripts/fetch_data.py --symbol BTCUSDT --interval 1m --year 2023

# 2. 一鍵訓練
python integrated_pipeline.py --symbol btc

# 3. 查看結果
python monitoring_dashboard.py

# 4. 如果成功，模型會自動保存到 models/btc_best_model.zip
```

### 場景 2：批量訓練多個幣種

```bash
# 訓練主流幣種
python integrated_pipeline.py --symbols btc eth bnb sol ada dot avax

# 系統會依序訓練每個幣種，自動重試失敗的
```

### 場景 3：持續監控與再訓練

```bash
# 終端 1：啟動 Web API
python web_dashboard.py

# 終端 2：定期訓練新數據
python integrated_pipeline.py --auto-discover

# 瀏覽器：打開 dashboard_enhanced.html 監控
```

### 場景 4：模型效能對比

```bash
# 1. 啟動 Web 儀表板
python web_dashboard.py

# 2. 打開 dashboard_enhanced.html

# 3. 使用「概覽」標籤對比所有幣種
# 4. 使用「幣種詳情」標籤查看單一幣種歷史
```

## 📊 儀表板功能

### Web 儀表板 (dashboard_enhanced.html)

#### 概覽標籤
- 所有幣種的訓練狀態一覽
- 成功率、最佳 PnL、評分對比
- 快速識別哪些幣種有可用模型

#### 幣種詳情標籤
- 選擇特定幣種查看詳細信息
- 訓練歷史趨勢
- 統計指標（平均 PnL、最佳評分等）

#### 訓練記錄標籤
- 最近的所有訓練記錄
- 實時狀態更新
- 失敗原因追蹤

#### 可用模型標籤
- 所有生產就緒的模型列表
- 模型路徑、效能指標
- 創建時間

## 🔧 進階配置

### 調整訓練標準

編輯 `configs/pipeline_config.yaml`：

```yaml
# 如果你想要更嚴格的標準
training_acceptance:
  min_pnl: 100           # 至少賺 $100
  min_win_rate: 0.50     # 至少 50% 勝率
  min_sharpe: 0.3        # 至少 0.3 Sharpe
  
production_readiness:
  min_sharpe: 1.0        # 更高的 Sharpe 要求
  min_win_rate: 0.52     # 更高的勝率
```

### 自定義參數調整策略

編輯 `configs/pipeline_config.yaml`：

```yaml
auto_tuning:
  enabled: true
  spread_ratio: 0.001      # 使用更大的價差（0.1%）
  cash_multiplier: 20.0    # 使用更多初始資金
```

### 增加訓練時間

```yaml
pipeline:
  total_timesteps: 500000  # 從 200k 增加到 500k
  training_timeout: 7200   # 2 小時超時
```

## 📈 效能優化建議

### 1. 數據準備
- 使用高質量、完整的數據
- 確保數據涵蓋不同市場條件
- 定期更新數據

### 2. 訓練策略
- 從少量重試開始（3-5次）
- 觀察結果後調整標準
- 使用早停以節省時間

### 3. 硬體資源
- GPU 加速（如果可用）
- 足夠的內存（建議 8GB+）
- SSD 硬碟以加快 I/O

## 🐛 故障排除

### 問題 1：訓練總是失敗

**原因：** 標準太嚴格或數據質量差

**解決：**
1. 檢查數據質量：`python -c "from auto_pipeline import *; pipeline = AutomatedPipeline(); pipeline.analyze_data(Path('data/your_data.csv'))"`
2. 降低標準：編輯 `pipeline_config.yaml` 中的 `training_acceptance`
3. 增加重試次數：`--retries 10`

### 問題 2：Web 儀表板顯示錯誤

**原因：** API 服務器未啟動

**解決：**
```bash
# 確保安裝 Flask
pip install flask flask-cors

# 啟動服務器
python web_dashboard.py
```

### 問題 3：模型效能不理想

**原因：** 超參數需要調整

**解決：**
1. 增加訓練步數：`total_timesteps: 500000`
2. 調整自動調參：修改 `spread_ratio`, `cash_multiplier`
3. 使用超參數搜索：`python scripts/hyperparameter_search.py`

### 問題 4：記憶體不足

**原因：** 並行訓練或數據太大

**解決：**
1. 不使用 `--parallel` 選項
2. 減少 `eval_episodes` 數量
3. 分批訓練幣種

## 🎯 最佳實踐

1. **從小開始** - 先用一個幣種測試流程
2. **驗證數據** - 確保數據質量後再批量訓練
3. **監控進度** - 使用 Web 儀表板追蹤效能
4. **定期備份** - 保存 `models/` 和 `logs/` 目錄
5. **記錄實驗** - 保存成功的配置文件
6. **生產測試** - 部署前先用歷史數據回測

## 🚀 部署到生產

找到滿意的模型後：

```bash
# 1. 確認模型位置
ls -lh models/btc_best_model.zip

# 2. 檢查模型指標
python monitoring_dashboard.py

# 3. 回測驗證（使用新數據）
python scripts/evaluate.py --model models/btc_best_model.zip --data data/btc_test.csv

# 4. 部署到生產環境
# 將模型和配置複製到生產服務器
# 使用生產環境的交易接口連接
```

## 📞 支援與貢獻

- 發現 Bug？請提交 Issue
- 有改進建議？歡迎 Pull Request
- 需要幫助？查看文檔或聯繫團隊

## 📝 版本歷史

### v2.0 - 整合式流水線
- ✨ 新增多幣種批量訓練
- ✨ 增強版 Web 儀表板
- ✨ 自動發現數據功能
- ✨ 改進質量檢查系統

### v1.0 - 自動化流程
- ✅ 自動參數調整
- ✅ 訓練結果驗證
- ✅ 生產就緒檢查
- ✅ 基礎監控儀表板

## 📄 授權

請參考項目根目錄的 LICENSE 文件

---

🎉 **開始使用：** `python integrated_pipeline.py --symbol btc`
