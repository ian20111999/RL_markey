# 自動化訓練流水線 (Automated Training Pipeline)

## 概述

這是一個完全自動化的強化學習交易模型訓練系統，可以：

✅ **自動判斷環境品質** - 驗證數據品質、環境參數合理性  
✅ **智能重試機制** - 訓練結果不佳時自動重訓練  
✅ **生產就緒檢查** - 多維度評估模型是否可用於實際交易  
✅ **自適應參數調整** - 根據資產特性自動調整環境參數  
✅ **全程監控追蹤** - 記錄所有訓練過程和結果  
✅ **可視化儀表板** - Web界面查看訓練進度和結果  

## 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 一鍵啟動訓練

```bash
# 訓練 BTC 模型
python start_pipeline.py --symbol btc

# 訓練 ETH 模型，最多重試 10 次
python start_pipeline.py --symbol eth --retries 10

# 訓練 BNB 模型，使用自定義配置
python start_pipeline.py --symbol bnb --config configs/custom_pipeline.yaml
```

### 3. 查看訓練監控儀表板

#### 方式 A: 終端查看
```bash
python monitoring_dashboard.py --mode console
```

#### 方式 B: Web 界面查看
```bash
# 1. 啟動 API 服務器
python monitoring_dashboard.py --mode server

# 2. 打開瀏覽器訪問
# 在瀏覽器中打開 dashboard.html 文件
# 或訪問 http://localhost:5000/api/dashboard (JSON 數據)
```

#### 方式 C: 導出 JSON
```bash
python monitoring_dashboard.py --mode export --output logs/dashboard.json
```

## 系統架構

```
┌─────────────────────────────────────────────────────────────┐
│                   自動化訓練流水線                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │  1. 數據獲取與驗證 (Data Layer)   │
        │  • 自動下載或查找數據              │
        │  • 數據品質檢查                   │
        │  • 價格/波動性/缺失值驗證          │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │  2. 環境配置 (Config Layer)       │
        │  • 自適應參數調整                 │
        │  • 根據資產特性優化配置            │
        │  • 環境健康檢查                   │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │  3. 訓練循環 (Training Layer)     │
        │  • 多次重試訓練                   │
        │  • 追蹤所有訓練運行                │
        │  • 自動保存最佳模型                │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │  4. 評估驗證 (Validation Layer)   │
        │  • 多指標評估 (PnL, Sharpe, etc.)  │
        │  • 生產就緒檢查                   │
        │  • 綜合評分計算                   │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │  5. 結果輸出 (Output Layer)       │
        │  • 保存最佳模型                   │
        │  • 記錄到數據庫                   │
        │  • 生成報告                       │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │  6. 監控儀表板 (Dashboard)        │
        │  • 實時查看訓練進度                │
        │  • 多幣種對比                     │
        │  • 歷史記錄追蹤                   │
        └───────────────────────────────────┘
```

## 核心功能詳解

### 1. 數據品質檢查 (`utils/validators.py`)

自動檢查：
- 樣本數量是否充足 (預設 ≥ 10,000)
- 缺失值比例 (預設 ≤ 1%)
- 價格異常 (零值、負值)
- 波動性合理性 (0.01% - 50%)
- 重複值檢測
- 成交量檢查

### 2. 自適應參數調整

根據資產平均價格自動調整：

| 參數 | 調整邏輯 | 範例 (BTC=100k, ETH=3k) |
|------|---------|------------------------|
| **base_spread** | 價格 × 0.05% (5 bps) | BTC: 50, ETH: 1.5 |
| **initial_cash** | 價格 × 10 | BTC: 1,000,000, ETH: 30,000 |
| **reward_scale** | 1e-6 × (100k / 價格) | BTC: 1e-6, ETH: 3.3e-5 |

### 3. 訓練結果驗證

#### 基本驗證指標：
- **PnL (盈虧)**: 必須 > 0
- **勝率**: 必須 > 45%
- **Sharpe Ratio**: 必須 > 0.0
- **最大回撤**: 必須 > -30%

#### 綜合評分 (0-100):
- PnL 貢獻: 30%
- 勝率貢獻: 20%
- Sharpe 貢獻: 25%
- 回撤貢獻: 15%
- 穩定性貢獻: 10%

### 4. 生產就緒檢查 (更嚴格)

確保模型可用於實際交易：
- **Sharpe Ratio** ≥ 0.5
- **最大回撤** > -25%
- **勝率** ≥ 48%
- **交易次數** ≥ 50
- **盈虧比** ≥ 1.2
- **波動性比率** ≤ 2.0
- **統計顯著性** t-stat > 1.5

### 5. 指標數據庫 (`logs/metrics.db`)

SQLite 數據庫記錄：
- **training_runs**: 所有訓練運行
- **training_results**: 訓練結果指標
- **best_models**: 每個幣種的最佳模型

## 配置文件

### `configs/pipeline_config.yaml`

```yaml
pipeline:
  max_retries: 5              # 最大重試次數
  training_timeout: 3600      # 訓練超時 (秒)
  evaluation_timeout: 600     # 評估超時 (秒)
  eval_episodes: 30           # 評估回合數
  total_timesteps: 200000     # 訓練步數
  early_stopping: true        # 找到好結果後提前停止
  cleanup_failed_runs: true   # 清理失敗的訓練

data_validation:
  min_samples: 10000          # 最少樣本數
  max_missing_ratio: 0.01     # 最大缺失值比例
  min_volatility: 0.0001      # 最小波動性
  max_volatility: 0.5         # 最大波動性

training_acceptance:
  min_pnl: 0                  # 最低 PnL
  min_win_rate: 0.45          # 最低勝率
  min_sharpe: 0.0             # 最低 Sharpe
  max_drawdown: -0.3          # 最大回撤
  min_composite_score: 60     # 最低綜合評分

production_readiness:
  min_sharpe: 0.5             # 生產環境最低 Sharpe
  max_drawdown: -0.25         # 生產環境最大回撤
  min_win_rate: 0.48          # 生產環境最低勝率
  min_trades: 50              # 最少交易次數
  min_profit_factor: 1.2      # 最低盈虧比
  max_volatility_ratio: 2.0   # 最大波動性比率

auto_tuning:
  enabled: true
  spread_ratio: 0.0005        # Spread 佔價格比例
  cash_multiplier: 10.0       # 初始資金倍數
  reward_scale_base: 1.0e-6   # 獎勵縮放基準
```

## 輸出結果

### 1. 最佳模型
```
models/
  ├── btc_best_model.zip      # 最佳 BTC 模型
  ├── btc_best_config.yaml    # 最佳 BTC 配置
  ├── eth_best_model.zip      # 最佳 ETH 模型
  └── eth_best_config.yaml    # 最佳 ETH 配置
```

### 2. 日誌和指標
```
logs/
  ├── metrics.db              # SQLite 數據庫
  ├── metrics.json            # JSON 導出
  ├── dashboard.json          # 儀表板數據
  └── pipeline/               # 訓練日誌
      ├── pipeline_20231212_143022.log
      └── pipeline_20231212_151533.log
```

### 3. 訓練運行目錄
```
runs/
  ├── run_btc_1702389022_v1/
  │   ├── config.yaml
  │   ├── best_model/
  │   └── evaluation_results.json
  └── run_btc_1702389155_v2/
      └── ...
```

## 使用場景

### 場景 1: 訓練單個幣種
```bash
python start_pipeline.py --symbol btc
```

### 場景 2: 批量訓練多個幣種
```bash
#!/bin/bash
for symbol in btc eth bnb sol ada
do
    echo "Training $symbol..."
    python start_pipeline.py --symbol $symbol --retries 5
done
```

### 場景 3: 自定義配置
```bash
# 1. 複製並修改配置文件
cp configs/pipeline_config.yaml configs/my_config.yaml

# 2. 使用自定義配置
python start_pipeline.py --symbol btc --config configs/my_config.yaml
```

### 場景 4: 定期自動訓練 (Cron)
```cron
# 每天凌晨 2 點訓練 BTC
0 2 * * * cd /path/to/RL_markey && python start_pipeline.py --symbol btc >> logs/cron.log 2>&1
```

## API 端點 (監控服務器)

啟動服務器：
```bash
python monitoring_dashboard.py --mode server --host 0.0.0.0 --port 5000
```

可用端點：
- `GET /api/dashboard` - 獲取儀表板數據
- `GET /api/symbol/<symbol>` - 獲取特定幣種詳情
- `GET /api/health` - 健康檢查

## 進階用法

### 直接使用 AutomatedPipeline 類

```python
from auto_pipeline import AutomatedPipeline
from pathlib import Path

# 初始化
pipeline = AutomatedPipeline(
    config_path=Path("configs/pipeline_config.yaml")
)

# 自定義重試次數
pipeline.config['pipeline']['max_retries'] = 10

# 運行
success = pipeline.run(symbol="btc", data_dir="data")

if success:
    print("Training successful!")
else:
    print("Training failed or not production-ready")
```

### 查詢指標數據庫

```python
from utils.metrics_db import MetricsDatabase
from pathlib import Path

db = MetricsDatabase(Path("logs/metrics.db"))

# 獲取最佳運行
best = db.get_best_run_for_symbol("btc")
print(f"Best BTC model: {best}")

# 獲取最近運行
recent = db.get_recent_runs(symbol="btc", limit=10)
for run in recent:
    print(f"{run['run_id']}: PnL=${run['mean_pnl']:.2f}")

# 導出 JSON
db.export_to_json(Path("logs/export.json"))
```

## 常見問題

### Q1: 訓練一直失敗怎麼辦？
**A**: 
1. 檢查數據品質: `python monitoring_dashboard.py --mode console`
2. 降低驗收標準: 修改 `pipeline_config.yaml` 中的 `training_acceptance`
3. 增加重試次數: `--retries 10`
4. 檢查日誌: `logs/pipeline/*.log`

### Q2: 如何調整訓練速度？
**A**: 修改 `pipeline_config.yaml`:
```yaml
pipeline:
  total_timesteps: 100000  # 減少步數加快訓練
  eval_episodes: 10        # 減少評估回合
```

### Q3: 生產就緒檢查太嚴格？
**A**: 調整 `pipeline_config.yaml` 中的 `production_readiness`:
```yaml
production_readiness:
  min_sharpe: 0.3          # 降低 Sharpe 要求
  max_drawdown: -0.35      # 放寬回撤限制
  min_win_rate: 0.45       # 降低勝率要求
```

### Q4: 如何添加自定義驗證邏輯？
**A**: 修改 `utils/validators.py` 或 `utils/production_checker.py`

### Q5: 數據從哪裡來？
**A**: 
- 自動從 Binance 下載 (需要 `scripts/fetch_data.py`)
- 或手動放置 CSV 文件到 `data/` 目錄

## 下一步開發

### 前端界面增強 (已有基礎)
- [x] 基本 HTML 儀表板
- [ ] React/Vue 完整前端應用
- [ ] 實時訓練進度追蹤
- [ ] 交互式圖表 (TradingView)
- [ ] 模型對比分析
- [ ] 參數調整界面

### 功能擴展
- [ ] 多進程並行訓練多個幣種
- [ ] Optuna 超參數優化集成
- [ ] 模型 A/B 測試框架
- [ ] 實盤交易接口
- [ ] 告警通知 (Email, Slack, Telegram)
- [ ] 模型版本管理

### 性能優化
- [ ] GPU 加速訓練
- [ ] 分布式訓練支持
- [ ] 增量訓練/遷移學習
- [ ] 模型壓縮與量化

## 貢獻指南

歡迎提交 Pull Request 或 Issue！

## 授權

MIT License
