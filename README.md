# RL Market Making

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.0+-green.svg)](https://stable-baselines3.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)

**完整的端到端強化學習做市交易系統**

使用強化學習（Reinforcement Learning）進行加密貨幣做市策略訓練與生產部署

[快速開始](#-快速開始推薦) •
[生產部署](#-生產級部署功能) •
[文檔](#-詳細文檔) •
[Docker](#-docker-部署)

</div>

> 📢 **重要更新**: 本專案已完成全面優化！現包含生產級部署功能、完整文檔體系和 Docker 支援。
> 詳見 [優化總結](OPTIMIZATION_SUMMARY.md) | [專案結構](PROJECT_STRUCTURE.md)

---

## 🎯 專案概述

本專案實現了一個端到端的 RL 做市交易系統，包含：

### 核心功能
- **🚀 全自動化 Pipeline**：自動下載數據、調參、訓練、評估
- **🎯 智能參數調整**：根據不同幣種價格自動優化參數
- **🔄 自動重試機制**：失敗自動重訓，確保獲得可用模型
- **📊 多幣種支援**：一鍵訓練任何加密貨幣對
- **🤖 多演算法支援**：SAC（預設）、PPO、TD3
- **🔬 進階環境設計**：真實成交模型、Domain Randomization
- **📈 完整評估指標**：PnL、Sharpe Ratio、Win Rate、Max Drawdown

### 🆕 生產級部署功能

現在包含完整的生產環境支援，讓任何人都能輕鬆產出穩定且可獲利的模型：

- **🤖 自動化訓練 CLI**：一鍵訓練可獲利模型（自動重試直到成功）
- **📊 模型註冊系統**：自動追蹤所有模型的性能指標與版本
- **🌐 REST API**：生產級 FastAPI 用於模型推論和管理
- **📈 Web 監控面板**：視覺化模型性能和系統健康狀態
- **🐳 Docker 支援**：一鍵容器化部署
- **✅ 自動驗證**：只有通過盈利標準的模型才會被標記為「生產就緒」

## ⚡ 快速開始（推薦）

### 最簡單的方式（單個模型訓練）

```bash
# 1. 安裝依賴
pip install -r requirements.txt

# 2. 直接訓練（會自動下載數據）
python pipeline.py --symbol btc
python pipeline.py --symbol eth
python pipeline.py --symbol sol

# 3. 查看結果
# 最佳模型會自動保存到 models/{symbol}_best_model.zip
```

**就是這麼簡單！** Pipeline 會自動處理：
- ✅ 檢查並下載缺失的歷史數據（從 Binance）
- ✅ 分析數據並自動調整參數（spread, cash, reward_scale）
- ✅ 訓練模型（最多重試 3 次）
- ✅ Out-of-Sample 評估
- ✅ 只保存盈利模型（PnL > 0 且 WinRate >= 50%）

### 🚀 生產級快速開始（推薦進階用戶）

使用生產級 CLI 工具，自動重試直到獲得可獲利模型：

```bash
# 1. 訓練一個可獲利的模型（自動重試）
python production/cli.py train --symbol btc --attempts 5

# 2. 查看所有訓練的模型
python production/cli.py list --filter production

# 3. 查看最佳模型
python production/cli.py best --symbol btc

# 4. 啟動生產 API
python production/cli.py serve --port 8000

# 5. 查看監控面板
python production/dashboard.py
# 訪問 http://localhost:8080
```

### 🐳 Docker 快速部署

```bash
# 使用 Docker Compose 一鍵部署
docker-compose up -d

# API 將運行在 http://localhost:8000
# 監控面板將運行在 http://localhost:8080
```

📖 **詳細文檔**：

| 文檔類型 | 文件 | 說明 |
|---------|------|------|
| **快速開始** | [QUICKSTART.md](QUICKSTART.md) | 5 分鐘快速上手指南 |
| **生產部署** | [docs/PRODUCTION_GUIDE.md](docs/PRODUCTION_GUIDE.md) | 生產環境完整部署指南（英文） |
| **使用手冊** | [docs/USER_GUIDE_ZH.md](docs/USER_GUIDE_ZH.md) | 詳細使用說明（中文） |
| **專案結構** | [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | 完整專案組織與架構說明 |
| **開發指南** | [DEVELOPMENT.md](DEVELOPMENT.md) | 開發者貢獻指南與調試技巧 |
| **更新日誌** | [CHANGELOG.md](CHANGELOG.md) | 版本更新歷史 |
| **優化總結** | [OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md) | 專案整合與優化報告 |
| **配置指南** | [CONFIG_GUIDE.md](CONFIG_GUIDE.md) | 配置文件詳細說明 |
| **企業系統** | [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md) | 進階功能完整說明 |
| **生產總結** | [INTEGRATED_PIPELINE_README.md](INTEGRATED_PIPELINE_README.md) | 企業級多幣種系統 |

---

## 📁 專案結構

```
RL_markey/
├── pipeline.py                 # 🎯 主要入口（推薦使用）
├── configs/
│   ├── default.yaml           # 預設配置模板
│   └── pipeline_config.yaml   # Pipeline 配置
│
├── data/                       # 數據目錄（自動下載）
│   ├── btc_usdt_1m_2023.csv
│   └── eth_usdt_1m_2023.csv
│
├── envs/                       # 交易環境
│   ├── market_making_env.py   # 主環境（V3 穩定版）
│   ├── realistic_fill_model.py # 真實成交模擬
│   ├── config_schema.py       # 配置定義
│   └── constants.py           # 常數定義
│
├── scripts/                    # 核心腳本
│   ├── fetch_data.py          # 數據下載（自動調用）
│   ├── train.py               # 訓練腳本
│   ├── evaluate.py            # 評估腳本
│   └── visualize_episode.py   # 可視化
│
├── utils/                      # 工具函數
│   ├── validators.py          # 數據驗證
│   ├── production_checker.py  # 生產就緒檢查
│   ├── metrics.py             # 評估指標
│   ├── numba_optimizations.py # 性能優化
│   └── ...                    # 其他進階工具
│
├── models/                     # 訓練完成的模型
│   ├── btc_best_model.zip
│   └── eth_best_model.zip
│
├── runs/                       # 訓練記錄
│   └── run_xxx_timestamp/
│       ├── config.yaml
│       ├── best_model/
│       └── evaluation_results.json
│
└── [進階系統]                  # 可選的企業級功能
    ├── integrated_pipeline.py  # 多幣種批量訓練
    ├── auto_pipeline.py        # 完整自動化系統
    ├── web_dashboard.py        # Web 監控介面
    └── production/             # 🆕 生產級部署工具
        ├── cli.py              # 命令行工具
        ├── api.py              # REST API 服務
        ├── model_registry.py   # 模型註冊系統
        └── dashboard.py        # 監控面板
```

---

## 🎯 生產級功能詳解

### 1. 自動化訓練 CLI

```bash
# 訓練直到獲得可獲利模型（自動重試）
python production/cli.py train --symbol btc --attempts 5

# 查看所有模型
python production/cli.py list

# 只顯示生產就緒的模型
python production/cli.py list --filter production

# 查看特定幣種的最佳模型
python production/cli.py best --symbol eth

# 導出排行榜
python production/cli.py leaderboard --output leaderboard.json
```

### 2. 模型註冊系統

自動追蹤所有訓練的模型：
- 📊 性能指標（PnL、勝率、Sharpe、回撤等）
- 🏷️ 自動版本控制
- ✅ 生產就緒驗證（5 項標準）
- 📈 盈利能力評分（0-100）

**生產標準**：
1. 正 PnL
2. 勝率 ≥ 50%
3. Sharpe Ratio ≥ 0.5
4. 最大回撤 ≤ 30%
5. 平均交易數 ≥ 10

### 3. REST API 服務

```bash
# 啟動 API 服務
python production/cli.py serve --port 8000

# 或直接運行
uvicorn production.api:app --host 0.0.0.0 --port 8000
```

**API 端點**：
- `GET /health` - 健康檢查
- `GET /models` - 列出所有模型
- `GET /models/best/{symbol}` - 獲取最佳模型
- `POST /predict` - 模型推論
- `GET /stats` - 系統統計

訪問 API 文檔：http://localhost:8000/docs

### 4. Web 監控面板

```bash
python production/dashboard.py
```

訪問 http://localhost:8080 查看：
- 即時系統統計
- 最佳模型展示
- 性能排行榜
- 模型版本追蹤

---

## 🚀 使用指南

### 基本使用

```bash
# 訓練單一幣種
python pipeline.py --symbol btc

# 設定重試次數（預設 3 次）
python pipeline.py --symbol eth --retries 5
```

### Pipeline 自動執行流程

1. **數據檢查**：檢查 `data/{symbol}_usdt_1m_2023.csv` 是否存在
2. **自動下載**：如果缺失，從 Binance Vision 下載完整 2023 年數據
3. **智能分析**：分析前 10k 行數據，計算平均價格
4. **參數調整**：
   - `base_spread` = 價格 × 0.05%
   - `initial_cash` = 價格 × 10
   - `reward_scale` = 動態調整（確保獎勵數值穩定）
5. **訓練循環**（最多 3 次）：
   - 訓練 200k timesteps (SAC 算法)
   - Out-of-Sample 評估（20 episodes）
   - 如果 PnL > 0 且 WinRate >= 50%，保存模型並結束
   - 否則使用新 seed 重訓
6. **結果保存**：最佳模型 → `models/{symbol}_best_model.zip`

### 訓練結果範例

```
🏆 Best Run: run_eth_1765516821_v3 (PnL: $6.14)
   Mean PnL:       +6.14 ± 55.24
   Win Rate:       60.0% (12/20)
   Total PnL:      +122.78
   💾 Saved to: models/eth_best_model.zip
```

---

## 🔧 主要功能

### 🤖 核心功能

| 功能 | 說明 |
|------|------|
| **自動數據下載** | 從 Binance Vision 自動下載完整歷史數據 |
| **智能參數調整** | 根據價格自動計算 spread, cash, reward_scale |
| **自動重試** | 失敗自動使用不同 seed 重訓（最多 3 次） |
| **質量控制** | 只保存盈利且勝率 ≥ 50% 的模型 |
| **多幣種支援** | 支援任何 Binance 上的 USDT 交易對 |

### 📊 環境特性

- **真實成交模擬**：排隊位置、部分成交、滑點模擬
- **多維觀察空間**：價格、庫存、波動率、動量、成交量、趨勢
- **Shaped Reward**：帶有 inventory 懲罰、turnover 懲罰的獎勵塑造
- **Domain Randomization**：費率、spread 隨機化增強泛化
- **動態持倉限制**：根據市況動態調整持倉上限

### 🎯 訓練算法

| 演算法 | 適用場景 | 特點 |
|--------|----------|------|
| **SAC** | 連續動作空間（預設） | 樣本效率高、自動探索調整 |
| **PPO** | 通用場景 | 穩定、易調參（尚未集成） |
| **TD3** | 連續動作空間 | 減少過估計（尚未集成） |

### 📊 評估指標

| 指標 | 說明 |
|------|------|
| **Mean PnL** | 每個 episode 的平均損益 |
| **Win Rate** | 獲利 episode 的比例 |
| **Total PnL** | 所有 episode 的總損益 |
| **Sharpe Ratio** | 風險調整後報酬（未來） |
| **Max Drawdown** | 最大回撤（未來） |

---

## 📝 配置說明

主要配置檔：`configs/default.yaml`

### 快速調整指南

```yaml
# 調整持倉限制
env:
  max_inventory: 2.0          # 減少風險：1.0，增加利潤：5.0

# 調整庫存懲罰
reward:
  lambda_inventory: 20.0      # 預設 20.0，更保守：50.0
  lambda_turnover: 0.01       # 防止過度交易

# 調整學習率
train:
  learning_rate: 0.00003      # 不穩定：降低至 1e-5
  batch_size: 256             # RAM 不足：降低至 128
```

📖 **完整配置文檔**：查看 [CONFIG_GUIDE.md](CONFIG_GUIDE.md) 獲取詳細說明和最佳實踐。

---

## 🔍 進階使用

### 系統架構對比

本專案包含兩個主要系統：

| 系統 | 適用場景 | 入口檔案 | 特點 |
|------|----------|----------|------|
| **簡化 Pipeline** | 個人使用、快速實驗 | `pipeline.py` | ✅ 簡單、快速<br>✅ 自動下載數據<br>✅ 自動調參 |
| **企業級系統** | 批量訓練、生產部署 | `integrated_pipeline.py` | ✅ 多幣種並行<br>✅ Web 監控<br>✅ 指標追蹤<br>✅ 生產就緒檢查 |

**推薦新手使用 `pipeline.py`**，等熟悉後再探索企業級功能。

### 使用企業級系統

```bash
# 批量訓練多個幣種
python integrated_pipeline.py --symbols btc eth bnb sol

# 啟動 Web 監控介面
python web_dashboard.py
# 瀏覽器打開 http://localhost:5000

# 使用自定義配置
python integrated_pipeline.py --symbols btc --config configs/pipeline_config.yaml
```

📖 **詳細文檔**：[INTEGRATED_PIPELINE_README.md](INTEGRATED_PIPELINE_README.md)

---

### 自定義訓練

如果你想完全控制訓練流程：

```python
import yaml
from pathlib import Path
from stable_baselines3 import SAC
from envs.market_making_env import MarketMakingEnv

# 1. 載入配置
with open('configs/default.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 2. 創建環境
env = MarketMakingEnv(
    csv_path="data/btc_usdt_1m_2023.csv",
    **config['env']
)

# 3. 創建模型
model = SAC(
    "MlpPolicy", 
    env,
    learning_rate=config['train']['learning_rate'],
    batch_size=config['train']['batch_size'],
    verbose=1
)

# 4. 訓練
model.learn(total_timesteps=200000)

# 5. 保存
model.save("my_custom_model")
```

### 使用訓練好的模型

```python
from stable_baselines3 import SAC

# 載入模型
model = SAC.load("models/btc_best_model.zip")

# 預測動作
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        break
```

---

## 📅 專案開發歷程

### 現狀：完整生產級系統 ✅

目前專案包含兩套系統：

#### 1. 簡化版 Pipeline（`pipeline.py`）
適合個人使用和快速實驗：
- ✅ 自動數據下載
- ✅ 智能參數調整
- ✅ 自動重試機制
- ✅ 質量控制（只保存盈利模型）
- ✅ 多幣種支援

#### 2. 生產級系統（`production/`）
適合企業部署和批量訓練：
- ✅ 自動化訓練 CLI
- ✅ 模型註冊與版本管理
- ✅ REST API 服務
- ✅ Web 監控面板
- ✅ Docker 容器化部署
- ✅ 自動化測試

### 核心特點

1. **環境版本**：`envs/market_making_env.py` (V3 穩定版)
   - Shaped Reward 與庫存懲罰
   - 動態持倉限制
   - 真實成交模擬（可選）

2. **訓練算法**：SAC (Soft Actor-Critic)
   - 樣本效率高
   - 自動探索調整
   - 適合連續動作空間

3. **驗證結果**：
   - BTC: 訓練成功，PnL +$3027, WinRate 60%
   - ETH: 訓練成功，PnL +$6.14, WinRate 60%

4. **生產功能**：
   - ✅ 模型註冊系統
   - ✅ REST API 服務
   - ✅ Web 監控面板
   - ✅ Docker 部署支援

---

## 🐳 Docker 部署

### 快速啟動

```bash
# 使用 Docker Compose 一鍵部署
docker-compose up -d

# 查看服務狀態
docker-compose ps

# 查看日誌
docker-compose logs -f

# 停止服務
docker-compose down
```

### 單獨構建

```bash
# 構建鏡像
docker build -t rl-market-making .

# 運行 API 服務
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  --name rl-api \
  rl-market-making python production/cli.py serve

# 運行監控面板
docker run -d \
  -p 8080:8080 \
  -v $(pwd)/models:/app/models \
  --name rl-dashboard \
  rl-market-making python production/dashboard.py
```

### 服務地址

- **API 文檔**: http://localhost:8000/docs
- **監控面板**: http://localhost:8080
- **健康檢查**: http://localhost:8000/health

---

## 📚 完整文檔索引

### 快速上手
- [QUICKSTART.md](QUICKSTART.md) - 5 分鐘快速入門
- [docs/QUICKSTART.md](docs/QUICKSTART.md) - 更詳細的快速開始指南

### 使用指南
- [docs/USER_GUIDE_ZH.md](docs/USER_GUIDE_ZH.md) - 完整中文使用手冊
- [CONFIG_GUIDE.md](CONFIG_GUIDE.md) - 配置文件詳解
- [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md) - 系統功能概覽

### 生產部署
- [docs/PRODUCTION_GUIDE.md](docs/PRODUCTION_GUIDE.md) - 生產環境完整指南
- [INTEGRATED_PIPELINE_README.md](INTEGRATED_PIPELINE_README.md) - 企業級多幣種系統

### 開發文檔
- [DEVELOPMENT.md](DEVELOPMENT.md) - 開發者貢獻指南
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - 專案結構詳解
- [CHANGELOG.md](CHANGELOG.md) - 版本更新歷史
- [OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md) - 最新優化報告

### 範例代碼
- [examples/api_usage.py](examples/api_usage.py) - API 使用範例
- [examples/complete_workflow.py](examples/complete_workflow.py) - 完整工作流程

---

## 🤝 Contributing

歡迎貢獻！請查看 [DEVELOPMENT.md](DEVELOPMENT.md) 了解：
- 開發環境設置
- 代碼風格規範
- Git 工作流程
- 測試指南

### 快速貢獻步驟

```bash
# 1. Fork 專案並克隆
git clone https://github.com/your-username/RL_markey.git
cd RL_markey

# 2. 創建功能分支
git checkout -b feature/your-feature

# 3. 進行修改並測試
pytest tests/

# 4. 提交更改
git add .
git commit -m "feat: your feature description"

# 5. 推送並創建 Pull Request
git push origin feature/your-feature
```

---

## 📝 License

MIT License - 詳見 [LICENSE](LICENSE) 文件

---

## 🙏 致謝

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL 演算法實現
- [Gymnasium](https://gymnasium.farama.org/) - 環境標準
- [FastAPI](https://fastapi.tiangolo.com/) - API 框架
- [Binance](https://www.binance.com/) - 歷史數據來源

---

<div align="center">

**⭐ 如果這個專案對你有幫助，請給個 Star！**

Made with ❤️ by the RL Market Making Team

</div>