# RL Market Making

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.0+-green.svg)](https://stable-baselines3.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)

**完整的端到端強化學習做市交易系統**

使用強化學習（Reinforcement Learning）進行加密貨幣做市策略訓練與生產部署

[快速開始](#-快速開始) •
[功能特色](#-專案概述) •
[文檔](#-詳細文檔) •
[生產部署](#-生產級部署功能)

</div>

> 📢 **重要更新**: 本專案已完成分支整合與全面優化！所有功能已統一至 main 分支，並提供完整的生產級部署方案。  
> 詳見 [優化總結](OPTIMIZATION_SUMMARY.md)

---

## 📑 目錄

- [專案概述](#-專案概述)
- [生產級部署功能](#-生產級部署功能)
- [專案結構](#-專案結構)
- [快速開始](#-快速開始)
- [主要功能](#-主要功能)
- [配置說明](#-配置說明)
- [評估指標](#-評估指標)
- [進階使用](#-進階使用)
- [專案開發階段](#-專案開發階段-development-phases)
- [詳細文檔](#-詳細文檔)
- [Docker 部署](#-docker-部署)
- [License](#-license)

---

## 🎯 專案概述

本專案實現了一個端到端的 RL 做市交易系統，包含：
- **多演算法支援**：SAC、PPO、TD3
- **進階環境設計**：V2 環境含 Potential-based Reward Shaping、Domain Randomization
- **風險敏感訓練**：CVaR、Mean-Variance 優化
- **課程學習**：漸進式難度訓練
- **專業回測框架**：Walk-Forward Analysis、Monte Carlo Simulation
- **自動化報告**：HTML/PDF 報告生成

## 🚀 **NEW: 生產級部署功能**

現在包含完整的生產環境支援，讓任何人都能輕鬆產出穩定且可獲利的模型：

- **🤖 自動化訓練 CLI**：一鍵訓練可獲利模型（自動重試直到成功）
- **📊 模型註冊系統**：自動追蹤所有模型的性能指標與版本
- **🌐 REST API**：生產級 HTTP API 用於模型推論和管理
- **📈 Web 監控面板**：視覺化模型性能和系統健康狀態
- **🐳 Docker 支援**：一鍵容器化部署
- **✅ 自動驗證**：只有通過盈利標準的模型才會被標記為「生產就緒」

### 快速開始（生產環境）

```bash
# 1. 安裝依賴（包含所有生產環境所需套件）
pip install -r requirements.txt

# 2. 訓練一個可獲利的模型（自動重試）
python production/cli.py train --symbol btc --attempts 3

# 3. 啟動生產 API
python production/cli.py serve --port 8000

# 4. 查看監控面板
python production/dashboard.py
# 訪問 http://localhost:8080
```

## 📚 詳細文檔

| 文檔 | 說明 |
|------|------|
| **[OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md)** | 🆕 專案整合與優化完成報告 |
| **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** | 完整專案組織、架構與最佳實踐 |
| **[DEVELOPMENT.md](DEVELOPMENT.md)** | 開發者指南、貢獻規範與調試技巧 |
| **[CHANGELOG.md](CHANGELOG.md)** | 版本更新歷史與升級指南 |
| **[docs/QUICKSTART.md](docs/QUICKSTART.md)** | 5分鐘快速上手教程 |
| **[docs/PRODUCTION_GUIDE.md](docs/PRODUCTION_GUIDE.md)** | 生產環境部署指南（英文） |
| **[docs/USER_GUIDE_ZH.md](docs/USER_GUIDE_ZH.md)** | 詳細使用說明（中文） |
| **[PRODUCTION_SUMMARY.md](PRODUCTION_SUMMARY.md)** | 生產級功能完整總結 |

---

## 🐳 Docker 部署

### 快速啟動

```bash
# 1. 使用 Docker Compose 一鍵啟動
docker-compose up -d

# 2. 檢查服務狀態
docker-compose ps

# 3. 查看日誌
docker-compose logs -f

# 4. 訪問服務
# API: http://localhost:8000
# 文檔: http://localhost:8000/docs
# 健康檢查: http://localhost:8000/health
```

### 手動構建

```bash
# 構建映像
docker build -t rl-market-making .

# 運行容器
docker run -d -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  rl-market-making
```

---

## 📁 專案結構

```
RL_markey/
├── configs/                    # 配置檔
│   ├── env_v3_full.yaml       # V3 完整配置（推薦）
│   ├── env_v2.yaml            # V2 基礎配置
│   └── env_baseline.yaml      # 基準配置
│
├── data/                       # 數據
│   └── btc_usdt_1m_2023.csv   # BTC/USDT 1分鐘K線
│
├── envs/                       # Gymnasium 環境
│   ├── market_making_env_v2.py     # V2 環境（主要）
│   ├── realistic_fill_model.py     # 真實成交模型
│   └── legacy/                     # 舊版環境（參考用）
│
├── scripts/                    # 執行腳本
│   ├── run_v3_pipeline.py     # V3 完整流程（推薦）
│   ├── train_v2.py            # V2 訓練腳本
│   ├── fetch_binance_ohlcv.py # 數據下載
│   └── legacy/                # 舊版腳本（參考用）
│
├── utils/                      # 工具模組
│   ├── algorithms.py          # 多演算法工廠
│   ├── risk_sensitive.py      # 風險敏感訓練
│   ├── curriculum.py          # 課程學習
│   ├── backtesting.py         # 回測框架
│   ├── ensemble.py            # 集成方法
│   ├── explainability.py      # 可解釋性分析
│   ├── online_adaptation.py   # 線上適應
│   ├── distributed_training.py # 分散式訓練
│   └── report_generator.py    # 報告生成
│
├── production/                 # 🆕 生產環境組件
│   ├── model_registry.py      # 模型註冊系統
│   ├── api.py                 # REST API 服務
│   ├── cli.py                 # 生產 CLI 工具
│   └── dashboard.py           # Web 監控面板
│
├── models/                     # 模型與參數
├── runs/                       # 訓練記錄
└── docs/                       # 文件
```

---

## 🚀 快速開始

### 1. 安裝依賴

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 下載數據

```bash
python scripts/fetch_binance_ohlcv.py --symbol BTC/USDT --timeframe 1m --days 365
```

### 3. 執行訓練

**標準訓練（推薦新手）：**
```bash
python scripts/run_v3_pipeline.py --config configs/env_v3_full.yaml --mode standard
```

**使用不同演算法：**
```bash
python scripts/run_v3_pipeline.py --algorithm PPO --total_timesteps 200000
python scripts/run_v3_pipeline.py --algorithm TD3 --total_timesteps 200000
```

**課程學習訓練：**
```bash
python scripts/run_v3_pipeline.py --mode curriculum
```

**完整流程（訓練 + 回測 + 報告）：**
```bash
python scripts/run_v3_pipeline.py --mode full --generate_report
```

---

## 🔧 主要功能

### 演算法支援

| 演算法 | 適用場景 | 特點 |
|--------|----------|------|
| **SAC** | 連續動作空間（預設） | 樣本效率高、自動探索調整 |
| **PPO** | 通用場景 | 穩定、易調參 |
| **TD3** | 連續動作空間 | 減少過估計、穩定性佳 |

### 環境特性（V2）

- **4 種獎勵模式**：`dense`, `sparse`, `shaped`, `hybrid`
- **Domain Randomization**：費率、spread、波動率隨機化
- **擴展觀察空間**：17+ 特徵（波動率、動量、成交量等）
- **靈活動作空間**：對稱/非對稱 spread + 數量控制

### 進階功能

```bash
# 風險敏感訓練
python scripts/run_v3_pipeline.py --use_risk_wrapper

# 分散式訓練（超參數搜尋 + 多種子驗證）
python scripts/run_v3_pipeline.py --mode distributed --n_hp_trials 30

# 回測分析
python scripts/run_v3_pipeline.py --run_backtest

# 可解釋性分析
python scripts/run_v3_pipeline.py --run_explainability
```

---

## 📊 配置說明

`configs/env_v3_full.yaml` 包含所有可調參數：

```yaml
env:
  initial_cash: 100000
  fee_rate: 0.0004
  max_inventory: 10.0
  
  reward_config:
    mode: "hybrid"              # dense, sparse, shaped, hybrid
    inventory_penalty: 0.0005
    
  domain_randomization:
    enabled: true
    fee_rate_range: [0.0003, 0.0005]

curriculum:
  enabled: false
  stages:
    - name: "easy"
      env_params: {fee_rate: 0.0002}
      advancement_threshold: 50.0

risk_sensitive:
  enabled: false
  risk_lambda: 0.1              # 風險厭惡係數
  risk_type: "variance"         # variance, cvar, downside_variance
```

---

## 📈 評估指標

| 指標 | 說明 |
|------|------|
| **Sharpe Ratio** | 風險調整報酬 |
| **Max Drawdown** | 最大回撤 |
| **Win Rate** | 勝率 |
| **Profit Factor** | 獲利因子 |
| **Avg Trade PnL** | 平均交易損益 |
| **Sortino Ratio** | 下行風險調整報酬 |

---

## 🛠️ 進階使用

### 使用演算法工廠

```python
from utils.algorithms import create_model

model = create_model(
    algo="sac",
    env=env,
    config_overrides={
        "learning_rate": 3e-4,
        "buffer_size": 100000
    }
)
```

### 課程學習

```python
from utils.curriculum import CurriculumScheduler, CurriculumEnvWrapper

# 建立課程排程器
scheduler = CurriculumScheduler(stages=[
    {"name": "easy", "env_params": {"fee_rate": 0.0002}},
    {"name": "hard", "env_params": {"fee_rate": 0.0004}}
])

# 包裝環境
env = CurriculumEnvWrapper(base_env, scheduler)
```

### 回測框架

```python
from utils.backtesting import BacktestEngine

engine = BacktestEngine(env_fn=make_env, policy=model)
results = engine.run_walk_forward_analysis(
    train_window=30, test_window=7
)
mc_results = engine.run_monte_carlo_simulation(n_simulations=1000)
```

### 報告生成

```python
from utils.report_generator import QuickReportBuilder

report = (
    QuickReportBuilder("My Report")
    .with_metric("Sharpe", 1.85)
    .with_equity_curve("path/to/equity.csv")
    .build("report.html")
)
```

---

## 📅 專案開發階段 (Development Phases)

本專案的開發歷程分為 5 個主要階段，目前處於 **Phase 4**。

### Phase 1: 基礎建設與基線 (Foundation & Baseline)
- **目標**: 建立環境並證明 RL 有學習潛力。
- **核心**: `envs/market_making_env_v2.py`, `configs/env_baseline.yaml`
- **成果**: 驗證 RL 表現優於隨機 (Random) 與固定價差 (Fixed Spread) 策略。

### Phase 2: 初步 RL 訓練 (Standard RL)
- **目標**: 讓 Agent 學會基本的做市行為（低買高賣）。
- **核心**: `configs/env_v3.yaml`, `scripts/run_v3_pipeline.py`
- **成果**: Agent 學會基本交易，但在極端行情下表現不穩。

### Phase 3: 穩健性與調優 (Robustness & Tuning)
- **目標**: 提高模型在不同市場狀況下的生存率。
- **核心**: `configs/env_v3_robust.yaml`, `tune_mm_sac.py`, `utils/risk_sensitive.py`
- **成果**: 引入動態持倉限制 (Dynamic Position Limit) 與 Optuna 自動調優。

### Phase 4: "核彈級" 庫存控制 (Nuclear / Stabilization) ⬅️ 目前階段
- **目標**: 強制 Agent 學會極致的庫存管理，實現穩定的正收益。
- **核心**: `configs/env_v3_stabilized.yaml`, `restart_training.sh`, `scripts/run_stabilization.py`
- **特點**: 
    - 使用極高的庫存懲罰 (`lambda_inventory=10.0`)。
    - 採用 `restart_training.sh` 進行斷點續訓，修復特定問題而不重練。
    - 實作 Reward Clipping 防止梯度爆炸。

### Phase 5: 評估與分析 (Evaluation & OOS)
- **目標**: 驗證模型在未見過數據 (Out-of-Sample) 上的表現。
- **核心**: `scripts/evaluate_v3_oos.py`, `scripts/analyze_experiments.py`
- **計畫**: 完成 Phase 4 穩定化訓練後，進行嚴格的 OOS 測試與實盤模擬。

---

## 📝 License

MIT License - 詳見 [LICENSE](LICENSE) 文件

---

## 🤝 Contributing

歡迎提交 Issue 和 Pull Request！

### 貢獻指南

1. **Fork 本專案**
2. **創建您的特性分支** (`git checkout -b feature/AmazingFeature`)
3. **提交您的改動** (`git commit -m 'Add some AmazingFeature'`)
4. **推送到分支** (`git push origin feature/AmazingFeature`)
5. **開啟 Pull Request**

### 開發規範

- 遵循 PEP 8 代碼風格
- 為新功能添加測試
- 更新相關文檔
- 確保所有測試通過

---

## 🙏 Acknowledgements

本專案使用以下優秀的開源項目：

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL 演算法實現
- [Gymnasium](https://gymnasium.farama.org/) - RL 環境標準
- [FastAPI](https://fastapi.tiangolo.com/) - 生產 API 框架
- [PyTorch](https://pytorch.org/) - 深度學習框架

---

## 📧 Contact

如有問題或建議，請透過以下方式聯繫：

- 提交 [GitHub Issue](https://github.com/ian20111999/RL_markey/issues)
- Email: [專案維護者郵箱]

---

## ⭐ Star History

如果這個專案對您有幫助，請給它一個 ⭐ Star！

---

<div align="center">

**Made with ❤️ by RL Market Making Team**

[回到頂部](#rl-market-making)

</div>
