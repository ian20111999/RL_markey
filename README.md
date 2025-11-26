# RL Market Making

使用強化學習（Reinforcement Learning）進行加密貨幣做市策略訓練的完整框架。

## 🎯 專案概述

本專案實現了一個端到端的 RL 做市交易系統，包含：
- **多演算法支援**：SAC、PPO、TD3
- **進階環境設計**：V2 環境含 Potential-based Reward Shaping、Domain Randomization
- **風險敏感訓練**：CVaR、Mean-Variance 優化
- **課程學習**：漸進式難度訓練
- **專業回測框架**：Walk-Forward Analysis、Monte Carlo Simulation
- **自動化報告**：HTML/PDF 報告生成

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
│   ├── historical_market_making_env.py  # V1 環境
│   └── realistic_fill_model.py     # 真實成交模型
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
from utils.algorithms import AlgorithmFactory

model = AlgorithmFactory.create(
    algorithm="SAC",
    env=env,
    learning_rate=3e-4,
    buffer_size=100000
)
```

### 課程學習

```python
from utils.curriculum import train_with_curriculum

model = train_with_curriculum(
    env_fn=make_env,
    stages=[
        {"name": "easy", "env_params": {"fee_rate": 0.0002}},
        {"name": "hard", "env_params": {"fee_rate": 0.0004}}
    ],
    total_timesteps=200000
)
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

## 📝 License

MIT License

## 🤝 Contributing

歡迎提交 Issue 和 Pull Request！