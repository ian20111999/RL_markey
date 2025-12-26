# RL Market - 強化學習交易系統

基於強化學習（Reinforcement Learning）的加密貨幣交易系統，支援完整的訓練、回測、評估和生產部署。

> 📖 **詳細文件**:
> - [專案結構說明](PROJECT_STRUCTURE.md) - 完整目錄結構和檔案說明
> - [清理建議](CLEANUP_GUIDE.md) - 資料清理和優化建議

---

## 🚀 快速開始

### 基本訓練

```bash
# 1. 安裝依賴
pip install -r requirements.txt

# 2. 訓練模型
python pipeline.py --symbol btc

# 3. 查看結果
# 模型保存在 models/{symbol}_best_model.zip
```

### Docker 部署

```bash
# 啟動所有服務（PostgreSQL + Dashboards）
docker-compose up -d

# 查看服務狀態
docker-compose ps

# 查看日誌
docker-compose logs -f
```

---

## 📊 資料庫連接

### 方式一：使用 Docker 內建的 pgAdmin（推薦）

```bash
# 1. 啟動 pgAdmin
docker-compose up -d pgadmin

# 2. 開啟瀏覽器訪問
open http://localhost:5050
```

**登入資訊：**
- Email: `admin@rlmarket.com`
- Password: `admin`

**新增 PostgreSQL Server：**
1. 登入後，右鍵 **Servers** → **Register** → **Server**
2. **General** 標籤：
   - Name: `RL Market`
3. **Connection** 標籤：
   - Host: `postgres` ⬅️ 使用 Docker 服務名稱
   - Port: `5432`
   - Maintenance database: `postgres`
   - Username: `rl_user`
   - Password: `rl_password`
   - ✅ Save password
4. **Save** → 展開：**Servers → RL Market → Databases → rl_market → Schemas → public → Tables**

### 方式二：本機 PostgreSQL 客戶端連接

如果使用本機安裝的 pgAdmin 或其他工具（DBeaver, TablePlus）：

- **Host**: `localhost` 或 `127.0.0.1`
- **Port**: `5432`
- **Database**: `rl_market`
- **Username**: `rl_user`
- **Password**: `rl_password`

### 常用查詢

```bash
# 查看所有表
docker-compose exec postgres psql -U rl_user -d rl_market -c "\dt"

# 查看資料統計
docker-compose exec postgres psql -U rl_user -d rl_market -c "
SELECT 'symbols' as table, COUNT(*) FROM symbols
UNION ALL SELECT 'training_runs', COUNT(*) FROM training_runs
UNION ALL SELECT 'models', COUNT(*) FROM models;"

# 查看訓練記錄
docker-compose exec postgres psql -U rl_user -d rl_market -c "
SELECT run_id, symbol, status, final_pnl FROM training_runs ORDER BY start_time DESC LIMIT 10;"
```

---

## 🗄️ 資料庫架構

系統使用 PostgreSQL 16，包含 8 個核心資料表：

| 表名 | 說明 | 主要欄位 |
|------|------|---------|
| **symbols** | 交易對管理 | symbol, base_currency, quote_currency |
| **market_data** | OHLCV 市場資料 | symbol_id, timestamp, open, high, low, close, volume |
| **training_runs** | 訓練執行記錄 | run_id, symbol, algorithm, status, final_pnl |
| **episodes** | Episode 詳細指標 | run_id, episode_num, reward, pnl, win_rate |
| **models** | 模型資訊 | model_name, symbol, model_path, performance_metrics |
| **backtest_runs** | 回測結果 | backtest_id, model_id, total_pnl, sharpe_ratio |
| **trades** | 交易明細 | symbol_id, timestamp, side, price, quantity, pnl |
| **system_logs** | 系統日誌 | timestamp, level, component, message |

### 資料庫視圖

- `v_latest_training_runs` - 最新訓練記錄
- `v_symbol_performance` - 幣種表現統計
- `v_model_leaderboard` - 模型排行榜
- `v_market_data_stats` - 市場資料統計

---

## 📁 專案結構

```
RL_markey/
├── README.md                      # 本文件
├── pipeline.py                    # 基礎訓練流程
├── integrated_pipeline.py         # 整合版 Pipeline
├── auto_pipeline.py               # 自動化多幣種訓練
├── docker-compose.yml             # Docker 配置
├── requirements.txt               # Python 依賴
│
├── configs/                       # 配置檔案
│   ├── default.yaml
│   └── pipeline_config.yaml
│
├── envs/                          # 交易環境
│   ├── market_making_env.py
│   └── env_factory.py
│
├── utils/                         # 工具模組
│   ├── database.py               # 資料庫抽象層
│   ├── sqlite_db.py              # SQLite 實現
│   ├── postgres_db.py            # PostgreSQL 實現
│   ├── algorithms.py             # RL 演算法
│   ├── backtesting.py            # 回測引擎
│   └── metrics.py                # 指標計算
│
├── scripts/                       # 腳本工具
│   ├── train.py                  # 訓練腳本
│   ├── evaluate.py               # 評估腳本
│   ├── fetch_data.py             # 資料獲取
│   ├── init_db.sql               # PostgreSQL Schema
│   └── init_db_sqlite.sql        # SQLite Schema
│
├── production/                    # 生產模組
│   ├── api.py                    # RESTful API
│   ├── cli.py                    # 命令列介面
│   └── dashboard.py              # Dashboard
│
├── data/                          # 資料目錄
├── models/                        # 模型目錄
├── logs/                          # 日誌目錄
│   └── metrics.db                # SQLite 資料庫（開發用）
└── runs/                          # 訓練記錄
```

---

## 🎯 核心功能

### 1. 訓練系統

#### 單一幣種訓練
```bash
python scripts/train.py --symbol BTCUSDT --algorithm PPO
```

#### 多幣種自動訓練
```bash
python auto_pipeline.py --symbols BTCUSDT ETHUSDT --version v2
```

#### 使用自定義配置
```bash
python scripts/train.py --symbol ETHUSDT --config configs/eth_best_config.yaml
```

### 2. 監控系統

#### Streamlit Dashboard
```bash
streamlit run monitoring_dashboard.py
# 訪問 http://localhost:8501
```

#### Flask Web Dashboard
```bash
python web_dashboard.py
# 訪問 http://localhost:5000
```

#### Production Dashboard
```bash
python production/dashboard.py
# 訪問 http://localhost:8080
```

### 3. 評估與回測

```python
from utils.backtesting import BacktestEngine

engine = BacktestEngine(
    model_path='models/ppo_btc_best.zip',
    data_file='data/btc_usdt_1m_2023.csv'
)

results = engine.run()
print(f"總回報: {results['total_return']:.2%}")
print(f"夏普比率: {results['sharpe_ratio']:.2f}")
```

### 4. 生產 API

```bash
# 啟動 API Server
python production/api.py

# API 端點
# GET  /api/models              - 獲取所有模型
# POST /api/predict             - 模型預測
# GET  /api/training_runs       - 訓練記錄
# POST /api/backtest            - 執行回測
```

---

## ⚙️ 配置說明

### 環境變數 (.env)

```bash
# 資料庫類型（sqlite 或 postgresql）
DB_TYPE=sqlite

# SQLite 設定
SQLITE_DB_PATH=logs/metrics.db

# PostgreSQL 設定（Docker 使用）
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=rl_market
POSTGRES_USER=rl_user
POSTGRES_PASSWORD=rl_password

# Dashboard Ports
WEB_DASHBOARD_PORT=5555
MONITORING_DASHBOARD_PORT=5556
PRODUCTION_DASHBOARD_PORT=8080

# Logging
LOG_LEVEL=DEBUG
```

### 訓練配置 (configs/default.yaml)

```yaml
training:
  algorithm: PPO
  total_timesteps: 100000
  n_steps: 2048
  learning_rate: 0.0003
  
environment:
  initial_balance: 10000
  fee_rate: 0.001
  max_position: 1.0
  
reward:
  type: risk_adjusted
  sharpe_weight: 0.3
  drawdown_penalty: 0.2
```

---

## 🐳 Docker 使用

### 服務組成

```yaml
services:
  postgres:              # PostgreSQL 資料庫
  pgadmin:               # pgAdmin Web UI (http://localhost:5050)
  dashboard_monitoring:  # Streamlit Dashboard
  dashboard_web:         # Flask API Dashboard
  dashboard_production:  # 生產 Dashboard
  training:              # 訓練服務
  nginx:                 # 反向代理
```

### 常用命令

```bash
# 啟動所有服務
docker-compose up -d

# 啟動特定服務
docker-compose up -d postgres

# 查看服務狀態
docker-compose ps

# 查看日誌
docker-compose logs -f postgres

# 停止所有服務
docker-compose down

# 停止並刪除所有資料
docker-compose down -v

# 重新構建
docker-compose build --no-cache
```

### 埠號映射

| 服務 | 內部埠號 | 外部埠號 | 說明 |
|------|---------|---------|------|
| PostgreSQL | 5432 | 5432 | 資料庫 |
| pgAdmin | 80 | 5050 | 資料庫管理介面 |
| Dashboard Monitoring | 8501 | 8501 | Streamlit |
| Dashboard Web | 5000 | 5001 | Flask API |
| Dashboard Production | 8502 | 8502 | 生產介面 |
| Nginx | 80 | 8080 | 反向代理 |

---

## 🔧 開發指南

### 安裝開發環境

```bash
# 建立虛擬環境
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 安裝依賴
pip install -r requirements.txt

# 安裝測試工具
pip install pytest pytest-cov
```

### 執行測試

```bash
# 方式 1: 使用測試腳本
./run_tests.sh

# 方式 2: 使用 Python 腳本
python run_tests.py

# 方式 3: 直接使用 pytest
pytest tests/ -v

# 執行特定測試
pytest tests/test_production.py -v
pytest tests/test_database.py -v

# 生成覆蓋率報告
pytest tests/ --cov=. --cov-report=html
# 報告位置: htmlcov/index.html
```

**測試套件包含:**
- ✅ 環境測試 (test_env_basic.py)
- ✅ 獎勵函數測試 (test_reward.py)
- ✅ 資料庫測試 (test_database.py)
- ✅ 生產模組測試 (test_production.py)
- ✅ 整合測試 (test_integration.py)
- ✅ API 測試 (test_api.py)

查看完整測試報告: [TEST_REPORT.md](TEST_REPORT.md)

### 資料庫測試

```bash
# 測試 SQLite
python scripts/test_database.py

# 測試 PostgreSQL（需先啟動 Docker）
docker-compose up -d postgres
python scripts/test_postgres.py
```

### 新增交易對

```python
from utils.database import get_database

db = get_database()

# 插入新交易對
db.execute("""
    INSERT INTO symbols (symbol, base_currency, quote_currency)
    VALUES ('ADAUSDT', 'ADA', 'USDT')
    ON CONFLICT (symbol) DO NOTHING
""")
db.commit()
```

### 查詢訓練記錄

```python
from utils.database import get_database

db = get_database()

# 查詢最新 10 次訓練
runs = db.fetchall("""
    SELECT run_id, symbol, algorithm, final_pnl, status
    FROM training_runs
    ORDER BY start_time DESC
    LIMIT 10
""")

for run in runs:
    print(f"{run['run_id']}: {run['symbol']} | PnL: {run['final_pnl']}")
```

---

## 📈 效能指標

系統計算以下效能指標：

- **PnL (Profit and Loss)**: 總損益
- **Total Return**: 總回報率
- **Sharpe Ratio**: 夏普比率（風險調整回報）
- **Sortino Ratio**: 索提諾比率
- **Max Drawdown**: 最大回撤
- **Win Rate**: 勝率
- **Total Trades**: 總交易次數
- **Avg Trade Duration**: 平均持倉時間

---

## 🛠️ 故障排除

### 1. PostgreSQL 連接失敗

**問題**: `connection refused` 或 `role does not exist`

**解決**:
```bash
# 重新啟動 PostgreSQL 和 pgAdmin
docker-compose down -v
docker-compose up -d postgres pgadmin

# 等待初始化完成（約 10-15 秒）
sleep 15

# 驗證資料庫
docker-compose exec postgres psql -U rl_user -d rl_market -c "\dt"

# 訪問 pgAdmin
open http://localhost:5050
```

**使用 Docker 內建的 pgAdmin（推薦）：**
- Host 使用 `postgres`（Docker 服務名稱）
- 不需要 localhost 或 127.0.0.1

### 2. 訓練無法開始

**問題**: 找不到資料檔案

**解決**:
```bash
# 下載資料
python scripts/fetch_data.py --symbol BTCUSDT --days 365

# 檢查資料
ls -lh data/btc_usdt_1m*.csv
```

### 3. Dashboard 無法啟動

**問題**: 埠號被佔用

**解決**:
```bash
# 檢查佔用的埠號
lsof -i :8501  # Streamlit
lsof -i :5000  # Flask

# 殺掉佔用的程序或修改 .env 中的埠號
```

### 4. Docker 記憶體不足

**問題**: 容器頻繁重啟

**解決**:
```bash
# 增加 Docker 記憶體配置（Docker Desktop → Settings → Resources）
# 建議至少 4GB RAM

# 或限制單個服務的記憶體使用
docker-compose up -d --scale training=0  # 停用訓練服務
```

---

## 🎓 使用範例

### 完整訓練流程

```bash
# 1. 準備資料
python scripts/fetch_data.py --symbol BTCUSDT --days 365

# 2. 訓練模型
python scripts/train.py --symbol BTCUSDT --algorithm PPO --episodes 1000

# 3. 評估模型
python scripts/evaluate.py --model models/ppo_btc_best.zip

# 4. 回測
python -c "
from utils.backtesting import BacktestEngine
engine = BacktestEngine('models/ppo_btc_best.zip', 'data/btc_usdt_1m_2023.csv')
results = engine.run()
print(results)
"

# 5. 部署到生產
python production/api.py
```

### 批次訓練多幣種

```bash
# 自動訓練並儲存最佳模型
python auto_pipeline.py \
    --symbols BTCUSDT ETHUSDT BNBUSDT SOLUSDT \
    --algorithm PPO \
    --version v2 \
    --episodes 1000
```

### 查看訓練進度

```bash
# 方法 1: Streamlit Dashboard（即時監控）
streamlit run monitoring_dashboard.py

# 方法 2: 命令列查詢
docker-compose exec postgres psql -U rl_user -d rl_market -c "
SELECT 
    run_id,
    symbol,
    algorithm,
    status,
    total_episodes,
    final_pnl
FROM v_latest_training_runs
WHERE status = 'running'
ORDER BY start_time DESC;"
```

---

## 📦 依賴套件

主要依賴：

- **stable-baselines3** - RL 演算法實現
- **gymnasium** - 環境標準
- **numpy / pandas** - 數據處理
- **streamlit** - 監控 Dashboard
- **flask** - Web API
- **psycopg2-binary** - PostgreSQL 連接
- **sqlalchemy** - ORM 支援
- **plotly** - 圖表視覺化

完整列表請見 `requirements.txt`

---

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

### 開發流程

```bash
# 1. Fork 專案
# 2. 建立分支
git checkout -b feature/your-feature

# 3. 開發並測試
python tests/test_*.py

# 4. 提交更改
git add .
git commit -m "feat: your feature description"

# 5. 推送並創建 Pull Request
git push origin feature/your-feature
```

---

## 📝 授權

MIT License - 詳見 LICENSE 檔案

---

## 🙏 致謝

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [Gymnasium](https://gymnasium.farama.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Binance](https://www.binance.com/)

---

<div align="center">

**Made with ❤️ by RL Market Team**

⭐ 如果這個專案對你有幫助，請給個 Star！

</div>
