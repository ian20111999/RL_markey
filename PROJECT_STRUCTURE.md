# RL Market 專案結構

> 更新時間: 2025-12-27

## 快速導覽

```
RL_market/
├── 📁 configs/          # 配置文件
├── 📁 data/             # 訓練資料
├── 📁 docs/             # 文檔
├── 📁 envs/             # RL 環境
├── 📁 frontend-next/    # Next.js Dashboard (新)
├── 📁 models/           # 訓練模型
├── 📁 production/       # 生產部署
├── 📁 scripts/          # 執行腳本
├── 📁 tests/            # 測試
└── 📁 utils/            # 工具模組
```

---

## 核心模組

### 🎮 `envs/` - RL 環境

| 檔案 | 功能 |
|------|------|
| `market_making_env.py` | 主環境實現 (MarketMakingEnv) |
| `env_factory.py` | 環境工廠 + 配置載入 |
| `config_schema.py` | Pydantic 配置驗證 |
| `constants.py` | 環境常數 |

### 🛠 `utils/` - 工具模組

| 檔案 | 功能 |
|------|------|
| `algorithms.py` | SAC/PPO/TD3 模型創建 |
| `backtesting.py` | 回測引擎 + Walk-Forward |
| `curriculum.py` | 課程學習 |
| `ensemble.py` | 集成學習 |
| `database.py` | 資料庫抽象層 |
| `cache_manager.py` | 快取管理 |
| `logging_config.py` | 日誌配置 |

### 📜 `scripts/` - 執行腳本

**訓練相關**
| 檔案 | 功能 |
|------|------|
| `train.py` | 核心訓練腳本 |
| `train_flexible.py` | 多演算法訓練 (新) |
| `hyperparameter_tuning.py` | Optuna HPO (新) |
| `compare_algorithms.py` | 演算法比較 (新) |
| `validate_model.py` | 模型驗證 (新) |

**Pipeline**
| 檔案 | 功能 |
|------|------|
| `pipeline.py` | 基礎訓練流程 |
| `optimized_pipeline.py` | 優化版流程 |

**資料庫**
| 檔案 | 功能 |
|------|------|
| `init_db.sql` | PostgreSQL Schema |
| `init_db_sqlite.sql` | SQLite Schema |

### 🏭 `production/` - 生產模組

| 檔案 | 功能 |
|------|------|
| `api.py` | FastAPI 服務 |
| `model_registry.py` | 模型註冊表 |
| `trader.py` | 交易執行 |

### 🧪 `tests/` - 測試

| 檔案 | 覆蓋範圍 |
|------|----------|
| `conftest.py` | 共用 Fixtures |
| `test_env_basic.py` | 環境基礎測試 |
| `test_reward.py` | 獎勵函數測試 |
| `test_integration.py` | 整合測試 |
| `test_database.py` | 資料庫測試 |
| `test_production.py` | 生產環境測試 |
| `test_utils.py` | 工具模組測試 |

### 🎨 `frontend-next/` - Next.js Dashboard (新)

```
frontend-next/
├── src/
│   ├── app/           # App Router 頁面
│   │   ├── layout.tsx
│   │   ├── page.tsx   # Dashboard 主頁
│   │   └── globals.css
│   └── components/    # UI 組件
│       ├── Sidebar.tsx
│       ├── MetricCard.tsx
│       ├── PerformanceChart.tsx
│       ├── ModelList.tsx
│       └── TrainingStatus.tsx
├── package.json
├── tsconfig.json
└── tailwind.config.js
```

---

## 配置與資料

### `configs/`
- `default.yaml` - 預設訓練配置

### `data/`
- `.gitkeep` - 資料目錄
- CSV 檔案由 `.gitignore` 忽略

---

## Docker 與部署

| 檔案 | 功能 |
|------|------|
| `docker-compose.yml` | 完整服務編排 |
| `Dockerfile` | 基礎映像 |
| `Dockerfile.training` | 訓練映像 |
| `Dockerfile.dashboard` | Dashboard 映像 |
| `nginx.conf` | Nginx 配置 |

---

## CI/CD

```
.github/
└── workflows/
    ├── test.yml      # 自動測試
    └── deploy.yml    # 自動部署
```

---

## Shell 腳本

| 檔案 | 功能 |
|------|------|
| `quickstart.sh` | 快速啟動 |
| `run_tests.sh` | 執行測試 |
| `run_training.sh` | 執行訓練 |
| `start_dashboard.sh` | 啟動 Dashboard |

---

## 文檔

| 檔案 | 內容 |
|------|------|
| `README.md` | 專案概覽 |
| `docs/API.md` | API 文檔 |
| `PROJECT_STRUCTURE.md` | 專案結構 (本文件) |
| `CLEANUP_GUIDE.md` | 清理指南 |
| `OPTIMIZATION_*.md` | 優化報告 |

---

## 使用流程

### 訓練模型
```bash
# 基本訓練
python scripts/train.py --config configs/default.yaml

# 多演算法訓練
python scripts/train_flexible.py --algorithm sac --curriculum

# 超參數優化
python scripts/hyperparameter_tuning.py --algorithm sac --n_trials 50
```

### 驗證模型
```bash
python scripts/validate_model.py --model models/xxx.zip
```

### 啟動前端
```bash
cd frontend-next && npm run dev
```

### 執行測試
```bash
pytest tests/ -v
```
