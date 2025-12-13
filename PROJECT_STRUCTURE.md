# 📂 Project Structure & Organization

## Overview

此文件詳細說明 RL Market Making 專案的完整結構、組織方式與最佳實踐。

---

## 🗂️ Directory Structure

```
RL_markey/
├── 📁 configs/                     # 配置檔案
│   ├── env_v3_full.yaml           # V3 完整環境配置（推薦）
│   ├── env_v2.yaml                # V2 基礎配置
│   └── env_baseline.yaml          # 基準配置
│
├── 📁 data/                        # 訓練數據（需自行下載）
│   └── btc_usdt_1m_2023.csv       # BTC/USDT 1分鐘K線數據
│
├── 📁 envs/                        # Gymnasium 交易環境
│   ├── market_making_env.py       # 主要做市環境
│   ├── realistic_fill_model.py    # 真實成交模型
│   ├── env_factory.py             # 環境工廠
│   ├── config_schema.py           # 配置結構定義
│   └── constants.py               # 環境常數
│
├── 📁 utils/                       # 工具模組
│   ├── algorithms.py              # 多演算法工廠 (SAC/PPO/TD3)
│   ├── risk_sensitive.py          # 風險敏感訓練
│   ├── curriculum.py              # 課程學習
│   ├── backtesting.py             # 專業回測框架
│   ├── ensemble.py                # 集成學習
│   ├── explainability.py          # 可解釋性分析
│   ├── online_adaptation.py       # 線上適應
│   ├── distributed_training.py    # 分散式訓練
│   ├── report_generator.py        # 報告生成器
│   ├── metrics.py                 # 性能指標計算
│   ├── config.py                  # 配置載入工具
│   ├── lstm_features.py           # LSTM特徵提取
│   └── numba_optimizations.py     # 性能優化
│
├── 📁 scripts/                     # 執行腳本
│   ├── train.py                   # 通用訓練腳本
│   ├── evaluate.py                # 評估腳本
│   └── visualize_episode.py       # 可視化工具
│
├── 📁 production/                  # 🆕 生產級功能
│   ├── model_registry.py          # 模型註冊與管理系統
│   ├── api.py                     # FastAPI REST API
│   ├── cli.py                     # 命令列介面
│   ├── dashboard.py               # Web 監控面板
│   └── __init__.py
│
├── 📁 tests/                       # 測試套件
│   └── test_production.py         # 生產功能測試
│
├── 📁 examples/                    # 🆕 使用範例
│   ├── api_usage.py               # API 使用範例
│   └── complete_workflow.py       # 完整工作流程範例
│
├── 📁 docs/                        # 文檔
│   ├── QUICKSTART.md              # 5分鐘快速開始
│   ├── PRODUCTION_GUIDE.md        # 生產部署指南（英文）
│   └── USER_GUIDE_ZH.md           # 用戶使用指南（中文）
│
├── 📁 models/                      # 訓練後的模型
│   └── registry/                  # 模型註冊庫
│
├── 📁 runs/                        # TensorBoard 訓練記錄
│
├── 📄 pipeline.py                  # 主要訓練管線
├── 📄 run_training.sh              # 訓練執行腳本
├── 📄 requirements.txt             # Python 依賴（包含生產環境）
├── 📄 README.md                    # 主要說明文件
├── 📄 PRODUCTION_SUMMARY.md        # 生產功能總結
├── 📄 PROJECT_STRUCTURE.md         # 本文件
├── 📄 Dockerfile                   # Docker 映像檔定義
├── 📄 docker-compose.yml           # Docker Compose 配置
├── 📄 .dockerignore                # Docker 忽略檔案
└── 📄 .gitignore                   # Git 忽略檔案
```

---

## 🎯 Core Components

### 1. Trading Environment (`envs/`)

**Purpose**: 實現符合 Gymnasium 標準的加密貨幣做市交易環境

**Key Features**:
- ✅ 真實的市場動態模擬
- ✅ 靈活的獎勵函數設計
- ✅ Domain Randomization 支援
- ✅ 可配置的交易規則

**Main Files**:
- `market_making_env.py`: 主要環境實現
- `realistic_fill_model.py`: 真實成交邏輯
- `env_factory.py`: 環境創建工廠

### 2. Training Utilities (`utils/`)

**Purpose**: 提供訓練所需的各種輔助功能

**Modules**:

#### 2.1 Algorithm Support
- `algorithms.py`: 統一的演算法介面 (SAC/PPO/TD3)

#### 2.2 Advanced Training
- `risk_sensitive.py`: CVaR、Mean-Variance 風險優化
- `curriculum.py`: 漸進式難度訓練
- `distributed_training.py`: 分散式訓練與超參數搜尋

#### 2.3 Evaluation & Analysis
- `backtesting.py`: Walk-Forward、Monte Carlo 回測
- `explainability.py`: SHAP、Attention 分析
- `metrics.py`: 專業交易指標計算

#### 2.4 Post-Processing
- `ensemble.py`: 模型集成
- `online_adaptation.py`: 線上學習適應
- `report_generator.py`: HTML/PDF 報告生成

### 3. Production System (`production/`) 🆕

**Purpose**: 提供完整的生產級部署與管理功能

**Components**:

#### 3.1 Model Registry (`model_registry.py`)
- 自動版本管理
- 性能指標追蹤
- 5項生產驗證標準
- 盈利能力評分系統

#### 3.2 REST API (`api.py`)
- FastAPI 實現
- 模型推論端點
- 健康檢查
- 自動文檔生成

#### 3.3 CLI Tool (`cli.py`)
- 訓練自動化
- 模型管理
- 排行榜導出
- API 服務啟動

#### 3.4 Web Dashboard (`dashboard.py`)
- 即時系統狀態
- 模型性能視覺化
- 響應式 UI

### 4. Configuration System (`configs/`)

**Purpose**: 集中管理環境與訓練參數

**Best Practices**:
- 使用 YAML 格式
- 分離不同實驗配置
- 包含完整註釋說明

### 5. Documentation (`docs/`)

**Files**:
- `QUICKSTART.md`: 新手5分鐘快速上手
- `PRODUCTION_GUIDE.md`: 完整生產部署文檔（英文）
- `USER_GUIDE_ZH.md`: 詳細使用指南（中文）

---

## 🔄 Typical Workflows

### Workflow 1: Research & Development

```bash
# 1. 準備數據
python scripts/fetch_binance_ohlcv.py --symbol BTC/USDT --days 365

# 2. 調整配置
vim configs/env_v3_full.yaml

# 3. 訓練模型
python scripts/train.py --config configs/env_v3_full.yaml --algorithm SAC

# 4. 評估模型
python scripts/evaluate.py --model models/sac_model.zip

# 5. 可視化
python scripts/visualize_episode.py --model models/sac_model.zip
```

### Workflow 2: Production Deployment

```bash
# 1. 安裝依賴
pip install -r requirements.txt

# 2. 訓練可獲利模型
python production/cli.py train --symbol btc --attempts 3

# 3. 查看已註冊模型
python production/cli.py list

# 4. 啟動 API 服務
python production/cli.py serve --port 8000

# 5. 啟動監控面板
python production/dashboard.py
```

### Workflow 3: Docker Deployment

```bash
# 1. 構建映像
docker-compose build

# 2. 啟動服務
docker-compose up -d

# 3. 檢查狀態
docker-compose ps
curl http://localhost:8000/health

# 4. 查看日誌
docker-compose logs -f
```

---

## 🧪 Testing Strategy

### Unit Tests
```bash
pytest tests/test_production.py -v
```

### Integration Tests
```bash
# 測試完整工作流程
python examples/complete_workflow.py
```

### API Tests
```bash
# 啟動 API
python production/cli.py serve &

# 測試端點
curl http://localhost:8000/health
curl http://localhost:8000/models
```

---

## 📦 Dependency Management

### Core Dependencies
- **RL Framework**: Stable-Baselines3, Gymnasium
- **Deep Learning**: PyTorch
- **Data Processing**: Pandas, NumPy
- **Optimization**: Optuna

### Production Dependencies
- **API**: FastAPI, Uvicorn
- **Validation**: Pydantic

### Optional Dependencies
- **Performance**: Numba (10-50x speedup)
- **Reporting**: WeasyPrint (PDF generation)
- **ML**: scikit-learn (ensemble methods)

**All dependencies are consolidated in `requirements.txt`**

---

## 🔐 Security Best Practices

1. **API Key Management**
   - Use environment variables for sensitive data
   - Never commit API keys to repository

2. **Model Validation**
   - All models must pass 5 production criteria
   - Automated profitability checks

3. **Docker Security**
   - Use slim Python images
   - Run as non-root user (if needed)
   - Health checks enabled

---

## 🚀 Performance Optimization

### Training Speed
- Use Numba JIT compilation (10-50x speedup)
- Enable vectorized environments
- Distributed training for hyperparameter search

### Inference Speed
- Model caching in production API
- Batch prediction support
- Efficient state preprocessing

### Memory Usage
- Rolling window for large datasets
- Efficient replay buffer management
- Model compression (if needed)

---

## 📊 Monitoring & Logging

### Training Monitoring
- TensorBoard integration (`runs/` directory)
- Real-time metric logging
- Episode statistics tracking

### Production Monitoring
- Web dashboard (port 8080)
- API health checks
- Performance metrics tracking

### Logging Strategy
```python
# Example logging setup
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
```

---

## 🔄 Version Control Strategy

### Branch Strategy
- `main`: Production-ready code (all features consolidated)
- Feature branches: Short-lived for specific features

### Commit Guidelines
- Use descriptive commit messages
- One logical change per commit
- Reference issues in commits

---

## 📚 Additional Resources

### Internal Documentation
- [README.md](README.md) - Main project overview
- [PRODUCTION_SUMMARY.md](PRODUCTION_SUMMARY.md) - Production features
- [docs/QUICKSTART.md](docs/QUICKSTART.md) - Quick start guide
- [docs/PRODUCTION_GUIDE.md](docs/PRODUCTION_GUIDE.md) - Deployment guide
- [docs/USER_GUIDE_ZH.md](docs/USER_GUIDE_ZH.md) - Chinese user guide

### Code Examples
- [examples/api_usage.py](examples/api_usage.py) - API usage examples
- [examples/complete_workflow.py](examples/complete_workflow.py) - Full workflow

### External Resources
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

## 🛠️ Maintenance Guidelines

### Regular Tasks
- [ ] Update dependencies monthly
- [ ] Run full test suite before releases
- [ ] Monitor model performance metrics
- [ ] Review and update documentation

### Troubleshooting
1. **Training fails**: Check data availability and config validity
2. **API errors**: Verify model registry and dependencies
3. **Performance issues**: Enable Numba, check system resources
4. **Docker issues**: Review logs with `docker-compose logs`

---

## 📝 Contributing

When contributing to this project:

1. **Code Style**: Follow PEP 8 guidelines
2. **Documentation**: Update relevant docs for any changes
3. **Testing**: Add tests for new features
4. **Configuration**: Document new config parameters

---

## 🎓 Learning Path

### Beginner
1. Read [QUICKSTART.md](docs/QUICKSTART.md)
2. Run basic training with default configs
3. Explore visualization tools

### Intermediate
1. Study [USER_GUIDE_ZH.md](docs/USER_GUIDE_ZH.md)
2. Experiment with different algorithms
3. Customize reward functions

### Advanced
1. Read [PRODUCTION_GUIDE.md](docs/PRODUCTION_GUIDE.md)
2. Implement custom features
3. Deploy to production
4. Contribute improvements

---

**Last Updated**: 2025-12-13
**Maintainer**: RL Market Making Team
