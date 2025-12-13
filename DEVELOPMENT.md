# 🛠️ Development Guide

## 開發環境設置

### 前置需求

- Python 3.10 或更高版本
- Git
- 至少 8GB RAM
- （可選）CUDA 支援的 GPU

### 本地開發設置

```bash
# 1. Clone 專案
git clone https://github.com/ian20111999/RL_markey.git
cd RL_markey

# 2. 創建虛擬環境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows

# 3. 安裝依賴
pip install -r requirements.txt

# 4. 安裝開發工具（可選）
pip install black flake8 mypy pytest-cov

# 5. 驗證安裝
python -c "import stable_baselines3; print('SB3 version:', stable_baselines3.__version__)"
```

---

## 🏗️ 專案架構

### 核心組件

```
┌─────────────────────────────────────────────────────────────┐
│                     RL Market Making System                  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   │
│  │   Training   │   │  Production  │   │  Monitoring  │   │
│  │   Pipeline   │──▶│     API      │──▶│  Dashboard   │   │
│  └──────────────┘   └──────────────┘   └──────────────┘   │
│         │                    │                    │         │
│         ▼                    ▼                    ▼         │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   │
│  │     RL       │   │    Model     │   │   Metrics    │   │
│  │  Algorithms  │   │   Registry   │   │   Tracking   │   │
│  └──────────────┘   └──────────────┘   └──────────────┘   │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐                                          │
│  │   Trading    │                                          │
│  │ Environment  │                                          │
│  └──────────────┘                                          │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 模組職責

| 模組 | 職責 | 主要文件 |
|------|------|----------|
| **envs/** | 交易環境實現 | `market_making_env.py` |
| **utils/** | 訓練工具與輔助功能 | `algorithms.py`, `backtesting.py` |
| **production/** | 生產級部署功能 | `api.py`, `model_registry.py` |
| **scripts/** | 執行腳本 | `train.py`, `evaluate.py` |
| **configs/** | 配置管理 | `*.yaml` |

---

## 🧪 測試策略

### 運行測試

```bash
# 運行所有測試
pytest tests/ -v

# 運行特定測試文件
pytest tests/test_production.py -v

# 運行測試並生成覆蓋率報告
pytest tests/ --cov=production --cov-report=html

# 運行特定測試
pytest tests/test_production.py::test_model_registry -v
```

### 測試結構

```
tests/
├── test_production.py      # 生產功能測試
├── test_environment.py     # 環境測試（待添加）
├── test_algorithms.py      # 演算法測試（待添加）
└── test_integration.py     # 整合測試（待添加）
```

### 編寫測試

```python
import pytest
from production.model_registry import ModelRegistry

def test_model_registration():
    """測試模型註冊功能"""
    registry = ModelRegistry()
    
    # 註冊模型
    metadata = {
        "symbol": "BTC/USDT",
        "algorithm": "SAC",
        "mean_pnl": 150.5,
        "win_rate": 0.65,
        "sharpe_ratio": 1.5
    }
    
    model_id = registry.register_model(
        model_path="models/test_model.zip",
        metadata=metadata
    )
    
    # 驗證
    assert model_id is not None
    models = registry.list_models()
    assert len(models) > 0
```

---

## 🎨 代碼風格

### Python 風格指南

遵循 [PEP 8](https://pep8.org/) 標準：

```bash
# 檢查代碼風格
flake8 production/ utils/ envs/

# 自動格式化代碼
black production/ utils/ envs/

# 類型檢查
mypy production/ --ignore-missing-imports
```

### 命名慣例

- **類名**: `PascalCase` (例: `ModelRegistry`, `MarketMakingEnv`)
- **函數/變量**: `snake_case` (例: `train_model`, `mean_pnl`)
- **常數**: `UPPER_SNAKE_CASE` (例: `MAX_INVENTORY`, `DEFAULT_FEE`)
- **私有成員**: `_leading_underscore` (例: `_validate_model`)

### 文檔字符串

使用 Google 風格的 docstring：

```python
def train_profitable_model(symbol: str, max_attempts: int = 3) -> Optional[str]:
    """訓練一個可獲利的模型。
    
    Args:
        symbol: 交易對符號 (例: 'BTC/USDT')
        max_attempts: 最大訓練嘗試次數
        
    Returns:
        成功時返回模型ID，失敗時返回None
        
    Raises:
        ValueError: 當symbol格式不正確時
        
    Example:
        >>> model_id = train_profitable_model("BTC/USDT", attempts=3)
        >>> print(f"Model trained: {model_id}")
    """
    pass
```

---

## 🔄 Git 工作流程

### 分支策略

- **main**: 穩定的生產代碼（所有功能已整合）
- **feature/xxx**: 新功能開發
- **bugfix/xxx**: Bug 修復
- **hotfix/xxx**: 緊急修復

### 提交規範

使用語義化提交訊息：

```
<type>(<scope>): <subject>

<body>

<footer>
```

**類型 (type)**:
- `feat`: 新功能
- `fix`: Bug 修復
- `docs`: 文檔更新
- `style`: 代碼格式調整
- `refactor`: 代碼重構
- `test`: 測試相關
- `chore`: 建構/工具相關

**範例**:
```
feat(production): add model profitability scoring

- Implement 0-100 scoring system
- Add automatic validation criteria
- Update dashboard to show scores

Closes #123
```

### Pull Request 流程

1. **創建分支**
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **開發並提交**
   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   ```

3. **推送分支**
   ```bash
   git push origin feature/amazing-feature
   ```

4. **開啟 PR**
   - 填寫清晰的 PR 描述
   - 關聯相關 Issue
   - 請求代碼審查

5. **代碼審查**
   - 回應審查意見
   - 進行必要的修改

6. **合併**
   - 確保所有測試通過
   - Squash and merge

---

## 🐛 調試技巧

### 訓練調試

```python
# 啟用詳細日誌
import logging
logging.basicConfig(level=logging.DEBUG)

# 使用小數據集測試
env_config = {
    'data_path': 'data/btc_sample_100.csv',  # 小數據集
    'max_steps': 100  # 減少步數
}

# 啟用環境渲染
env.render()

# 檢查觀察空間
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")
print(f"Observation: {obs}")
```

### API 調試

```bash
# 啟用 FastAPI 調試模式
uvicorn production.api:app --reload --log-level debug

# 測試 API 端點
curl -X GET http://localhost:8000/health
curl -X GET http://localhost:8000/models

# 查看 API 文檔
open http://localhost:8000/docs
```

### 性能分析

```python
import cProfile
import pstats

# 性能分析
profiler = cProfile.Profile()
profiler.enable()

# 執行代碼
model.learn(total_timesteps=10000)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # 顯示前20個最慢的函數
```

---

## 📊 性能優化

### 訓練加速

1. **使用 Numba JIT 編譯**
   ```python
   from numba import jit
   
   @jit(nopython=True)
   def compute_reward(price, inventory):
       # 快速計算
       return price * inventory
   ```

2. **向量化環境**
   ```python
   from stable_baselines3.common.vec_env import SubprocVecEnv
   
   env = SubprocVecEnv([make_env() for _ in range(4)])
   ```

3. **使用 GPU**
   ```python
   model = SAC("MlpPolicy", env, device="cuda")
   ```

### 記憶體優化

```python
# 限制 Replay Buffer 大小
model = SAC(
    "MlpPolicy",
    env,
    buffer_size=50000,  # 減少記憶體使用
    optimize_memory_usage=True
)

# 使用滾動窗口讀取大數據
df = pd.read_csv('data.csv', chunksize=10000)
for chunk in df:
    process_chunk(chunk)
```

---

## 📦 發布流程

### 版本號規範

遵循 [Semantic Versioning](https://semver.org/)：

- **MAJOR**: 不相容的 API 變更
- **MINOR**: 向下相容的新功能
- **PATCH**: 向下相容的 Bug 修復

### 發布檢查清單

- [ ] 所有測試通過
- [ ] 文檔已更新
- [ ] CHANGELOG 已更新
- [ ] 版本號已更新
- [ ] 代碼已審查
- [ ] 性能測試完成
- [ ] 安全檢查完成

---

## 🔐 安全性

### 敏感資訊管理

```bash
# 使用環境變量
export BINANCE_API_KEY="your_key_here"
export BINANCE_API_SECRET="your_secret_here"

# 或使用 .env 文件
cat > .env << EOF
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
EOF

# 確保 .env 在 .gitignore 中
echo ".env" >> .gitignore
```

### 代碼掃描

```bash
# 掃描依賴漏洞
pip install safety
safety check

# 掃描代碼安全問題
pip install bandit
bandit -r production/ utils/ envs/
```

---

## 📚 學習資源

### 推薦閱讀

1. **強化學習基礎**
   - [Sutton & Barto - Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html)
   - [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)

2. **市場做市**
   - Market Making and Algorithmic Trading
   - High-Frequency Trading: A Practical Guide

3. **軟件工程**
   - Clean Code by Robert C. Martin
   - The Pragmatic Programmer

### 相關工具

- **RL 訓練**: Stable-Baselines3, RLlib
- **回測**: Backtrader, Zipline
- **可視化**: TensorBoard, Plotly
- **API**: FastAPI, Flask
- **容器化**: Docker, Kubernetes

---

## 🤝 社群

### 獲取幫助

1. **查看文檔**: 先查閱 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
2. **搜尋 Issues**: 檢查是否有類似問題
3. **提問**: 創建新的 Issue 並提供：
   - 清晰的問題描述
   - 重現步驟
   - 錯誤訊息
   - 環境資訊

### 貢獻者行為準則

- 尊重所有參與者
- 接受建設性批評
- 關注對社群最有利的事
- 展現對其他社群成員的同理心

---

## 📝 待辦事項

### 高優先級
- [ ] 添加更多單元測試
- [ ] 實現 CI/CD 管線
- [ ] 優化訓練速度
- [ ] 添加更多示例

### 中優先級
- [ ] 支援更多交易對
- [ ] 實現模型 A/B 測試
- [ ] 添加更多風險指標
- [ ] 改進文檔

### 低優先級
- [ ] 添加更多可視化
- [ ] 實現移動端支援
- [ ] 多語言支援
- [ ] 社群論壇

---

## 🎓 常見問題

### Q: 訓練需要多長時間？
A: 取決於配置，通常 10-30 分鐘可以完成一次訓練。

### Q: 需要 GPU 嗎？
A: 不是必需的，但使用 GPU 可以顯著加速訓練（2-5x）。

### Q: 如何提高模型性能？
A: 嘗試：
1. 調整獎勵函數
2. 使用課程學習
3. 增加訓練時間
4. 啟用風險敏感訓練
5. 使用超參數優化

### Q: 可以用於實盤交易嗎？
A: 本專案僅供研究和教育用途。實盤交易需要額外的風險控制和測試。

---

**Last Updated**: 2025-12-13

**Maintainer**: RL Market Making Team

[回到 README](README.md)
