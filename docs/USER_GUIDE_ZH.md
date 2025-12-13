# 🚀 生產級 RL 做市系統 - 使用指南

## 概述

此系統讓任何人都能輕鬆訓練出穩定且可獲利的做市模型。系統會自動驗證模型品質，確保只有真正能賺錢的模型才會被部署到生產環境。

---

## 🎯 核心特性

### 1. 自動化訓練
- ✅ 一鍵訓練，自動重試直到獲得可獲利模型
- ✅ 自動驗證 5 大盈利標準
- ✅ 智能評分系統（0-100分）
- ✅ 失敗自動重試機制

### 2. 生產級部署
- ✅ REST API 服務（FastAPI）
- ✅ Docker 容器化支援
- ✅ Web 監控面板
- ✅ 健康檢查與監控

### 3. 模型管理
- ✅ 中央化模型註冊系統
- ✅ 版本控制與元數據追蹤
- ✅ 自動選擇最佳模型
- ✅ 性能排行榜

---

## 🚀 快速開始（5分鐘）

### 步驟 1: 安裝依賴

```bash
pip install -r requirements.txt
pip install -r requirements-production.txt
```

### 步驟 2: 訓練第一個模型

```bash
# 自動訓練直到獲得可獲利模型（最多重試3次）
python production/cli.py train --symbol btc --attempts 3
```

系統會：
1. 訓練模型
2. 在驗證集上測試
3. 檢查是否符合盈利標準
4. 如果不符合，自動重試
5. 註冊最佳模型

### 步驟 3: 啟動服務

**選項 A - API 服務：**
```bash
python production/cli.py serve --port 8000
# 訪問 http://localhost:8000/docs 查看 API 文檔
```

**選項 B - Web 監控面板：**
```bash
python production/dashboard.py
# 訪問 http://localhost:8080 查看儀表板
```

**選項 C - Docker 部署：**
```bash
docker-compose up -d
```

---

## 📊 盈利標準

模型必須同時滿足以下 5 個條件才算「生產就緒」：

| 標準 | 要求 | 說明 |
|------|------|------|
| **平均損益** | > 0 | 必須是獲利的 |
| **勝率** | > 50% | 多數交易要賺錢 |
| **夏普比率** | > 1.0 | 風險調整後的報酬要好 |
| **最大回撤** | < 20% | 風險控制要好 |
| **綜合評分** | > 60/100 | 整體表現要達標 |

### 綜合評分計算方式

分數範圍 0-100，權重分配：
- 平均損益：30%
- 勝率：25%
- 夏普比率：25%
- 最大回撤：20%

**評分等級：**
- 🏆 優秀：70-100 分
- ✅ 良好：60-69 分
- ⚠️ 不及格：< 60 分

---

## 🛠️ 命令列工具（CLI）

### 訓練模型

```bash
# 基本訓練
python production/cli.py train --symbol btc

# 進階選項
python production/cli.py train \
  --symbol btc \
  --algorithm SAC \
  --attempts 5 \
  --timesteps 300000 \
  --config configs/default.yaml
```

**參數說明：**
- `--symbol`: 交易標的（btc, eth 等）
- `--algorithm`: 演算法（SAC, PPO, TD3）
- `--attempts`: 最大重試次數
- `--timesteps`: 每次訓練步數
- `--config`: 配置文件路徑

### 查看模型

```bash
# 列出所有模型
python production/cli.py list

# 只顯示生產就緒的模型
python production/cli.py list --production-only

# 查看最佳模型
python production/cli.py best

# 導出排行榜
python production/cli.py leaderboard
```

### 啟動服務

```bash
# 啟動 API 服務器
python production/cli.py serve --port 8000

# 自定義 host 和 port
python production/cli.py serve --host 0.0.0.0 --port 8080
```

---

## 🌐 API 使用

### 健康檢查

```bash
curl http://localhost:8000/health
```

回應：
```json
{
  "status": "healthy",
  "models_loaded": 1,
  "registry_size": 5,
  "production_models": 3
}
```

### 獲取預測

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "observation": [0.5, 0.2, 0.1, ...],
    "deterministic": true
  }'
```

回應：
```json
{
  "action": [0.3, 0.7],
  "model_id": "sac_20231215_143022",
  "version": "v1.0.0"
}
```

### 列出模型

```bash
# 列出所有模型
curl http://localhost:8000/models

# 只列出生產就緒的模型
curl http://localhost:8000/models?production_ready_only=true

# 獲取最佳模型
curl http://localhost:8000/models/best/current
```

### 模型管理

```bash
# 載入模型到記憶體
curl -X POST http://localhost:8000/models/{model_id}/load

# 卸載模型
curl -X POST http://localhost:8000/models/{model_id}/unload

# 刪除模型
curl -X DELETE http://localhost:8000/models/{model_id}
```

---

## 🐳 Docker 部署

### 方法 1: Docker Compose（推薦）

```bash
# 啟動
docker-compose up -d

# 查看狀態
docker-compose ps

# 查看日誌
docker-compose logs -f

# 停止
docker-compose down
```

### 方法 2: 手動 Docker

```bash
# 建構映像
docker build -t rl-market-making:latest .

# 運行容器
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  --name rl-market-making \
  rl-market-making:latest

# 查看日誌
docker logs -f rl-market-making

# 停止容器
docker stop rl-market-making
```

---

## 📈 實際使用流程

### 場景 1: 訓練並部署新模型

```bash
# 1. 訓練模型
python production/cli.py train --symbol btc --attempts 3

# 2. 查看結果
python production/cli.py best

# 3. 啟動 API
python production/cli.py serve

# 4. 開始使用
curl http://localhost:8000/models/best/current
```

### 場景 2: 比較不同演算法

```bash
# 訓練 SAC
python production/cli.py train --algorithm SAC --attempts 3

# 訓練 PPO
python production/cli.py train --algorithm PPO --attempts 3

# 訓練 TD3
python production/cli.py train --algorithm TD3 --attempts 3

# 查看排行榜
python production/cli.py leaderboard
```

### 場景 3: 生產環境部署

```bash
# 1. 訓練多個模型
for i in {1..5}; do
  python production/cli.py train --symbol btc --attempts 2
done

# 2. 查看最佳模型
python production/cli.py best

# 3. Docker 部署
docker-compose up -d

# 4. 驗證服務
curl http://localhost:8000/health
```

---

## 🔍 監控與維護

### 查看性能指標

```bash
# 獲取排行榜
curl http://localhost:8000/metrics/leaderboard

# 查看特定模型
curl http://localhost:8000/models/{model_id}
```

### Web 監控面板

啟動面板：
```bash
python production/dashboard.py
```

面板顯示：
- 系統統計（總模型數、生產就緒數等）
- 最佳模型詳情
- 性能排行榜
- 視覺化圖表

### 健康檢查

API 提供健康檢查端點：
```bash
curl http://localhost:8000/health
```

在生產環境中，設置定時檢查（如每 30 秒）。

---

## ⚠️ 常見問題

### Q: 訓練失敗怎麼辦？
**A:** 增加重試次數和訓練步數：
```bash
python production/cli.py train --attempts 5 --timesteps 300000
```

### Q: 找不到數據文件？
**A:** 先下載數據：
```bash
python scripts/fetch_binance_ohlcv.py --symbol BTC/USDT --timeframe 1m --days 365
```

### Q: 沒有生產就緒的模型？
**A:** 嘗試調整配置或增加訓練時間：
```bash
# 使用不同配置
python production/cli.py train --config configs/default.yaml --timesteps 500000

# 或嘗試不同演算法
python production/cli.py train --algorithm TD3
```

### Q: API 無法啟動？
**A:** 檢查端口是否被佔用：
```bash
lsof -i :8000

# 使用其他端口
python production/cli.py serve --port 8080
```

### Q: Docker 容器運行失敗？
**A:** 查看日誌：
```bash
docker-compose logs -f
```

---

## 📚 進階功能

### 自定義配置

編輯 `configs/default.yaml` 調整訓練參數：

```yaml
env:
  initial_cash: 10000.0
  max_inventory: 2.0
  fee_rate: 0.0004

reward:
  lambda_inventory: 20.0
  mode: "shaped"

train:
  learning_rate: 0.00003
  batch_size: 256
```

### 集成到現有系統

Python 代碼範例：

```python
from production import ModelRegistry
import requests

# 使用註冊系統
registry = ModelRegistry()
best_model_id = registry.get_best_model()
model_info = registry.get_model_metadata(best_model_id)

# 使用 API
response = requests.post(
    "http://localhost:8000/predict",
    json={"observation": market_data, "deterministic": True}
)
action = response.json()["action"]
```

---

## 🎓 學習資源

1. **快速入門**: `docs/QUICKSTART.md`
2. **完整指南**: `docs/PRODUCTION_GUIDE.md`
3. **工作流程**: `docs/COMPLETE_WORKFLOW.md`
4. **範例代碼**: `examples/` 目錄
5. **API 文檔**: http://localhost:8000/docs

---

## 💡 最佳實踐

1. **定期重新訓練**: 每週或每月使用最新數據訓練
2. **保留備份**: 定期備份 `models/registry` 目錄
3. **監控性能**: 持續追蹤模型表現
4. **分散風險**: 使用多個模型組合
5. **逐步部署**: 先小額測試再全面部署

---

## 📞 支援

- **文檔**: 查看 `docs/` 目錄
- **範例**: 查看 `examples/` 目錄
- **問題**: 提交 GitHub Issue

---

**祝交易順利！📈🚀**
