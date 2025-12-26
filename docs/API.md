# RL Market API 文檔

## 概述

RL Market 提供 RESTful API 用於模型推論和管理。

## Base URL

```
http://localhost:8000/api/v1
```

---

## 認證

暫不需要認證（開發環境）。生產環境建議使用 API Key 或 OAuth2。

---

## Endpoints

### 健康檢查

#### GET /health

檢查 API 服務狀態。

**Response**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-12-26T23:50:00Z"
}
```

---

### 模型管理

#### GET /models

列出所有已註冊的模型。

**Query Parameters**
| 參數 | 類型 | 說明 |
|------|------|------|
| symbol | string | 過濾交易對 |
| algorithm | string | 過濾演算法 (sac, ppo, td3) |
| deployed | boolean | 只顯示已部署模型 |

**Response**
```json
{
  "models": [
    {
      "model_id": "sac_btc_20251226",
      "symbol": "BTCUSDT",
      "algorithm": "SAC",
      "sharpe_ratio": 2.5,
      "is_deployed": true,
      "created_at": "2025-12-26T10:00:00Z"
    }
  ]
}
```

---

#### GET /models/{model_id}

取得單一模型詳細資訊。

**Response**
```json
{
  "model_id": "sac_btc_20251226",
  "symbol": "BTCUSDT",
  "algorithm": "SAC",
  "metrics": {
    "sharpe_ratio": 2.5,
    "max_drawdown": 0.12,
    "win_rate": 0.65,
    "total_trades": 1500
  },
  "config": {
    "learning_rate": 0.0003,
    "batch_size": 256
  },
  "is_deployed": true,
  "production_ready": true
}
```

---

#### POST /models/{model_id}/deploy

部署模型到生產環境。

**Response**
```json
{
  "success": true,
  "message": "Model deployed successfully",
  "deployed_at": "2025-12-26T23:50:00Z"
}
```

---

### 預測

#### POST /predict

使用模型進行預測。

**Request Body**
```json
{
  "model_id": "sac_btc_20251226",
  "observation": {
    "price": 42000.0,
    "inventory": 0.5,
    "volatility": 0.02,
    "momentum": 0.001
  }
}
```

**Response**
```json
{
  "action": {
    "bid_spread": 0.0015,
    "ask_spread": 0.0012,
    "quote_flag": 1.0
  },
  "confidence": 0.85,
  "inference_time_ms": 2.3
}
```

---

### 訓練狀態

#### GET /training/runs

列出訓練執行記錄。

**Response**
```json
{
  "runs": [
    {
      "run_id": "run_btc_20251226_v1",
      "symbol": "BTCUSDT",
      "algorithm": "SAC",
      "status": "completed",
      "final_pnl": 250.5,
      "total_episodes": 1000
    }
  ]
}
```

---

#### GET /training/runs/{run_id}

取得單一訓練執行詳情。

---

### 回測

#### POST /backtest

執行模型回測。

**Request Body**
```json
{
  "model_id": "sac_btc_20251226",
  "start_date": "2025-01-01",
  "end_date": "2025-06-30",
  "initial_balance": 10000
}
```

**Response**
```json
{
  "backtest_id": "bt_123456",
  "status": "running"
}
```

#### GET /backtest/{backtest_id}

取得回測結果。

---

## 錯誤碼

| 狀態碼 | 說明 |
|--------|------|
| 200 | 成功 |
| 400 | 請求格式錯誤 |
| 404 | 資源不存在 |
| 500 | 伺服器錯誤 |

**錯誤回應格式**
```json
{
  "error": {
    "code": "MODEL_NOT_FOUND",
    "message": "Model with ID 'xxx' not found"
  }
}
```

---

## 使用範例

### Python

```python
import requests

# 列出模型
response = requests.get("http://localhost:8000/api/v1/models")
models = response.json()["models"]

# 進行預測
prediction = requests.post(
    "http://localhost:8000/api/v1/predict",
    json={
        "model_id": "sac_btc_20251226",
        "observation": {
            "price": 42000.0,
            "inventory": 0.5
        }
    }
)
print(prediction.json())
```

### cURL

```bash
# 健康檢查
curl http://localhost:8000/api/v1/health

# 列出模型
curl http://localhost:8000/api/v1/models?symbol=BTCUSDT

# 預測
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"model_id": "sac_btc_20251226", "observation": {"price": 42000}}'
```

---

## 開發指南

### 啟動 API 服務

```bash
# 開發模式
uvicorn production.api:app --reload --port 8000

# 生產模式
uvicorn production.api:app --host 0.0.0.0 --port 8000 --workers 4
```

### 自動文檔

FastAPI 自動生成的互動式文檔：
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
