# data/ 目錄說明

此目錄用於存放訓練用的市場資料 CSV 檔案。

## 資料來源

使用 `scripts/fetch_data.py` 下載歷史資料：

```bash
# 下載單一幣種
python scripts/fetch_data.py --symbol BTCUSDT --days 365

# 下載多個幣種
python scripts/fetch_data.py --symbol BTCUSDT ETHUSDT BNBUSDT --days 365
```

## 資料格式

CSV 檔案需包含以下欄位：
- `timestamp` - Unix 時間戳（毫秒）
- `datetime` - 日期時間字串
- `open` - 開盤價
- `high` - 最高價
- `low` - 最低價
- `close` - 收盤價
- `volume` - 成交量

## 注意事項

- CSV 檔案會被 `.gitignore` 排除（檔案較大）
- 資料會自動導入 PostgreSQL 資料庫
- 訓練時系統會從資料庫讀取，不需要保留 CSV
