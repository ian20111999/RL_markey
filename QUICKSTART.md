# 🚀 RL Market Making - 5 分鐘快速上手

歡迎使用 RL Market Making！這份指南將幫助你在 5 分鐘內開始訓練你的第一個 AI 做市機器人。

## 📋 前置需求

- Python 3.8 或以上
- 8GB+ RAM（推薦 16GB）
- 穩定的網路連接（用於下載歷史數據）

## ⚡ 三步驟開始

### 第 1 步：安裝依賴（1 分鐘）

```bash
# 克隆專案（如果還沒有）
git clone <your-repo-url>
cd RL_markey

# 創建虛擬環境（推薦）
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安裝依賴
pip install -r requirements.txt
```

### 第 2 步：運行你的第一個訓練（2 分鐘）

```bash
# 訓練 BTC 做市機器人（會自動下載數據）
python pipeline.py --symbol btc
```

**會發生什麼？**
1. ✅ Pipeline 檢測到沒有 BTC 數據
2. ⬇️ 自動從 Binance 下載 2023 年完整歷史數據（~500k 筆）
3. 🔍 分析數據，計算平均價格
4. ⚙️ 自動調整參數（spread、資金、獎勵縮放）
5. 🏋️ 開始訓練（200,000 timesteps）
6. 📊 評估模型（20 個 episode）
7. 💾 保存最佳模型（如果盈利）

### 第 3 步：查看結果（30 秒）

```bash
# 檢查訓練日誌
ls runs/run_btc_*/

# 查看評估結果
cat runs/run_btc_*/evaluation_results.json

# 檢查保存的模型
ls models/
```

## 🎯 典型輸出

成功的訓練會顯示類似這樣的結果：

```
🏆 Best Run: run_eth_1765516821_v3 (PnL: $6.14)
   Mean PnL:       +6.14 ± 55.24
   Mean Trades:    48.8
   Win Rate:       60.0% (12/20)
   Total PnL:      +122.78
   💾 Saved to: models/eth_best_model.zip
```

**這代表什麼？**
- **Mean PnL**: 每個 episode（1440 分鐘）平均賺 $6.14
- **Win Rate**: 60% 的 episode 獲利
- **Total PnL**: 在 20 個測試 episode 中總共賺 $122.78

## 📚 下一步

### 訓練其他幣種

```bash
python pipeline.py --symbol eth  # 以太坊
python pipeline.py --symbol sol  # Solana
python pipeline.py --symbol bnb  # Binance Coin
```

### 調整重試次數

```bash
# 如果結果不滿意，增加重試次數（預設 3 次）
python pipeline.py --symbol btc --retries 5
```

### 使用訓練好的模型

```python
from stable_baselines3 import SAC

# 載入模型
model = SAC.load("models/btc_best_model.zip")

# 用於實際交易（需要自行實現執行邏輯）
# obs = env.reset()
# action, _ = model.predict(obs, deterministic=True)
```

## 🔧 常見問題

### Q: 訓練需要多久？

**A:** 取決於你的硬體：
- CPU（4核）：~30-60 分鐘
- GPU（RTX 3060+）：~10-20 分鐘

### Q: 為什麼有時候訓練失敗？

**A:** RL 訓練有隨機性。Pipeline 會自動重試（預設 3 次），每次使用不同的 random seed。如果 3 次都失敗，可以：
1. 增加 `--retries` 參數
2. 調整 `configs/default.yaml` 中的超參數
3. 檢查數據質量

### Q: 模型為什麼沒有保存？

**A:** Pipeline 只保存**盈利**的模型（PnL > 0 且 Win Rate >= 50%）。如果所有重試都未達標，不會保存模型。這是為了確保只部署生產級別的模型。

### Q: 可以用自己的數據嗎？

**A:** 可以！將你的 CSV 文件放入 `data/` 目錄，格式如下：

```csv
timestamp,open,high,low,close,volume
1640995200000,46222.5,46250.0,46200.0,46230.0,123.45
```

然後運行：
```bash
python pipeline.py --symbol your_symbol
```

### Q: 如何調整訓練參數？

**A:** 編輯 `configs/default.yaml`：

```yaml
train:
  learning_rate: 0.00003  # 學習率
  batch_size: 256         # 批次大小
  buffer_size: 100000     # 經驗回放緩衝區

env:
  max_inventory: 2.0      # 最大持倉
  fee_rate: 0.0004        # 手續費率
```

## 🎓 進階功能

準備好探索更多功能了嗎？查看：

- [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md) - 完整系統說明
- [INTEGRATED_PIPELINE_README.md](INTEGRATED_PIPELINE_README.md) - 企業級多幣種系統
- [README.md](README.md) - 進階配置和 API

## 💡 最佳實踐

1. **先從 BTC 或 ETH 開始**：流動性最好的幣種通常更容易訓練成功
2. **保持數據新鮮**：定期更新歷史數據
3. **多次訓練**：RL 有隨機性，多訓練幾次找最佳模型
4. **監控指標**：不只看 PnL，也要關注 Sharpe Ratio、Max Drawdown
5. **回測驗證**：在不同時間段的數據上測試模型的穩健性

## 🆘 需要幫助？

- 📖 查看完整文檔：[README.md](README.md)
- 🐛 報告問題：[GitHub Issues](https://github.com/your-repo/issues)
- 💬 討論交流：[GitHub Discussions](https://github.com/your-repo/discussions)

---

**準備好了嗎？現在就運行你的第一個訓練！**

```bash
python pipeline.py --symbol btc
```

祝你訓練順利！🎉
