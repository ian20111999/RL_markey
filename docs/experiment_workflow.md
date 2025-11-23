# 全流程實驗工作流與效能指南

本文件將「tuning → training → evaluate → analyze」整套流程串成可平行執行的批次作業，並提供 Mac Apple Silicon 上的資源利用建議。照此操作可在 1–2 個工作階段內產生多組可比較的策略結果。

## 0. 實驗矩陣

| Config | 重點 | 建議用途 |
| --- | --- | --- |
| `configs/env_baseline.yaml` | 原始環境，lambda_turnover=0 | 當作對照組，驗證新流程是否穩定 |
| `configs/env_turnover_penalty.yaml` | 追加 turnover 懲罰，beta 略高 | 抑制刷單並觀察成交稀疏情況 |
| `configs/env_aggressive_spread.yaml` | 縮小 spread、提高 beta | 壓榨流動性、測試高頻動態 |
| `configs/env_conservative_inventory.yaml` | 嚴控庫存、base_spread 较寬 | 保守策略、檢查穩定性 |

> 需要更多變體時，可複製其中一份 YAML，調整 `env` / `train` / `run.short_name` 後再加入下列流程即可。

## 1. 批次調參（Optuna）

1. 建立一份 config 清單：
   ```bash
   cat <<'EOF' > configs/tuning_targets.txt
   configs/env_baseline.yaml
   configs/env_turnover_penalty.yaml
   configs/env_aggressive_spread.yaml
   configs/env_conservative_inventory.yaml
   EOF
   ```

2. 以 2~3 個平行行程跑 Optuna，優先壓榨 CPU：
   ```bash
   # 每組 25 trials、每 trial 80k steps，依機器調整
   xargs -I{} -P 2 bash -c '
     source .venv/bin/activate && \
     python tune_mm_sac.py \
       --config {} \
       --n_trials 25 \
       --train_timesteps 80000 \
       --eval_episode_length 600 \
       --eval_episodes 5 \
       --save_best_params \
       --device mps > logs/$(basename {} .yaml)_tune.log 2>&1
   ' < configs/tuning_targets.txt
   ```

   - `-P 2` 表示同時兩組 tuning；視 CPU/GPU 負載可調成 3。
   - log 會存進 `logs/`，方便日後對照。

3. Tuning 完成後，`models/best_sac_params.json` 會被覆寫成最新結果；若要各 config 各存一份，可在指令裡改 `--best_params_path runs/<algo>/<config_name>/best_params.json`。

## 2. 大規模訓練

- 依 config 矩陣開啟 2~3 個訓練行程，讓 GPU (MPS) 一樣吃滿：

```bash
cat configs/tuning_targets.txt | xargs -I{} -P 2 bash -c '
  source .venv/bin/activate && \
  python train_mm_sac.py --config {} --device mps > logs/$(basename {} .yaml)_train.log 2>&1
'
```

- `train_mm_sac.py` 會自動生成 `runs/SAC/<timestamp>_<short_name>/`。稍後的評估/分析都直接讀 run 目錄，互不覆蓋。

- 若需要更快收斂，可把 config 裡的 `train.total_timesteps` 調高為 500k~1M，並確保 `train.log_interval` 不要太小，避免 IO 成瓶頸。

## 3. 分段評估 (Train/Valid/Test)

訓練完成後，立即對所有 model.zip 做分段測試：

```bash
find runs/SAC -name model.zip | sort | xargs -I{} -P 3 bash -c '
  source .venv/bin/activate && \
  python scripts/evaluate_policy.py \
    --config configs/env_baseline.yaml \
    --model_path {} \
    --episodes 5 \
    --device mps > logs/$(basename $(dirname {}))_eval.log 2>&1
'
```

- `--config` 可換成與 model 相同的 YAML，以確保資料切割一致。
- 輸出會寫在 `runs/.../evaluation/` 下，包含 `metrics.json`（train/valid/test 指標）、每段 CSV、test 曲線圖。

## 4. 整體分析

1. 整理所有 run 的 metrics（建議在每輪評估後執行）：
   ```bash
   source .venv/bin/activate
   python scripts/analyze_experiments.py --runs_dir runs --sort_by test_sharpe --top_k 10 --plot_param lambda_inv --plot_metric test_sharpe
   ```

2. 產出的 `runs/runs_summary.csv` 建議開在 spreadsheet/Notion，進一步做 pivot 或篩選。

3. `plots/analysis_<param>_vs_<metric>.png` 可快速判斷參數與績效的關係，例如調整 `lambda_turnover` 是否能換來更好 Sharpe。

## 5. 資源配置 Tips

- **CPU vs GPU**：tuning 偏 CPU 密集（Optuna + env step），training/ evaluation 則會吃到 GPU/MPS。可同時跑 2 個 training + 1 個 tuning，以充分利用兩者。
- **Log 管理**：所有指令都導向 `logs/*.log`，可用 `tail -f` 即時查看。建議配合 `grep -i warning` 監控錯誤。
- **熱管理**：若 Activity Monitor 顯示 SoC 溫度過高，可把 `xargs -P` 調小或在晚上排程；M 系列長時間 80% 以上負載仍可穩定運作。
- **中斷續跑**：Optuna 若使用 `--storage sqlite:///optuna.db`，即使中途中斷也能接續；run 輸出不會被覆蓋。

## 6. 預期產物（每組 config）

1. `runs/SAC/<timestamp>_<short_name>/`：完整訓練、評估資料。
2. `logs/<config>_{tune,train,eval}.log`：排錯用。
3. `runs/runs_summary.csv` + `plots/analysis_*.png`：跨實驗比較視覺化。

依照上述腳本，可形成以下時間線：

1. 上午：啟動批次 tuning（~2hr）。
2. 中午：挑選最佳參數後立即開跑訓練 2~3 組（~3hr）。
3. 下午：訓練完成即刻進行 evaluate + analyze（<1hr），當天就能得到 test 段穩定度排名。

有了這套流程，可以在單日內跑出一個 mini grid search + 回測報告，並確保所有輸出都有 config/metrics 追蹤，方便之後微調特定環境參數或 reward 組件。
