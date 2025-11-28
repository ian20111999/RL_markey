"""scan_env_baseline.py: 掃描環境參數並使用固定 Baseline 策略進行測試，尋找合理參數組合。"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# 確保可以 import envs
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from envs.historical_market_making_env import HistoricalMarketMakingEnv
from utils.config import load_config

# 建議參數範圍
# fee_rate: 固定 0.0004
# base_spread: 10.0 ~ 30.0 (因為 0.0004 * 30000 = 12，Spread 必須大於 12 才能覆蓋手續費)
# lambda_inv: 0.0002, 0.0005
# alpha: 固定 0.5

FEE_RATES = [0.0004]
BASE_SPREADS = [10.0, 15.0, 20.0, 25.0, 30.0]
LAMBDA_INVS = [0.0002, 0.0005]
ALPHA = 0.5

DEFAULT_CONFIG_PATH = ROOT / "configs" / "env_baseline.yaml"

def run_baseline_episode(env: HistoricalMarketMakingEnv) -> Dict[str, float]:
    """執行單個 episode 的 baseline 策略 (固定 spread/skew)。"""
    obs, _ = env.reset()
    done = False
    
    # Baseline 策略：始終保持中性
    # action[0] (spread) = 0.0 -> 使用 base_spread * (1 + alpha * 0) = base_spread
    # action[1] (skew) = 0.0 -> skew = beta * 0 = 0
    action = np.array([0.0, 0.0], dtype=np.float32)
    
    final_info = {}
    
    while not done:
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            final_info = info
            
    return {
        "gross_pnl": final_info.get("episode_gross_pnl", 0.0),
        "fees": final_info.get("episode_fees", 0.0),
        "penalty_inv": final_info.get("episode_penalty_inv", 0.0),
        "net_pnl": final_info.get("episode_net_pnl", 0.0),
    }

def main():
    parser = argparse.ArgumentParser(description="掃描環境參數並檢查 Baseline 表現")
    parser.add_argument("--episodes", type=int, default=5, help="每個組合測試的 episode 數")
    parser.add_argument("--output", type=str, default="env_baseline_scan_results.csv", help="結果輸出 CSV 檔名")
    args = parser.parse_args()

    # 載入預設 config 以獲取資料路徑與分割設定
    base_config = load_config(DEFAULT_CONFIG_PATH)
    
    # 準備測試資料段
    test_split = base_config.data_split
    date_range = (test_split.get("test_start"), test_split.get("test_end"))
    
    results = []
    
    # 產生所有參數組合
    combinations = list(itertools.product(FEE_RATES, BASE_SPREADS, LAMBDA_INVS))
    total_combos = len(combinations)
    
    print(f"🚀 開始掃描 {total_combos} 組參數組合...")
    print(f"📅 測試資料段: {date_range}")
    print(f"{'Fee':<10} | {'Spread':<10} | {'L_Inv':<10} | {'Gross PnL':<12} | {'Fees':<10} | {'Net PnL':<12}")
    print("-" * 80)

    for i, (fee, spread, l_inv) in enumerate(combinations, 1):
        # 建立環境
        env_kwargs = {
            "csv_path": base_config.env["csv_path"],
            "episode_length": base_config.env["episode_length"],
            "fee_rate": fee,
            "base_spread": spread,
            "lambda_inv": l_inv,
            "alpha": ALPHA,
            "max_inventory": base_config.env["max_inventory"],
            "random_start": True,
            "date_range": date_range,
        }
        
        # 為了避免每次都重新讀取 CSV，理想上應該重用環境，但為了確保參數乾淨，這裡每次重建
        # 若效能太差可優化
        env = HistoricalMarketMakingEnv(**env_kwargs)
        
        metrics_accum = {"gross_pnl": [], "fees": [], "penalty_inv": [], "net_pnl": []}
        
        for ep in range(args.episodes):
            # 設定 seed 確保可重現性，但不同 episode 要不同
            env.reset(seed=1000 + ep)
            res = run_baseline_episode(env)
            for k, v in res.items():
                metrics_accum[k].append(v)
        
        env.close()
        
        # 計算平均
        avg_gross = np.mean(metrics_accum["gross_pnl"])
        avg_fees = np.mean(metrics_accum["fees"])
        avg_penalty = np.mean(metrics_accum["penalty_inv"])
        avg_net = np.mean(metrics_accum["net_pnl"])
        
        results.append({
            "fee_rate": fee,
            "base_spread": spread,
            "lambda_inv": l_inv,
            "avg_gross_pnl": avg_gross,
            "avg_fees": avg_fees,
            "avg_penalty_inv": avg_penalty,
            "avg_net_pnl": avg_net
        })
        
        print(f"{fee:<10.4f} | {spread:<10.1f} | {l_inv:<10.4f} | {avg_gross:<12.2f} | {avg_fees:<10.2f} | {avg_net:<12.2f}")

    # 轉為 DataFrame 並排序
    df = pd.DataFrame(results)
    df = df.sort_values(by="avg_net_pnl", ascending=False)
    
    # 存檔
    output_path = ROOT / args.output
    df.to_csv(output_path, index=False)
    print("-" * 80)
    print(f"✅ 掃描完成！結果已儲存至: {output_path}")
    
    print("\n🏆 Top 5 最佳參數組合 (依 Net PnL):")
    print(df.head(5).to_string(index=False))
    
    print("\n💡 建議：")
    print("1. 觀察 avg_gross_pnl 是否接近 0 或為正，表示做市本身有獲利潛力。")
    print("2. 若 avg_fees 過高導致 Net PnL 大幅為負，考慮調高 base_spread 或降低 fee_rate (若可選)。")
    print("3. 若 avg_penalty_inv 過高，表示庫存控制不易，可調整 lambda_inv 或檢查策略。")

if __name__ == "__main__":
    main()
