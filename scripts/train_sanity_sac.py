"""train_sanity_sac.py: 固定環境 Sanity Check 訓練腳本。
目的：確認在固定環境 v2 下，SAC 至少能學出「比 Random 好」甚至「接近或略優於 Baseline」的策略。
支援自動重試與結果判斷。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from envs.historical_market_making_env import HistoricalMarketMakingEnv
from utils.config import build_env_kwargs, export_config, load_config
from utils.metrics import max_drawdown, sharpe_ratio

# -----------------------------------------------------------------------------
# Simple Agents for Comparison
# -----------------------------------------------------------------------------

class RandomAgent:
    """隨機策略 Agent"""
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, None]:
        return np.random.uniform(-1, 1, size=(2,)), None

class BaselineAgent:
    """固定 Baseline 策略 Agent (Spread=Base, Skew=0)"""
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, None]:
        return np.array([0.0, 0.0], dtype=np.float32), None

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def rollout_episode(model: Any, env_kwargs: Dict[str, Any], seed: int) -> Dict[str, float]:
    """執行單個 episode 並回傳統計指標。"""
    local_kwargs = dict(env_kwargs)
    local_kwargs["seed"] = seed
    local_kwargs["random_start"] = True
    
    env = HistoricalMarketMakingEnv(**local_kwargs)
    obs, _ = env.reset()
    done = False
    
    pvs = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        pvs.append(float(info.get("portfolio_value", 0.0)))
        if done:
            final_info = info
            
    env.close()
    
    pv_array = np.array(pvs)
    returns = np.diff(pv_array, prepend=pv_array[0])
    
    return {
        "net_pnl": final_info.get("episode_net_pnl", 0.0),
        "gross_pnl": final_info.get("episode_gross_pnl", 0.0),
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(pv_array),
    }

def evaluate_agent(agent: Any, agent_name: str, env_kwargs: Dict[str, Any], n_episodes: int) -> Dict[str, float]:
    """評估指定 Agent 多個 episodes 取平均。"""
    print(f"正在評估 {agent_name} ({n_episodes} episodes)...")
    metrics_accum = {"net_pnl": [], "gross_pnl": [], "sharpe": [], "max_drawdown": []}
    
    for i in range(n_episodes):
        res = rollout_episode(agent, env_kwargs, seed=10000 + i)
        for k in metrics_accum:
            metrics_accum[k].append(res[k])
            
    return {k: float(np.mean(v)) for k, v in metrics_accum.items()}

def check_sanity_criteria(metrics: pd.DataFrame, criteria: Dict[str, Any]) -> Tuple[bool, str]:
    """檢查是否通過 Sanity Criteria"""
    rl_row = metrics[metrics["agent"] == "Sanity_RL"].iloc[0]
    rand_row = metrics[metrics["agent"] == "Random"].iloc[0]
    base_row = metrics[metrics["agent"] == "Baseline"].iloc[0]
    
    rl_pnl = rl_row["net_pnl"]
    rand_pnl = rand_row["net_pnl"]
    base_pnl = base_row["net_pnl"]
    
    # 1. Min Net PnL
    if rl_pnl < criteria.get("min_rl_net_pnl", -999999):
        return False, f"RL Net PnL ({rl_pnl:.2f}) < Min ({criteria['min_rl_net_pnl']})"
        
    # 2. RL vs Random
    # 若 Random 為負，RL 只要大於 Random 即可
    # 若 Random 為正，RL 需大於 Random * ratio
    ratio = criteria.get("min_rl_vs_random_ratio", 1.0)
    if rand_pnl > 0:
        if rl_pnl < rand_pnl * ratio:
            return False, f"RL ({rl_pnl:.2f}) < {ratio} * Random ({rand_pnl:.2f})"
    else:
        if rl_pnl <= rand_pnl:
             return False, f"RL ({rl_pnl:.2f}) <= Random ({rand_pnl:.2f})"

    # 3. RL vs Baseline
    # 只有當 Baseline 為正時才嚴格檢查 ratio
    base_ratio = criteria.get("min_rl_vs_baseline_ratio", 0.7)
    if base_pnl > 0:
        if rl_pnl < base_pnl * base_ratio:
            return False, f"RL ({rl_pnl:.2f}) < {base_ratio} * Baseline ({base_pnl:.2f})"
            
    return True, "Pass"

# -----------------------------------------------------------------------------
# Main Training Logic
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sanity Check Training")
    parser.add_argument("--config", type=Path, required=True, help="環境設定檔")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "runs" / "sanity_v2", help="輸出目錄")
    args = parser.parse_args()

    config = load_config(args.config)
    env_cfg = config.env
    train_cfg = config.train
    criteria = getattr(config, "sanity_criteria", {})
    base_seed = getattr(config, "base_seed", 42)
    
    max_retries = criteria.get("max_retries", 1)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_config(config, args.output_dir / "config.yaml")
    
    status_file = args.output_dir / "sanity_status.json"
    
    for attempt in range(1, max_retries + 1):
        current_seed = base_seed + attempt
        print(f"\n🔄 Sanity Attempt {attempt}/{max_retries} (Seed: {current_seed})")
        
        # Setup Env
        train_env_kwargs = build_env_kwargs(env_cfg, root_dir=ROOT, seed=current_seed)
        n_envs = 4
        train_env = SubprocVecEnv([
            lambda: Monitor(HistoricalMarketMakingEnv(**train_env_kwargs)) for _ in range(n_envs)
        ])
        
        # Setup Model
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=train_cfg.get("learning_rate", 3e-4),
            buffer_size=train_cfg.get("buffer_size", 200000),
            batch_size=train_cfg.get("batch_size", 256),
            tau=train_cfg.get("tau", 0.02),
            gamma=train_cfg.get("gamma", 0.99),
            train_freq=train_cfg.get("train_freq", 1),
            gradient_steps=train_cfg.get("gradient_steps", 1),
            policy_kwargs={"net_arch": train_cfg.get("net_arch", [256, 256])},
            verbose=0,
            device="auto"
        )
        
        # Train
        total_timesteps = train_cfg.get("total_timesteps", 100000)
        print(f"🚀 Training {total_timesteps} steps...")
        model.learn(total_timesteps=total_timesteps)
        train_env.close()
        
        # Evaluate
        print("📊 Evaluating...")
        test_split = config.data_split
        test_date_range = (test_split.get("test_start"), test_split.get("test_end"))
        test_env_kwargs = build_env_kwargs(env_cfg, root_dir=ROOT, date_range=test_date_range)
        
        agents = {
            "Random": RandomAgent(),
            "Baseline": BaselineAgent(),
            "Sanity_RL": model
        }
        
        results = []
        for name, agent in agents.items():
            metrics = evaluate_agent(agent, name, test_env_kwargs, n_episodes=5)
            metrics["agent"] = name
            results.append(metrics)
            
        df = pd.DataFrame(results)
        print(df[["agent", "net_pnl", "gross_pnl", "sharpe"]])
        
        # Check Criteria
        passed, reason = check_sanity_criteria(df, criteria)
        
        if passed:
            print(f"✅ Sanity Check Passed! (Reason: {reason})")
            model.save(args.output_dir / "model.zip")
            df.to_csv(args.output_dir / "eval_summary.csv", index=False)
            
            with open(status_file, "w") as f:
                json.dump({"status": "success", "reason": reason, "attempt": attempt}, f)
            return
        else:
            print(f"❌ Sanity Check Failed. (Reason: {reason})")
            
    # If all retries failed
    print("💀 All Sanity attempts failed.")
    with open(status_file, "w") as f:
        json.dump({"status": "failed", "reason": "Max retries exceeded"}, f)
    sys.exit(1)

if __name__ == "__main__":
    main()
