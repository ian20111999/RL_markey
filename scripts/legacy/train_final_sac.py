"""train_final_sac.py: 使用最佳超參數進行最終長訓。
流程：
1. 讀取 env_v2.yaml
2. 讀取 best_sac_params.json (若有)
3. 執行長時訓練 (e.g. 500k steps) + Early Stopping
4. 儲存最終模型
5. 執行最終評估 (RL vs Baseline)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from envs.historical_market_making_env import HistoricalMarketMakingEnv
from utils.config import build_env_kwargs, export_config, load_config
from utils.metrics import max_drawdown, sharpe_ratio

# -----------------------------------------------------------------------------
# Agents
# -----------------------------------------------------------------------------

class BaselineAgent:
    """固定 Baseline 策略 Agent (Spread=Base, Skew=0)"""
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, None]:
        return np.array([0.0, 0.0], dtype=np.float32), None

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def rollout_episode(model: Any, env_kwargs: Dict[str, Any], seed: int) -> Dict[str, float]:
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

def evaluate_agent(agent: Any, agent_name: str, env_kwargs: Dict[str, Any], n_episodes: int, quiet: bool = False) -> Dict[str, float]:
    if not quiet:
        print(f"正在評估 {agent_name} ({n_episodes} episodes)...")
    metrics_accum = {"net_pnl": [], "gross_pnl": [], "sharpe": [], "max_drawdown": []}
    for i in range(n_episodes):
        res = rollout_episode(agent, env_kwargs, seed=20000 + i)
        for k in metrics_accum:
            metrics_accum[k].append(res[k])
    return {k: float(np.mean(v)) for k, v in metrics_accum.items()}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Final Training with Best Params")
    parser.add_argument("--config", type=Path, required=True, help="環境設定檔")
    parser.add_argument("--params", type=Path, default=ROOT / "models" / "best_sac_params.json", help="最佳超參數 JSON")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "runs" / "final_env_v2_sac", help="輸出目錄")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子")
    parser.add_argument("--quiet", action="store_true", help="減少輸出訊息")
    args = parser.parse_args()

    verbose = 0 if args.quiet else 1

    # 1. Load Config & Params
    config = load_config(args.config)
    env_cfg = config.env
    train_cfg = dict(config.train)  # 複製一份以便修改
    
    # Load Best Params if exists
    if args.params.exists():
        if not args.quiet:
            print(f"📥 載入最佳超參數: {args.params}")
        with open(args.params, "r", encoding="utf-8") as f:
            best_params = json.load(f)
        # Override train_cfg
        for k, v in best_params.items():
            if k == "policy_kwargs":
                 if "net_arch" in v:
                     train_cfg["net_arch"] = v["net_arch"]
            elif k == "net_arch":
                # Handle direct net_arch from Optuna (string like "256x2")
                if isinstance(v, str):
                    net_arch_map = {"64x2": [64, 64], "128x2": [128, 128], "256x2": [256, 256]}
                    train_cfg["net_arch"] = net_arch_map.get(v, [256, 256])
                else:
                    train_cfg["net_arch"] = v
            else:
                train_cfg[k] = v
    else:
        if not args.quiet:
            print("⚠️ 未找到最佳超參數檔案，將使用 Config 預設值進行訓練。")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_config(config, args.output_dir / "config.yaml")
    
    # 鎖定 Config Hash
    with open(args.config, 'rb') as f:
        config_hash = hashlib.md5(f.read()).hexdigest()
    with open(args.output_dir / "config_hash.txt", 'w') as f:
        f.write(config_hash)

    # 2. Setup Env
    seed = args.seed
    train_env_kwargs = build_env_kwargs(env_cfg, root_dir=ROOT, seed=seed)
    n_envs = 4
    
    def make_train_env(seed_offset):
        def _init():
            kwargs = dict(train_env_kwargs)
            kwargs["seed"] = seed + seed_offset
            return Monitor(HistoricalMarketMakingEnv(**kwargs))
        return _init
    
    train_env = SubprocVecEnv([make_train_env(i) for i in range(n_envs)])
    
    # Setup Eval Env for Early Stopping
    eval_env_kwargs = build_env_kwargs(env_cfg, root_dir=ROOT, seed=seed + 100)
    eval_env = DummyVecEnv([lambda: Monitor(HistoricalMarketMakingEnv(**eval_env_kwargs))])

    # 3. Setup Model
    policy_kwargs = {}
    if "net_arch" in train_cfg:
        policy_kwargs["net_arch"] = train_cfg["net_arch"]

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
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=verbose,
        device="auto"
    )

    # 4. Setup Callbacks for Early Stopping
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,  # Final 用更耐心的設定
        min_evals=10,
        verbose=verbose
    )
    
    total_timesteps = train_cfg.get("total_timesteps", 500000)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.output_dir / "best_model"),
        log_path=str(args.output_dir / "eval_logs"),
        eval_freq=max(total_timesteps // 50, 5000),  # 評估 50 次
        n_eval_episodes=5,
        callback_after_eval=stop_callback,
        deterministic=True,
        verbose=verbose
    )

    # 5. Train with Early Stopping
    if not args.quiet:
        print(f"🚀 開始最終長訓 (Up to {total_timesteps} steps, Early Stopping enabled)...")
    
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, log_interval=100 if not args.quiet else 0)
    
    train_env.close()
    eval_env.close()
    
    # Load best model if available
    best_model_path = args.output_dir / "best_model" / "best_model.zip"
    if best_model_path.exists():
        if not args.quiet:
            print("📥 Loading best model from Early Stopping...")
        model = SAC.load(best_model_path)
    
    model_path = args.output_dir / "model.zip"
    model.save(model_path)
    if not args.quiet:
        print(f"✅ 最終模型已儲存至 {model_path}")

    # 6. Final Evaluation
    if not args.quiet:
        print("\n📊 執行最終評估 (Test Set)...")
    test_split = config.data_split
    test_date_range = (test_split.get("test_start"), test_split.get("test_end"))
    test_env_kwargs = build_env_kwargs(env_cfg, root_dir=ROOT, date_range=test_date_range)

    agents = {
        "Baseline": BaselineAgent(),
        "Final_RL": model
    }
    
    results = []
    for name, agent in agents.items():
        metrics = evaluate_agent(agent, name, test_env_kwargs, n_episodes=10, quiet=args.quiet)
        metrics["agent"] = name
        results.append(metrics)
        
    df = pd.DataFrame(results)
    df = df[["agent", "net_pnl", "gross_pnl", "sharpe", "max_drawdown"]]
    
    if not args.quiet:
        print("\n🏆 最終評估結果:")
        print(df.to_string(index=False))
    
    df.to_csv(args.output_dir / "final_eval_summary.csv", index=False)

if __name__ == "__main__":
    main()
