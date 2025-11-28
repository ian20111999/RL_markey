"""train_sanity_sac.py: 固定環境 Sanity Check 訓練腳本。
目的：確認在固定環境 v2 下，SAC 至少能學出「比 Random 好」甚至「接近或略優於 Baseline」的策略。
支援自動重試與結果判斷、Early Stopping。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

def check_sanity_criteria(metrics: pd.DataFrame, criteria: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """檢查是否通過 Sanity Criteria，回傳 (passed, reason, details)"""
    rl_row = metrics[metrics["agent"] == "Sanity_RL"].iloc[0]
    rand_row = metrics[metrics["agent"] == "Random"].iloc[0]
    base_row = metrics[metrics["agent"] == "Baseline"].iloc[0]
    
    rl_pnl = float(rl_row["net_pnl"])
    rl_sharpe = float(rl_row["sharpe"])
    rand_pnl = float(rand_row["net_pnl"])
    base_pnl = float(base_row["net_pnl"])
    
    details = {
        "rl_net_pnl": rl_pnl,
        "rl_sharpe": rl_sharpe,
        "random_net_pnl": rand_pnl,
        "baseline_net_pnl": base_pnl,
        "rl_vs_random_gap": rl_pnl - rand_pnl,
        "rl_vs_baseline_gap": rl_pnl - base_pnl,
    }
    
    # 1. Min Net PnL (絕對門檻)
    min_pnl = criteria.get("min_rl_net_pnl", -999999)
    if rl_pnl < min_pnl:
        return False, f"RL Net PnL ({rl_pnl:.2f}) < Min ({min_pnl})", details
    
    # 2. RL vs Random (使用 Gap 差值，更穩健)
    min_gap_random = criteria.get("min_rl_vs_random_gap", None)
    if min_gap_random is not None:
        actual_gap = rl_pnl - rand_pnl
        if actual_gap < min_gap_random:
            return False, f"RL-Random Gap ({actual_gap:.2f}) < Min Gap ({min_gap_random})", details
    else:
        # 舊邏輯：使用 ratio（向後相容）
        ratio = criteria.get("min_rl_vs_random_ratio", 1.0)
        if rand_pnl > 0:
            if rl_pnl < rand_pnl * ratio:
                return False, f"RL ({rl_pnl:.2f}) < {ratio} * Random ({rand_pnl:.2f})", details
        else:
            if rl_pnl <= rand_pnl:
                return False, f"RL ({rl_pnl:.2f}) <= Random ({rand_pnl:.2f})", details

    # 3. RL vs Baseline (使用 Gap 差值)
    min_gap_baseline = criteria.get("min_rl_vs_baseline_gap", None)
    if min_gap_baseline is not None:
        actual_gap = rl_pnl - base_pnl
        if actual_gap < min_gap_baseline:
            return False, f"RL-Baseline Gap ({actual_gap:.2f}) < Min Gap ({min_gap_baseline})", details
    else:
        # 舊邏輯：只有當 Baseline > 0 時才檢查 ratio
        base_ratio = criteria.get("min_rl_vs_baseline_ratio", 0.7)
        if base_pnl > 0:
            if rl_pnl < base_pnl * base_ratio:
                return False, f"RL ({rl_pnl:.2f}) < {base_ratio} * Baseline ({base_pnl:.2f})", details
    
    # 4. 可選：Sharpe > 0
    if criteria.get("require_positive_sharpe", False):
        if rl_sharpe <= 0:
            return False, f"RL Sharpe ({rl_sharpe:.4f}) <= 0", details
            
    return True, "All criteria passed", details


def compute_config_hash(config_path: Path) -> str:
    """計算 Config 檔案的 MD5 Hash"""
    with open(config_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def lock_config(config_path: Path, run_dir: Path) -> str:
    """複製 Config 到 run 目錄並記錄 Hash"""
    shutil.copy(config_path, run_dir / "config_used.yaml")
    config_hash = compute_config_hash(config_path)
    with open(run_dir / "config_hash.txt", 'w') as f:
        f.write(config_hash)
    return config_hash

# -----------------------------------------------------------------------------
# Main Training Logic
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sanity Check Training")
    parser.add_argument("--config", type=Path, required=True, help="環境設定檔")
    parser.add_argument("--output_dir", type=Path, default=None, help="輸出目錄（預設為 runs/exp_<timestamp>/sanity）")
    parser.add_argument("--exp_dir", type=Path, default=None, help="實驗根目錄（若提供，sanity 輸出在其下）")
    args = parser.parse_args()

    config = load_config(args.config)
    env_cfg = config.env
    train_cfg = config.train
    criteria = config.raw.get("sanity_criteria", {})
    base_seed = config.raw.get("base_seed", 42)
    
    max_retries = criteria.get("max_retries", 3)
    
    # 決定輸出目錄
    if args.exp_dir:
        exp_dir = args.exp_dir
        output_dir = exp_dir / "sanity"
    elif args.output_dir:
        output_dir = args.output_dir
        exp_dir = output_dir.parent
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = ROOT / "runs" / f"exp_{timestamp}"
        output_dir = exp_dir / "sanity"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 鎖定 Config（確保可追溯）
    config_hash = lock_config(args.config, output_dir)
    print(f"📋 Config Hash: {config_hash}")
    
    export_config(config, output_dir / "config.yaml")
    
    status_file = output_dir / "sanity_status.json"
    
    for attempt in range(1, max_retries + 1):
        current_seed = base_seed + attempt
        print(f"\n🔄 Sanity Attempt {attempt}/{max_retries} (Seed: {current_seed})")
        
        # Setup Train Env
        train_env_kwargs = build_env_kwargs(env_cfg, root_dir=ROOT, seed=current_seed)
        n_envs = 4
        
        # 使用 lambda 閉包問題修正
        def make_train_env(seed_offset):
            def _init():
                kwargs = dict(train_env_kwargs)
                kwargs["seed"] = current_seed + seed_offset
                return Monitor(HistoricalMarketMakingEnv(**kwargs))
            return _init
        
        train_env = SubprocVecEnv([make_train_env(i) for i in range(n_envs)])
        
        # Setup Eval Env for Early Stopping
        eval_env_kwargs = build_env_kwargs(env_cfg, root_dir=ROOT, seed=current_seed + 100)
        eval_env = DummyVecEnv([lambda: Monitor(HistoricalMarketMakingEnv(**eval_env_kwargs))])
        
        # Setup Callbacks for Early Stopping
        stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=5,
            min_evals=5,
            verbose=1
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(output_dir / "best_model"),
            log_path=str(output_dir / "eval_logs"),
            eval_freq=max(train_cfg.get("total_timesteps", 100000) // 20, 1000),
            n_eval_episodes=3,
            callback_after_eval=stop_callback,
            deterministic=True,
            verbose=0
        )
        
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
            seed=current_seed,
            device="auto"
        )
        
        # Train with Early Stopping
        total_timesteps = train_cfg.get("total_timesteps", 100000)
        print(f"🚀 Training up to {total_timesteps} steps (Early Stopping enabled)...")
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        
        train_env.close()
        eval_env.close()
        
        # Load best model if available
        best_model_path = output_dir / "best_model" / "best_model.zip"
        if best_model_path.exists():
            print("📥 Loading best model from Early Stopping...")
            model = SAC.load(best_model_path)
        
        # Evaluate on Test Set
        print("📊 Evaluating on Test Set...")
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
        print("\n📈 Evaluation Results:")
        print(df[["agent", "net_pnl", "gross_pnl", "sharpe"]].to_string(index=False))
        
        # Check Criteria
        passed, reason, details = check_sanity_criteria(df, criteria)
        
        if passed:
            print(f"\n✅ Sanity Check Passed! ({reason})")
            model.save(output_dir / "model.zip")
            df.to_csv(output_dir / "eval_summary.csv", index=False)
            
            # 檢查是否可以跳過 Tuning
            skip_tuning_threshold = criteria.get("skip_tuning_if_exceed_baseline", None)
            skip_tuning = False
            if skip_tuning_threshold and details["baseline_net_pnl"] > 0:
                if details["rl_net_pnl"] > details["baseline_net_pnl"] * skip_tuning_threshold:
                    skip_tuning = True
                    print(f"🎯 RL exceeds Baseline by {skip_tuning_threshold}x, can skip Tuning!")
            
            status = {
                "status": "success",
                "reason": reason,
                "attempt": attempt,
                "config_hash": config_hash,
                "skip_tuning": skip_tuning,
                "details": details
            }
            
            with open(status_file, "w") as f:
                json.dump(status, f, indent=2)
            
            # 寫入實驗根目錄的 exp_dir（若有）
            if exp_dir and exp_dir != output_dir:
                with open(exp_dir / "sanity_status.json", "w") as f:
                    json.dump(status, f, indent=2)
            
            return
        else:
            print(f"\n❌ Sanity Check Failed. ({reason})")
            
    # If all retries failed
    print("\n💀 All Sanity attempts failed.")
    status = {
        "status": "failed",
        "reason": "Max retries exceeded",
        "config_hash": config_hash,
        "details": details if 'details' in dir() else {}
    }
    with open(status_file, "w") as f:
        json.dump(status, f, indent=2)
    sys.exit(1)

if __name__ == "__main__":
    main()
