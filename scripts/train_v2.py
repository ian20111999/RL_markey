"""train_v2.py: 使用改良版環境 MarketMakingEnvV2 的訓練腳本。

支援：
- 多種 Reward 模式（Dense/Sparse/Shaped/Hybrid）
- 擴展的 Observation/Action 空間
- Domain Randomization
- 完整的行為與風險指標追蹤
- 多演算法支援（SAC/PPO/TD3）
- Curriculum Learning
- Risk-Sensitive Training
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, EvalCallback, StopTrainingOnNoModelImprovement
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from envs.market_making_env_v2 import MarketMakingEnvV2, RewardConfig, RewardMode
from utils.config import (
    ExperimentConfig, load_config, export_config, load_data, split_data,
    create_env, create_env_from_config
)
from utils.algorithms import create_model, get_algo_class, ALGO_CONFIGS
from utils.curriculum import CurriculumScheduler, CurriculumEnvWrapper, create_market_making_curriculum
from utils.risk_sensitive import RiskAwareRewardWrapper, CVaRCallback, DrawdownEarlyStopping


# =============================================================================
# Comparison Agents
# =============================================================================

class RandomAgent:
    """隨機策略（用於比較）"""
    def __init__(self, action_dim: int = 3):
        self.action_dim = action_dim
    
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, None]:
        return np.random.uniform(-1, 1, size=(self.action_dim,)).astype(np.float32), None


class BaselineAgent:
    """Baseline 策略：中性報價，不 Skew"""
    def __init__(self, action_dim: int = 3):
        self.action_dim = action_dim
    
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, None]:
        # 對於 asymmetric action: [bid_spread=0, ask_spread=0, quote=1]
        action = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return action[:self.action_dim], None


class InventorySkewAgent:
    """庫存 Skew 策略：根據庫存方向調整報價"""
    def __init__(self, action_dim: int = 3, skew_factor: float = 0.3):
        self.action_dim = action_dim
        self.skew_factor = skew_factor
    
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, None]:
        # obs 中 inventory 是第 2 個特徵（index 1）
        inventory_norm = obs[1] if len(obs) > 1 else 0.0
        
        # 庫存為正時，更願意賣（縮小 ask spread）
        # 庫存為負時，更願意買（縮小 bid spread）
        bid_adj = self.skew_factor * inventory_norm  # 正庫存 -> 增大 bid spread
        ask_adj = -self.skew_factor * inventory_norm  # 正庫存 -> 減小 ask spread
        
        action = np.array([bid_adj, ask_adj, 1.0], dtype=np.float32)
        return action[:self.action_dim], None


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_agent(
    agent: Any,
    agent_name: str,
    config: ExperimentConfig,
    root_dir: Path,
    date_range: Tuple[str, str],
    n_episodes: int = 5,
    quiet: bool = False,
) -> Dict[str, Any]:
    """評估 Agent 並回傳完整指標。"""
    if not quiet:
        print(f"正在評估 {agent_name} ({n_episodes} episodes)...")
    
    all_metrics = []
    
    for i in range(n_episodes):
        # 建立環境（不啟用 Domain Randomization）
        env = create_env_from_config(
            config, root_dir,
            seed=20000 + i,
            date_range=date_range,
            enable_domain_rand=False,
        )
        
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        # 取得 episode 結束時的指標
        if "metrics" in info:
            episode_metrics = info["metrics"]
        else:
            episode_metrics = {"net_pnl": info.get("episode_net_pnl", 0.0)}
        
        all_metrics.append(episode_metrics)
        env.close()
    
    # 彙總所有 episodes
    summary = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if key in m]
        summary[key] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    
    summary["agent"] = agent_name
    return summary


def run_evaluation(
    model: SAC,
    config: ExperimentConfig,
    root_dir: Path,
    output_dir: Path,
    n_episodes: int = 10,
    quiet: bool = False,
) -> pd.DataFrame:
    """執行完整評估，比較 RL vs Baseline vs Random。"""
    test_split = config.data_split
    test_range = (test_split.get("test_start"), test_split.get("test_end"))
    
    action_dim = 3  # asymmetric mode
    action_config = config.env.get("action_config", {})
    if action_config.get("mode") == "symmetric":
        action_dim = 2
    
    agents = {
        "Random": RandomAgent(action_dim),
        "Baseline": BaselineAgent(action_dim),
        "InventorySkew": InventorySkewAgent(action_dim),
        "RL": model,
    }
    
    results = []
    for name, agent in agents.items():
        metrics = evaluate_agent(
            agent, name, config, root_dir,
            date_range=test_range,
            n_episodes=n_episodes,
            quiet=quiet,
        )
        results.append(metrics)
    
    df = pd.DataFrame(results)
    
    # 重新排列欄位
    primary_cols = ["agent", "net_pnl", "sharpe", "max_drawdown", "calmar_ratio"]
    behavior_cols = ["avg_spread", "fill_rate", "quote_rate", "inventory_turnover"]
    risk_cols = ["var_95", "expected_shortfall_95", "adverse_selection_rate"]
    
    all_cols = primary_cols + behavior_cols + risk_cols
    existing_cols = [c for c in all_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in all_cols and not c.endswith("_std")]
    
    df = df[existing_cols + other_cols]
    
    # 儲存
    df.to_csv(output_dir / "eval_summary.csv", index=False)
    
    if not quiet:
        print("\n📊 Evaluation Results:")
        print(df[existing_cols].to_string(index=False))
    
    return df


# =============================================================================
# Training Callbacks
# =============================================================================

class MetricsLoggingCallback(BaseCallback):
    """記錄訓練過程中的行為指標。"""
    
    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.metrics_history = []
    
    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            # 從 info 中收集指標
            infos = self.locals.get("infos", [])
            for info in infos:
                if "metrics" in info:
                    self.metrics_history.append({
                        "step": self.n_calls,
                        **info["metrics"]
                    })
        return True


# =============================================================================
# Main Training Function
# =============================================================================

def train(
    config_path: Path,
    output_dir: Path,
    seed: int = 42,
    total_timesteps: Optional[int] = None,
    enable_domain_rand: bool = True,
    quiet: bool = False,
    algo: str = "sac",
    use_curriculum: bool = False,
    use_risk_aware: bool = False,
    risk_lambda: float = 0.1,
) -> Tuple[Any, pd.DataFrame]:
    """執行訓練流程。
    
    Args:
        config_path: 配置檔路徑
        output_dir: 輸出目錄
        seed: 隨機種子
        total_timesteps: 總訓練步數
        enable_domain_rand: 是否啟用 Domain Randomization
        quiet: 靜默模式
        algo: 演算法 ("sac", "ppo", "td3")
        use_curriculum: 是否使用課程學習
        use_risk_aware: 是否使用風險感知訓練
        risk_lambda: 風險厭惡係數
    """
    
    config = load_config(config_path)
    train_cfg = config.train
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 鎖定 Config
    shutil.copy(config_path, output_dir / "config_used.yaml")
    with open(config_path, 'rb') as f:
        config_hash = hashlib.md5(f.read()).hexdigest()
    with open(output_dir / "config_hash.txt", 'w') as f:
        f.write(config_hash)
    
    if not quiet:
        print(f"📋 Config Hash: {config_hash}")
        print(f"📁 Output Dir: {output_dir}")
    
    # 訓練資料範圍
    train_split = config.data_split
    train_range = (train_split.get("train_start"), train_split.get("train_end"))
    valid_range = (train_split.get("valid_start"), train_split.get("valid_end"))
    
    # 建立訓練環境
    n_envs = train_cfg.get("n_envs", 4)
    
    def make_train_env(env_seed: int):
        def _init():
            env = create_env_from_config(
                config, ROOT,
                seed=env_seed,
                date_range=train_range,
                enable_domain_rand=enable_domain_rand,
            )
            
            # 課程學習包裝
            if use_curriculum:
                scheduler = CurriculumScheduler(
                    stages=create_market_making_curriculum("normal"),
                    verbose=0 if quiet else 1,
                )
                env = CurriculumEnvWrapper(env, scheduler)
            
            # 風險感知包裝
            if use_risk_aware:
                env = RiskAwareRewardWrapper(
                    env,
                    risk_lambda=risk_lambda,
                    risk_type="variance",
                )
            
            return Monitor(env)
        return _init
    
    train_env = SubprocVecEnv([make_train_env(seed + i) for i in range(n_envs)])
    
    # 建立評估環境（不啟用 Domain Rand）
    eval_env = DummyVecEnv([lambda: Monitor(create_env_from_config(
        config, ROOT,
        seed=seed + 100,
        date_range=valid_range,
        enable_domain_rand=False,
    ))])
    
    # 設定 Model
    net_arch = train_cfg.get("net_arch", [256, 256])
    
    # 使用演算法工廠建立模型
    algo_lower = algo.lower()
    
    config_overrides = {
        "learning_rate": train_cfg.get("learning_rate", 3e-4),
        "buffer_size": train_cfg.get("buffer_size", 200000),
        "batch_size": train_cfg.get("batch_size", 256),
        "gamma": train_cfg.get("gamma", 0.99),
        "policy_kwargs": {"net_arch": net_arch},
    }
    
    # SAC 特有參數
    if algo_lower == "sac":
        config_overrides.update({
            "tau": train_cfg.get("tau", 0.02),
            "train_freq": train_cfg.get("train_freq", 1),
            "gradient_steps": train_cfg.get("gradient_steps", 1),
            "ent_coef": train_cfg.get("ent_coef", "auto"),
        })
    # PPO 特有參數
    elif algo_lower == "ppo":
        config_overrides.update({
            "n_steps": train_cfg.get("n_steps", 2048),
            "n_epochs": train_cfg.get("n_epochs", 10),
            "clip_range": train_cfg.get("clip_range", 0.2),
        })
    # TD3 特有參數
    elif algo_lower == "td3":
        config_overrides.update({
            "tau": train_cfg.get("tau", 0.005),
            "policy_delay": train_cfg.get("policy_delay", 2),
        })
    
    model = create_model(
        algo=algo_lower,
        env=train_env,
        config_overrides=config_overrides,
        tensorboard_log=str(output_dir / "tb_logs"),
        verbose=0 if quiet else 1,
        seed=seed,
    )
    
    if not quiet:
        print(f"🤖 Algorithm: {algo.upper()}")
    
    # Callbacks
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,
        min_evals=10,
        verbose=0 if quiet else 1,
    )
    
    timesteps = total_timesteps or train_cfg.get("total_timesteps", 200000)
    eval_freq = train_cfg.get("eval_freq", max(timesteps // 20, 5000))
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=eval_freq,
        n_eval_episodes=train_cfg.get("n_eval_episodes", 5),
        callback_after_eval=stop_callback,
        deterministic=True,
        verbose=0 if quiet else 1,
    )
    
    metrics_callback = MetricsLoggingCallback(log_freq=5000)
    
    callbacks = CallbackList([eval_callback, metrics_callback])
    
    # 訓練
    if not quiet:
        print(f"\n🚀 Training {timesteps} steps (Early Stopping enabled)...")
    
    model.learn(total_timesteps=timesteps, callback=callbacks)
    
    train_env.close()
    eval_env.close()
    
    # 載入最佳模型
    best_model_path = output_dir / "best_model" / "best_model.zip"
    if best_model_path.exists():
        if not quiet:
            print("📥 Loading best model...")
        algo_class = get_algo_class(algo)
        model = algo_class.load(best_model_path)
    
    model.save(output_dir / "model.zip")
    
    # 評估
    if not quiet:
        print("\n📊 Running evaluation...")
    
    eval_df = run_evaluation(model, config, ROOT, output_dir, n_episodes=10, quiet=quiet)
    
    return model, eval_df


# =============================================================================
# Sanity Check
# =============================================================================

def sanity_check(
    config_path: Path,
    output_dir: Path,
    max_retries: int = 3,
    base_seed: int = 42,
) -> Tuple[bool, Dict[str, Any]]:
    """執行 Sanity Check：確認 RL 能學到比 Random 好的策略。"""
    
    config = load_config(config_path)
    criteria = config.raw.get("sanity_criteria", {})
    
    for attempt in range(1, max_retries + 1):
        seed = base_seed + attempt
        print(f"\n🔄 Sanity Attempt {attempt}/{max_retries} (Seed: {seed})")
        
        attempt_dir = output_dir / f"attempt_{attempt}"
        
        model, eval_df = train(
            config_path=config_path,
            output_dir=attempt_dir,
            seed=seed,
            total_timesteps=config.train.get("total_timesteps", 100000),
            enable_domain_rand=True,
            quiet=False,
        )
        
        # 檢查條件
        rl_row = eval_df[eval_df["agent"] == "RL"].iloc[0]
        rand_row = eval_df[eval_df["agent"] == "Random"].iloc[0]
        base_row = eval_df[eval_df["agent"] == "Baseline"].iloc[0]
        
        rl_pnl = float(rl_row["net_pnl"])
        rand_pnl = float(rand_row["net_pnl"])
        base_pnl = float(base_row["net_pnl"])
        
        details = {
            "rl_net_pnl": rl_pnl,
            "random_net_pnl": rand_pnl,
            "baseline_net_pnl": base_pnl,
            "rl_vs_random_gap": rl_pnl - rand_pnl,
            "rl_vs_baseline_gap": rl_pnl - base_pnl,
            "attempt": attempt,
        }
        
        # 檢查條件
        min_pnl = criteria.get("min_rl_net_pnl", -999999)
        min_gap_random = criteria.get("min_rl_vs_random_gap", 0)
        min_gap_baseline = criteria.get("min_rl_vs_baseline_gap", -999999)
        
        passed = True
        reason = ""
        
        if rl_pnl < min_pnl:
            passed = False
            reason = f"RL PnL ({rl_pnl:.2f}) < Min ({min_pnl})"
        elif rl_pnl - rand_pnl < min_gap_random:
            passed = False
            reason = f"RL-Random Gap ({rl_pnl - rand_pnl:.2f}) < Min ({min_gap_random})"
        elif rl_pnl - base_pnl < min_gap_baseline:
            passed = False
            reason = f"RL-Baseline Gap ({rl_pnl - base_pnl:.2f}) < Min ({min_gap_baseline})"
        
        if passed:
            print(f"\n✅ Sanity Check Passed!")
            
            # 檢查是否可跳過 Tuning
            skip_threshold = criteria.get("skip_tuning_if_exceed_baseline", None)
            skip_tuning = False
            if skip_threshold and base_pnl > 0:
                if rl_pnl > base_pnl * skip_threshold:
                    skip_tuning = True
                    print(f"🎯 RL exceeds Baseline by {skip_threshold}x, can skip Tuning!")
            
            # 複製最佳模型到主目錄
            shutil.copy(attempt_dir / "model.zip", output_dir / "model.zip")
            shutil.copy(attempt_dir / "eval_summary.csv", output_dir / "eval_summary.csv")
            
            status = {
                "status": "success",
                "reason": "All criteria passed",
                "skip_tuning": skip_tuning,
                "details": details,
            }
            
            with open(output_dir / "sanity_status.json", "w") as f:
                json.dump(status, f, indent=2)
            
            return True, status
        else:
            print(f"❌ Failed: {reason}")
    
    # 全部重試失敗
    print("\n💀 All Sanity attempts failed.")
    status = {
        "status": "failed",
        "reason": "Max retries exceeded",
        "details": details,
    }
    
    with open(output_dir / "sanity_status.json", "w") as f:
        json.dump(status, f, indent=2)
    
    return False, status


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="V2 Environment Training")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "env_v3.yaml")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["train", "sanity", "eval"], default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--no_domain_rand", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    # 新增參數
    parser.add_argument("--algo", type=str, default="sac", choices=["sac", "ppo", "td3"],
                        help="Algorithm to use")
    parser.add_argument("--use_curriculum", action="store_true",
                        help="Enable curriculum learning")
    parser.add_argument("--use_risk_aware", action="store_true",
                        help="Enable risk-aware training")
    parser.add_argument("--risk_lambda", type=float, default=0.1,
                        help="Risk aversion coefficient")
    args = parser.parse_args()
    
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = ROOT / "runs" / f"v2_{args.mode}_{args.algo}_{timestamp}"
    
    if args.mode == "train":
        train(
            config_path=args.config,
            output_dir=args.output_dir,
            seed=args.seed,
            total_timesteps=args.timesteps,
            enable_domain_rand=not args.no_domain_rand,
            quiet=args.quiet,
            algo=args.algo,
            use_curriculum=args.use_curriculum,
            use_risk_aware=args.use_risk_aware,
            risk_lambda=args.risk_lambda,
        )
    
    elif args.mode == "sanity":
        sanity_check(
            config_path=args.config,
            output_dir=args.output_dir,
            max_retries=3,
            base_seed=args.seed,
        )
    
    elif args.mode == "eval":
        config = load_config(args.config)
        model_path = args.output_dir / "model.zip"
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            sys.exit(1)
        
        algo_class = get_algo_class(args.algo)
        model = algo_class.load(model_path)
        run_evaluation(model, config, ROOT, args.output_dir, n_episodes=20, quiet=args.quiet)


if __name__ == "__main__":
    main()
