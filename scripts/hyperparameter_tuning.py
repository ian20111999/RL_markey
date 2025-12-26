"""
scripts/hyperparameter_tuning.py
使用 Optuna 進行超參數優化

功能:
- 自動搜索最佳超參數組合
- 支援多種演算法 (SAC, PPO, TD3)
- 使用 Walk-Forward 驗證作為目標函數
- 儲存最佳配置和訓練結果

用法:
    python scripts/hyperparameter_tuning.py --symbol BTCUSDT --n_trials 50
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import optuna
import pandas as pd
import yaml
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.market_making_env import (
    MarketMakingEnv, RewardConfig, ObservationConfig, ActionConfig, RewardMode
)
from utils.algorithms import create_model, ALGO_CONFIGS
from utils.backtesting import BacktestEngine, WalkForwardAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Hyperparameter Search Spaces
# =============================================================================

def get_sac_params(trial: optuna.Trial) -> Dict[str, Any]:
    """SAC 超參數搜索空間"""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "buffer_size": trial.suggest_categorical("buffer_size", [50000, 100000, 200000]),
        "gamma": trial.suggest_float("gamma", 0.95, 0.9999),
        "tau": trial.suggest_float("tau", 0.001, 0.05),
        "train_freq": trial.suggest_categorical("train_freq", [1, 2, 4]),
        "gradient_steps": trial.suggest_categorical("gradient_steps", [1, 2, 4]),
        "ent_coef": "auto",  # 使用自動調整
        "policy_kwargs": {
            "net_arch": trial.suggest_categorical(
                "net_arch", 
                [[64, 64], [128, 128], [256, 256], [256, 128], [128, 64]]
            ),
        },
    }


def get_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """PPO 超參數搜索空間"""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096]),
        "n_epochs": trial.suggest_int("n_epochs", 3, 20),
        "gamma": trial.suggest_float("gamma", 0.95, 0.9999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
        "ent_coef": trial.suggest_float("ent_coef", 0.0001, 0.1, log=True),
        "vf_coef": trial.suggest_float("vf_coef", 0.1, 0.9),
        "policy_kwargs": {
            "net_arch": dict(
                pi=trial.suggest_categorical("pi_arch", [[64, 64], [128, 128], [256, 256]]),
                vf=trial.suggest_categorical("vf_arch", [[64, 64], [128, 128], [256, 256]]),
            ),
        },
    }


def get_td3_params(trial: optuna.Trial) -> Dict[str, Any]:
    """TD3 超參數搜索空間"""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "buffer_size": trial.suggest_categorical("buffer_size", [50000, 100000, 200000]),
        "gamma": trial.suggest_float("gamma", 0.95, 0.9999),
        "tau": trial.suggest_float("tau", 0.001, 0.05),
        "policy_delay": trial.suggest_int("policy_delay", 1, 3),
        "target_policy_noise": trial.suggest_float("target_policy_noise", 0.1, 0.3),
        "target_noise_clip": trial.suggest_float("target_noise_clip", 0.3, 0.6),
        "policy_kwargs": {
            "net_arch": trial.suggest_categorical(
                "net_arch", 
                [[64, 64], [128, 128], [256, 256], [256, 128]]
            ),
        },
    }


def get_reward_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Reward 函數超參數搜索空間"""
    return {
        "reward_scale": trial.suggest_float("reward_scale", 1e-7, 1e-5, log=True),
        "lambda_inventory": trial.suggest_float("lambda_inventory", 1.0, 50.0),
        "lambda_turnover": trial.suggest_float("lambda_turnover", 0.0, 0.1),
        "lambda_inventory_age": trial.suggest_float("lambda_inventory_age", 0.0, 0.5),
        "terminal_bonus_weight": trial.suggest_float("terminal_bonus_weight", 0.1, 0.5),
        "spread_capture_bonus": trial.suggest_float("spread_capture_bonus", 0.0, 1.0),
    }


PARAM_SAMPLERS = {
    "sac": get_sac_params,
    "ppo": get_ppo_params,
    "td3": get_td3_params,
}


# =============================================================================
# Environment Factory
# =============================================================================

def create_env_factory(
    data: pd.DataFrame,
    reward_params: Dict[str, Any],
    env_config: Dict[str, Any],
):
    """創建環境工廠函數"""
    
    def make_env():
        reward_cfg = RewardConfig(
            mode=RewardMode.SHAPED,
            reward_scale=reward_params.get("reward_scale", 1e-6),
            lambda_inventory=reward_params.get("lambda_inventory", 20.0),
            lambda_turnover=reward_params.get("lambda_turnover", 0.01),
            lambda_inventory_age=reward_params.get("lambda_inventory_age", 0.1),
            terminal_bonus_weight=reward_params.get("terminal_bonus_weight", 0.3),
            spread_capture_bonus=reward_params.get("spread_capture_bonus", 0.5),
        )
        
        obs_cfg = ObservationConfig(
            include_price=True,
            include_inventory=True,
            include_time=True,
            include_volatility=True,
            include_momentum=True,
            include_volume=True,
            include_inventory_age=True,
            include_trend=True,
        )
        
        action_cfg = ActionConfig(
            mode="asymmetric",
            allow_no_quote=True,
            max_spread_multiplier=2.0,
            min_spread_multiplier=0.5,
        )
        
        return MarketMakingEnv(
            df=data,
            initial_cash=env_config.get("initial_cash", 10000),
            fee_rate=env_config.get("fee_rate", 0.0004),
            max_inventory=env_config.get("max_inventory", 2.0),
            episode_length=env_config.get("episode_length", 1440),
            base_spread=env_config.get("base_spread", 60.0),
            random_start=True,
            reward_config=reward_cfg,
            obs_config=obs_cfg,
            action_config=action_cfg,
        )
    
    return make_env


# =============================================================================
# Objective Function
# =============================================================================

class HyperparameterOptimizer:
    """超參數優化器"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        algorithm: str = "sac",
        train_timesteps: int = 50000,
        n_eval_episodes: int = 10,
        use_walk_forward: bool = True,
        env_config: Optional[Dict] = None,
        output_dir: str = "hpo_results",
    ):
        """
        Args:
            data: 訓練數據
            algorithm: 演算法 (sac, ppo, td3)
            train_timesteps: 訓練步數
            n_eval_episodes: 評估 episode 數
            use_walk_forward: 是否使用 Walk-Forward 驗證
            env_config: 環境配置
            output_dir: 輸出目錄
        """
        self.data = data
        self.algorithm = algorithm.lower()
        self.train_timesteps = train_timesteps
        self.n_eval_episodes = n_eval_episodes
        self.use_walk_forward = use_walk_forward
        self.env_config = env_config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 分割數據
        train_end = int(len(data) * 0.6)
        valid_end = int(len(data) * 0.8)
        
        self.train_data = data.iloc[:train_end].reset_index(drop=True)
        self.valid_data = data.iloc[train_end:valid_end].reset_index(drop=True)
        self.test_data = data.iloc[valid_end:].reset_index(drop=True)
        
        logger.info(f"Data split: train={len(self.train_data)}, valid={len(self.valid_data)}, test={len(self.test_data)}")
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna 目標函數"""
        try:
            # 取得超參數
            param_sampler = PARAM_SAMPLERS.get(self.algorithm)
            if param_sampler is None:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")
            
            algo_params = param_sampler(trial)
            reward_params = get_reward_params(trial)
            
            # 創建環境
            env_factory = create_env_factory(self.train_data, reward_params, self.env_config)
            eval_env_factory = create_env_factory(self.valid_data, reward_params, self.env_config)
            
            train_env = DummyVecEnv([env_factory])
            eval_env = DummyVecEnv([eval_env_factory])
            
            # 創建模型
            model = create_model(
                algo=self.algorithm,
                env=train_env,
                config_overrides=algo_params,
                verbose=0,
            )
            
            # 訓練
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=None,
                log_path=None,
                eval_freq=max(self.train_timesteps // 10, 1000),
                n_eval_episodes=3,
                deterministic=True,
                render=False,
                verbose=0,
            )
            
            model.learn(
                total_timesteps=self.train_timesteps,
                callback=eval_callback,
                progress_bar=False,
            )
            
            # 評估
            if self.use_walk_forward:
                score = self._evaluate_walk_forward(model, reward_params)
            else:
                score = self._evaluate_simple(model, eval_env)
            
            # 清理
            train_env.close()
            eval_env.close()
            
            # 保存 trial 信息
            trial.set_user_attr("algo_params", str(algo_params))
            trial.set_user_attr("reward_params", str(reward_params))
            
            return score
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return float("-inf")
    
    def _evaluate_simple(self, model, eval_env) -> float:
        """簡單評估：計算平均獎勵"""
        rewards = []
        for _ in range(self.n_eval_episodes):
            obs = eval_env.reset()
            done = False
            ep_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                ep_reward += reward[0]
            rewards.append(ep_reward)
        
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards) if len(rewards) > 1 else 1.0
        
        # Sharpe-like metric
        return avg_reward / (std_reward + 1e-8)
    
    def _evaluate_walk_forward(self, model, reward_params: Dict) -> float:
        """Walk-Forward 評估"""
        env_factory = create_env_factory(self.valid_data, reward_params, self.env_config)
        engine = BacktestEngine(env_factory)
        
        try:
            result = engine.run_backtest(model, n_episodes=self.n_eval_episodes)
            # 使用 Sharpe Ratio 作為目標
            return result.sharpe_ratio
        except Exception as e:
            logger.warning(f"Walk-forward evaluation failed: {e}")
            return float("-inf")
    
    def optimize(
        self,
        n_trials: int = 50,
        n_jobs: int = 1,
        study_name: Optional[str] = None,
    ) -> optuna.Study:
        """執行優化"""
        study_name = study_name or f"hpo_{self.algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )
        
        logger.info(f"Starting optimization: {n_trials} trials, algorithm={self.algorithm}")
        
        study.optimize(
            self.objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True,
        )
        
        # 保存結果
        self._save_results(study)
        
        return study
    
    def _save_results(self, study: optuna.Study):
        """保存優化結果"""
        # 最佳參數
        best_params = study.best_params
        best_value = study.best_value
        
        results = {
            "algorithm": self.algorithm,
            "best_value": best_value,
            "best_params": best_params,
            "n_trials": len(study.trials),
            "timestamp": datetime.now().isoformat(),
        }
        
        # 保存 JSON
        results_path = self.output_dir / f"best_params_{self.algorithm}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        # 生成最佳配置 YAML
        config = self._generate_config(best_params)
        config_path = self.output_dir / f"best_config_{self.algorithm}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Best value: {best_value:.4f}")
        logger.info(f"Best params saved to: {results_path}")
        logger.info(f"Best config saved to: {config_path}")
        
        # 打印最佳參數
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Algorithm: {self.algorithm.upper()}")
        print(f"Best Sharpe: {best_value:.4f}")
        print("\nBest Parameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        print("=" * 60)
    
    def _generate_config(self, params: Dict) -> Dict:
        """生成訓練配置 YAML"""
        # 分離演算法參數和獎勵參數
        algo_keys = ["learning_rate", "batch_size", "buffer_size", "gamma", "tau",
                     "train_freq", "gradient_steps", "n_steps", "n_epochs",
                     "gae_lambda", "clip_range", "ent_coef", "vf_coef",
                     "policy_delay", "target_policy_noise", "target_noise_clip",
                     "net_arch", "pi_arch", "vf_arch"]
        
        reward_keys = ["reward_scale", "lambda_inventory", "lambda_turnover",
                       "lambda_inventory_age", "terminal_bonus_weight", "spread_capture_bonus"]
        
        train_config = {k: v for k, v in params.items() if k in algo_keys}
        reward_config = {k: v for k, v in params.items() if k in reward_keys}
        
        # 處理網路架構
        if "net_arch" in train_config and isinstance(train_config["net_arch"], str):
            train_config["net_arch"] = eval(train_config["net_arch"])
        
        return {
            "train": train_config,
            "reward": {
                "mode": "shaped",
                **reward_config,
            },
            "env": self.env_config,
        }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization with Optuna")
    parser.add_argument("--data", type=str, default="data/btc_usdt_1m_2023.csv",
                        help="Path to training data CSV")
    parser.add_argument("--algorithm", type=str, default="sac", choices=["sac", "ppo", "td3"],
                        help="RL algorithm to optimize")
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Number of optimization trials")
    parser.add_argument("--train_timesteps", type=int, default=50000,
                        help="Training timesteps per trial")
    parser.add_argument("--n_eval_episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--use_walk_forward", action="store_true", default=True,
                        help="Use Walk-Forward validation")
    parser.add_argument("--output_dir", type=str, default="hpo_results",
                        help="Output directory for results")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of parallel jobs (use -1 for all CPUs)")
    
    args = parser.parse_args()
    
    # 載入數據
    logger.info(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    
    if "close" not in df.columns and "price" in df.columns:
        df["close"] = df["price"]
    
    logger.info(f"Data loaded: {len(df)} rows")
    
    # 環境配置
    env_config = {
        "initial_cash": 10000,
        "fee_rate": 0.0004,
        "max_inventory": 2.0,
        "episode_length": 1440,
        "base_spread": 60.0,
    }
    
    # 創建優化器
    optimizer = HyperparameterOptimizer(
        data=df,
        algorithm=args.algorithm,
        train_timesteps=args.train_timesteps,
        n_eval_episodes=args.n_eval_episodes,
        use_walk_forward=args.use_walk_forward,
        env_config=env_config,
        output_dir=args.output_dir,
    )
    
    # 執行優化
    study = optimizer.optimize(
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
    )
    
    return study


if __name__ == "__main__":
    main()
