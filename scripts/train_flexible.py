"""
scripts/train_flexible.py
靈活訓練腳本 - 支援多演算法和 Curriculum Learning

功能:
- 支援 SAC, PPO, TD3 演算法選擇
- 可選的 Curriculum Learning (漸進式難度)
- 整合驗證流程
- 從超參數優化結果載入配置

用法:
    # 基本訓練
    python scripts/train_flexible.py --algorithm sac --symbol btc
    
    # 使用優化後的配置
    python scripts/train_flexible.py --algorithm sac --config hpo_results/best_config_sac.yaml
    
    # 啟用 Curriculum Learning
    python scripts/train_flexible.py --algorithm ppo --curriculum
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from envs.market_making_env import (
    MarketMakingEnv, RewardConfig, ObservationConfig, ActionConfig, RewardMode
)
from utils.algorithms import create_model, load_model, ALGO_CONFIGS
from utils.curriculum import CurriculumScheduler, CurriculumEnvWrapper, CurriculumStage
from utils.risk_sensitive import DynamicPositionLimitWrapper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    "env": {
        "initial_cash": 10000,
        "fee_rate": 0.0004,
        "max_inventory": 2.0,
        "episode_length": 1440,
        "base_spread": 60.0,
        "random_start": True,
    },
    "reward": {
        "mode": "shaped",
        "reward_scale": 1e-6,
        "lambda_inventory": 20.0,
        "lambda_turnover": 0.01,
        "lambda_inventory_age": 0.1,
        "terminal_bonus_weight": 0.3,
        "spread_capture_bonus": 0.5,
    },
    "train": {
        "total_timesteps": 200000,
        "learning_rate": 3e-4,
        "batch_size": 256,
    },
    "curriculum": {
        "enabled": False,
        "stages": [
            {"name": "easy", "fee_rate": 0.0002, "max_inventory": 3.0, "min_episodes": 50},
            {"name": "medium", "fee_rate": 0.0003, "max_inventory": 2.5, "min_episodes": 100},
            {"name": "hard", "fee_rate": 0.0004, "max_inventory": 2.0, "min_episodes": 150},
        ]
    },
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """載入配置"""
    config = DEFAULT_CONFIG.copy()
    
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            loaded = yaml.safe_load(f)
        # 深度合併
        for key, value in loaded.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
        logger.info(f"Loaded config from {config_path}")
    
    return config


# =============================================================================
# Environment Creation
# =============================================================================

def create_env(
    data: pd.DataFrame,
    config: Dict[str, Any],
    seed: int = 42,
    curriculum_params: Optional[Dict] = None,
) -> MarketMakingEnv:
    """創建環境"""
    env_cfg = config.get("env", {})
    reward_cfg_dict = config.get("reward", {})
    
    # 如果有 curriculum 參數覆蓋
    if curriculum_params:
        env_cfg = {**env_cfg, **curriculum_params}
    
    reward_cfg = RewardConfig(
        mode=RewardMode(reward_cfg_dict.get("mode", "shaped")),
        reward_scale=reward_cfg_dict.get("reward_scale", 1e-6),
        lambda_inventory=reward_cfg_dict.get("lambda_inventory", 20.0),
        lambda_turnover=reward_cfg_dict.get("lambda_turnover", 0.01),
        lambda_inventory_age=reward_cfg_dict.get("lambda_inventory_age", 0.1),
        terminal_bonus_weight=reward_cfg_dict.get("terminal_bonus_weight", 0.3),
        spread_capture_bonus=reward_cfg_dict.get("spread_capture_bonus", 0.5),
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
    
    env = MarketMakingEnv(
        df=data,
        initial_cash=env_cfg.get("initial_cash", 10000),
        fee_rate=env_cfg.get("fee_rate", 0.0004),
        max_inventory=env_cfg.get("max_inventory", 2.0),
        episode_length=env_cfg.get("episode_length", 1440),
        base_spread=env_cfg.get("base_spread", 60.0),
        random_start=env_cfg.get("random_start", True),
        seed=seed,
        reward_config=reward_cfg,
        obs_config=obs_cfg,
        action_config=action_cfg,
    )
    
    # 動態倉位限制
    if config.get("dynamic_position_limit", {}).get("enabled", False):
        env = DynamicPositionLimitWrapper(
            env,
            base_max_inventory=env_cfg.get("max_inventory", 2.0),
        )
    
    return env


def create_curriculum_stages(config: Dict[str, Any]) -> CurriculumScheduler:
    """創建課程學習階段"""
    curriculum_cfg = config.get("curriculum", {})
    stages_cfg = curriculum_cfg.get("stages", [])
    
    if not stages_cfg:
        # 預設三階段
        stages = [
            CurriculumStage(
                name="easy",
                env_params={"fee_rate": 0.0002, "max_inventory": 3.0},
                advancement_threshold=50.0,
                min_episodes=50,
            ),
            CurriculumStage(
                name="medium",
                env_params={"fee_rate": 0.0003, "max_inventory": 2.5},
                advancement_threshold=30.0,
                min_episodes=100,
            ),
            CurriculumStage(
                name="hard",
                env_params={"fee_rate": 0.0004, "max_inventory": 2.0},
                advancement_threshold=0.0,
                min_episodes=150,
            ),
        ]
    else:
        stages = [
            CurriculumStage(
                name=s.get("name", f"stage_{i}"),
                env_params={k: v for k, v in s.items() if k not in ["name", "min_episodes", "threshold"]},
                advancement_threshold=s.get("threshold", 30.0),
                min_episodes=s.get("min_episodes", 50),
            )
            for i, s in enumerate(stages_cfg)
        ]
    
    return CurriculumScheduler(stages=stages)


# =============================================================================
# Training
# =============================================================================

def train(
    algorithm: str,
    data_path: str,
    config: Dict[str, Any],
    output_dir: Path,
    seed: int = 42,
    use_curriculum: bool = False,
    n_envs: int = 4,
    model_path: Optional[str] = None,
):
    """執行訓練"""
    
    # 載入數據
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    if "close" not in df.columns and "price" in df.columns:
        df["close"] = df["price"]
    
    # 分割數據
    train_end = int(len(df) * 0.7)
    train_data = df.iloc[:train_end].reset_index(drop=True)
    eval_data = df.iloc[train_end:].reset_index(drop=True)
    
    logger.info(f"Data: train={len(train_data)}, eval={len(eval_data)}")
    
    # 課程學習設置
    curriculum_scheduler = None
    if use_curriculum:
        curriculum_scheduler = create_curriculum_stages(config)
        logger.info(f"Curriculum Learning enabled with {len(curriculum_scheduler.stages)} stages")
    
    # 創建環境
    def make_train_env(rank):
        def _init():
            env = create_env(train_data, config, seed=seed + rank)
            if curriculum_scheduler:
                env = CurriculumEnvWrapper(env, curriculum_scheduler)
            return env
        return _init
    
    def make_eval_env():
        return create_env(eval_data, config, seed=seed + 100)
    
    # 向量化環境
    if n_envs > 1:
        vec_env = SubprocVecEnv([make_train_env(i) for i in range(n_envs)])
    else:
        vec_env = DummyVecEnv([make_train_env(0)])
    
    eval_env = DummyVecEnv([make_eval_env])
    
    # 創建或載入模型
    train_cfg = config.get("train", {})
    
    if model_path and Path(model_path).exists():
        logger.info(f"Loading model from {model_path}...")
        model = load_model(algorithm, model_path, env=vec_env)
    else:
        logger.info(f"Creating new {algorithm.upper()} model...")
        
        # 準備超參數覆蓋
        config_overrides = {}
        if "learning_rate" in train_cfg:
            config_overrides["learning_rate"] = train_cfg["learning_rate"]
        if "batch_size" in train_cfg:
            config_overrides["batch_size"] = train_cfg["batch_size"]
        if "net_arch" in train_cfg:
            config_overrides["policy_kwargs"] = {"net_arch": train_cfg["net_arch"]}
        
        model = create_model(
            algo=algorithm,
            env=vec_env,
            config_overrides=config_overrides if config_overrides else None,
            tensorboard_log=str(output_dir / "logs"),
            verbose=1,
            seed=seed,
        )
    
    # 回調
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(output_dir / "checkpoints"),
        name_prefix=f"{algorithm}_model",
    )
    
    callbacks = [eval_callback, checkpoint_callback]
    
    # 訓練
    total_timesteps = train_cfg.get("total_timesteps", 200000)
    logger.info(f"Starting training: {algorithm.upper()}, {total_timesteps} timesteps")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    
    # 保存最終模型
    final_model_path = output_dir / f"final_{algorithm}_model"
    model.save(str(final_model_path))
    logger.info(f"Model saved to: {final_model_path}")
    
    # 保存訓練配置
    config_save_path = output_dir / "training_config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    # 清理
    vec_env.close()
    eval_env.close()
    
    return model


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Flexible RL Training Script")
    parser.add_argument("--algorithm", "-a", type=str, default="sac",
                        choices=["sac", "ppo", "td3"],
                        help="RL algorithm")
    parser.add_argument("--data", "-d", type=str, default="data/btc_usdt_1m_2023.csv",
                        help="Path to training data")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Path to config YAML (optional)")
    parser.add_argument("--timesteps", "-t", type=int, default=200000,
                        help="Total training timesteps")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--n_envs", type=int, default=4,
                        help="Number of parallel environments")
    parser.add_argument("--curriculum", action="store_true",
                        help="Enable Curriculum Learning")
    parser.add_argument("--model_path", "-m", type=str, default=None,
                        help="Path to pre-trained model (optional)")
    parser.add_argument("--output_dir", "-o", type=str, default=None,
                        help="Output directory (auto-generated if not specified)")
    
    args = parser.parse_args()
    
    # 載入配置
    config = load_config(args.config)
    
    # 覆蓋 timesteps
    if "train" not in config:
        config["train"] = {}
    config["train"]["total_timesteps"] = args.timesteps
    
    # 輸出目錄
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"runs/{args.algorithm}_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # 執行訓練
    model = train(
        algorithm=args.algorithm,
        data_path=args.data,
        config=config,
        output_dir=output_dir,
        seed=args.seed,
        use_curriculum=args.curriculum,
        n_envs=args.n_envs,
        model_path=args.model_path,
    )
    
    logger.info("Training complete!")
    
    return model


if __name__ == "__main__":
    main()
