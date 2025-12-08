
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import sys
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from envs.market_making_env_v2 import (
    MarketMakingEnvV2, RewardConfig, ObservationConfig, ActionConfig,
    RewardMode, FillModelEnvConfig, AdvancedObservationConfig
)
from utils.risk_sensitive import DynamicPositionLimitWrapper

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_env(data, config, seed=None):
    env_config = config['env']
    reward_config = config.get('reward', {})
    obs_config = config.get('observation', {})
    action_config = config.get('action', {})
    dr_config = config.get('domain_randomization', {})
    fill_config = config.get('fill_model', {})
    adv_obs_config = config.get('advanced_observation', {})
    
    # Reward Config
    reward_cfg = RewardConfig(
        mode=RewardMode(reward_config.get('mode', 'shaped')),
        lambda_inventory=reward_config.get('lambda_inventory', 0.005),
        lambda_turnover=reward_config.get('lambda_turnover', 0.0),
        gamma=reward_config.get('gamma', 0.99),
        sparse_scale=reward_config.get('sparse_scale', 0.01),
        terminal_bonus_weight=reward_config.get('terminal_bonus_weight', 0.5),
        spread_capture_bonus=reward_config.get('spread_capture_bonus', 0.0),
        round_trip_bonus=reward_config.get('round_trip_bonus', 0.0),
        inventory_revert_bonus=reward_config.get('inventory_revert_bonus', 0.0),
        reward_scale=reward_config.get('reward_scale', 1.0),
    )
    
    # Action Config
    action_cfg = ActionConfig(
        mode=action_config.get('mode', 'asymmetric'),
        allow_no_quote=action_config.get('allow_no_quote', False),
        max_spread_multiplier=action_config.get('max_spread_multiplier', 2.0),
        min_spread_multiplier=action_config.get('min_spread_multiplier', 0.5),
    )
    
    # Observation Config
    obs_cfg = ObservationConfig(
        include_price=obs_config.get('include_price', True),
        include_inventory=obs_config.get('include_inventory', True),
        include_time=obs_config.get('include_time', True),
        include_volatility=obs_config.get('include_volatility', True),
        include_momentum=obs_config.get('include_momentum', True),
        include_volume=obs_config.get('include_volume', True),
        include_inventory_age=obs_config.get('include_inventory_age', True),
        include_trend=obs_config.get('include_trend', False),
        volatility_windows=obs_config.get('volatility_windows', [5, 15, 60]),
        momentum_windows=obs_config.get('momentum_windows', [5, 15]),
        trend_windows=obs_config.get('trend_windows', [60, 240, 1440]),
    )
    
    env = MarketMakingEnvV2(
        df=data,
        initial_cash=env_config.get('initial_cash', 10000),
        fee_rate=env_config.get('fee_rate', 0.0004),
        max_inventory=env_config.get('max_inventory', 5.0),
        episode_length=env_config.get('episode_length', 1440),
        base_spread=env_config.get('base_spread', 25.0),
        random_start=env_config.get('random_start', True),
        seed=seed,
        reward_config=reward_cfg,
        obs_config=obs_cfg,
        action_config=action_cfg,
    )
    
    # Dynamic Position Limit
    dpl_config = config.get('dynamic_position_limit', {})
    if dpl_config.get('enabled', False):
        env = DynamicPositionLimitWrapper(
            env,
            base_max_inventory=env_config.get('max_inventory', 5.0),
            volatility_threshold=dpl_config.get('volatility_threshold', 0.02),
            min_inventory_ratio=dpl_config.get('min_inventory_ratio', 0.3),
            volatility_window=dpl_config.get('volatility_window', 60),
        )
        
    return env

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_timesteps", type=int, default=300000)
    parser.add_argument("--config", type=str, default='configs/env_v3_normalized.yaml')
    parser.add_argument("--model_path", type=str, default='runs/v3_nuclear_continued_20251206_122315/best_model/best_model.zip')
    args = parser.parse_args()

    # Paths
    config_path = args.config
    model_path = args.model_path
    
    # Create new run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"runs/v3_nuclear_long_run_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Load data
    logger.info("Loading data...")
    df = pd.read_csv(config['env']['data_file'])
    if 'close' not in df.columns and 'price' in df.columns:
        df['close'] = df['price']
        
    # Split data
    split_cfg = config.get('data_split', {})
    train_end = split_cfg.get('train_end', 0.5)
    valid_end = split_cfg.get('valid_end', 0.67)
    train_idx = int(len(df) * train_end)
    valid_idx = int(len(df) * valid_end)
    
    train_data = df.iloc[:train_idx].reset_index(drop=True)
    valid_data = df.iloc[train_idx:valid_idx].reset_index(drop=True)
    
    # Data Augmentation (Flip)
    if config.get('data_augmentation', {}).get('enable_price_flip', False):
        logger.info("Applying price flip augmentation...")
        flipped_train = train_data.copy()
        flipped_train['close'] = 1.0 / flipped_train['close']
        flipped_train['high'] = 1.0 / flipped_train['low']
        flipped_train['low'] = 1.0 / flipped_train['high']
        flipped_train['open'] = 1.0 / flipped_train['open']
        # Re-normalize prices to start at same level
        start_price = train_data['close'].iloc[0]
        flipped_start = flipped_train['close'].iloc[0]
        ratio = start_price / flipped_start
        flipped_train[['close', 'high', 'low', 'open']] *= ratio
        
        train_data = pd.concat([train_data, flipped_train], ignore_index=True)
    
    # Create Envs
    env = create_env(train_data, config)
    eval_env = create_env(valid_data, config)
    
    vec_env = DummyVecEnv([lambda: env])
    eval_vec_env = DummyVecEnv([lambda: eval_env])
    
    # Load Model
    logger.info(f"Loading model from {model_path}...")
    model = SAC.load(model_path, env=vec_env)
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_vec_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="sac_model"
    )
    
    # Continue Training
    logger.info(f"Continuing training for {args.total_timesteps} steps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        reset_num_timesteps=False
    )
    
    # Save Final Model
    model.save(str(output_dir / "final_model"))
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
