
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import sys
from datetime import datetime
import argparse

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
class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    # Simplified format: Time - Message
    format_str = "%(asctime)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: green + "%(message)s" + reset,
        logging.WARNING: yellow + "%(asctime)s - WARNING - %(message)s" + reset,
        logging.ERROR: red + "%(asctime)s - ERROR - %(message)s" + reset,
        logging.CRITICAL: bold_red + "%(asctime)s - CRITICAL - %(message)s" + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return formatter.format(record)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Remove existing handlers if any
if root_logger.hasHandlers():
    root_logger.handlers.clear()

console_handler = logging.StreamHandler()
console_handler.setFormatter(CustomFormatter())
root_logger.addHandler(console_handler)

# Suppress noisy libraries
logging.getLogger("stable_baselines3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_timesteps", type=int, default=200000)
    parser.add_argument("--config", type=str, default='configs/env_v3_stabilized.yaml')
    parser.add_argument("--model_path", type=str, default='runs/v3_nuclear_continued_20251206_122315/best_model/best_model.zip')
    # Add missing arguments to match restart_training.sh
    parser.add_argument("--base_model", type=str, dest="model_path", help="Alias for model_path")
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    args = parser.parse_args()

    # Paths
    config_path = args.config
    model_path = args.model_path
    
    # Create new run directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"runs/v3_stabilized_run_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with CLI args
    if args.learning_rate:
        if 'train' not in config: config['train'] = {}
        config['train']['learning_rate'] = args.learning_rate
        
    if args.batch_size:
        if 'train' not in config: config['train'] = {}
        config['train']['batch_size'] = args.batch_size
        
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
    # We load the model but we need to update the learning rate and other hyperparameters
    # SAC.load will load the saved parameters. We need to override them.
    # However, stable-baselines3 load() doesn't easily allow overriding optimizer params directly in the load call for everything.
    # But we can modify the model object after loading.
    
    custom_objects = {
        "learning_rate": config['train']['learning_rate'],
        "batch_size": config['train']['batch_size'],
        "ent_coef": config['train']['ent_coef'],
        "target_entropy": config['train']['target_entropy']
    }
    
    # Note: changing LR after load requires updating the optimizer's param groups if the optimizer is already loaded.
    # But SB3 re-creates the optimizer on learn() if it's not fully restored? No, it restores it.
    # A safer way is to pass `custom_objects` to load, but that's for pickle compatibility.
    # The best way to change LR is to set `model.learning_rate` to a float or schedule.
    
    model = SAC.load(model_path, env=vec_env)
    
    # UPDATE HYPERPARAMETERS
    new_lr = config['train']['learning_rate']
    new_batch_size = config['train']['batch_size']
    new_ent_coef = config['train']['ent_coef']
    
    logger.info(f"Updating Hyperparameters: LR={new_lr}, Batch={new_batch_size}, EntCoef={new_ent_coef}")
    
    # 1. Update Learning Rate - ROBUST METHOD
    # Step 1: Update model attribute
    model.learning_rate = new_lr
    
    # Step 2: Force update all optimizers (handle various SB3 internal structures)
    optimizers_updated = []
    
    # Update Actor Optimizer
    if hasattr(model, 'actor'):
        if hasattr(model.actor, 'optimizer'):
            for param_group in model.actor.optimizer.param_groups:
                param_group['lr'] = new_lr
            optimizers_updated.append(f"Actor: {new_lr}")
        
    # Update Critic Optimizer
    if hasattr(model, 'critic'):
        if hasattr(model.critic, 'optimizer'):
            for param_group in model.critic.optimizer.param_groups:
                param_group['lr'] = new_lr
            optimizers_updated.append(f"Critic: {new_lr}")
    
    # Update Critic Target (if exists)
    if hasattr(model, 'critic_target'):
        if hasattr(model.critic_target, 'optimizer'):
            for param_group in model.critic_target.optimizer.param_groups:
                param_group['lr'] = new_lr
            optimizers_updated.append(f"Critic Target: {new_lr}")
            
    # Update Entropy Optimizer (if exists)
    if hasattr(model, 'ent_coef_optimizer') and model.ent_coef_optimizer is not None:
        for param_group in model.ent_coef_optimizer.param_groups:
            param_group['lr'] = new_lr
        optimizers_updated.append(f"Entropy: {new_lr}")
    
    # Verify updates
    logger.info("=" * 60)
    logger.info("Learning Rate Update Verification:")
    for update_msg in optimizers_updated:
        logger.info(f"  ✓ {update_msg}")
    
    # Additional verification: check actual optimizer state
    if hasattr(model, 'actor') and hasattr(model.actor, 'optimizer'):
        actual_lr = model.actor.optimizer.param_groups[0]['lr']
        logger.info(f"  → Verified Actor LR: {actual_lr}")
        if abs(actual_lr - new_lr) > 1e-9:
            logger.warning(f"  ⚠️  LR mismatch! Expected {new_lr}, got {actual_lr}")
    logger.info("=" * 60)
        
    # 2. Update Batch Size
    model.batch_size = new_batch_size
    
    # 3. Update Entropy Coefficient
    # If it was 'auto', ent_coef_optimizer might exist. If it was fixed, it might not.
    # The previous model had ent_coef=0.1 (fixed).
    # The new config has ent_coef="auto".
    # This is tricky. Switching from fixed to auto requires initializing log_ent_coef and ent_coef_optimizer.
    
    if new_ent_coef == 'auto' and not isinstance(model.ent_coef, str):
        logger.info("Switching from Fixed Entropy to Auto Entropy...")
        # We need to re-initialize entropy optimization
        model.ent_coef = 'auto'
        model.target_entropy = config['train']['target_entropy']
        if model.target_entropy == 'auto':
            model.target_entropy = float(-np.prod(env.action_space.shape).astype(np.float32))
            
        # Initialize log_ent_coef
        import torch
        model.log_ent_coef = torch.log(torch.ones(1, device=model.device)).requires_grad_(True)
        model.ent_coef_optimizer = torch.optim.Adam([model.log_ent_coef], lr=new_lr)
    
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
    logger.info(f"Starting Stabilization Run for {args.total_timesteps} steps...")
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
