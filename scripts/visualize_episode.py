
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import SAC
from envs.market_making_env_v2 import (
    MarketMakingEnvV2, RewardConfig, ObservationConfig, ActionConfig, RewardMode
)
from utils.risk_sensitive import DynamicPositionLimitWrapper

def visualize(run_folder, episode_idx=0):
    run_path = Path(run_folder)
    config_path = run_path / 'config.yaml'
    if not config_path.exists():
        config_path = run_path / 'env_config.yaml'
    model_path = run_path / 'best_model' / 'best_model.zip'
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load data
    data_file = config['env']['data_file']
    df = pd.read_csv(data_file, parse_dates=['timestamp'])
    if 'close' not in df.columns and 'price' in df.columns:
        df['close'] = df['price']

    # OOS Data
    split_cfg = config.get('data_split', {})
    valid_end = split_cfg.get('valid_end', 0.67)
    valid_idx = int(len(df) * valid_end)
    test_df = df.iloc[valid_idx:].reset_index(drop=True)
    
    # Create Environment
    env_cfg = config['env']
    obs_cfg = config.get('observation', {})
    reward_cfg = config.get('reward', {})
    action_cfg = config.get('action', {})

    obs_config = ObservationConfig(
        include_trend=obs_cfg.get('include_trend', False),
        trend_windows=obs_cfg.get('trend_windows', [60, 240, 1440]),
        include_price=obs_cfg.get('include_price', True),
        include_inventory=obs_cfg.get('include_inventory', True),
        include_time=obs_cfg.get('include_time', True),
        include_volatility=obs_cfg.get('include_volatility', True),
        include_momentum=obs_cfg.get('include_momentum', True),
        include_volume=obs_cfg.get('include_volume', True),
        include_inventory_age=obs_cfg.get('include_inventory_age', True),
        volatility_windows=obs_cfg.get('volatility_windows', [5, 15, 60]),
        momentum_windows=obs_cfg.get('momentum_windows', [5, 15]),
    )

    reward_config = RewardConfig(
        mode=RewardMode(reward_cfg.get('mode', 'shaped')),
        lambda_inventory=reward_cfg.get('lambda_inventory', 0.005),
        reward_scale=reward_cfg.get('reward_scale', 1.0),
    )

    action_config = ActionConfig(
        mode=action_cfg.get('mode', 'asymmetric'),
        allow_no_quote=action_cfg.get('allow_no_quote', False),
    )

    env = MarketMakingEnvV2(
        df=test_df,
        initial_cash=env_cfg.get('initial_cash', 10000),
        fee_rate=env_cfg.get('fee_rate', 0.0004),
        max_inventory=env_cfg.get('max_inventory', 5.0),
        episode_length=env_cfg.get('episode_length', 1440),
        base_spread=env_cfg.get('base_spread', 25.0),
        random_start=True,
        obs_config=obs_config,
        reward_config=reward_config,
        action_config=action_config,
        seed=42 + episode_idx, # Different seed for different episodes
    )
    
    # Dynamic Position Limit
    dpl_config = config.get('dynamic_position_limit', {})
    if dpl_config.get('enabled', False):
        env = DynamicPositionLimitWrapper(
            env,
            base_max_inventory=env_cfg.get('max_inventory', 5.0),
            volatility_threshold=dpl_config.get('volatility_threshold', 0.02),
            min_inventory_ratio=dpl_config.get('min_inventory_ratio', 0.3),
            volatility_window=dpl_config.get('volatility_window', 60),
        )

    # Load Model
    model = SAC.load(model_path)
    
    # Run Episode
    obs, _ = env.reset()
    done = False
    
    prices = []
    inventories = []
    pnls = []
    buy_trades = [] # (step, price)
    sell_trades = [] # (step, price)
    
    step = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        prices.append(info.get('mid_price', 0))
        inventories.append(info.get('inventory', 0))
        pnls.append(info.get('portfolio_value', 10000) - 10000)
        
        if info.get('trade_executed', False): # Need to check how to detect trade in wrapper
            # Wrapper might hide info, let's check env.unwrapped
            pass
            
        # Manual trade detection
        if len(inventories) > 1:
            delta = inventories[-1] - inventories[-2]
            if delta > 0.01:
                buy_trades.append((step, prices[-1]))
            elif delta < -0.01:
                sell_trades.append((step, prices[-1]))
        
        step += 1
        
    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Price & Trades
    ax1.plot(prices, label='Price', color='gray', alpha=0.5)
    if buy_trades:
        bx, by = zip(*buy_trades)
        ax1.scatter(bx, by, marker='^', color='green', label='Buy', s=30)
    if sell_trades:
        sx, sy = zip(*sell_trades)
        ax1.scatter(sx, sy, marker='v', color='red', label='Sell', s=30)
    ax1.set_title(f'Price & Trades (Ep {episode_idx})')
    ax1.legend()
    
    # Inventory
    ax2.plot(inventories, label='Inventory', color='blue')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.set_title('Inventory')
    ax2.set_ylabel('Contracts')
    
    # PnL
    ax3.plot(pnls, label='PnL', color='green')
    ax3.set_title(f'Cumulative PnL: {pnls[-1]:.2f}')
    ax3.set_ylabel('USDT')
    
    plt.tight_layout()
    output_file = f'plots/episode_{episode_idx}_vis.png'
    plt.savefig(output_file)
    print(f'Saved visualization to {output_file}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_folder', type=str, required=True)
    parser.add_argument('--episode', type=int, default=0)
    args = parser.parse_args()
    
    visualize(args.run_folder, args.episode)
