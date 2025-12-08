
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from stable_baselines3 import SAC
from envs.market_making_env_v2 import (
    MarketMakingEnvV2, RewardConfig, ObservationConfig, ActionConfig
)

def evaluate(run_folder, n_episodes=30, model_file=None):
    run_path = Path(run_folder)
    config_path = run_path / 'config.yaml'
    
    if model_file:
        model_path = Path(model_file)
    else:
        model_path = run_path / 'best_model' / 'best_model.zip'
    
    if not config_path.exists():
        print(f"Config not found at {config_path}")
        return
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load data
    data_file = config['env']['data_file']
    df = pd.read_csv(data_file, parse_dates=['timestamp'])
    if 'close' not in df.columns and 'price' in df.columns:
        df['close'] = df['price']

    # OOS Data (last 33%)
    split_cfg = config.get('data_split', {})
    valid_end = split_cfg.get('valid_end', 0.67)
    valid_idx = int(len(df) * valid_end)
    test_df = df.iloc[valid_idx:].reset_index(drop=True)
    print(f'📊 OOS Data: {len(test_df)} rows (from index {valid_idx})')

    # Create Environment
    env_cfg = config['env']
    obs_cfg = config.get('observation', {})
    reward_cfg = config.get('reward', {})
    action_cfg = config.get('action', {})

    obs_config = ObservationConfig(
        include_trend=obs_cfg.get('include_trend', True),
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
        mode=reward_cfg.get('mode', 'shaped'),
        reward_scale=reward_cfg.get('reward_scale', 1.0),
        lambda_inventory=reward_cfg.get('lambda_inventory', 0.005),
        lambda_turnover=reward_cfg.get('lambda_turnover', 0.0),
        spread_capture_bonus=reward_cfg.get('spread_capture_bonus', 0.0),
        round_trip_bonus=reward_cfg.get('round_trip_bonus', 0.0),
        inventory_revert_bonus=reward_cfg.get('inventory_revert_bonus', 0.0),
    )

    action_config = ActionConfig(
        mode=action_cfg.get('mode', 'asymmetric'),
        allow_no_quote=action_cfg.get('allow_no_quote', False),
        max_spread_multiplier=action_cfg.get('max_spread_multiplier', 2.0),
        min_spread_multiplier=action_cfg.get('min_spread_multiplier', 0.5),
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
        seed=42,
    )

    print(f'✅ Environment created. Obs dim: {env.observation_space.shape[0]}')

    # Load Model
    model = SAC.load(model_path)
    print(f'✅ Model loaded from {model_path}')

    # Run Evaluation
    print()
    print(f'🔬 Running OOS Evaluation ({n_episodes} episodes)...')
    print('-' * 80)
    print(f"{'Ep':<4} | {'PnL':>10} | {'Trades':>6} | {'MaxInv':>6} | {'Sharpe':>6} | {'Hold%':>6}")
    print('-' * 80)

    all_pnls = []
    all_trades = []
    all_max_inv = []
    all_sharpes = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_trades = 0
        max_inv = 0
        prev_inv = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            curr_inv = abs(env.inventory)
            if abs(curr_inv - prev_inv) > 0.01:
                ep_trades += 1
            prev_inv = curr_inv
            max_inv = max(max_inv, curr_inv)
        
        ep_pnl = info.get('episode_net_pnl', 0)
        metrics = info.get('metrics', {})
        sharpe = metrics.get('sharpe', 0)
        
        all_pnls.append(ep_pnl)
        all_trades.append(ep_trades)
        all_max_inv.append(max_inv)
        all_sharpes.append(sharpe)
        
        print(f"{ep+1:<4} | {ep_pnl:>10.2f} | {ep_trades:>6} | {max_inv:>6.2f} | {sharpe:>6.2f} | {'N/A':>6}")

    print('-' * 80)
    print()
    print('📊 Summary:')
    print(f'   Mean PnL:       {np.mean(all_pnls):+,.2f} ± {np.std(all_pnls):,.2f}')
    print(f'   Mean Trades:    {np.mean(all_trades):.1f}')
    print(f'   Mean MaxInv:    {np.mean(all_max_inv):.2f}')
    print(f'   Positive Eps:   {sum(1 for p in all_pnls if p > 0)} / {n_episodes} ({sum(1 for p in all_pnls if p > 0)/n_episodes*100:.0f}%)')
    print(f'   Total PnL:      {sum(all_pnls):+,.2f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_folder', type=str, required=True, help='Path to the run folder')
    parser.add_argument('--episodes', type=int, default=30, help='Number of episodes')
    parser.add_argument('--model_path', type=str, default=None, help='Path to specific model zip file')
    args = parser.parse_args()
    
    evaluate(args.run_folder, args.episodes, args.model_path)
