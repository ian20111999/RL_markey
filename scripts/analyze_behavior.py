import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from stable_baselines3 import SAC
from envs.market_making_env_v2 import (
    MarketMakingEnvV2, RewardConfig, ObservationConfig, ActionConfig
)
import matplotlib.pyplot as plt
# import seaborn as sns

def analyze_behavior(run_folder, config_path, episodes=20):
    # 1. Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Load Data (OOS)
    df = pd.read_csv('data/btc_usdt_1m_2023.csv', parse_dates=['timestamp'])
    if 'close' not in df.columns and 'price' in df.columns:
        df['close'] = df['price']
        
    # Use OOS split
    split_cfg = config.get('data_split', {})
    valid_end = split_cfg.get('valid_end', 0.67)
    valid_idx = int(len(df) * valid_end)
    test_df = df.iloc[valid_idx:].reset_index(drop=True)
    
    print(f"Loading model from: {run_folder}")
    print(f"Testing on {len(test_df)} rows of OOS data")

    # 3. Setup Environment
    env_cfg = config['env']
    obs_cfg = config.get('observation', {})
    reward_cfg = config.get('reward', {})
    action_cfg = config.get('action', {})

    obs_config = ObservationConfig(
        include_trend=obs_cfg.get('include_trend', True),
        trend_windows=obs_cfg.get('trend_windows', [60, 240, 1440]),
    )

    reward_config = RewardConfig(
        mode=reward_cfg.get('mode', 'shaped'),
        lambda_inventory=reward_cfg.get('lambda_inventory', 0.005),
        spread_capture_bonus=reward_cfg.get('spread_capture_bonus', 0.1),
        round_trip_bonus=reward_cfg.get('round_trip_bonus', 2.0),
        inventory_revert_bonus=reward_cfg.get('inventory_revert_bonus', 1.0),
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

    # 4. Load Model
    model_path = Path(run_folder) / "best_model/best_model.zip"
    if not model_path.exists():
        # Try searching for any zip file
        zips = list(Path(run_folder).glob("**/*.zip"))
        if zips:
            model_path = zips[0]
        else:
            raise FileNotFoundError(f"No model found in {run_folder}")
            
    model = SAC.load(model_path)

    # 5. Run Analysis
    stats = {
        'holding_times': [],
        'trade_pnls': [],
        'spreads': [],
        'inventory_levels': [],
        'trade_sides': [],
        'episode_pnls': []
    }

    print(f"\nRunning analysis for {episodes} episodes...")
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        
        # Episode tracking
        entry_times = {} # id -> step
        trades = []
        
        step_count = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            stats['inventory_levels'].append(env.inventory)
            
            # Track trades
            trades_count = info.get('trades_count', 0)
            if trades_count > 0:
                # Infer side from inventory change
                current_inv = env.inventory
                # We need last_inventory, but env.inventory is already updated.
                # We can track it manually in the loop
                pass

            # Manual inventory tracking to detect side
            if len(stats['inventory_levels']) >= 2:
                prev_inv = stats['inventory_levels'][-2]
                curr_inv = stats['inventory_levels'][-1]
                delta = curr_inv - prev_inv
                
                if delta > 0.01:
                    stats['trade_sides'].append('buy')
                elif delta < -0.01:
                    stats['trade_sides'].append('sell')


            step_count += 1
            
        stats['episode_pnls'].append(info.get('episode_net_pnl', 0))
        print(f"Ep {ep+1}: PnL={info.get('episode_net_pnl', 0):.2f}")

    # 6. Calculate Metrics
    avg_pnl = np.mean(stats['episode_pnls'])
    win_rate = np.mean([p > 0 for p in stats['episode_pnls']])
    
    inv_levels = np.array(stats['inventory_levels'])
    avg_abs_inv = np.mean(np.abs(inv_levels))
    max_inv = np.max(np.abs(inv_levels))
    
    buy_count = stats['trade_sides'].count('buy')
    sell_count = stats['trade_sides'].count('sell')
    total_trades = buy_count + sell_count
    
    print("\n" + "="*50)
    print("BEHAVIOR ANALYSIS REPORT")
    print("="*50)
    print(f"Performance:")
    print(f"  Mean PnL:        {avg_pnl:+.2f}")
    print(f"  Win Rate:        {win_rate*100:.1f}%")
    print(f"\nInventory Management:")
    print(f"  Avg |Inventory|: {avg_abs_inv:.2f} (Target: Low)")
    print(f"  Max |Inventory|: {max_inv:.2f} (Limit: 5.0)")
    print(f"  Zero Inv %:      {np.mean(inv_levels == 0)*100:.1f}%")
    print(f"\nTrading Activity (Total {episodes} eps):")
    print(f"  Total Trades:    {total_trades}")
    print(f"  Avg Trades/Ep:   {total_trades/episodes:.1f}")
    print(f"  Buy/Sell Ratio:  {buy_count}/{sell_count} ({buy_count/max(1, sell_count):.2f})")
    
    # Interpretation
    print("\n" + "-"*50)
    print("STRATEGY CLASSIFICATION:")
    if avg_abs_inv < 1.0 and total_trades/episodes > 50:
        print(">> TYPE: High-Frequency Market Maker (Ideal)")
        print("   (Low inventory, high turnover)")
    elif avg_abs_inv > 2.5:
        print(">> TYPE: Directional/Inventory Hoarder (Risky)")
        print("   (Holding large positions, likely betting on trends)")
    elif total_trades/episodes < 10:
        print(">> TYPE: Passive/Inactive")
        print("   (Not trading enough)")
    else:
        print(">> TYPE: Hybrid / Balanced")
    print("-"*50)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_folder", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/env_v3_normalized.yaml")
    args = parser.parse_args()
    
    analyze_behavior(args.run_folder, args.config)
