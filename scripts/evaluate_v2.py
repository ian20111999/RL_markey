#!/usr/bin/env python3
"""
scripts/evaluate_v2.py
V2 環境的綜合評估腳本
- 行為指標：平均 Spread、Fill Rate、Inventory Turnover、Quote 意願分佈
- 風險指標：VaR、Expected Shortfall、Max Drawdown、Max Inventory Time
- 統計顯著性：Bootstrap Confidence Interval

用法:
    python scripts/evaluate_v2.py --model_path runs/exp_v2_xxx/final/model.zip \
                                  --config configs/env_v3.yaml \
                                  --n_episodes 20 \
                                  --output_dir runs/exp_v2_xxx/evaluation
"""

import argparse
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

from stable_baselines3 import SAC
import gymnasium as gym

# 添加專案路徑
import sys
sys.path.append(str(Path(__file__).parent.parent))

from envs.market_making_env_v2 import MarketMakingEnvV2
from utils.config import load_config


def evaluate_episode(env, model, deterministic: bool = True) -> Dict:
    """評估單一 Episode 並收集詳細指標"""
    obs, _ = env.reset()
    done = False
    
    # 收集指標
    episode_data = {
        'pnl_history': [],
        'inventory_history': [],
        'spread_history': [],
        'actions': [],
        'fills': [],
        'step_pnls': [],
    }
    
    step = 0
    cumulative_pnl = 0
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 解析 action
        bid_spread = action[0]
        ask_spread = action[1]
        bid_qty = action[2]
        ask_qty = action[3]
        
        # 記錄 spread（只有在真正下單時）
        if bid_qty > 0.1:
            episode_data['spread_history'].append(bid_spread)
        if ask_qty > 0.1:
            episode_data['spread_history'].append(ask_spread)
        
        # 使用 info 中的資訊
        step_pnl = info.get('step_pnl', 0)
        cumulative_pnl += step_pnl
        
        episode_data['pnl_history'].append(cumulative_pnl)
        episode_data['inventory_history'].append(info.get('inventory', 0))
        episode_data['actions'].append(action.copy())
        episode_data['step_pnls'].append(step_pnl)
        
        if info.get('trade_executed', False):
            episode_data['fills'].append({
                'step': step,
                'side': info.get('trade_side', 'unknown'),
                'price': info.get('trade_price', 0),
            })
        
        step += 1
    
    return episode_data, env._total_pnl


def compute_behavior_metrics(episodes_data: List[Dict]) -> Dict:
    """計算行為指標"""
    all_spreads = []
    all_inventories = []
    bid_quote_rates = []
    ask_quote_rates = []
    fill_counts = []
    
    for ep in episodes_data:
        if ep['spread_history']:
            all_spreads.extend(ep['spread_history'])
        all_inventories.extend([abs(inv) for inv in ep['inventory_history']])
        
        actions = np.array(ep['actions'])
        if len(actions) > 0:
            bid_quote_rates.append(np.mean(actions[:, 2] > 0.1))  # bid quantity > 0
            ask_quote_rates.append(np.mean(actions[:, 3] > 0.1))  # ask quantity > 0
        
        fill_counts.append(len(ep['fills']))
    
    # Inventory Turnover: 平均每 Episode 成交次數 / 平均持倉
    avg_fills = np.mean(fill_counts) if fill_counts else 0
    avg_inventory = np.mean(all_inventories) if all_inventories else 1
    inventory_turnover = avg_fills / max(avg_inventory, 0.01)
    
    return {
        'avg_spread': float(np.mean(all_spreads)) if all_spreads else 0,
        'spread_std': float(np.std(all_spreads)) if all_spreads else 0,
        'avg_bid_quote_rate': float(np.mean(bid_quote_rates)) if bid_quote_rates else 0,
        'avg_ask_quote_rate': float(np.mean(ask_quote_rates)) if ask_quote_rates else 0,
        'inventory_turnover': float(inventory_turnover),
        'avg_fill_count': float(avg_fills),
    }


def compute_risk_metrics(episodes_data: List[Dict], pnls: List[float]) -> Dict:
    """計算風險指標"""
    all_step_pnls = []
    for ep in episodes_data:
        all_step_pnls.extend(ep['step_pnls'])
    
    # VaR (5%)
    var_5 = np.percentile(all_step_pnls, 5) if all_step_pnls else 0
    
    # Expected Shortfall (平均低於 VaR 的損失)
    below_var = [p for p in all_step_pnls if p < var_5]
    es_5 = np.mean(below_var) if below_var else var_5
    
    # Max Drawdown
    max_drawdowns = []
    for ep in episodes_data:
        pnl_history = ep['pnl_history']
        if pnl_history:
            peak = np.maximum.accumulate(pnl_history)
            drawdown = peak - pnl_history
            max_drawdowns.append(np.max(drawdown))
    
    # Max Inventory Holding Time
    max_inventory_times = []
    for ep in episodes_data:
        inv_history = ep['inventory_history']
        if inv_history:
            # 計算連續持有庫存的最長時間
            current_streak = 0
            max_streak = 0
            for inv in inv_history:
                if abs(inv) > 0.1:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 0
            max_inventory_times.append(max_streak)
    
    # 總體 PnL 統計
    pnl_std = float(np.std(pnls)) if len(pnls) > 1 else 0
    
    return {
        'var_5pct': float(var_5),
        'expected_shortfall_5pct': float(es_5),
        'max_drawdown': float(np.mean(max_drawdowns)) if max_drawdowns else 0,
        'max_drawdown_std': float(np.std(max_drawdowns)) if len(max_drawdowns) > 1 else 0,
        'max_inventory_time_avg': float(np.mean(max_inventory_times)) if max_inventory_times else 0,
        'episode_pnl_std': pnl_std,
    }


def bootstrap_confidence_interval(
    data: List[float], 
    n_bootstrap: int = 1000, 
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """Bootstrap 方法計算信賴區間"""
    if len(data) < 2:
        mean = np.mean(data) if data else 0
        return mean, mean, mean
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return float(np.mean(data)), float(lower), float(upper)


def main():
    parser = argparse.ArgumentParser(description="V2 Environment Evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/env_v3.yaml")
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--deterministic", action="store_true", default=True)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 載入配置
    cfg = load_config(args.config)
    
    # 載入數據
    df = pd.read_csv(cfg['data']['path'], parse_dates=['timestamp'])
    test_start = int(len(df) * cfg['data_split']['train_end'])
    test_df = df.iloc[test_start:].reset_index(drop=True)
    
    print(f"📊 Test Data: {len(test_df)} rows")
    
    # 建立 V2 環境
    env_cfg = cfg.get('env', {})
    reward_cfg = env_cfg.get('reward_config', {})
    obs_cfg = env_cfg.get('obs_config', {})
    action_cfg = env_cfg.get('action_config', {})
    
    env = MarketMakingEnvV2(
        df=test_df,
        initial_cash=cfg['env'].get('initial_cash', 100_000),
        fee_rate=cfg['env'].get('fee_rate', 0.0004),
        max_inventory=cfg['env'].get('max_inventory', 10.0),
        lookback=cfg['env'].get('lookback', 60),
        # Reward
        reward_mode=reward_cfg.get('mode', 'dense'),
        inventory_penalty=reward_cfg.get('inventory_penalty', 0.01),
        shaping_lambda=reward_cfg.get('shaping_lambda', 0.1),
        # Observation
        include_volatility=obs_cfg.get('include_volatility', True),
        include_momentum=obs_cfg.get('include_momentum', True),
        include_time_features=obs_cfg.get('include_time_features', True),
        volatility_windows=obs_cfg.get('volatility_windows', [5, 60]),
        # Action
        spread_range=tuple(action_cfg.get('spread_range', [0.0001, 0.01])),
        allow_asymmetric_spread=action_cfg.get('allow_asymmetric_spread', True),
        quantity_range=tuple(action_cfg.get('quantity_range', [0.0, 1.0])),
        # Disable Domain Randomization for evaluation
        enable_domain_randomization=False,
    )
    
    # 載入模型
    model = SAC.load(args.model_path)
    print(f"✅ Model loaded: {args.model_path}")
    
    # 評估
    print(f"\n🔄 Running {args.n_episodes} evaluation episodes...")
    episodes_data = []
    episode_pnls = []
    
    for ep in range(args.n_episodes):
        ep_data, pnl = evaluate_episode(env, model, deterministic=args.deterministic)
        episodes_data.append(ep_data)
        episode_pnls.append(pnl)
        print(f"  Episode {ep+1:3d}: PnL = {pnl:,.2f}")
    
    # 計算指標
    print("\n📈 Computing Metrics...")
    behavior_metrics = compute_behavior_metrics(episodes_data)
    risk_metrics = compute_risk_metrics(episodes_data, episode_pnls)
    
    # Bootstrap CI for PnL
    pnl_mean, pnl_lower, pnl_upper = bootstrap_confidence_interval(episode_pnls)
    
    # 彙整結果
    results = {
        'n_episodes': args.n_episodes,
        'pnl': {
            'mean': pnl_mean,
            'ci_lower_95': pnl_lower,
            'ci_upper_95': pnl_upper,
        },
        'behavior': behavior_metrics,
        'risk': risk_metrics,
    }
    
    # 儲存
    with open(output_dir / 'evaluation_report.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 儲存每 Episode 的 PnL
    pd.DataFrame({
        'episode': list(range(1, args.n_episodes + 1)),
        'pnl': episode_pnls
    }).to_csv(output_dir / 'episode_pnls.csv', index=False)
    
    # 打印報告
    print("\n" + "=" * 60)
    print("📊 EVALUATION REPORT")
    print("=" * 60)
    print(f"\n📈 PnL Performance:")
    print(f"   Mean: ${pnl_mean:,.2f}")
    print(f"   95% CI: [${pnl_lower:,.2f}, ${pnl_upper:,.2f}]")
    print(f"\n🎯 Behavior Metrics:")
    print(f"   Average Spread: {behavior_metrics['avg_spread']:.6f}")
    print(f"   Bid Quote Rate: {behavior_metrics['avg_bid_quote_rate']:.2%}")
    print(f"   Ask Quote Rate: {behavior_metrics['avg_ask_quote_rate']:.2%}")
    print(f"   Avg Fill Count: {behavior_metrics['avg_fill_count']:.1f}")
    print(f"   Inventory Turnover: {behavior_metrics['inventory_turnover']:.2f}")
    print(f"\n⚠️ Risk Metrics:")
    print(f"   VaR (5%): ${risk_metrics['var_5pct']:,.4f}")
    print(f"   Expected Shortfall (5%): ${risk_metrics['expected_shortfall_5pct']:,.4f}")
    print(f"   Avg Max Drawdown: ${risk_metrics['max_drawdown']:,.2f}")
    print(f"   Episode PnL Std: ${risk_metrics['episode_pnl_std']:,.2f}")
    print(f"   Avg Max Inventory Time: {risk_metrics['max_inventory_time_avg']:.0f} steps")
    print("=" * 60)
    print(f"\n💾 Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
