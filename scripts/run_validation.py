#!/usr/bin/env python3
"""
完整驗證腳本
包含：
- Walk-Forward Analysis
- Monte Carlo Simulation
- Robustness Testing
- Ensemble Training
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

# 添加專案路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from envs.market_making_env_v2 import (
    MarketMakingEnvV2,
    RewardConfig,
    RewardMode,
    ActionConfig,
    DomainRandomizationConfig,
    FillModelEnvConfig,
    AdvancedObservationConfig,
)
from utils.backtesting import (
    BacktestEngine,
    WalkForwardAnalyzer,
    MonteCarloSimulator,
    RobustnessTester,
    BacktestReportGenerator,
)
from utils.ensemble import EnsembleBuilder, train_diverse_ensemble
from utils.risk_sensitive import RiskMetricsCalculator
from utils.algorithms import create_model, load_model

# 設定 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from stable_baselines3 import SAC, PPO, TD3
    from stable_baselines3.common.evaluation import evaluate_policy
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    logger.warning("stable_baselines3 not found")


def load_config(config_path: str) -> dict:
    """載入配置"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_env_from_config(data: pd.DataFrame, config: dict, seed: int = None):
    """從配置建立環境"""
    env_config = config.get('env', {})
    reward_config = config.get('reward', {})
    action_config = config.get('action', {})
    fill_config = config.get('fill_model', {})
    adv_obs_config = config.get('advanced_observation', {})
    
    reward_cfg = RewardConfig(
        mode=RewardMode(reward_config.get('mode', 'shaped')),
        lambda_inventory=reward_config.get('lambda_inventory', 0.005),
        lambda_turnover=reward_config.get('lambda_turnover', 0.0001),
    )
    
    action_cfg = ActionConfig(
        mode=action_config.get('mode', 'asymmetric'),
        allow_no_quote=action_config.get('allow_no_quote', False),
    )
    
    fill_model_cfg = FillModelEnvConfig(
        enabled=fill_config.get('enabled', False),
        mode=fill_config.get('mode', 'moderate'),
    )
    
    adv_obs_cfg = AdvancedObservationConfig(
        include_order_flow_imbalance=adv_obs_config.get('include_order_flow_imbalance', False),
        include_vwap_deviation=adv_obs_config.get('include_vwap_deviation', False),
        include_multi_timeframe_momentum=adv_obs_config.get('include_multi_timeframe_momentum', False),
    )
    
    return MarketMakingEnvV2(
        df=data,
        initial_cash=env_config.get('initial_cash', 10000),
        fee_rate=env_config.get('fee_rate', 0.0004),
        max_inventory=env_config.get('max_inventory', 5.0),
        episode_length=env_config.get('episode_length', 1440),
        base_spread=env_config.get('base_spread', 25.0),
        random_start=env_config.get('random_start', True),
        seed=seed,
        reward_config=reward_cfg,
        action_config=action_cfg,
        fill_model_config=fill_model_cfg,
        advanced_obs_config=adv_obs_cfg,
    )


def run_walk_forward(
    model_path: str,
    data: pd.DataFrame,
    config: dict,
    output_dir: Path,
    algorithm: str = "SAC",
    train_window_days: int = 30,
    test_window_days: int = 7,
    step_days: int = 7,
    train_timesteps: int = 50000,
):
    """執行 Walk-Forward Analysis"""
    logger.info("=" * 60)
    logger.info("Running Walk-Forward Analysis")
    logger.info("=" * 60)
    
    def env_factory_with_data(df):
        return create_env_from_config(df, config)
    
    def model_factory(env):
        return create_model(
            algo=algorithm.lower(),
            env=env,
            verbose=0
        )
    
    analyzer = WalkForwardAnalyzer(
        data=data,
        env_factory_with_data=env_factory_with_data,
        model_factory=model_factory,
        train_timesteps=train_timesteps,
    )
    
    results = analyzer.run(
        train_window_days=train_window_days,
        test_window_days=test_window_days,
        step_days=step_days,
        verbose=True,
    )
    
    # 保存結果
    with open(output_dir / "walk_forward_results.json", 'w') as f:
        json.dump(results.to_dict(), f, indent=2, default=str)
    
    logger.info(f"\nWalk-Forward Results:")
    logger.info(f"  Windows: {results.aggregate_metrics.get('n_windows', 0)}")
    logger.info(f"  Overall Sharpe: {results.aggregate_metrics.get('overall_sharpe', 0):.4f}")
    logger.info(f"  Positive Windows: {results.aggregate_metrics.get('positive_windows', 0):.2%}")
    logger.info(f"  Stability Score: {results.stability_score:.4f}")
    
    return results


def run_monte_carlo(
    model_path: str,
    data: pd.DataFrame,
    config: dict,
    output_dir: Path,
    algorithm: str = "SAC",
    n_simulations: int = 1000,
    n_periods: int = 252,
):
    """執行 Monte Carlo 模擬"""
    logger.info("=" * 60)
    logger.info("Running Monte Carlo Simulation")
    logger.info("=" * 60)
    
    if not HAS_SB3:
        logger.error("stable_baselines3 required for Monte Carlo")
        return None
    
    # 載入模型
    algo_class = {'SAC': SAC, 'PPO': PPO, 'TD3': TD3}[algorithm]
    model = algo_class.load(model_path)
    
    # 收集 PnL 分佈
    env = create_env_from_config(data, config)
    pnls = []
    
    logger.info("Collecting PnL distribution...")
    for i in range(100):  # 100 episodes
        obs, _ = env.reset()
        done = False
        episode_pnl = 0.0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_pnl += reward
        
        pnls.append(episode_pnl)
        if (i + 1) % 20 == 0:
            logger.info(f"  Collected {i + 1}/100 episodes")
    
    env.close()
    
    # Monte Carlo 模擬
    simulator = MonteCarloSimulator(
        base_pnl_distribution=np.array(pnls),
        n_simulations=n_simulations,
        n_periods=n_periods,
    )
    
    results = simulator.run()
    
    # 保存結果（不包含大型數組）
    results_to_save = {k: v for k, v in results.items() if k != 'paths'}
    with open(output_dir / "monte_carlo_results.json", 'w') as f:
        json.dump(results_to_save, f, indent=2, default=str)
    
    # 保存路徑數據（可選）
    np.save(output_dir / "monte_carlo_paths.npy", results['paths'])
    
    logger.info(f"\nMonte Carlo Results:")
    fp = results['final_pnl']
    logger.info(f"  Mean Final PnL: ${fp['mean']:,.2f}")
    logger.info(f"  5th Percentile: ${fp['percentile_5']:,.2f}")
    logger.info(f"  95th Percentile: ${fp['percentile_95']:,.2f}")
    logger.info(f"  Prob. Positive: {fp['prob_positive']:.2%}")
    
    return results


def run_robustness_test(
    model_path: str,
    data: pd.DataFrame,
    config: dict,
    output_dir: Path,
    algorithm: str = "SAC",
):
    """執行穩健性測試"""
    logger.info("=" * 60)
    logger.info("Running Robustness Tests")
    logger.info("=" * 60)
    
    if not HAS_SB3:
        logger.error("stable_baselines3 required for robustness testing")
        return None
    
    # 載入模型
    algo_class = {'SAC': SAC, 'PPO': PPO, 'TD3': TD3}[algorithm]
    model = algo_class.load(model_path)
    
    def env_factory():
        return create_env_from_config(data, config)
    
    tester = RobustnessTester(model, env_factory)
    
    results = {}
    
    # 手續費敏感度
    logger.info("Testing fee sensitivity...")
    fee_results = tester.test_fee_sensitivity(
        fee_rate_range=[0.0001, 0.0002, 0.0004, 0.0006, 0.001],
        n_episodes=10,
    )
    results['fee_sensitivity'] = fee_results
    
    # 滑點敏感度（如果環境支援）
    logger.info("Testing slippage sensitivity...")
    slippage_results = tester.test_slippage_sensitivity(
        slippage_bps_range=[0, 1, 2, 5, 10],
        n_episodes=10,
    )
    results['slippage_sensitivity'] = slippage_results
    
    # 保存結果
    with open(output_dir / "robustness_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("\nRobustness Test Results:")
    logger.info("  Fee Sensitivity:")
    for fee, pnl in zip(fee_results['param_values'], fee_results['avg_pnls']):
        logger.info(f"    Fee {fee:.4f}: PnL = {pnl:.2f}")
    
    return results


def run_ensemble_training(
    data: pd.DataFrame,
    config: dict,
    output_dir: Path,
    algorithm: str = "SAC",
    n_models: int = 3,
    total_timesteps: int = 100000,
):
    """訓練 Ensemble 模型"""
    logger.info("=" * 60)
    logger.info("Training Ensemble Models")
    logger.info("=" * 60)
    
    def env_factory():
        return create_env_from_config(data, config)
    
    ensemble_dir = output_dir / "ensemble_models"
    
    ensemble = train_diverse_ensemble(
        env_factory=env_factory,
        algo=algorithm.lower(),
        n_models=n_models,
        total_timesteps=total_timesteps,
        diversity_seeds=list(range(n_models)),
        output_dir=str(ensemble_dir),
    )
    
    # 評估 ensemble
    logger.info("\nEvaluating ensemble...")
    env = env_factory()
    
    pnls = []
    for _ in range(20):
        obs, _ = env.reset()
        done = False
        episode_pnl = 0.0
        
        while not done:
            action, _ = ensemble.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_pnl += reward
        
        pnls.append(episode_pnl)
    
    env.close()
    
    results = {
        'n_models': n_models,
        'mean_pnl': float(np.mean(pnls)),
        'std_pnl': float(np.std(pnls)),
        'sharpe': float(np.mean(pnls) / np.std(pnls)) if np.std(pnls) > 0 else 0,
    }
    
    with open(output_dir / "ensemble_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nEnsemble Results:")
    logger.info(f"  Models: {n_models}")
    logger.info(f"  Mean PnL: {results['mean_pnl']:.2f}")
    logger.info(f"  Sharpe: {results['sharpe']:.4f}")
    
    return ensemble, results


def generate_full_report(
    output_dir: Path,
    config: dict,
    wf_results=None,
    mc_results=None,
    robustness_results=None,
    ensemble_results=None,
):
    """生成完整報告"""
    logger.info("=" * 60)
    logger.info("Generating Full Validation Report")
    logger.info("=" * 60)
    
    report_gen = BacktestReportGenerator(str(output_dir))
    
    # 彙總報告
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
    }
    
    if wf_results:
        report['walk_forward'] = wf_results.to_dict() if hasattr(wf_results, 'to_dict') else wf_results
    
    if mc_results:
        report['monte_carlo'] = {k: v for k, v in mc_results.items() if k != 'paths'}
    
    if robustness_results:
        report['robustness'] = robustness_results
    
    if ensemble_results:
        report['ensemble'] = ensemble_results
    
    # 生成 HTML 報告
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>RL Market Making - Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
        .metric {{ margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 5px; }}
        .metric-label {{ font-weight: bold; color: #333; }}
        .metric-value {{ color: #0066cc; font-size: 1.2em; }}
        .section {{ margin: 30px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .good {{ color: green; }}
        .bad {{ color: red; }}
    </style>
</head>
<body>
    <h1>🤖 RL Market Making - Validation Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""
    
    # Walk-Forward Section
    if wf_results:
        wf_data = wf_results.to_dict() if hasattr(wf_results, 'to_dict') else wf_results
        agg = wf_data.get('aggregate_metrics', {})
        html_content += f"""
    <div class="section">
        <h2>📊 Walk-Forward Analysis</h2>
        <div class="metric">
            <span class="metric-label">Windows Tested:</span>
            <span class="metric-value">{agg.get('n_windows', 'N/A')}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Overall Sharpe:</span>
            <span class="metric-value">{agg.get('overall_sharpe', 0):.4f}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Positive Windows:</span>
            <span class="metric-value {'good' if agg.get('positive_windows', 0) > 0.6 else 'bad'}">{agg.get('positive_windows', 0):.2%}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Stability Score:</span>
            <span class="metric-value">{wf_data.get('stability_score', 0):.4f}</span>
        </div>
    </div>
"""
    
    # Monte Carlo Section
    if mc_results:
        fp = mc_results.get('final_pnl', {})
        html_content += f"""
    <div class="section">
        <h2>🎲 Monte Carlo Simulation</h2>
        <div class="metric">
            <span class="metric-label">Mean Final PnL:</span>
            <span class="metric-value {'good' if fp.get('mean', 0) > 0 else 'bad'}">${fp.get('mean', 0):,.2f}</span>
        </div>
        <div class="metric">
            <span class="metric-label">5th Percentile (Worst Case):</span>
            <span class="metric-value">${fp.get('percentile_5', 0):,.2f}</span>
        </div>
        <div class="metric">
            <span class="metric-label">95th Percentile (Best Case):</span>
            <span class="metric-value">${fp.get('percentile_95', 0):,.2f}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Probability of Profit:</span>
            <span class="metric-value {'good' if fp.get('prob_positive', 0) > 0.5 else 'bad'}">{fp.get('prob_positive', 0):.2%}</span>
        </div>
    </div>
"""
    
    # Robustness Section
    if robustness_results:
        html_content += f"""
    <div class="section">
        <h2>🛡️ Robustness Tests</h2>
        <h3>Fee Sensitivity</h3>
        <table>
            <tr><th>Fee Rate</th><th>Mean PnL</th><th>Sharpe</th></tr>
"""
        fee_data = robustness_results.get('fee_sensitivity', {})
        for i, fee in enumerate(fee_data.get('param_values', [])):
            pnl = fee_data.get('avg_pnls', [])[i] if i < len(fee_data.get('avg_pnls', [])) else 0
            sharpe = fee_data.get('sharpes', [])[i] if i < len(fee_data.get('sharpes', [])) else 0
            html_content += f"            <tr><td>{fee:.4f}</td><td>${pnl:,.2f}</td><td>{sharpe:.4f}</td></tr>\n"
        
        html_content += """        </table>
    </div>
"""
    
    # Ensemble Section
    if ensemble_results:
        html_content += f"""
    <div class="section">
        <h2>🎯 Ensemble Model</h2>
        <div class="metric">
            <span class="metric-label">Number of Models:</span>
            <span class="metric-value">{ensemble_results.get('n_models', 0)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Mean PnL:</span>
            <span class="metric-value">${ensemble_results.get('mean_pnl', 0):,.2f}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Sharpe Ratio:</span>
            <span class="metric-value">{ensemble_results.get('sharpe', 0):.4f}</span>
        </div>
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    # 保存 HTML
    report_path = output_dir / "validation_report.html"
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    # 保存 JSON
    with open(output_dir / "validation_summary.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Report saved to {report_path}")
    
    return report_path


def main():
    parser = argparse.ArgumentParser(description='Complete Validation Pipeline')
    
    parser.add_argument('--config', type=str, default='configs/env_v3.yaml')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--algorithm', type=str, default='SAC')
    
    # 驗證選項
    parser.add_argument('--run_walk_forward', action='store_true')
    parser.add_argument('--run_monte_carlo', action='store_true')
    parser.add_argument('--run_robustness', action='store_true')
    parser.add_argument('--run_ensemble', action='store_true')
    parser.add_argument('--run_all', action='store_true')
    
    # Walk-Forward 參數
    parser.add_argument('--wf_train_days', type=int, default=30)
    parser.add_argument('--wf_test_days', type=int, default=7)
    parser.add_argument('--wf_step_days', type=int, default=7)
    parser.add_argument('--wf_train_timesteps', type=int, default=50000)
    
    # Monte Carlo 參數
    parser.add_argument('--mc_simulations', type=int, default=1000)
    parser.add_argument('--mc_periods', type=int, default=252)
    
    # Ensemble 參數
    parser.add_argument('--ensemble_models', type=int, default=3)
    parser.add_argument('--ensemble_timesteps', type=int, default=100000)
    
    args = parser.parse_args()
    
    # 設定輸出目錄
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f"runs/validation_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 載入配置
    config = load_config(args.config)
    
    # 載入數據
    data_path = config.get('data', {}).get('path', 'data/btc_usdt_1m_2023.csv')
    data = pd.read_csv(data_path)
    
    logger.info(f"Loaded {len(data)} rows from {data_path}")
    
    # 結果收集
    wf_results = None
    mc_results = None
    robustness_results = None
    ensemble_results = None
    
    # 執行驗證
    if args.run_all or args.run_walk_forward:
        wf_results = run_walk_forward(
            model_path=args.model_path,
            data=data,
            config=config,
            output_dir=output_dir,
            algorithm=args.algorithm,
            train_window_days=args.wf_train_days,
            test_window_days=args.wf_test_days,
            step_days=args.wf_step_days,
            train_timesteps=args.wf_train_timesteps,
        )
    
    if (args.run_all or args.run_monte_carlo) and args.model_path:
        mc_results = run_monte_carlo(
            model_path=args.model_path,
            data=data,
            config=config,
            output_dir=output_dir,
            algorithm=args.algorithm,
            n_simulations=args.mc_simulations,
            n_periods=args.mc_periods,
        )
    
    if (args.run_all or args.run_robustness) and args.model_path:
        robustness_results = run_robustness_test(
            model_path=args.model_path,
            data=data,
            config=config,
            output_dir=output_dir,
            algorithm=args.algorithm,
        )
    
    if args.run_all or args.run_ensemble:
        _, ensemble_results = run_ensemble_training(
            data=data,
            config=config,
            output_dir=output_dir,
            algorithm=args.algorithm,
            n_models=args.ensemble_models,
            total_timesteps=args.ensemble_timesteps,
        )
    
    # 生成報告
    generate_full_report(
        output_dir=output_dir,
        config=config,
        wf_results=wf_results,
        mc_results=mc_results,
        robustness_results=robustness_results,
        ensemble_results=ensemble_results,
    )
    
    logger.info("=" * 60)
    logger.info(f"Validation complete! Results saved to {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
