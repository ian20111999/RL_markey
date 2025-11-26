#!/usr/bin/env python3
"""
V3 完整訓練流程腳本
整合所有進階功能：
- Multi-Algorithm Support (SAC/PPO/TD3)
- Risk-Sensitive Training
- Curriculum Learning
- Realistic Fill Model
- Ensemble Methods
- Explainability
- Online Adaptation
- Backtesting Framework
- Distributed Training
- Report Generation
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

# 設定 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加專案路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 核心導入
from envs.market_making_env_v2 import MarketMakingEnvV2
from utils.algorithms import AlgorithmFactory, get_default_hyperparameters
from utils.risk_sensitive import RiskAwareRewardWrapper, RiskMetricsCallback
from utils.curriculum import CurriculumScheduler, train_with_curriculum
from utils.backtesting import BacktestEngine
from utils.report_generator import ReportGenerator, QuickReportBuilder, ReportConfig

# 可選導入
try:
    from stable_baselines3 import SAC, PPO, TD3
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.evaluation import evaluate_policy
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    logger.warning("stable_baselines3 not found. Training features disabled.")

try:
    from utils.ensemble import create_ensemble
    HAS_ENSEMBLE = True
except ImportError:
    HAS_ENSEMBLE = False

try:
    from utils.explainability import PolicyAnalyzer
    HAS_EXPLAINABILITY = True
except ImportError:
    HAS_EXPLAINABILITY = False

try:
    from utils.online_adaptation import AdaptiveTrainer
    HAS_ADAPTATION = True
except ImportError:
    HAS_ADAPTATION = False

try:
    from utils.distributed_training import (
        DistributedTrainingManager,
        MultiSeedValidator,
        HyperparameterSearch
    )
    HAS_DISTRIBUTED = True
except ImportError:
    HAS_DISTRIBUTED = False


def load_config(config_path: str) -> dict:
    """載入 YAML 配置檔"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_env(
    data: pd.DataFrame,
    config: dict,
    use_realistic_fill: bool = False,
    use_risk_wrapper: bool = False
):
    """
    建立環境
    
    Args:
        data: 價格數據
        config: 配置字典
        use_realistic_fill: 是否使用真實填充模型
        use_risk_wrapper: 是否使用風險感知包裝器
    
    Returns:
        環境實例
    """
    env_config = config.get('env', {})
    
    env = MarketMakingEnvV2(
        data=data,
        initial_cash=env_config.get('initial_cash', 100000),
        fee_rate=env_config.get('fee_rate', 0.0004),
        max_inventory=env_config.get('max_inventory', 10.0),
        episode_length=env_config.get('episode_length', 1000),
        lookback=env_config.get('lookback', 60),
        reward_config=env_config.get('reward_config', {}),
        obs_config=env_config.get('obs_config', {}),
        action_config=env_config.get('action_config', {}),
        domain_randomization=env_config.get('domain_randomization', {}),
        random_start=env_config.get('random_start', True)
    )
    
    if use_risk_wrapper:
        risk_config = config.get('risk_sensitive', {})
        env = RiskAwareRewardWrapper(
            env,
            risk_lambda=risk_config.get('risk_lambda', 0.1),
            risk_type=risk_config.get('risk_type', 'variance'),
            window_size=risk_config.get('window_size', 100)
        )
    
    return env


def run_standard_training(
    config: dict,
    train_data: pd.DataFrame,
    valid_data: pd.DataFrame,
    output_dir: Path,
    algorithm: str = "SAC",
    total_timesteps: int = 100000
):
    """執行標準訓練流程"""
    if not HAS_SB3:
        raise ImportError("stable_baselines3 required for training")
    
    logger.info(f"Starting standard training with {algorithm}...")
    
    # 建立環境
    env = create_env(train_data, config)
    eval_env = create_env(valid_data, config)
    
    vec_env = DummyVecEnv([lambda: env])
    
    # 取得超參數
    train_config = config.get('train', {})
    hyperparams = get_default_hyperparameters(algorithm)
    hyperparams.update({
        'learning_rate': train_config.get('learning_rate', 3e-4),
        'batch_size': train_config.get('batch_size', 256),
        'gamma': train_config.get('gamma', 0.99),
    })
    
    # 建立模型
    model = AlgorithmFactory.create(
        algorithm=algorithm,
        env=vec_env,
        **hyperparams
    )
    
    # 設定回調
    callbacks = []
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=train_config.get('eval_freq', 10000),
        n_eval_episodes=train_config.get('n_eval_episodes', 5),
        deterministic=True,
        render=False
    )
    callbacks.append(eval_callback)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=train_config.get('eval_freq', 10000),
        save_path=str(output_dir / "checkpoints"),
        name_prefix="model"
    )
    callbacks.append(checkpoint_callback)
    
    # 訓練
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    # 保存最終模型
    model.save(str(output_dir / "final_model"))
    
    # 評估
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10
    )
    
    logger.info(f"Training complete. Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    env.close()
    eval_env.close()
    
    return model, {'mean_reward': mean_reward, 'std_reward': std_reward}


def run_curriculum_training(
    config: dict,
    train_data: pd.DataFrame,
    valid_data: pd.DataFrame,
    output_dir: Path,
    algorithm: str = "SAC",
    total_timesteps: int = 100000
):
    """執行課程學習訓練"""
    if not HAS_SB3:
        raise ImportError("stable_baselines3 required for training")
    
    logger.info("Starting curriculum training...")
    
    curriculum_config = config.get('curriculum', {})
    stages = curriculum_config.get('stages', [])
    
    if not stages:
        # 使用預設階段
        stages = [
            {
                'name': 'easy',
                'env_params': {'fee_rate': 0.0002, 'max_inventory': 3.0},
                'advancement_threshold': 50.0,
                'min_episodes': 50
            },
            {
                'name': 'medium',
                'env_params': {'fee_rate': 0.0003, 'max_inventory': 5.0},
                'advancement_threshold': 30.0,
                'min_episodes': 100
            },
            {
                'name': 'hard',
                'env_params': {'fee_rate': 0.0004, 'max_inventory': 10.0},
                'advancement_threshold': 0.0,
                'min_episodes': 0
            }
        ]
    
    # 建立 env_fn 用於課程學習
    def make_env_fn(difficulty_params=None):
        env_config = config.copy()
        if difficulty_params:
            env_config['env'].update(difficulty_params)
        return create_env(train_data, env_config)
    
    # 使用課程學習訓練
    model = train_with_curriculum(
        env_fn=make_env_fn,
        stages=stages,
        algorithm=algorithm,
        total_timesteps=total_timesteps,
        save_path=str(output_dir),
        verbose=1
    )
    
    # 評估
    eval_env = create_env(valid_data, config)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    
    logger.info(f"Curriculum training complete. Mean reward: {mean_reward:.2f}")
    
    return model, {'mean_reward': mean_reward, 'std_reward': std_reward}


def run_distributed_training(
    config: dict,
    train_data: pd.DataFrame,
    valid_data: pd.DataFrame,
    output_dir: Path,
    algorithm: str = "SAC",
    n_hp_trials: int = 20,
    validation_seeds: list = None
):
    """執行分散式訓練（超參數搜尋 + 多種子驗證）"""
    if not HAS_DISTRIBUTED:
        raise ImportError("Distributed training modules not available")
    
    logger.info("Starting distributed training pipeline...")
    
    def make_env():
        return create_env(train_data, config)
    
    manager = DistributedTrainingManager(
        env_fn=make_env,
        algorithm=algorithm,
        output_dir=str(output_dir)
    )
    
    validation_seeds = validation_seeds or [42, 43, 44, 45, 46]
    
    results = manager.run_full_pipeline(
        n_hp_trials=n_hp_trials,
        hp_timesteps=50000,
        validation_seeds=validation_seeds,
        validation_timesteps=100000,
        final_timesteps=config.get('train', {}).get('total_timesteps', 200000)
    )
    
    return results


def run_backtesting(
    model,
    test_data: pd.DataFrame,
    config: dict,
    output_dir: Path
):
    """執行回測分析"""
    logger.info("Running backtesting analysis...")
    
    def make_env():
        return create_env(test_data, config)
    
    engine = BacktestEngine(
        env_fn=make_env,
        policy=model
    )
    
    backtest_config = config.get('backtest', {})
    
    # 基本回測
    backtest_results = engine.run_backtest(
        n_episodes=backtest_config.get('n_episodes', 20)
    )
    
    results = {'basic_backtest': backtest_results}
    
    # Walk-forward 分析
    if backtest_config.get('walk_forward', {}).get('enabled', False):
        wf_config = backtest_config['walk_forward']
        wf_results = engine.run_walk_forward_analysis(
            train_window=wf_config.get('train_window_days', 30),
            test_window=wf_config.get('test_window_days', 7),
            step_size=wf_config.get('step_days', 7)
        )
        results['walk_forward'] = wf_results
    
    # Monte Carlo 模擬
    if backtest_config.get('monte_carlo', {}).get('enabled', False):
        mc_config = backtest_config['monte_carlo']
        mc_results = engine.run_monte_carlo_simulation(
            n_simulations=mc_config.get('n_simulations', 1000),
            n_periods=mc_config.get('n_periods', 252)
        )
        results['monte_carlo'] = mc_results
    
    # 交易成本分析
    if backtest_config.get('robustness', {}).get('enabled', False):
        tc_results = engine.analyze_transaction_costs(
            fee_rates=backtest_config['robustness'].get('test_fee_rates', 
                      [0.0001, 0.0002, 0.0004, 0.0006, 0.001])
        )
        results['transaction_cost_analysis'] = tc_results
    
    # 保存結果
    with open(output_dir / "backtest_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def run_explainability_analysis(
    model,
    test_data: pd.DataFrame,
    config: dict,
    output_dir: Path
):
    """執行可解釋性分析"""
    if not HAS_EXPLAINABILITY:
        logger.warning("Explainability module not available")
        return None
    
    logger.info("Running explainability analysis...")
    
    env = create_env(test_data, config)
    analyzer = PolicyAnalyzer(model, env)
    
    explainability_config = config.get('explainability', {})
    n_samples = explainability_config.get('n_samples', 500)
    
    # 特徵重要性
    importance = analyzer.compute_feature_importance(
        n_samples=n_samples,
        method='permutation'
    )
    
    # 動作分佈分析
    action_analysis = analyzer.analyze_action_distribution(n_samples=n_samples)
    
    # 保存結果
    results = {
        'feature_importance': importance,
        'action_analysis': action_analysis
    }
    
    with open(output_dir / "explainability_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # 生成視覺化
    try:
        plot_path = output_dir / "explainability_plots"
        plot_path.mkdir(exist_ok=True)
        
        analyzer.create_feature_importance_plot(
            importance,
            save_path=str(plot_path / "feature_importance.png")
        )
        
        analyzer.create_action_distribution_plot(
            action_analysis,
            save_path=str(plot_path / "action_distribution.png")
        )
    except Exception as e:
        logger.warning(f"Failed to create plots: {e}")
    
    env.close()
    return results


def generate_report(
    output_dir: Path,
    config: dict,
    training_results: dict,
    backtest_results: dict = None,
    explainability_results: dict = None
):
    """生成完整報告"""
    logger.info("Generating report...")
    
    report = QuickReportBuilder(config.get('title', 'RL Market Making Report'))
    
    # 添加訓練指標
    if training_results:
        report.with_metric("Mean Reward", training_results.get('mean_reward', 0))
        report.with_metric("Std Reward", training_results.get('std_reward', 0))
    
    # 添加回測指標
    if backtest_results and 'basic_backtest' in backtest_results:
        bt = backtest_results['basic_backtest']
        if isinstance(bt, dict):
            for key, value in bt.items():
                if isinstance(value, (int, float)):
                    report.with_metric(key, value)
    
    # 添加配置說明
    report.with_section(
        "Configuration",
        f"<pre>{json.dumps(config, indent=2)}</pre>"
    )
    
    # 添加策略比較（如果有多個策略結果）
    if backtest_results and 'transaction_cost_analysis' in backtest_results:
        tc_data = backtest_results['transaction_cost_analysis']
        if tc_data:
            report.with_comparison(
                {f"{k:.4f}": v.get('mean_reward', 0) for k, v in tc_data.items()},
                title="Fee Rate Sensitivity",
                ylabel="Mean Reward"
            )
    
    # 生成報告
    report_path = str(output_dir / "report.html")
    report.build(report_path)
    
    logger.info(f"Report saved to {report_path}")
    return report_path


def main():
    """主程式"""
    parser = argparse.ArgumentParser(description='V3 Complete Training Pipeline')
    
    parser.add_argument('--config', type=str, default='configs/env_v3_full.yaml',
                        help='Configuration file path')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--algorithm', type=str, default='SAC',
                        choices=['SAC', 'PPO', 'TD3'],
                        help='RL algorithm to use')
    parser.add_argument('--total_timesteps', type=int, default=100000,
                        help='Total training timesteps')
    
    # 訓練模式
    parser.add_argument('--mode', type=str, default='standard',
                        choices=['standard', 'curriculum', 'distributed', 'full'],
                        help='Training mode')
    
    # 功能開關
    parser.add_argument('--use_risk_wrapper', action='store_true',
                        help='Use risk-sensitive reward wrapper')
    parser.add_argument('--run_backtest', action='store_true',
                        help='Run backtesting after training')
    parser.add_argument('--run_explainability', action='store_true',
                        help='Run explainability analysis')
    parser.add_argument('--generate_report', action='store_true',
                        help='Generate HTML report')
    
    # 分散式訓練參數
    parser.add_argument('--n_hp_trials', type=int, default=20,
                        help='Number of hyperparameter search trials')
    parser.add_argument('--validation_seeds', type=str, default='42,43,44',
                        help='Comma-separated validation seeds')
    
    args = parser.parse_args()
    
    # 載入配置
    config = load_config(args.config)
    
    # 設定輸出目錄
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f"runs/v3_{args.mode}_{args.algorithm}_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    logger.info(f"Output directory: {output_dir}")
    
    # 載入數據
    data_config = config.get('data', {})
    data_path = data_config.get('path', 'data/btc_usdt_1m_2023.csv')
    
    logger.info(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    
    # 分割數據
    split_config = config.get('data_split', {})
    n = len(data)
    
    train_end = int(n * split_config.get('train_end', 0.7))
    valid_end = int(n * split_config.get('valid_end', 0.85))
    
    train_data = data.iloc[:train_end].reset_index(drop=True)
    valid_data = data.iloc[train_end:valid_end].reset_index(drop=True)
    test_data = data.iloc[valid_end:].reset_index(drop=True)
    
    logger.info(f"Data split: train={len(train_data)}, valid={len(valid_data)}, test={len(test_data)}")
    
    # 執行訓練
    model = None
    training_results = {}
    
    if args.mode == 'standard':
        model, training_results = run_standard_training(
            config, train_data, valid_data, output_dir,
            algorithm=args.algorithm,
            total_timesteps=args.total_timesteps
        )
    
    elif args.mode == 'curriculum':
        model, training_results = run_curriculum_training(
            config, train_data, valid_data, output_dir,
            algorithm=args.algorithm,
            total_timesteps=args.total_timesteps
        )
    
    elif args.mode == 'distributed':
        validation_seeds = [int(s) for s in args.validation_seeds.split(',')]
        distributed_results = run_distributed_training(
            config, train_data, valid_data, output_dir,
            algorithm=args.algorithm,
            n_hp_trials=args.n_hp_trials,
            validation_seeds=validation_seeds
        )
        training_results = distributed_results.get('final_training', {})
        
        # 載入最終模型
        final_model_path = output_dir / "final_model" / "final_model"
        if final_model_path.exists():
            algo_class = {'SAC': SAC, 'PPO': PPO, 'TD3': TD3}[args.algorithm]
            model = algo_class.load(str(final_model_path))
    
    elif args.mode == 'full':
        # 完整流程：標準訓練 + 所有分析
        model, training_results = run_standard_training(
            config, train_data, valid_data, output_dir,
            algorithm=args.algorithm,
            total_timesteps=args.total_timesteps
        )
        args.run_backtest = True
        args.run_explainability = True
        args.generate_report = True
    
    # 保存訓練結果
    with open(output_dir / "training_results.json", 'w') as f:
        json.dump(training_results, f, indent=2, default=str)
    
    # 回測
    backtest_results = None
    if args.run_backtest and model is not None:
        backtest_results = run_backtesting(model, test_data, config, output_dir)
    
    # 可解釋性分析
    explainability_results = None
    if args.run_explainability and model is not None:
        explainability_results = run_explainability_analysis(
            model, test_data, config, output_dir
        )
    
    # 生成報告
    if args.generate_report:
        generate_report(
            output_dir, config, training_results,
            backtest_results, explainability_results
        )
    
    logger.info("Pipeline complete!")
    logger.info(f"Results saved to: {output_dir}")
    
    return {
        'output_dir': str(output_dir),
        'training_results': training_results,
        'backtest_results': backtest_results,
        'explainability_results': explainability_results
    }


if __name__ == "__main__":
    main()
