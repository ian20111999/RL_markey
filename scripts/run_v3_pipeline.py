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
from envs.market_making_env_v2 import (
    MarketMakingEnvV2,
    RewardConfig,
    RewardMode,
    ObservationConfig,
    ActionConfig,
    DomainRandomizationConfig,
    FillModelEnvConfig,
    AdvancedObservationConfig,
)
from utils.config import load_config as load_yaml_config, load_data, create_env as create_base_env
from utils.algorithms import create_model, get_algo_class, get_default_config, AlgorithmComparator
from utils.risk_sensitive import (
    RiskAwareRewardWrapper, 
    RiskMetricsCalculator,
    CVaRCallback,
    DrawdownEarlyStopping,
    DynamicPositionLimitWrapper,
)
from utils.curriculum import CurriculumScheduler, CurriculumCallback, CurriculumEnvWrapper, create_curriculum_env
from utils.backtesting import BacktestEngine, WalkForwardAnalyzer, MonteCarloSimulator, RobustnessTester
from utils.report_generator import ReportGenerator, QuickReportBuilder, ReportConfig

# 可選導入
try:
    from stable_baselines3 import SAC, PPO, TD3
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, StopTrainingOnNoModelImprovement, CallbackList
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


# =============================================================================
# 🆕 數據增強函數
# =============================================================================

def flip_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    翻轉價格數據，將上漲市場轉換為下跌市場
    
    原理：
    - 計算價格的對數收益率
    - 反轉收益率（乘以 -1）
    - 重建價格序列
    
    這樣可以從牛市數據創造熊市數據，增加訓練多樣性
    
    Args:
        df: 原始 DataFrame (需包含 open, high, low, close, volume)
    
    Returns:
        翻轉後的 DataFrame
    """
    df = df.copy()
    
    # 使用第一個價格作為基準
    base_price = df['close'].iloc[0]
    
    # 計算對數收益率
    log_returns = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    
    # 反轉收益率
    flipped_returns = -log_returns
    
    # 重建價格 (從最後一個價格開始，確保不會出現負價格)
    flipped_close = np.zeros(len(df))
    flipped_close[0] = base_price
    for i in range(1, len(df)):
        flipped_close[i] = flipped_close[i-1] * np.exp(flipped_returns.iloc[i])
    
    # 計算 OHLC 的比例關係，保持相對結構
    close_ratio = flipped_close / df['close'].values
    
    df['open'] = df['open'] * close_ratio
    df['high'] = df['close'] * close_ratio + (df['high'] - df['close']).abs() * close_ratio  # 高點變低點概念
    df['low'] = df['close'] * close_ratio - (df['close'] - df['low']).abs() * close_ratio   # 低點變高點概念
    df['close'] = flipped_close
    
    # 交換 high 和 low 如果順序不對
    high_low_swap = df['high'] < df['low']
    df.loc[high_low_swap, ['high', 'low']] = df.loc[high_low_swap, ['low', 'high']].values
    
    # 確保價格為正
    min_price = df[['open', 'high', 'low', 'close']].min().min()
    if min_price <= 0:
        adjustment = abs(min_price) + 1
        df[['open', 'high', 'low', 'close']] += adjustment
    
    # Volume 保持不變（或可以輕微調整）
    # df['volume'] = df['volume']  # 保持原樣
    
    return df


def load_config(config_path: str) -> dict:
    """載入 YAML 配置檔（包裝 utils.config）"""
    cfg = load_yaml_config(config_path)
    return cfg.raw  # 返回原始 dict 以便向後兼容


def create_env(
    data: pd.DataFrame,
    config: dict,
    use_realistic_fill: bool = False,
    use_risk_wrapper: bool = False,
    use_dynamic_position_limit: bool = False,
    seed: int = None
):
    """
    建立環境
    
    Args:
        data: 價格數據
        config: 配置字典
        use_realistic_fill: 是否使用真實填充模型
        use_risk_wrapper: 是否使用風險感知包裝器
        use_dynamic_position_limit: 是否使用動態倉位限制
    
    Returns:
        環境實例
    """
    env_config = config.get('env', {})
    reward_config = config.get('reward', {})
    action_config = config.get('action', {})
    obs_config = config.get('observation', {})
    dr_config = config.get('domain_randomization', {})
    fill_config = config.get('fill_model', {})
    adv_obs_config = config.get('advanced_observation', {})
    
    # 建構 Reward Config
    reward_mode_str = reward_config.get('mode', 'shaped')
    reward_cfg = RewardConfig(
        mode=RewardMode(reward_mode_str),
        lambda_inventory=reward_config.get('lambda_inventory', 0.005),
        lambda_turnover=reward_config.get('lambda_turnover', 0.0001),
        gamma=reward_config.get('gamma', 0.99),
        sparse_scale=reward_config.get('sparse_scale', 0.01),
        terminal_bonus_weight=reward_config.get('terminal_bonus_weight', 0.3),
        # 🆕 做市獎勵參數
        spread_capture_bonus=reward_config.get('spread_capture_bonus', 0.0),
        round_trip_bonus=reward_config.get('round_trip_bonus', 0.0),
        inventory_revert_bonus=reward_config.get('inventory_revert_bonus', 0.0),
        asymmetric_penalty=reward_config.get('asymmetric_penalty', 0.0),
        # 🆕 v3: 獎勵縮放（穩定訓練）
        reward_scale=reward_config.get('reward_scale', 1.0),
    )
    
    # 建構 Action Config
    action_cfg = ActionConfig(
        mode=action_config.get('mode', 'asymmetric'),
        allow_no_quote=action_config.get('allow_no_quote', False),
        max_spread_multiplier=action_config.get('max_spread_multiplier', 3.0),
        min_spread_multiplier=action_config.get('min_spread_multiplier', 0.3),
    )
    
    # 建構 Domain Randomization Config
    dr_cfg = DomainRandomizationConfig(
        enabled=dr_config.get('enabled', False),
        fee_rate_range=tuple(dr_config.get('fee_rate_range', [0.0003, 0.0005])),
        base_spread_range=tuple(dr_config.get('base_spread_range', [20.0, 30.0])),
        volatility_multiplier_range=tuple(dr_config.get('volatility_multiplier_range', [0.8, 1.2])),
        fill_probability_noise=dr_config.get('fill_probability_noise', 0.05),
    )
    
    # 建構 Fill Model Config
    fill_model_cfg = FillModelEnvConfig(
        enabled=fill_config.get('enabled', False) or use_realistic_fill,
        mode=fill_config.get('mode', 'moderate'),
        enable_queue_position=fill_config.get('enable_queue_position', True),
        enable_slippage=fill_config.get('enable_slippage', True),
        slippage_bps=fill_config.get('slippage_bps', 1.0),
        enable_market_impact=fill_config.get('enable_market_impact', False),
        enable_adverse_selection=fill_config.get('enable_adverse_selection', True),
        adverse_selection_prob=fill_config.get('adverse_selection_prob', 0.1),
    )
    
    # 建構 Advanced Observation Config
    adv_obs_cfg = AdvancedObservationConfig(
        include_order_flow_imbalance=adv_obs_config.get('include_order_flow_imbalance', False),
        order_flow_window=adv_obs_config.get('order_flow_window', 20),
        include_vwap_deviation=adv_obs_config.get('include_vwap_deviation', False),
        vwap_window=adv_obs_config.get('vwap_window', 60),
        include_multi_timeframe_momentum=adv_obs_config.get('include_multi_timeframe_momentum', False),
        mtf_windows=adv_obs_config.get('mtf_windows', [15, 60, 240]),
        include_volatility_forecast=adv_obs_config.get('include_volatility_forecast', False),
        ewma_span=adv_obs_config.get('ewma_span', 20),
        include_microstructure=adv_obs_config.get('include_microstructure', False),
    )
    
    # 🆕 建構 Observation Config (包含趨勢特徵)
    obs_cfg = ObservationConfig(
        include_price=obs_config.get('include_price', True),
        include_inventory=obs_config.get('include_inventory', True),
        include_time=obs_config.get('include_time', True),
        include_volatility=obs_config.get('include_volatility', True),
        include_momentum=obs_config.get('include_momentum', True),
        include_volume=obs_config.get('include_volume', True),
        include_inventory_age=obs_config.get('include_inventory_age', True),
        include_trend=obs_config.get('include_trend', False),  # 🆕 趨勢特徵
        volatility_windows=obs_config.get('volatility_windows', [5, 15, 60]),
        momentum_windows=obs_config.get('momentum_windows', [5, 15]),
        trend_windows=obs_config.get('trend_windows', [60, 240, 1440]),  # 🆕 趨勢窗口
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
        obs_config=obs_cfg,  # 🆕 加入 ObservationConfig
        action_config=action_cfg,
        domain_rand_config=dr_cfg,
        fill_model_config=fill_model_cfg,
        advanced_obs_config=adv_obs_cfg,
    )
    
    # 動態倉位限制
    if use_dynamic_position_limit:
        dpl_config = config.get('dynamic_position_limit', {})
        env = DynamicPositionLimitWrapper(
            env,
            base_max_inventory=dpl_config.get('base_max_inventory', env_config.get('max_inventory', 5.0)),
            volatility_threshold=dpl_config.get('volatility_threshold', 0.02),
            min_inventory_ratio=dpl_config.get('min_inventory_ratio', 0.3),
            volatility_window=dpl_config.get('volatility_window', 20),
        )
    
    # 風險感知包裝器
    if use_risk_wrapper:
        risk_config = config.get('risk_sensitive', {})
        env = RiskAwareRewardWrapper(
            env,
            risk_lambda=risk_config.get('risk_lambda', 0.1),
            risk_type=risk_config.get('risk_type', 'variance'),
            window_size=risk_config.get('window_size', 100),
            cvar_alpha=risk_config.get('cvar_alpha', 0.05),
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
    
    # 讀取配置
    train_config = config.get('train', {})
    risk_config = config.get('risk_sensitive', {})
    fill_config = config.get('fill_model', {})
    
    # 決定是否使用進階功能
    use_risk_wrapper = risk_config.get('enabled', False)
    use_realistic_fill = fill_config.get('enabled', False)
    use_dynamic_position_limit = config.get('dynamic_position_limit', {}).get('enabled', False)
    
    # 建立環境
    env = create_env(
        train_data, config, 
        use_realistic_fill=use_realistic_fill,
        use_risk_wrapper=use_risk_wrapper,
        use_dynamic_position_limit=use_dynamic_position_limit
    )
    eval_env = create_env(
        valid_data, config, 
        use_realistic_fill=use_realistic_fill,
        use_risk_wrapper=False,  # 評估時不用風險包裝器
        use_dynamic_position_limit=use_dynamic_position_limit
    )
    
    vec_env = DummyVecEnv([lambda: env])
    
    # 取得超參數
    hyperparams = get_default_config(algorithm)
    hyperparams.update({
        'learning_rate': train_config.get('learning_rate', 3e-4),
        'batch_size': train_config.get('batch_size', 256),
        'gamma': train_config.get('gamma', 0.99),
    })
    
    # 🔧 支援更多 SAC 超參數
    if algorithm.upper() == 'SAC':
        if 'ent_coef' in train_config:
            hyperparams['ent_coef'] = train_config['ent_coef']
        if 'learning_starts' in train_config:
            hyperparams['learning_starts'] = train_config['learning_starts']
        if 'tau' in train_config:
            hyperparams['tau'] = train_config['tau']
        if 'target_entropy' in train_config:
            hyperparams['target_entropy'] = train_config['target_entropy']
        if 'buffer_size' in train_config:
            hyperparams['buffer_size'] = train_config['buffer_size']
    
    # 建立模型
    model = create_model(
        algo=algorithm.lower(),
        env=vec_env,
        config_overrides=hyperparams,
        verbose=1
    )
    
    # 設定回調
    callbacks = []
    
    # CVaR 監控回調
    if risk_config.get('cvar_monitoring', False):
        cvar_callback = CVaRCallback(
            alpha=risk_config.get('cvar_alpha', 0.05),
            cvar_threshold=risk_config.get('cvar_threshold', -500.0),
            window_size=risk_config.get('cvar_window', 1000),
            check_freq=train_config.get('eval_freq', 10000),
            verbose=1,
        )
        callbacks.append(cvar_callback)
        logger.info("CVaR monitoring enabled")
    
    # Drawdown 提前停止
    if risk_config.get('drawdown_early_stopping', False):
        dd_callback = DrawdownEarlyStopping(
            max_drawdown_threshold=risk_config.get('max_drawdown_threshold', 0.2),
            check_freq=train_config.get('eval_freq', 10000),
            eval_env=eval_env,
            n_eval_episodes=5,
            verbose=1,
        )
        callbacks.append(dd_callback)
        logger.info("Drawdown early stopping enabled")
    
    # 早停設定
    early_stopping_config = train_config.get('early_stopping', {})
    stop_callback = None
    if early_stopping_config.get('enabled', False):
        stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=early_stopping_config.get('patience', 15),
            min_evals=5,  # 至少評估 5 次才開始檢查早停
            verbose=1
        )
        logger.info(f"Early stopping enabled with patience={early_stopping_config.get('patience', 15)}")
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=train_config.get('eval_freq', 10000),
        n_eval_episodes=train_config.get('n_eval_episodes', 5),
        deterministic=True,
        render=False,
        callback_after_eval=stop_callback  # 在評估後檢查是否早停
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
    # 建立基礎環境
    base_env = make_env_fn()
    vec_env = DummyVecEnv([lambda: base_env])
    
    # 建立模型
    model = create_model(
        algo=algorithm.lower(),
        env=vec_env,
        verbose=1
    )
    
    # 建立課程回調
    curriculum_callback = CurriculumCallback(
        curriculum_config=curriculum_config,
        verbose=1
    )
    
    # 訓練
    model.learn(
        total_timesteps=total_timesteps,
        callback=curriculum_callback,
        progress_bar=True
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
        return create_env(data=test_data, config=config)
    
    engine = BacktestEngine(
        env_factory=make_env,
        initial_capital=config.get('env', {}).get('initial_capital', 100_000),
        transaction_cost_bps=config.get('env', {}).get('fee_rate', 0.0004) * 10000,
    )
    
    backtest_config = config.get('backtest', {})
    
    # 基本回測
    backtest_results = engine.run_backtest(
        model=model,
        n_episodes=backtest_config.get('n_episodes', 20)
    )
    
    results = {'basic_backtest': backtest_results}
    
    # Walk-forward 分析（需要 WalkForwardAnalyzer 類別）
    # if backtest_config.get('walk_forward', {}).get('enabled', False):
    #     wf_config = backtest_config['walk_forward']
    #     # 需要使用 WalkForwardAnalyzer，暫時跳過
    #     pass
    
    # Monte Carlo 模擬（需要 MonteCarloSimulator 類別）
    # if backtest_config.get('monte_carlo', {}).get('enabled', False):
    #     mc_config = backtest_config['monte_carlo']
    #     # 需要使用 MonteCarloSimulator，暫時跳過
    #     pass
    
    # 交易成本分析（需要 RobustnessTester 類別）
    # if backtest_config.get('robustness', {}).get('enabled', False):
    #     # 需要使用 RobustnessTester，暫時跳過
    #     pass
    
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
    
    env = create_env(df=test_data, config=config)
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
    
    # 分割數據（支援日期格式和比例格式）
    split_config = config.get('data_split', {})
    n = len(data)
    
    train_end_cfg = split_config.get('train_end', 0.7)
    valid_end_cfg = split_config.get('valid_end', 0.85)
    
    # 檢查是否使用日期格式
    if isinstance(train_end_cfg, str):
        # 使用日期格式分割
        if data["timestamp"].dtype in ["int64", "float64"]:
            data["_datetime"] = pd.to_datetime(data["timestamp"], unit="ms")
        else:
            data["_datetime"] = pd.to_datetime(data["timestamp"])
        
        train_start = split_config.get("train_start", "2023-01-01")
        train_end = split_config.get("train_end", "2023-06-30")
        valid_start = split_config.get("valid_start", "2023-07-01")
        valid_end = split_config.get("valid_end", "2023-08-31")
        test_start = split_config.get("test_start", "2023-09-01")
        test_end = split_config.get("test_end", "2023-12-31")
        
        train_mask = (data["_datetime"] >= train_start) & (data["_datetime"] <= train_end)
        valid_mask = (data["_datetime"] >= valid_start) & (data["_datetime"] <= valid_end)
        test_mask = (data["_datetime"] >= test_start) & (data["_datetime"] <= test_end)
        
        train_data = data[train_mask].drop(columns=["_datetime"]).reset_index(drop=True)
        valid_data = data[valid_mask].drop(columns=["_datetime"]).reset_index(drop=True)
        test_data = data[test_mask].drop(columns=["_datetime"]).reset_index(drop=True)
    else:
        # 使用比例格式分割
        train_end = int(n * train_end_cfg)
        valid_end = int(n * valid_end_cfg)
        
        train_data = data.iloc[:train_end].reset_index(drop=True)
        valid_data = data.iloc[train_end:valid_end].reset_index(drop=True)
        test_data = data.iloc[valid_end:].reset_index(drop=True)
    
    logger.info(f"Data split: train={len(train_data)}, valid={len(valid_data)}, test={len(test_data)}")
    
    # 🆕 數據增強：翻轉數據模擬下跌市場
    augment_config = config.get('data_augmentation', {})
    if augment_config.get('enable_price_flip', False):
        logger.info("Applying price flip augmentation...")
        flipped_train = flip_price_data(train_data.copy())
        train_data = pd.concat([train_data, flipped_train], ignore_index=True)
        logger.info(f"Augmented train data size: {len(train_data)}")
    
    # 執行訓練
    model = None
    training_results = {}
    
    # 從配置文件讀取 total_timesteps，如果沒有則用命令行參數
    config_timesteps = config.get('train', {}).get('total_timesteps', args.total_timesteps)
    
    if args.mode == 'standard':
        model, training_results = run_standard_training(
            config, train_data, valid_data, output_dir,
            algorithm=args.algorithm,
            total_timesteps=config_timesteps
        )
    
    elif args.mode == 'curriculum':
        model, training_results = run_curriculum_training(
            config, train_data, valid_data, output_dir,
            algorithm=args.algorithm,
            total_timesteps=config_timesteps
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
            total_timesteps=config_timesteps
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
