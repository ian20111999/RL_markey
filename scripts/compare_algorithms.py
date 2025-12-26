"""
scripts/compare_algorithms.py
多演算法比較實驗腳本

功能:
- 自動比較 SAC, PPO, TD3 三種演算法
- 多次隨機種子訓練
- 生成對比報告和視覺化

用法:
    python scripts/compare_algorithms.py --data data/btc_usdt_1m_2023.csv
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3.common.vec_env import DummyVecEnv

from envs.market_making_env import (
    MarketMakingEnv, RewardConfig, ObservationConfig, ActionConfig, RewardMode
)
from utils.algorithms import AlgorithmComparator, create_model

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Environment Setup
# =============================================================================

def create_env_factory(data: pd.DataFrame, env_config: Dict[str, Any]):
    """創建環境工廠"""
    
    def make_env():
        reward_cfg = RewardConfig(
            mode=RewardMode.SHAPED,
            reward_scale=env_config.get("reward_scale", 1e-6),
            lambda_inventory=env_config.get("lambda_inventory", 20.0),
        )
        
        obs_cfg = ObservationConfig(
            include_price=True,
            include_inventory=True,
            include_time=True,
            include_volatility=True,
            include_momentum=True,
            include_volume=True,
            include_inventory_age=True,
        )
        
        action_cfg = ActionConfig(
            mode="asymmetric",
            allow_no_quote=True,
        )
        
        return MarketMakingEnv(
            df=data,
            initial_cash=env_config.get("initial_cash", 10000),
            fee_rate=env_config.get("fee_rate", 0.0004),
            max_inventory=env_config.get("max_inventory", 2.0),
            episode_length=env_config.get("episode_length", 1440),
            base_spread=env_config.get("base_spread", 60.0),
            random_start=True,
            reward_config=reward_cfg,
            obs_config=obs_cfg,
            action_config=action_cfg,
        )
    
    return make_env


# =============================================================================
# Comparison Runner
# =============================================================================

class AlgorithmComparisonRunner:
    """演算法比較執行器"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        algorithms: List[str] = ["sac", "ppo", "td3"],
        n_seeds: int = 3,
        total_timesteps: int = 100000,
        n_eval_episodes: int = 10,
        output_dir: str = "algo_comparison",
        env_config: Dict[str, Any] = None,
    ):
        self.data = data
        self.algorithms = algorithms
        self.n_seeds = n_seeds
        self.total_timesteps = total_timesteps
        self.n_eval_episodes = n_eval_episodes
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.env_config = env_config or {}
        
        # 分割數據
        train_end = int(len(data) * 0.7)
        self.train_data = data.iloc[:train_end].reset_index(drop=True)
        self.eval_data = data.iloc[train_end:].reset_index(drop=True)
    
    def run(self) -> Dict[str, Any]:
        """執行比較實驗"""
        env_factory = create_env_factory(self.train_data, self.env_config)
        eval_env_factory = create_env_factory(self.eval_data, self.env_config)
        
        comparator = AlgorithmComparator(
            env_factory=env_factory,
            algos=self.algorithms,
            n_seeds=self.n_seeds,
            total_timesteps=self.total_timesteps,
        )
        
        results = comparator.run_comparison(
            eval_env_factory=eval_env_factory,
            n_eval_episodes=self.n_eval_episodes,
            output_dir=str(self.output_dir),
        )
        
        # 生成報告
        self._generate_report(results)
        self._generate_plots(results)
        
        return results
    
    def _generate_report(self, results: Dict[str, Any]):
        """生成文字報告"""
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("ALGORITHM COMPARISON REPORT")
        report_lines.append("=" * 70)
        report_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Algorithms: {', '.join(self.algorithms)}")
        report_lines.append(f"Seeds per algorithm: {self.n_seeds}")
        report_lines.append(f"Training timesteps: {self.total_timesteps:,}")
        report_lines.append(f"Evaluation episodes: {self.n_eval_episodes}")
        report_lines.append("")
        
        # 排名表
        report_lines.append("-" * 70)
        report_lines.append(f"{'Algorithm':<10} {'Mean Reward':>15} {'Std':>12} {'Best':>12} {'Worst':>12}")
        report_lines.append("-" * 70)
        
        sorted_results = sorted(
            results.items(), 
            key=lambda x: x[1].get('mean', float('-inf')), 
            reverse=True
        )
        
        for algo, res in sorted_results:
            mean_val = res.get('mean', 0)
            std_val = res.get('std', 0)
            max_val = res.get('max', 0)
            min_val = res.get('min', 0)
            report_lines.append(
                f"{algo.upper():<10} {mean_val:>15.2f} {std_val:>12.2f} {max_val:>12.2f} {min_val:>12.2f}"
            )
        
        report_lines.append("-" * 70)
        
        # 推薦
        best_algo = sorted_results[0][0] if sorted_results else "N/A"
        report_lines.append("")
        report_lines.append(f"🏆 RECOMMENDED ALGORITHM: {best_algo.upper()}")
        report_lines.append("")
        report_lines.append("=" * 70)
        
        # 保存報告
        report_path = self.output_dir / "comparison_report.txt"
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))
        
        # 也打印到控制台
        print("\n".join(report_lines))
        
        # 保存 JSON
        json_path = self.output_dir / "comparison_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Report saved to: {report_path}")
        logger.info(f"Results saved to: {json_path}")
    
    def _generate_plots(self, results: Dict[str, Any]):
        """生成比較圖表"""
        try:
            # 準備數據
            algorithms = list(results.keys())
            means = [results[algo].get('mean', 0) for algo in algorithms]
            stds = [results[algo].get('std', 0) for algo in algorithms]
            
            # 條形圖
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = np.arange(len(algorithms))
            bars = ax.bar(x, means, yerr=stds, capsize=5, color=['#2ecc71', '#3498db', '#e74c3c'][:len(algorithms)])
            
            ax.set_xlabel('Algorithm', fontsize=12)
            ax.set_ylabel('Mean Reward', fontsize=12)
            ax.set_title('Algorithm Comparison: Mean Reward ± Std', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels([a.upper() for a in algorithms])
            
            # 添加數值標籤
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax.annotate(f'{mean:.1f}±{std:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            
            # 保存圖表
            plot_path = self.output_dir / "comparison_chart.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            
            logger.info(f"Chart saved to: {plot_path}")
            
        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")


# =============================================================================
# Quick Compare Function
# =============================================================================

def quick_compare(
    data_path: str = "data/btc_usdt_1m_2023.csv",
    algorithms: List[str] = ["sac", "ppo", "td3"],
    n_seeds: int = 2,
    timesteps: int = 50000,
    output_dir: str = "algo_comparison",
) -> Dict[str, Any]:
    """快速比較函數"""
    
    # 載入數據
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    if "close" not in df.columns and "price" in df.columns:
        df["close"] = df["price"]
    
    # 環境配置
    env_config = {
        "initial_cash": 10000,
        "fee_rate": 0.0004,
        "max_inventory": 2.0,
        "episode_length": 1440,
        "base_spread": 60.0,
    }
    
    # 執行比較
    runner = AlgorithmComparisonRunner(
        data=df,
        algorithms=algorithms,
        n_seeds=n_seeds,
        total_timesteps=timesteps,
        n_eval_episodes=10,
        output_dir=output_dir,
        env_config=env_config,
    )
    
    return runner.run()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare RL Algorithms")
    parser.add_argument("--data", type=str, default="data/btc_usdt_1m_2023.csv",
                        help="Path to training data CSV")
    parser.add_argument("--algorithms", type=str, nargs="+", default=["sac", "ppo", "td3"],
                        help="Algorithms to compare")
    parser.add_argument("--n_seeds", type=int, default=3,
                        help="Number of random seeds per algorithm")
    parser.add_argument("--timesteps", type=int, default=100000,
                        help="Training timesteps per run")
    parser.add_argument("--n_eval_episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--output_dir", type=str, default="algo_comparison",
                        help="Output directory")
    
    args = parser.parse_args()
    
    # 載入數據
    logger.info(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    if "close" not in df.columns and "price" in df.columns:
        df["close"] = df["price"]
    
    logger.info(f"Data loaded: {len(df)} rows")
    
    # 環境配置
    env_config = {
        "initial_cash": 10000,
        "fee_rate": 0.0004,
        "max_inventory": 2.0,
        "episode_length": 1440,
        "base_spread": 60.0,
    }
    
    # 執行比較
    runner = AlgorithmComparisonRunner(
        data=df,
        algorithms=args.algorithms,
        n_seeds=args.n_seeds,
        total_timesteps=args.timesteps,
        n_eval_episodes=args.n_eval_episodes,
        output_dir=args.output_dir,
        env_config=env_config,
    )
    
    results = runner.run()
    
    return results


if __name__ == "__main__":
    main()
