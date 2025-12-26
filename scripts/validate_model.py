"""
scripts/validate_model.py
模型驗證腳本

功能:
- Walk-Forward Analysis (滾動窗口驗證)
- Monte Carlo Simulation (尾部風險估計)
- Robustness Testing (參數敏感度)
- 生成完整驗證報告

用法:
    python scripts/validate_model.py --model models/sac_best_model.zip --data data/btc_usdt_1m_2023.csv
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3.common.vec_env import DummyVecEnv

from envs.market_making_env import (
    MarketMakingEnv, RewardConfig, ObservationConfig, ActionConfig, RewardMode
)
from utils.algorithms import load_model
from utils.backtesting import (
    BacktestEngine, WalkForwardAnalyzer, MonteCarloSimulator, 
    RobustnessTester, BacktestReportGenerator
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Environment Factory
# =============================================================================

def create_env_factory(data: pd.DataFrame, config: Dict[str, Any] = None):
    """創建環境工廠"""
    config = config or {}
    
    def make_env():
        reward_cfg = RewardConfig(
            mode=RewardMode.SHAPED,
            reward_scale=config.get("reward_scale", 1e-6),
            lambda_inventory=config.get("lambda_inventory", 20.0),
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
            initial_cash=config.get("initial_cash", 10000),
            fee_rate=config.get("fee_rate", 0.0004),
            max_inventory=config.get("max_inventory", 2.0),
            episode_length=config.get("episode_length", 1440),
            base_spread=config.get("base_spread", 60.0),
            random_start=True,
            reward_config=reward_cfg,
            obs_config=obs_cfg,
            action_config=action_cfg,
        )
    
    return make_env


# =============================================================================
# Model Validator
# =============================================================================

class ModelValidator:
    """模型驗證器"""
    
    def __init__(
        self,
        model_path: str,
        algorithm: str,
        data: pd.DataFrame,
        output_dir: str = "validation_results",
        config: Dict[str, Any] = None,
    ):
        self.model_path = Path(model_path)
        self.algorithm = algorithm.lower()
        self.data = data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        
        # 載入模型
        self.model = load_model(self.algorithm, str(self.model_path))
        logger.info(f"Loaded model from {self.model_path}")
    
    def run_full_validation(
        self,
        run_backtest: bool = True,
        run_walk_forward: bool = True,
        run_monte_carlo: bool = True,
        run_robustness: bool = True,
    ) -> Dict[str, Any]:
        """執行完整驗證"""
        results = {
            "model_path": str(self.model_path),
            "algorithm": self.algorithm,
            "timestamp": datetime.now().isoformat(),
        }
        
        # 1. 基本回測
        if run_backtest:
            logger.info("Running backtesting...")
            results["backtest"] = self._run_backtest()
        
        # 2. Walk-Forward 驗證
        if run_walk_forward:
            logger.info("Running Walk-Forward analysis...")
            results["walk_forward"] = self._run_walk_forward()
        
        # 3. Monte Carlo 模擬
        if run_monte_carlo and "backtest" in results:
            logger.info("Running Monte Carlo simulation...")
            results["monte_carlo"] = self._run_monte_carlo(
                results["backtest"].get("pnl_series", [])
            )
        
        # 4. 穩健性測試
        if run_robustness:
            logger.info("Running robustness testing...")
            results["robustness"] = self._run_robustness()
        
        # 生成報告
        self._generate_report(results)
        
        return results
    
    def _run_backtest(self) -> Dict[str, Any]:
        """執行基本回測"""
        # 使用後 30% 數據作為測試集
        test_start = int(len(self.data) * 0.7)
        test_data = self.data.iloc[test_start:].reset_index(drop=True)
        
        env_factory = create_env_factory(test_data, self.config)
        engine = BacktestEngine(env_factory)
        
        result = engine.run_backtest(self.model, n_episodes=20)
        
        return {
            "total_pnl": result.total_pnl,
            "avg_pnl": result.avg_pnl,
            "std_pnl": result.std_pnl,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "calmar_ratio": result.calmar_ratio,
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "var_95": result.var_95,
            "cvar_95": result.cvar_95,
            "pnl_series": result.pnl_series,
        }
    
    def _run_walk_forward(self) -> Dict[str, Any]:
        """執行 Walk-Forward 驗證"""
        
        def env_factory_with_data(data_slice):
            return create_env_factory(data_slice, self.config)()
        
        def model_factory(env):
            # 這裡我們直接評估現有模型而非重新訓練
            return self.model
        
        # 計算窗口 
        data_len = len(self.data)
        
        # 簡化版：手動執行滾動評估
        n_windows = 5
        window_size = data_len // (n_windows + 1)
        
        window_results = []
        
        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            
            window_data = self.data.iloc[start_idx:end_idx].reset_index(drop=True)
            
            if len(window_data) < 1000:  # 至少需要一些數據
                continue
            
            env_factory = create_env_factory(window_data, self.config)
            engine = BacktestEngine(env_factory)
            
            try:
                result = engine.run_backtest(self.model, n_episodes=5)
                window_results.append({
                    "window": i + 1,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "avg_pnl": result.avg_pnl,
                    "sharpe": result.sharpe_ratio,
                })
            except Exception as e:
                logger.warning(f"Window {i+1} failed: {e}")
        
        # 計算彙總
        if window_results:
            avg_pnls = [w["avg_pnl"] for w in window_results]
            sharpes = [w["sharpe"] for w in window_results]
            
            return {
                "n_windows": len(window_results),
                "windows": window_results,
                "overall_avg_pnl": float(np.mean(avg_pnls)),
                "overall_std_pnl": float(np.std(avg_pnls)),
                "overall_sharpe": float(np.mean(sharpes)),
                "sharpe_consistency": float(np.std(sharpes)),
                "positive_windows_ratio": sum(1 for p in avg_pnls if p > 0) / len(avg_pnls),
            }
        
        return {"error": "No valid windows"}
    
    def _run_monte_carlo(self, pnl_series: list) -> Dict[str, Any]:
        """執行 Monte Carlo 模擬"""
        if not pnl_series or len(pnl_series) < 5:
            return {"error": "Insufficient PnL data"}
        
        pnl_array = np.array(pnl_series)
        simulator = MonteCarloSimulator(
            base_pnl_distribution=pnl_array,
            n_simulations=1000,
            n_periods=252,  # 一年交易日
        )
        
        result = simulator.run()
        
        return {
            "final_pnl_mean": result["final_pnl"]["mean"],
            "final_pnl_median": result["final_pnl"]["median"],
            "final_pnl_p5": result["final_pnl"]["percentile_5"],
            "final_pnl_p95": result["final_pnl"]["percentile_95"],
            "prob_positive": result["final_pnl"]["prob_positive"],
            "max_drawdown_mean": result["max_drawdown"]["mean"],
            "max_drawdown_p95": result["max_drawdown"]["percentile_95"],
        }
    
    def _run_robustness(self) -> Dict[str, Any]:
        """執行穩健性測試"""
        test_start = int(len(self.data) * 0.7)
        test_data = self.data.iloc[test_start:].reset_index(drop=True)
        
        env_factory = create_env_factory(test_data, self.config)
        tester = RobustnessTester(self.model, env_factory)
        
        results = {}
        
        # 測試手續費敏感度
        try:
            fee_sensitivity = tester.test_fee_sensitivity(
                fee_rate_range=[0.0002, 0.0004, 0.0006, 0.0008, 0.001],
                n_episodes=5,
            )
            results["fee_sensitivity"] = {
                "fee_rates": fee_sensitivity["param_values"],
                "sharpes": fee_sensitivity["sharpes"],
                "avg_pnls": fee_sensitivity["avg_pnls"],
            }
        except Exception as e:
            logger.warning(f"Fee sensitivity test failed: {e}")
        
        return results
    
    def _generate_report(self, results: Dict[str, Any]):
        """生成驗證報告"""
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("MODEL VALIDATION REPORT")
        report_lines.append("=" * 70)
        report_lines.append(f"Model: {results.get('model_path', 'N/A')}")
        report_lines.append(f"Algorithm: {results.get('algorithm', 'N/A').upper()}")
        report_lines.append(f"Date: {results.get('timestamp', 'N/A')}")
        report_lines.append("")
        
        # 回測結果
        if "backtest" in results:
            bt = results["backtest"]
            report_lines.append("-" * 70)
            report_lines.append("📊 BACKTEST RESULTS")
            report_lines.append("-" * 70)
            report_lines.append(f"  Total PnL: ${bt.get('total_pnl', 0):,.2f}")
            report_lines.append(f"  Average PnL: ${bt.get('avg_pnl', 0):,.2f}")
            report_lines.append(f"  Sharpe Ratio: {bt.get('sharpe_ratio', 0):.4f}")
            report_lines.append(f"  Max Drawdown: {bt.get('max_drawdown', 0):.2%}")
            report_lines.append(f"  Win Rate: {bt.get('win_rate', 0):.2%}")
            report_lines.append(f"  Profit Factor: {bt.get('profit_factor', 0):.2f}")
            report_lines.append(f"  VaR (95%): ${bt.get('var_95', 0):,.2f}")
            report_lines.append("")
        
        # Walk-Forward 結果
        if "walk_forward" in results and "error" not in results["walk_forward"]:
            wf = results["walk_forward"]
            report_lines.append("-" * 70)
            report_lines.append("🔄 WALK-FORWARD ANALYSIS")
            report_lines.append("-" * 70)
            report_lines.append(f"  Windows Tested: {wf.get('n_windows', 0)}")
            report_lines.append(f"  Overall Sharpe: {wf.get('overall_sharpe', 0):.4f}")
            report_lines.append(f"  Sharpe Consistency (std): {wf.get('sharpe_consistency', 0):.4f}")
            report_lines.append(f"  Positive Windows: {wf.get('positive_windows_ratio', 0):.2%}")
            report_lines.append("")
        
        # Monte Carlo 結果
        if "monte_carlo" in results and "error" not in results["monte_carlo"]:
            mc = results["monte_carlo"]
            report_lines.append("-" * 70)
            report_lines.append("🎲 MONTE CARLO SIMULATION (1-Year)")
            report_lines.append("-" * 70)
            report_lines.append(f"  Expected Final PnL: ${mc.get('final_pnl_mean', 0):,.2f}")
            report_lines.append(f"  5th Percentile: ${mc.get('final_pnl_p5', 0):,.2f}")
            report_lines.append(f"  95th Percentile: ${mc.get('final_pnl_p95', 0):,.2f}")
            report_lines.append(f"  Prob. Positive: {mc.get('prob_positive', 0):.2%}")
            report_lines.append(f"  Expected Max DD: {mc.get('max_drawdown_mean', 0):.2%}")
            report_lines.append("")
        
        # 總結
        report_lines.append("=" * 70)
        
        # 計算總評分
        score = self._calculate_overall_score(results)
        grade = "🟢 PASSED" if score >= 70 else "🟡 CAUTION" if score >= 50 else "🔴 FAILED"
        
        report_lines.append(f"OVERALL SCORE: {score:.0f}/100 {grade}")
        report_lines.append("=" * 70)
        
        # 保存報告
        report_str = "\n".join(report_lines)
        
        report_path = self.output_dir / "validation_report.txt"
        with open(report_path, "w") as f:
            f.write(report_str)
        
        json_path = self.output_dir / "validation_results.json"
        with open(json_path, "w") as f:
            # 移除不可序列化的項目
            clean_results = {k: v for k, v in results.items() if k != "pnl_series"}
            json.dump(clean_results, f, indent=2, default=str)
        
        # 打印報告
        print(report_str)
        
        logger.info(f"Report saved to: {report_path}")
        logger.info(f"Results saved to: {json_path}")
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """計算總評分 (0-100)"""
        score = 0
        
        # 回測績效 (40分)
        if "backtest" in results:
            bt = results["backtest"]
            sharpe = bt.get("sharpe_ratio", 0)
            win_rate = bt.get("win_rate", 0)
            
            # Sharpe > 1 得滿分
            score += min(sharpe * 20, 25)
            # Win rate > 50% 得滿分
            score += min(win_rate * 30, 15)
        
        # Walk-Forward (30分)
        if "walk_forward" in results and "error" not in results["walk_forward"]:
            wf = results["walk_forward"]
            positive_ratio = wf.get("positive_windows_ratio", 0)
            consistency = 1 / (1 + wf.get("sharpe_consistency", 1))
            
            score += positive_ratio * 20
            score += consistency * 10
        
        # Monte Carlo (30分)
        if "monte_carlo" in results and "error" not in results["monte_carlo"]:
            mc = results["monte_carlo"]
            prob_positive = mc.get("prob_positive", 0)
            
            score += prob_positive * 30
        
        return min(score, 100)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Model Validation Script")
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Path to model file (.zip)")
    parser.add_argument("--algorithm", "-a", type=str, default="sac",
                        choices=["sac", "ppo", "td3"],
                        help="Algorithm used to train the model")
    parser.add_argument("--data", "-d", type=str, default="data/btc_usdt_1m_2023.csv",
                        help="Path to data file")
    parser.add_argument("--output_dir", "-o", type=str, default="validation_results",
                        help="Output directory")
    parser.add_argument("--skip_backtest", action="store_true",
                        help="Skip basic backtesting")
    parser.add_argument("--skip_walk_forward", action="store_true",
                        help="Skip Walk-Forward analysis")
    parser.add_argument("--skip_monte_carlo", action="store_true",
                        help="Skip Monte Carlo simulation")
    parser.add_argument("--skip_robustness", action="store_true",
                        help="Skip robustness testing")
    
    args = parser.parse_args()
    
    # 載入數據
    logger.info(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    if "close" not in df.columns and "price" in df.columns:
        df["close"] = df["price"]
    
    logger.info(f"Data loaded: {len(df)} rows")
    
    # 創建驗證器
    validator = ModelValidator(
        model_path=args.model,
        algorithm=args.algorithm,
        data=df,
        output_dir=args.output_dir,
    )
    
    # 執行驗證
    results = validator.run_full_validation(
        run_backtest=not args.skip_backtest,
        run_walk_forward=not args.skip_walk_forward,
        run_monte_carlo=not args.skip_monte_carlo,
        run_robustness=not args.skip_robustness,
    )
    
    logger.info("Validation complete!")
    
    return results


if __name__ == "__main__":
    main()
