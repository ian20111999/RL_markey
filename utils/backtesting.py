"""
utils/backtesting.py
完整回測框架

實作專業級回測功能:
- Walk-Forward Analysis: 滾動窗口驗證
- Monte Carlo Simulation: 蒙地卡羅模擬
- Transaction Cost Analysis: 交易成本分析
- Robustness Testing: 穩健性測試

用法:
    from utils.backtesting import BacktestEngine, WalkForwardAnalyzer
    
    engine = BacktestEngine(env_factory)
    results = engine.run_backtest(model, start_date, end_date)
    
    analyzer = WalkForwardAnalyzer(engine)
    wf_results = analyzer.run(train_window=30, test_window=7)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd
from stable_baselines3.common.base_class import BaseAlgorithm


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BacktestResult:
    """回測結果"""
    # 基本資訊
    start_date: str
    end_date: str
    n_episodes: int
    
    # 績效指標
    total_pnl: float
    avg_pnl: float
    std_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # 交易統計
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # 風險指標
    var_95: float
    cvar_95: float
    
    # 詳細數據
    pnl_series: List[float] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    drawdown_series: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "n_episodes": self.n_episodes,
            "total_pnl": self.total_pnl,
            "avg_pnl": self.avg_pnl,
            "std_pnl": self.std_pnl,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self.calmar_ratio,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
        }


@dataclass
class WalkForwardResult:
    """Walk-Forward 分析結果"""
    windows: List[Dict[str, Any]]
    aggregate_metrics: Dict[str, float]
    stability_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "windows": self.windows,
            "aggregate_metrics": self.aggregate_metrics,
            "stability_score": self.stability_score,
        }


# =============================================================================
# Backtest Engine
# =============================================================================

class BacktestEngine:
    """回測引擎"""
    
    def __init__(
        self,
        env_factory: Callable,
        initial_capital: float = 100_000,
        transaction_cost_bps: float = 4.0,  # 交易成本 (bps)
    ):
        """
        Args:
            env_factory: 建立環境的函數
            initial_capital: 初始資金
            transaction_cost_bps: 交易成本（基點）
        """
        self.env_factory = env_factory
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps
    
    def run_backtest(
        self,
        model: BaseAlgorithm,
        n_episodes: int = 10,
        deterministic: bool = True,
    ) -> BacktestResult:
        """
        執行回測
        
        Args:
            model: RL 模型
            n_episodes: 回測 episode 數
            deterministic: 是否使用確定性策略
        
        Returns:
            回測結果
        """
        env = self.env_factory()
        
        all_pnls = []
        all_trades = []
        all_equity_curves = []
        
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            
            episode_pnl = 0.0
            episode_trades = []
            episode_equity = [self.initial_capital]
            
            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # 記錄交易
                if info.get("trades_count", 0) > 0:
                    episode_trades.append({
                        "step": info.get("step", 0),
                        "pnl": reward,
                        "inventory": info.get("inventory", 0),
                    })
                
                episode_pnl += reward
                episode_equity.append(episode_equity[-1] + reward)
            
            all_pnls.append(episode_pnl)
            all_trades.extend(episode_trades)
            all_equity_curves.append(episode_equity)
        
        env.close()
        
        # 計算指標
        return self._compute_metrics(all_pnls, all_trades, all_equity_curves, n_episodes)
    
    def _compute_metrics(
        self,
        pnls: List[float],
        trades: List[Dict],
        equity_curves: List[List[float]],
        n_episodes: int,
    ) -> BacktestResult:
        """計算回測指標"""
        pnls = np.array(pnls)
        
        # 基本統計
        total_pnl = float(np.sum(pnls))
        avg_pnl = float(np.mean(pnls))
        std_pnl = float(np.std(pnls)) if len(pnls) > 1 else 0.0
        
        # Sharpe Ratio
        sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0.0
        
        # 合併 equity curve
        max_len = max(len(eq) for eq in equity_curves)
        padded_curves = []
        for eq in equity_curves:
            if len(eq) < max_len:
                eq = eq + [eq[-1]] * (max_len - len(eq))
            padded_curves.append(eq)
        
        avg_equity = np.mean(padded_curves, axis=0)
        
        # Max Drawdown
        peak = np.maximum.accumulate(avg_equity)
        drawdown = (peak - avg_equity) / peak
        max_drawdown = float(np.max(drawdown))
        
        # Calmar Ratio
        total_return = (avg_equity[-1] - avg_equity[0]) / avg_equity[0]
        calmar = total_return / max_drawdown if max_drawdown > 0 else 0.0
        
        # 交易統計
        trade_pnls = [t["pnl"] for t in trades] if trades else [0]
        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p < 0]
        
        win_rate = len(wins) / len(trade_pnls) if trade_pnls else 0.0
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
        
        # VaR 和 CVaR
        var_95 = float(np.percentile(pnls, 5))
        cvar_95 = float(np.mean(pnls[pnls <= var_95])) if len(pnls[pnls <= var_95]) > 0 else var_95
        
        return BacktestResult(
            start_date="",
            end_date="",
            n_episodes=n_episodes,
            total_pnl=total_pnl,
            avg_pnl=avg_pnl,
            std_pnl=std_pnl,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar,
            total_trades=len(trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            var_95=var_95,
            cvar_95=cvar_95,
            pnl_series=pnls.tolist(),
            equity_curve=avg_equity.tolist(),
            drawdown_series=drawdown.tolist(),
        )


# =============================================================================
# Walk-Forward Analyzer
# =============================================================================

class WalkForwardAnalyzer:
    """
    Walk-Forward 分析器
    
    滾動窗口驗證策略的穩定性
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        env_factory_with_data: Callable[[pd.DataFrame], Any],
        model_factory: Callable[[Any], BaseAlgorithm],
        train_timesteps: int = 50_000,
    ):
        """
        Args:
            data: 完整數據
            env_factory_with_data: 使用指定數據建立環境的函數
            model_factory: 建立模型的函數
            train_timesteps: 每個窗口的訓練步數
        """
        self.data = data
        self.env_factory_with_data = env_factory_with_data
        self.model_factory = model_factory
        self.train_timesteps = train_timesteps
    
    def run(
        self,
        train_window_days: int = 30,
        test_window_days: int = 7,
        step_days: int = 7,
        verbose: bool = True,
    ) -> WalkForwardResult:
        """
        執行 Walk-Forward 分析
        
        Args:
            train_window_days: 訓練窗口（天）
            test_window_days: 測試窗口（天）
            step_days: 步進（天）
            verbose: 是否輸出進度
        
        Returns:
            分析結果
        """
        # 假設數據是分鐘級，計算索引
        minutes_per_day = 24 * 60
        train_window = train_window_days * minutes_per_day
        test_window = test_window_days * minutes_per_day
        step = step_days * minutes_per_day
        
        windows = []
        total_len = len(self.data)
        
        start_idx = 0
        window_num = 0
        
        while start_idx + train_window + test_window <= total_len:
            window_num += 1
            
            # 切割數據
            train_end = start_idx + train_window
            test_end = train_end + test_window
            
            train_data = self.data.iloc[start_idx:train_end].reset_index(drop=True)
            test_data = self.data.iloc[train_end:test_end].reset_index(drop=True)
            
            if verbose:
                print(f"\n[Window {window_num}] "
                      f"Train: {start_idx}-{train_end}, "
                      f"Test: {train_end}-{test_end}")
            
            # 訓練
            train_env = self.env_factory_with_data(train_data)
            model = self.model_factory(train_env)
            model.learn(total_timesteps=self.train_timesteps, progress_bar=verbose)
            train_env.close()
            
            # 測試
            test_env = self.env_factory_with_data(test_data)
            test_result = self._evaluate_window(model, test_env)
            test_env.close()
            
            windows.append({
                "window": window_num,
                "train_start": start_idx,
                "train_end": train_end,
                "test_start": train_end,
                "test_end": test_end,
                "metrics": test_result,
            })
            
            if verbose:
                print(f"  Test PnL: {test_result['avg_pnl']:.2f}, "
                      f"Sharpe: {test_result['sharpe']:.4f}")
            
            start_idx += step
        
        # 計算彙總指標
        aggregate = self._compute_aggregate_metrics(windows)
        stability = self._compute_stability_score(windows)
        
        return WalkForwardResult(
            windows=windows,
            aggregate_metrics=aggregate,
            stability_score=stability,
        )
    
    def _evaluate_window(self, model: BaseAlgorithm, env, n_episodes: int = 5) -> Dict[str, float]:
        """評估單一窗口"""
        pnls = []
        
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_pnl = 0.0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_pnl += reward
            
            pnls.append(episode_pnl)
        
        pnls = np.array(pnls)
        avg_pnl = float(np.mean(pnls))
        std_pnl = float(np.std(pnls)) if len(pnls) > 1 else 0.01
        
        return {
            "avg_pnl": avg_pnl,
            "std_pnl": std_pnl,
            "sharpe": avg_pnl / std_pnl if std_pnl > 0 else 0.0,
            "max_pnl": float(np.max(pnls)),
            "min_pnl": float(np.min(pnls)),
        }
    
    def _compute_aggregate_metrics(self, windows: List[Dict]) -> Dict[str, float]:
        """計算彙總指標"""
        avg_pnls = [w["metrics"]["avg_pnl"] for w in windows]
        sharpes = [w["metrics"]["sharpe"] for w in windows]
        
        return {
            "overall_avg_pnl": float(np.mean(avg_pnls)),
            "overall_std_pnl": float(np.std(avg_pnls)),
            "overall_sharpe": float(np.mean(sharpes)),
            "sharpe_std": float(np.std(sharpes)),
            "positive_windows": sum(1 for p in avg_pnls if p > 0) / len(avg_pnls),
            "n_windows": len(windows),
        }
    
    def _compute_stability_score(self, windows: List[Dict]) -> float:
        """
        計算穩定性分數
        
        基於績效的一致性
        """
        if len(windows) < 2:
            return 0.0
        
        sharpes = [w["metrics"]["sharpe"] for w in windows]
        avg_pnls = [w["metrics"]["avg_pnl"] for w in windows]
        
        # 方向一致性
        direction_consistency = sum(1 for p in avg_pnls if p > 0) / len(avg_pnls)
        
        # Sharpe 變異係數（越低越穩定）
        sharpe_cv = np.std(sharpes) / (np.abs(np.mean(sharpes)) + 0.01)
        sharpe_stability = 1.0 / (1.0 + sharpe_cv)
        
        # 綜合分數
        stability = 0.5 * direction_consistency + 0.5 * sharpe_stability
        
        return float(stability)


# =============================================================================
# Monte Carlo Simulator
# =============================================================================

class MonteCarloSimulator:
    """
    蒙地卡羅模擬器
    
    模擬策略在不同市場條件下的表現
    """
    
    def __init__(
        self,
        base_pnl_distribution: np.ndarray,
        n_simulations: int = 1000,
        n_periods: int = 252,  # 交易日
    ):
        """
        Args:
            base_pnl_distribution: 基礎 PnL 分佈
            n_simulations: 模擬次數
            n_periods: 模擬期數
        """
        self.base_distribution = base_pnl_distribution
        self.n_simulations = n_simulations
        self.n_periods = n_periods
    
    def run(self) -> Dict[str, Any]:
        """
        執行蒙地卡羅模擬
        
        Returns:
            模擬結果
        """
        simulated_paths = []
        final_pnls = []
        max_drawdowns = []
        
        for _ in range(self.n_simulations):
            # 從分佈中抽樣
            daily_pnls = np.random.choice(self.base_distribution, size=self.n_periods, replace=True)
            
            # 計算累積 PnL
            cumulative = np.cumsum(daily_pnls)
            
            # 計算最大回撤
            peak = np.maximum.accumulate(cumulative)
            drawdown = peak - cumulative
            max_dd = np.max(drawdown) / (np.max(peak) + 1e-8)
            
            simulated_paths.append(cumulative)
            final_pnls.append(cumulative[-1])
            max_drawdowns.append(max_dd)
        
        final_pnls = np.array(final_pnls)
        max_drawdowns = np.array(max_drawdowns)
        
        return {
            "final_pnl": {
                "mean": float(np.mean(final_pnls)),
                "std": float(np.std(final_pnls)),
                "median": float(np.median(final_pnls)),
                "percentile_5": float(np.percentile(final_pnls, 5)),
                "percentile_95": float(np.percentile(final_pnls, 95)),
                "prob_positive": float(np.mean(final_pnls > 0)),
            },
            "max_drawdown": {
                "mean": float(np.mean(max_drawdowns)),
                "percentile_95": float(np.percentile(max_drawdowns, 95)),
            },
            "paths": np.array(simulated_paths),
        }


# =============================================================================
# Robustness Tester
# =============================================================================

class RobustnessTester:
    """
    穩健性測試器
    
    測試策略對參數變化的敏感度
    """
    
    def __init__(
        self,
        model: BaseAlgorithm,
        base_env_factory: Callable,
    ):
        """
        Args:
            model: RL 模型
            base_env_factory: 基礎環境工廠
        """
        self.model = model
        self.base_env_factory = base_env_factory
    
    def test_parameter_sensitivity(
        self,
        param_name: str,
        param_values: List[float],
        n_episodes: int = 10,
    ) -> Dict[str, List[float]]:
        """
        測試參數敏感度
        
        Args:
            param_name: 參數名稱
            param_values: 參數值列表
            n_episodes: 每個值的測試 episode 數
        
        Returns:
            各參數值的績效
        """
        results = {
            "param_values": param_values,
            "avg_pnls": [],
            "std_pnls": [],
            "sharpes": [],
        }
        
        for value in param_values:
            # 建立環境（修改參數）
            env = self.base_env_factory()
            if hasattr(env, param_name):
                setattr(env, param_name, value)
            elif hasattr(env.unwrapped, param_name):
                setattr(env.unwrapped, param_name, value)
            
            # 評估
            pnls = []
            for _ in range(n_episodes):
                obs, _ = env.reset()
                done = False
                episode_pnl = 0.0
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    episode_pnl += reward
                
                pnls.append(episode_pnl)
            
            env.close()
            
            avg_pnl = np.mean(pnls)
            std_pnl = np.std(pnls)
            sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0.0
            
            results["avg_pnls"].append(float(avg_pnl))
            results["std_pnls"].append(float(std_pnl))
            results["sharpes"].append(float(sharpe))
        
        return results
    
    def test_slippage_sensitivity(
        self,
        slippage_bps_range: List[float] = [0, 1, 2, 5, 10],
        n_episodes: int = 10,
    ) -> Dict[str, Any]:
        """測試滑點敏感度"""
        return self.test_parameter_sensitivity("slippage_bps", slippage_bps_range, n_episodes)
    
    def test_fee_sensitivity(
        self,
        fee_rate_range: List[float] = [0.0001, 0.0002, 0.0004, 0.0006, 0.001],
        n_episodes: int = 10,
    ) -> Dict[str, Any]:
        """測試手續費敏感度"""
        return self.test_parameter_sensitivity("fee_rate", fee_rate_range, n_episodes)


# =============================================================================
# Report Generator
# =============================================================================

class BacktestReportGenerator:
    """回測報告生成器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(
        self,
        backtest_result: BacktestResult,
        wf_result: WalkForwardResult = None,
        mc_result: Dict = None,
        robustness_result: Dict = None,
    ) -> str:
        """
        生成完整報告
        
        Returns:
            報告檔案路徑
        """
        report = {
            "backtest": backtest_result.to_dict(),
        }
        
        if wf_result:
            report["walk_forward"] = wf_result.to_dict()
        
        if mc_result:
            # 移除大型數組
            mc_summary = {k: v for k, v in mc_result.items() if k != "paths"}
            report["monte_carlo"] = mc_summary
        
        if robustness_result:
            report["robustness"] = robustness_result
        
        # 儲存 JSON
        report_path = self.output_dir / "backtest_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        # 生成文字報告
        text_report = self._generate_text_report(report)
        text_path = self.output_dir / "backtest_report.txt"
        with open(text_path, "w") as f:
            f.write(text_report)
        
        return str(report_path)
    
    def _generate_text_report(self, report: Dict) -> str:
        """生成文字報告"""
        lines = []
        lines.append("=" * 60)
        lines.append("BACKTEST REPORT")
        lines.append("=" * 60)
        
        bt = report.get("backtest", {})
        lines.append("\n📊 Performance Metrics:")
        lines.append(f"  Total PnL: ${bt.get('total_pnl', 0):,.2f}")
        lines.append(f"  Average PnL: ${bt.get('avg_pnl', 0):,.2f}")
        lines.append(f"  Sharpe Ratio: {bt.get('sharpe_ratio', 0):.4f}")
        lines.append(f"  Max Drawdown: {bt.get('max_drawdown', 0):.2%}")
        lines.append(f"  Calmar Ratio: {bt.get('calmar_ratio', 0):.4f}")
        
        lines.append("\n📈 Trading Statistics:")
        lines.append(f"  Total Trades: {bt.get('total_trades', 0)}")
        lines.append(f"  Win Rate: {bt.get('win_rate', 0):.2%}")
        lines.append(f"  Profit Factor: {bt.get('profit_factor', 0):.2f}")
        
        lines.append("\n⚠️ Risk Metrics:")
        lines.append(f"  VaR (95%): ${bt.get('var_95', 0):,.2f}")
        lines.append(f"  CVaR (95%): ${bt.get('cvar_95', 0):,.2f}")
        
        if "walk_forward" in report:
            wf = report["walk_forward"]
            agg = wf.get("aggregate_metrics", {})
            lines.append("\n🔄 Walk-Forward Analysis:")
            lines.append(f"  Windows: {agg.get('n_windows', 0)}")
            lines.append(f"  Overall Sharpe: {agg.get('overall_sharpe', 0):.4f}")
            lines.append(f"  Positive Windows: {agg.get('positive_windows', 0):.2%}")
            lines.append(f"  Stability Score: {wf.get('stability_score', 0):.4f}")
        
        if "monte_carlo" in report:
            mc = report["monte_carlo"]
            fp = mc.get("final_pnl", {})
            lines.append("\n🎲 Monte Carlo Simulation:")
            lines.append(f"  Mean Final PnL: ${fp.get('mean', 0):,.2f}")
            lines.append(f"  5th Percentile: ${fp.get('percentile_5', 0):,.2f}")
            lines.append(f"  95th Percentile: ${fp.get('percentile_95', 0):,.2f}")
            lines.append(f"  Prob. Positive: {fp.get('prob_positive', 0):.2%}")
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)
