"""
Production Readiness Checker
Ensures model is safe for real trading
"""
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class ProductionReadinessChecker:
    """
    Check if a model is ready for production deployment
    """
    
    def __init__(
        self,
        min_sharpe=0.5,
        max_drawdown_threshold=-0.25,
        min_win_rate=0.48,
        min_trades=50,
        min_profit_factor=1.2,
        max_volatility_ratio=2.0
    ):
        self.min_sharpe = min_sharpe
        self.max_drawdown_threshold = max_drawdown_threshold
        self.min_win_rate = min_win_rate
        self.min_trades = min_trades
        self.min_profit_factor = min_profit_factor
        self.max_volatility_ratio = max_volatility_ratio
    
    def check(self, results: Dict, episode_pnls: List[float] = None) -> Tuple[bool, List[str], Dict]:
        """
        Comprehensive production readiness check
        
        Returns: (is_ready, warnings, details)
        """
        warnings = []
        details = {}
        
        # 1. Profitability Check
        mean_pnl = results.get('mean_pnl', 0)
        details['mean_pnl'] = mean_pnl
        if mean_pnl <= 0:
            warnings.append(f"Model not profitable: PnL = {mean_pnl:.2f}")
        
        # 2. Sharpe Ratio Check
        sharpe = results.get('sharpe_ratio', None)
        details['sharpe_ratio'] = sharpe
        if sharpe is not None:
            if sharpe < self.min_sharpe:
                warnings.append(f"Sharpe too low: {sharpe:.2f} < {self.min_sharpe:.2f}")
        else:
            warnings.append("Sharpe ratio not available")
        
        # 3. Max Drawdown Check
        max_dd = results.get('max_drawdown', 0)
        details['max_drawdown'] = max_dd
        if max_dd < self.max_drawdown_threshold:
            warnings.append(f"Drawdown too large: {max_dd:.2%} < {self.max_drawdown_threshold:.2%}")
        
        # 4. Win Rate Check
        win_rate = results.get('win_rate', 0)
        details['win_rate'] = win_rate
        if win_rate < self.min_win_rate:
            warnings.append(f"Win rate low: {win_rate:.2%} < {self.min_win_rate:.2%}")
        
        # 5. Trading Activity Check
        total_trades = results.get('total_trades', 0)
        details['total_trades'] = total_trades
        if total_trades < self.min_trades:
            warnings.append(f"Insufficient trades: {total_trades} < {self.min_trades}")
        
        # 6. Profit Factor Check (if available)
        profit_factor = results.get('profit_factor', None)
        if profit_factor is not None:
            details['profit_factor'] = profit_factor
            if profit_factor < self.min_profit_factor:
                warnings.append(f"Profit factor low: {profit_factor:.2f} < {self.min_profit_factor:.2f}")
        
        # 7. Consistency Check - analyze episode PnLs if available
        if episode_pnls and len(episode_pnls) > 0:
            episode_pnls = np.array(episode_pnls)
            positive_ratio = (episode_pnls > 0).sum() / len(episode_pnls)
            details['positive_episode_ratio'] = positive_ratio
            
            if positive_ratio < 0.4:
                warnings.append(f"Low positive episode ratio: {positive_ratio:.2%}")
            
            # Check for stability (coefficient of variation)
            if mean_pnl > 0:
                cv = np.std(episode_pnls) / mean_pnl
                details['coefficient_of_variation'] = cv
                if cv > self.max_volatility_ratio:
                    warnings.append(f"High PnL volatility: CV = {cv:.2f}")
        
        # 8. Risk-Adjusted Return
        std_pnl = results.get('std_pnl', 0)
        if mean_pnl > 0 and std_pnl > 0:
            risk_adjusted_return = mean_pnl / std_pnl
            details['risk_adjusted_return'] = risk_adjusted_return
            if risk_adjusted_return < 0.3:
                warnings.append(f"Low risk-adjusted return: {risk_adjusted_return:.2f}")
        
        # 9. Statistical Significance (basic check)
        if episode_pnls and len(episode_pnls) > 5:
            # Simple t-test equivalent: is mean significantly > 0?
            t_stat = mean_pnl / (np.std(episode_pnls) / np.sqrt(len(episode_pnls)))
            details['t_statistic'] = t_stat
            if t_stat < 1.5:  # Rough threshold
                warnings.append(f"Returns not statistically significant: t-stat = {t_stat:.2f}")
        
        # Production ready if no critical warnings
        is_ready = len(warnings) == 0
        
        return is_ready, warnings, details
    
    def generate_report(self, is_ready: bool, warnings: List[str], details: Dict) -> str:
        """Generate human-readable report"""
        report = ["=" * 60]
        report.append("PRODUCTION READINESS REPORT")
        report.append("=" * 60)
        report.append("")
        
        status = "✅ READY FOR PRODUCTION" if is_ready else "❌ NOT READY"
        report.append(f"Status: {status}")
        report.append("")
        
        report.append("Metrics:")
        report.append("-" * 40)
        for key, value in details.items():
            if value is not None:
                if isinstance(value, float):
                    if abs(value) < 0.01:
                        report.append(f"  {key}: {value:.6f}")
                    else:
                        report.append(f"  {key}: {value:.4f}")
                else:
                    report.append(f"  {key}: {value}")
        report.append("")
        
        if warnings:
            report.append("Warnings:")
            report.append("-" * 40)
            for warning in warnings:
                report.append(f"  ⚠️  {warning}")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)
