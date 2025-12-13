"""
Data Quality and Environment Validation Module
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class DataQualityChecker:
    """Check data quality before training"""
    
    def __init__(self, min_samples=10000, max_missing_ratio=0.01):
        self.min_samples = min_samples
        self.max_missing_ratio = max_missing_ratio
    
    def validate(self, data_path: Path) -> Tuple[bool, Dict, str]:
        """
        Returns: (is_valid, metrics, message)
        """
        try:
            df = pd.read_csv(data_path)
            
            metrics = {}
            issues = []
            
            # 1. Check sample size
            metrics['total_samples'] = len(df)
            if len(df) < self.min_samples:
                issues.append(f"Insufficient data: {len(df)} < {self.min_samples}")
            
            # 2. Check missing values
            missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
            metrics['missing_ratio'] = missing_ratio
            if missing_ratio > self.max_missing_ratio:
                issues.append(f"Too many missing values: {missing_ratio:.2%}")
            
            # 3. Check price columns
            price_col = 'close' if 'close' in df.columns else 'Close' if 'Close' in df.columns else None
            if price_col is None:
                issues.append("No 'close' or 'Close' column found")
                return False, metrics, "; ".join(issues)
            
            prices = df[price_col].dropna()
            
            # 4. Check for zero/negative prices
            if (prices <= 0).any():
                issues.append("Found zero or negative prices")
            
            # 5. Check price volatility
            returns = prices.pct_change().dropna()
            metrics['mean_return'] = returns.mean()
            metrics['volatility'] = returns.std()
            metrics['avg_price'] = prices.mean()
            metrics['price_range'] = (prices.min(), prices.max())
            
            # Too low volatility means no trading opportunity
            if metrics['volatility'] < 0.0001:
                issues.append(f"Very low volatility: {metrics['volatility']:.6f}")
            
            # Unreasonably high volatility might indicate data issues
            if metrics['volatility'] > 0.5:
                issues.append(f"Suspiciously high volatility: {metrics['volatility']:.4f}")
            
            # 6. Check for suspicious patterns (e.g., constant values)
            if prices.nunique() < len(prices) * 0.1:
                issues.append("Too many duplicate price values")
            
            # 7. Check volume if available
            volume_col = 'volume' if 'volume' in df.columns else 'Volume' if 'Volume' in df.columns else None
            if volume_col:
                volumes = df[volume_col].dropna()
                metrics['avg_volume'] = volumes.mean()
                if (volumes == 0).sum() / len(volumes) > 0.3:
                    issues.append("More than 30% zero volume periods")
            
            is_valid = len(issues) == 0
            message = "Data quality check passed" if is_valid else "; ".join(issues)
            
            return is_valid, metrics, message
            
        except Exception as e:
            logger.error(f"Data validation error: {e}")
            return False, {}, f"Validation error: {str(e)}"


class EnvironmentHealthChecker:
    """Check if environment parameters are reasonable"""
    
    def __init__(self):
        pass
    
    def validate(self, config: Dict, data_metrics: Dict) -> Tuple[bool, List[str]]:
        """
        Check if environment config makes sense given the data
        Returns: (is_healthy, warnings)
        """
        warnings = []
        
        env_cfg = config.get('env', {})
        avg_price = data_metrics.get('avg_price', 0)
        volatility = data_metrics.get('volatility', 0)
        
        # 1. Check spread vs price
        base_spread = env_cfg.get('base_spread', 0)
        if base_spread > 0 and avg_price > 0:
            spread_ratio = base_spread / avg_price
            if spread_ratio < 0.00001:  # < 0.001%
                warnings.append(f"Base spread too small: {spread_ratio:.6%} of price")
            elif spread_ratio > 0.01:  # > 1%
                warnings.append(f"Base spread too large: {spread_ratio:.2%} of price")
        
        # 2. Check initial cash vs price
        initial_cash = env_cfg.get('initial_cash', 0)
        if initial_cash > 0 and avg_price > 0:
            max_units = initial_cash / avg_price
            if max_units < 1:
                warnings.append(f"Initial cash too low: can only buy {max_units:.2f} units")
        
        # 3. Check max inventory
        max_inventory = env_cfg.get('max_inventory', 0)
        if max_inventory <= 0:
            warnings.append("Max inventory not set or invalid")
        
        # 4. Check episode length
        episode_length = env_cfg.get('episode_length', 0)
        total_samples = data_metrics.get('total_samples', 0)
        if episode_length > total_samples * 0.5:
            warnings.append(f"Episode length too long: {episode_length} vs {total_samples} samples")
        
        # 5. Check reward scale
        reward_cfg = config.get('reward', {})
        reward_scale = reward_cfg.get('reward_scale', 1.0)
        if reward_scale <= 0:
            warnings.append("Invalid reward scale")
        
        # Health is OK if no critical warnings
        is_healthy = len(warnings) == 0
        
        return is_healthy, warnings


class TrainingResultValidator:
    """Validate training results with multiple metrics"""
    
    def __init__(
        self,
        min_pnl=0,
        min_win_rate=0.45,
        min_sharpe=0.0,
        max_drawdown=-0.3,
        min_episodes=10
    ):
        self.min_pnl = min_pnl
        self.min_win_rate = min_win_rate
        self.min_sharpe = min_sharpe
        self.max_drawdown = max_drawdown
        self.min_episodes = min_episodes
    
    def validate(self, results: Dict) -> Tuple[bool, float, str]:
        """
        Returns: (is_acceptable, score, reason)
        Score: 0-100 composite score
        """
        score_components = []
        issues = []
        
        # 1. PnL Check
        pnl = results.get('mean_pnl', 0)
        if pnl >= self.min_pnl:
            pnl_score = min(100, max(0, 50 + pnl / 100))  # Scale PnL to score
            score_components.append(('pnl', pnl_score, 0.3))
        else:
            issues.append(f"PnL too low: {pnl:.2f} < {self.min_pnl}")
            score_components.append(('pnl', 0, 0.3))
        
        # 2. Win Rate Check
        win_rate = results.get('win_rate', 0)
        if win_rate >= self.min_win_rate:
            wr_score = min(100, win_rate * 100)
            score_components.append(('win_rate', wr_score, 0.2))
        else:
            issues.append(f"Win rate too low: {win_rate:.2%} < {self.min_win_rate:.2%}")
            score_components.append(('win_rate', win_rate * 100, 0.2))
        
        # 3. Sharpe Ratio Check (if available)
        sharpe = results.get('sharpe_ratio', None)
        if sharpe is not None:
            if sharpe >= self.min_sharpe:
                sharpe_score = min(100, max(0, (sharpe + 2) * 25))  # Scale Sharpe to 0-100
                score_components.append(('sharpe', sharpe_score, 0.25))
            else:
                issues.append(f"Sharpe too low: {sharpe:.2f} < {self.min_sharpe:.2f}")
                score_components.append(('sharpe', 0, 0.25))
        
        # 4. Max Drawdown Check (if available)
        drawdown = results.get('max_drawdown', 0)
        if drawdown < self.max_drawdown:
            issues.append(f"Drawdown too large: {drawdown:.2%} < {self.max_drawdown:.2%}")
            dd_score = 0
        else:
            dd_score = min(100, max(0, (1 + drawdown) * 100))
        score_components.append(('drawdown', dd_score, 0.15))
        
        # 5. Consistency Check
        pnl_std = results.get('std_pnl', 0)
        if pnl > 0 and pnl_std > 0:
            consistency = pnl / pnl_std  # Signal-to-noise ratio
            consistency_score = min(100, consistency * 20)
            score_components.append(('consistency', consistency_score, 0.1))
        else:
            score_components.append(('consistency', 0, 0.1))
        
        # Calculate composite score
        total_score = sum(score * weight for _, score, weight in score_components)
        
        # Determine if acceptable
        is_acceptable = len(issues) == 0 and total_score >= 60
        
        reason = "Acceptable" if is_acceptable else ("; ".join(issues) if issues else "Score too low")
        
        return is_acceptable, total_score, reason
