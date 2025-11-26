"""
utils/risk_sensitive.py
風險敏感強化學習模組

實作風險控制機制:
- Mean-Variance Optimization: 最大化 E[R] - λ·Var[R]
- CVaR (Conditional Value at Risk): 考慮尾部風險
- Drawdown Constraints: 控制最大回撤
- Position Limits: 動態倉位限制

用法:
    from utils.risk_sensitive import (
        RiskAwareRewardWrapper,
        CVaRCallback,
        DrawdownEarlyStopping,
    )
    
    # 包裝環境
    env = RiskAwareRewardWrapper(env, risk_lambda=0.1)
    
    # 加入風險回調
    callbacks = [CVaRCallback(alpha=0.05)]
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


# =============================================================================
# Risk-Aware Reward Wrapper
# =============================================================================

class RiskAwareRewardWrapper(gym.Wrapper):
    """
    風險感知獎勵包裝器
    
    將原始獎勵轉換為風險調整後的獎勵:
    r_adj = r - λ * risk_penalty
    
    支援多種風險度量:
    - variance: 報酬變異數
    - downside_variance: 下行變異數 (只考慮負報酬)
    - cvar: 條件風險值
    """
    
    def __init__(
        self,
        env: gym.Env,
        risk_lambda: float = 0.1,
        risk_type: str = "variance",
        window_size: int = 100,
        cvar_alpha: float = 0.05,
    ):
        """
        Args:
            env: 原始環境
            risk_lambda: 風險厭惡係數
            risk_type: 風險度量類型 ("variance", "downside_variance", "cvar")
            window_size: 計算風險的滑動窗口大小
            cvar_alpha: CVaR 的分位數 (例如 0.05 = 5%)
        """
        super().__init__(env)
        self.risk_lambda = risk_lambda
        self.risk_type = risk_type
        self.window_size = window_size
        self.cvar_alpha = cvar_alpha
        
        self.reward_history: deque = deque(maxlen=window_size)
        self.episode_rewards: List[float] = []
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        self.episode_rewards = []
        return self.env.reset(**kwargs)
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 記錄獎勵
        self.reward_history.append(reward)
        self.episode_rewards.append(reward)
        
        # 計算風險懲罰
        risk_penalty = self._compute_risk_penalty()
        
        # 調整獎勵
        adjusted_reward = reward - self.risk_lambda * risk_penalty
        
        # 記錄原始獎勵和風險到 info
        info["original_reward"] = reward
        info["risk_penalty"] = risk_penalty
        info["adjusted_reward"] = adjusted_reward
        
        return obs, adjusted_reward, terminated, truncated, info
    
    def _compute_risk_penalty(self) -> float:
        """計算風險懲罰"""
        if len(self.reward_history) < 2:
            return 0.0
        
        rewards = np.array(self.reward_history)
        
        if self.risk_type == "variance":
            return float(np.var(rewards))
        
        elif self.risk_type == "downside_variance":
            # 只考慮低於平均的報酬
            mean_r = np.mean(rewards)
            downside = rewards[rewards < mean_r]
            if len(downside) < 2:
                return 0.0
            return float(np.var(downside))
        
        elif self.risk_type == "cvar":
            # 條件風險值 = 低於 VaR 的期望損失
            var = np.percentile(rewards, self.cvar_alpha * 100)
            below_var = rewards[rewards <= var]
            if len(below_var) == 0:
                return -var
            return -float(np.mean(below_var))
        
        else:
            raise ValueError(f"Unknown risk type: {self.risk_type}")


# =============================================================================
# CVaR Constraint Callback
# =============================================================================

class CVaRCallback(BaseCallback):
    """
    CVaR 監控回調
    
    追蹤訓練過程中的尾部風險，並可在風險過高時停止訓練
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        cvar_threshold: float = -100.0,
        window_size: int = 1000,
        check_freq: int = 1000,
        verbose: int = 1,
    ):
        """
        Args:
            alpha: CVaR 的分位數 (例如 0.05 = 最差 5%)
            cvar_threshold: CVaR 閾值，低於此值則警告
            window_size: 計算 CVaR 的窗口大小
            check_freq: 檢查頻率
            verbose: 輸出詳細程度
        """
        super().__init__(verbose)
        self.alpha = alpha
        self.cvar_threshold = cvar_threshold
        self.window_size = window_size
        self.check_freq = check_freq
        
        self.episode_rewards: deque = deque(maxlen=window_size)
        self.cvar_history: List[float] = []
    
    def _on_step(self) -> bool:
        # 收集 episode 結束時的總獎勵
        if self.locals.get("dones", [False])[0]:
            infos = self.locals.get("infos", [{}])
            if len(infos) > 0 and "episode" in infos[0]:
                ep_reward = infos[0]["episode"]["r"]
                self.episode_rewards.append(ep_reward)
        
        # 定期檢查 CVaR
        if self.n_calls % self.check_freq == 0 and len(self.episode_rewards) >= 10:
            cvar = self._compute_cvar()
            self.cvar_history.append(cvar)
            
            if self.verbose >= 1:
                print(f"[CVaR] Step {self.n_calls}: CVaR({self.alpha:.0%}) = {cvar:.2f}")
            
            if cvar < self.cvar_threshold:
                if self.verbose >= 1:
                    print(f"[CVaR] ⚠️ Warning: CVaR below threshold ({cvar:.2f} < {self.cvar_threshold})")
                # 可選：return False 來停止訓練
        
        return True
    
    def _compute_cvar(self) -> float:
        """計算 CVaR"""
        rewards = np.array(self.episode_rewards)
        var = np.percentile(rewards, self.alpha * 100)
        below_var = rewards[rewards <= var]
        if len(below_var) == 0:
            return var
        return float(np.mean(below_var))


# =============================================================================
# Drawdown Early Stopping
# =============================================================================

class DrawdownEarlyStopping(BaseCallback):
    """
    最大回撤提前停止
    
    當評估時的最大回撤超過閾值時停止訓練
    """
    
    def __init__(
        self,
        max_drawdown_threshold: float = 0.2,
        check_freq: int = 5000,
        eval_env=None,
        n_eval_episodes: int = 5,
        verbose: int = 1,
    ):
        """
        Args:
            max_drawdown_threshold: 最大回撤閾值 (例如 0.2 = 20%)
            check_freq: 檢查頻率
            eval_env: 評估環境
            n_eval_episodes: 評估 episode 數
            verbose: 輸出詳細程度
        """
        super().__init__(verbose)
        self.max_drawdown_threshold = max_drawdown_threshold
        self.check_freq = check_freq
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        
        self.drawdown_history: List[float] = []
    
    def _on_step(self) -> bool:
        if self.eval_env is None:
            return True
        
        if self.n_calls % self.check_freq == 0:
            max_dd = self._evaluate_drawdown()
            self.drawdown_history.append(max_dd)
            
            if self.verbose >= 1:
                print(f"[Drawdown] Step {self.n_calls}: Max Drawdown = {max_dd:.2%}")
            
            if max_dd > self.max_drawdown_threshold:
                if self.verbose >= 1:
                    print(f"[Drawdown] ❌ Max drawdown exceeded threshold "
                          f"({max_dd:.2%} > {self.max_drawdown_threshold:.2%})")
                return False
        
        return True
    
    def _evaluate_drawdown(self) -> float:
        """評估最大回撤"""
        max_drawdowns = []
        
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            portfolio_values = []
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                
                pv = info.get("portfolio_value", 0)
                portfolio_values.append(pv)
            
            if portfolio_values:
                pv_array = np.array(portfolio_values)
                peak = np.maximum.accumulate(pv_array)
                drawdown = (peak - pv_array) / np.maximum(peak, 1e-8)
                max_drawdowns.append(np.max(drawdown))
        
        return float(np.mean(max_drawdowns)) if max_drawdowns else 0.0


# =============================================================================
# Dynamic Position Limit
# =============================================================================

class DynamicPositionLimitWrapper(gym.Wrapper):
    """
    動態倉位限制包裝器
    
    根據市場波動率動態調整最大倉位:
    - 高波動時降低倉位上限
    - 低波動時允許較大倉位
    """
    
    def __init__(
        self,
        env: gym.Env,
        base_max_inventory: float = 10.0,
        volatility_threshold: float = 0.02,
        min_inventory_ratio: float = 0.3,
        volatility_window: int = 20,
    ):
        """
        Args:
            env: 原始環境
            base_max_inventory: 基礎最大倉位
            volatility_threshold: 波動率閾值
            min_inventory_ratio: 最小倉位比例 (高波動時)
            volatility_window: 波動率計算窗口
        """
        super().__init__(env)
        self.base_max_inventory = base_max_inventory
        self.volatility_threshold = volatility_threshold
        self.min_inventory_ratio = min_inventory_ratio
        self.volatility_window = volatility_window
        
        self.price_history: deque = deque(maxlen=volatility_window)
        self.current_max_inventory = base_max_inventory
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        self.price_history.clear()
        self.current_max_inventory = self.base_max_inventory
        return self.env.reset(**kwargs)
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 更新價格歷史
        if "mid_price" in info:
            self.price_history.append(info["mid_price"])
        
        # 動態調整倉位上限
        self._update_position_limit()
        
        # 將動態上限傳給環境（如果環境支援）
        if hasattr(self.env, "max_inventory"):
            self.env.max_inventory = self.current_max_inventory
        
        info["dynamic_max_inventory"] = self.current_max_inventory
        
        return obs, reward, terminated, truncated, info
    
    def _update_position_limit(self):
        """根據波動率更新倉位上限"""
        if len(self.price_history) < 2:
            return
        
        prices = np.array(self.price_history)
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) if len(returns) > 1 else 0
        
        # 根據波動率調整倉位
        if volatility > self.volatility_threshold:
            ratio = self.min_inventory_ratio + (1 - self.min_inventory_ratio) * (
                self.volatility_threshold / volatility
            )
            self.current_max_inventory = self.base_max_inventory * ratio
        else:
            self.current_max_inventory = self.base_max_inventory


# =============================================================================
# Risk Metrics Calculator
# =============================================================================

class RiskMetricsCalculator:
    """風險指標計算器"""
    
    @staticmethod
    def compute_var(returns: np.ndarray, alpha: float = 0.05) -> float:
        """計算 Value at Risk"""
        return float(np.percentile(returns, alpha * 100))
    
    @staticmethod
    def compute_cvar(returns: np.ndarray, alpha: float = 0.05) -> float:
        """計算 Conditional Value at Risk (Expected Shortfall)"""
        var = np.percentile(returns, alpha * 100)
        below_var = returns[returns <= var]
        return float(np.mean(below_var)) if len(below_var) > 0 else var
    
    @staticmethod
    def compute_max_drawdown(portfolio_values: np.ndarray) -> float:
        """計算最大回撤"""
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / np.maximum(peak, 1e-8)
        return float(np.max(drawdown))
    
    @staticmethod
    def compute_calmar_ratio(
        returns: np.ndarray,
        portfolio_values: np.ndarray,
        annual_factor: float = 252,
    ) -> float:
        """計算 Calmar Ratio (年化報酬 / 最大回撤)"""
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        max_dd = RiskMetricsCalculator.compute_max_drawdown(portfolio_values)
        if max_dd < 1e-8:
            return 0.0
        return float(total_return / max_dd)
    
    @staticmethod
    def compute_sortino_ratio(
        returns: np.ndarray,
        target_return: float = 0.0,
        annual_factor: float = 252,
    ) -> float:
        """計算 Sortino Ratio (使用下行標準差)"""
        excess = returns - target_return
        downside = returns[returns < target_return]
        
        if len(downside) < 2:
            return 0.0
        
        downside_std = np.std(downside)
        if downside_std < 1e-8:
            return 0.0
        
        return float(np.mean(excess) / downside_std * np.sqrt(annual_factor))
    
    @staticmethod
    def compute_ulcer_index(portfolio_values: np.ndarray) -> float:
        """
        計算 Ulcer Index
        衡量回撤的深度和持續時間
        """
        peak = np.maximum.accumulate(portfolio_values)
        drawdown_pct = (portfolio_values - peak) / peak * 100
        return float(np.sqrt(np.mean(drawdown_pct ** 2)))
    
    @staticmethod
    def compute_all_metrics(
        returns: np.ndarray,
        portfolio_values: np.ndarray,
        alpha: float = 0.05,
    ) -> Dict[str, float]:
        """計算所有風險指標"""
        calc = RiskMetricsCalculator
        
        return {
            "var": calc.compute_var(returns, alpha),
            "cvar": calc.compute_cvar(returns, alpha),
            "max_drawdown": calc.compute_max_drawdown(portfolio_values),
            "calmar_ratio": calc.compute_calmar_ratio(returns, portfolio_values),
            "sortino_ratio": calc.compute_sortino_ratio(returns),
            "ulcer_index": calc.compute_ulcer_index(portfolio_values),
        }
