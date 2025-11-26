"""
utils/online_adaptation.py
線上適應模組

實作市場狀態偵測與策略動態切換:
- Regime Detection: 偵測市場狀態（趨勢、震盪、高波動等）
- Strategy Switching: 根據市場狀態切換策略
- Continuous Fine-tuning: 持續微調模型

用法:
    from utils.online_adaptation import RegimeDetector, AdaptivePolicy
    
    detector = RegimeDetector()
    regime = detector.detect(market_data)
    
    policy = AdaptivePolicy(models_by_regime)
    action = policy.predict(obs, regime)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm


# =============================================================================
# Market Regime Definitions
# =============================================================================

class MarketRegime(Enum):
    """市場狀態"""
    TRENDING_UP = "trending_up"          # 上升趨勢
    TRENDING_DOWN = "trending_down"      # 下降趨勢
    RANGING = "ranging"                  # 震盪盤整
    HIGH_VOLATILITY = "high_volatility"  # 高波動
    LOW_VOLATILITY = "low_volatility"    # 低波動
    BREAKOUT = "breakout"                # 突破
    MEAN_REVERTING = "mean_reverting"    # 均值回歸
    UNKNOWN = "unknown"


@dataclass
class RegimeInfo:
    """市場狀態資訊"""
    regime: MarketRegime
    confidence: float
    features: Dict[str, float]
    timestamp: int = 0


# =============================================================================
# Regime Detector
# =============================================================================

class RegimeDetector:
    """
    市場狀態偵測器
    
    使用技術指標判斷當前市場狀態
    """
    
    def __init__(
        self,
        lookback: int = 100,
        volatility_threshold: float = 0.02,
        trend_threshold: float = 0.005,
        ranging_threshold: float = 0.002,
    ):
        """
        Args:
            lookback: 計算指標的回顧窗口
            volatility_threshold: 高/低波動的閾值
            trend_threshold: 趨勢判定閾值
            ranging_threshold: 震盪判定閾值
        """
        self.lookback = lookback
        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold
        self.ranging_threshold = ranging_threshold
        
        # 價格歷史
        self.price_history: deque = deque(maxlen=lookback)
        self.volume_history: deque = deque(maxlen=lookback)
        
        # 狀態歷史
        self.regime_history: List[RegimeInfo] = []
        self.current_step = 0
    
    def update(self, price: float, volume: float = 0.0):
        """更新市場資料"""
        self.price_history.append(price)
        self.volume_history.append(volume)
        self.current_step += 1
    
    def detect(self) -> RegimeInfo:
        """
        偵測當前市場狀態
        
        Returns:
            市場狀態資訊
        """
        if len(self.price_history) < self.lookback // 2:
            return RegimeInfo(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                features={},
                timestamp=self.current_step,
            )
        
        prices = np.array(self.price_history)
        
        # 計算特徵
        features = self._compute_features(prices)
        
        # 判斷狀態
        regime, confidence = self._classify_regime(features)
        
        info = RegimeInfo(
            regime=regime,
            confidence=confidence,
            features=features,
            timestamp=self.current_step,
        )
        
        self.regime_history.append(info)
        
        return info
    
    def _compute_features(self, prices: np.ndarray) -> Dict[str, float]:
        """計算特徵"""
        returns = np.diff(prices) / prices[:-1]
        
        features = {
            # 波動率
            "volatility": float(np.std(returns)),
            "volatility_short": float(np.std(returns[-20:])) if len(returns) >= 20 else 0,
            
            # 趨勢
            "momentum": float((prices[-1] - prices[0]) / prices[0]),
            "momentum_short": float((prices[-1] - prices[-20]) / prices[-20]) if len(prices) >= 20 else 0,
            
            # 均值回歸
            "mean_deviation": float((prices[-1] - np.mean(prices)) / np.std(prices)) if np.std(prices) > 0 else 0,
            
            # 價格範圍
            "range_ratio": float((np.max(prices) - np.min(prices)) / np.mean(prices)),
            
            # 趨勢強度 (簡化的 ADX)
            "trend_strength": self._compute_trend_strength(prices),
            
            # 突破指標
            "breakout_score": self._compute_breakout_score(prices),
        }
        
        return features
    
    def _compute_trend_strength(self, prices: np.ndarray) -> float:
        """計算趨勢強度"""
        if len(prices) < 14:
            return 0.0
        
        # 使用線性回歸斜率
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        
        # 正規化
        normalized_slope = slope / np.mean(prices) * 100
        
        return float(np.abs(normalized_slope))
    
    def _compute_breakout_score(self, prices: np.ndarray) -> float:
        """計算突破分數"""
        if len(prices) < 20:
            return 0.0
        
        recent_high = np.max(prices[-20:])
        recent_low = np.min(prices[-20:])
        historical_high = np.max(prices[:-20]) if len(prices) > 20 else recent_high
        historical_low = np.min(prices[:-20]) if len(prices) > 20 else recent_low
        
        # 檢查是否突破歷史高/低點
        if prices[-1] > historical_high:
            return 1.0
        elif prices[-1] < historical_low:
            return -1.0
        else:
            return 0.0
    
    def _classify_regime(self, features: Dict[str, float]) -> Tuple[MarketRegime, float]:
        """分類市場狀態"""
        vol = features["volatility"]
        momentum = features["momentum"]
        trend_strength = features["trend_strength"]
        breakout = features["breakout_score"]
        mean_dev = features["mean_deviation"]
        
        # 高波動
        if vol > self.volatility_threshold:
            return MarketRegime.HIGH_VOLATILITY, min(vol / self.volatility_threshold, 1.0)
        
        # 突破
        if abs(breakout) > 0.5:
            if breakout > 0:
                return MarketRegime.BREAKOUT, abs(breakout)
            else:
                return MarketRegime.BREAKOUT, abs(breakout)
        
        # 趨勢
        if abs(momentum) > self.trend_threshold and trend_strength > 0.1:
            if momentum > 0:
                return MarketRegime.TRENDING_UP, min(abs(momentum) / self.trend_threshold, 1.0)
            else:
                return MarketRegime.TRENDING_DOWN, min(abs(momentum) / self.trend_threshold, 1.0)
        
        # 均值回歸
        if abs(mean_dev) > 1.5:
            return MarketRegime.MEAN_REVERTING, min(abs(mean_dev) / 3.0, 1.0)
        
        # 低波動震盪
        if vol < self.volatility_threshold * 0.5:
            return MarketRegime.LOW_VOLATILITY, min((self.volatility_threshold - vol) / self.volatility_threshold, 1.0)
        
        # 震盪
        if abs(momentum) < self.ranging_threshold:
            return MarketRegime.RANGING, 0.7
        
        return MarketRegime.UNKNOWN, 0.5
    
    def get_regime_distribution(self, window: int = 100) -> Dict[str, float]:
        """
        取得近期狀態分佈
        
        Returns:
            各狀態的佔比
        """
        if not self.regime_history:
            return {}
        
        recent = self.regime_history[-window:]
        counts = {}
        
        for info in recent:
            regime_name = info.regime.value
            counts[regime_name] = counts.get(regime_name, 0) + 1
        
        total = len(recent)
        return {k: v / total for k, v in counts.items()}


# =============================================================================
# Adaptive Policy
# =============================================================================

class AdaptivePolicy:
    """
    自適應策略
    
    根據市場狀態切換不同的策略
    """
    
    def __init__(
        self,
        models: Dict[MarketRegime, BaseAlgorithm] = None,
        default_model: BaseAlgorithm = None,
        blend_actions: bool = True,
        blend_window: int = 10,
    ):
        """
        Args:
            models: 各狀態對應的模型
            default_model: 預設模型
            blend_actions: 是否混合動作（平滑過渡）
            blend_window: 混合窗口大小
        """
        self.models = models or {}
        self.default_model = default_model
        self.blend_actions = blend_actions
        self.blend_window = blend_window
        
        # 動作歷史（用於平滑）
        self.action_history: deque = deque(maxlen=blend_window)
        self.last_regime: Optional[MarketRegime] = None
    
    def predict(
        self,
        observation: np.ndarray,
        regime_info: RegimeInfo,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, None]:
        """
        根據市場狀態預測動作
        
        Args:
            observation: 觀察值
            regime_info: 市場狀態資訊
            deterministic: 是否使用確定性策略
        
        Returns:
            (action, state)
        """
        regime = regime_info.regime
        confidence = regime_info.confidence
        
        # 選擇模型
        if regime in self.models:
            model = self.models[regime]
        elif self.default_model is not None:
            model = self.default_model
        else:
            raise ValueError(f"No model for regime {regime} and no default model")
        
        # 預測動作
        action, _ = model.predict(observation, deterministic=deterministic)
        
        # 狀態切換時的平滑處理
        if self.blend_actions and self.last_regime is not None and self.last_regime != regime:
            action = self._blend_action(action, confidence)
        
        self.action_history.append(action)
        self.last_regime = regime
        
        return action, None
    
    def _blend_action(self, new_action: np.ndarray, confidence: float) -> np.ndarray:
        """混合動作（平滑過渡）"""
        if len(self.action_history) == 0:
            return new_action
        
        # 加權平均：新動作權重由信心決定
        old_action = np.mean(list(self.action_history), axis=0)
        blended = confidence * new_action + (1 - confidence) * old_action
        
        return blended
    
    def add_model(self, regime: MarketRegime, model: BaseAlgorithm):
        """添加狀態對應的模型"""
        self.models[regime] = model
    
    def set_default_model(self, model: BaseAlgorithm):
        """設定預設模型"""
        self.default_model = model


# =============================================================================
# Online Fine-tuner
# =============================================================================

class OnlineFineTuner:
    """
    線上微調器
    
    在部署時持續微調模型
    """
    
    def __init__(
        self,
        model: BaseAlgorithm,
        buffer_size: int = 10000,
        update_freq: int = 1000,
        learning_rate: float = 1e-5,
        min_samples: int = 1000,
    ):
        """
        Args:
            model: RL 模型
            buffer_size: 經驗緩衝區大小
            update_freq: 更新頻率（步數）
            learning_rate: 微調學習率
            min_samples: 最少樣本數才開始更新
        """
        self.model = model
        self.buffer_size = buffer_size
        self.update_freq = update_freq
        self.learning_rate = learning_rate
        self.min_samples = min_samples
        
        # 經驗緩衝區
        self.buffer: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = []
        self.step_count = 0
        self.update_count = 0
        
        # 效能追蹤
        self.reward_history: List[float] = []
    
    def observe(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """
        觀察新經驗
        
        Args:
            obs: 觀察值
            action: 動作
            reward: 獎勵
            next_obs: 下一個觀察值
            done: 是否結束
        """
        # 添加到緩衝區
        self.buffer.append((obs, action, reward, next_obs, done))
        
        # 維持緩衝區大小
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        
        self.step_count += 1
        self.reward_history.append(reward)
        
        # 定期更新
        if self.step_count % self.update_freq == 0 and len(self.buffer) >= self.min_samples:
            self._update()
    
    def _update(self):
        """執行模型更新"""
        # 這裡是簡化版本，實際實作需要根據具體演算法調整
        print(f"[OnlineFineTuner] Update #{self.update_count + 1}, "
              f"buffer_size={len(self.buffer)}, "
              f"avg_reward={np.mean(self.reward_history[-100:]):.4f}")
        
        # 對於 SAC 等 off-policy 演算法，可以直接從緩衝區採樣更新
        # 這裡省略具體實作，因為需要直接操作模型內部的 replay buffer
        
        self.update_count += 1
    
    def get_performance_trend(self, window: int = 100) -> Dict[str, float]:
        """取得效能趨勢"""
        if len(self.reward_history) < window:
            return {"trend": 0.0, "recent_avg": 0.0}
        
        recent = self.reward_history[-window:]
        older = self.reward_history[-2*window:-window] if len(self.reward_history) >= 2*window else self.reward_history[:window]
        
        return {
            "trend": np.mean(recent) - np.mean(older),
            "recent_avg": np.mean(recent),
            "older_avg": np.mean(older),
        }


# =============================================================================
# Regime-Aware Callback
# =============================================================================

class RegimeAwareCallback:
    """
    狀態感知回調
    
    在訓練過程中根據市場狀態調整訓練
    """
    
    def __init__(
        self,
        detector: RegimeDetector,
        regime_weights: Dict[MarketRegime, float] = None,
    ):
        """
        Args:
            detector: 市場狀態偵測器
            regime_weights: 各狀態的採樣權重
        """
        self.detector = detector
        self.regime_weights = regime_weights or {}
        
        # 預設權重
        default_weight = 1.0
        for regime in MarketRegime:
            if regime not in self.regime_weights:
                self.regime_weights[regime] = default_weight
    
    def get_sample_weight(self, price: float) -> float:
        """
        根據當前市場狀態取得採樣權重
        
        用於優先學習某些市場狀態
        """
        self.detector.update(price)
        regime_info = self.detector.detect()
        
        return self.regime_weights.get(regime_info.regime, 1.0)


# =============================================================================
# 便利函數
# =============================================================================

def create_adaptive_system(
    default_model: BaseAlgorithm,
    regime_models: Dict[str, BaseAlgorithm] = None,
    lookback: int = 100,
) -> Tuple[RegimeDetector, AdaptivePolicy]:
    """
    便利函數：建立自適應系統
    
    Args:
        default_model: 預設模型
        regime_models: 各狀態對應的模型 {"regime_name": model}
        lookback: 偵測器回顧窗口
    
    Returns:
        (detector, policy)
    """
    detector = RegimeDetector(lookback=lookback)
    
    # 轉換狀態名稱到 Enum
    models = {}
    if regime_models:
        for name, model in regime_models.items():
            try:
                regime = MarketRegime(name)
                models[regime] = model
            except ValueError:
                print(f"Warning: Unknown regime {name}, skipping")
    
    policy = AdaptivePolicy(
        models=models,
        default_model=default_model,
        blend_actions=True,
    )
    
    return detector, policy
