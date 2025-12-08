from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple, Union, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator

# Import constants for default values
from envs.constants import (
    DEFAULT_VOLATILITY_WINDOWS,
    DEFAULT_MOMENTUM_WINDOWS,
    DEFAULT_TREND_WINDOWS,
    DEFAULT_METRICS_BUFFER_SIZE,
)

# =============================================================================
# Enums
# =============================================================================

class RewardMode(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    SHAPED = "shaped"
    HYBRID = "hybrid"

class ActionMode(str, Enum):
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    DISCRETE = "discrete"

class FillModelMode(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    REALISTIC = "realistic"
    CUSTOM = "custom"

# =============================================================================
# Config Models
# =============================================================================

class EnvConfig(BaseModel):
    """Basic environment configuration"""
    id: str = "MarketMakingEnvV2"
    data_file: str = "data/btc_usdt_1m_2023.csv"
    
    # Basic parameters
    episode_length: int = Field(default=1440, gt=0)
    initial_cash: float = Field(default=10000.0, gt=0)
    max_inventory: float = Field(default=5.0, gt=0)
    random_start: bool = True
    
    # Market parameters
    fee_rate: float = Field(default=0.0004, ge=0)
    base_spread: float = Field(default=25.0, gt=0)

class DataAugmentationConfig(BaseModel):
    """Data augmentation configuration"""
    enable_price_flip: bool = False
    enable_noise_injection: bool = False
    noise_std: float = Field(default=0.001, ge=0)

class FillModelConfig(BaseModel):
    """Fill model configuration"""
    enabled: bool = False
    mode: FillModelMode = FillModelMode.SIMPLE
    
    # Custom configuration
    enable_queue_position: bool = True
    enable_slippage: bool = True
    slippage_bps: float = Field(default=1.0, ge=0)
    max_slippage_bps: float = Field(default=5.0, ge=0)
    enable_market_impact: bool = False
    market_impact_factor: float = Field(default=0.1, ge=0)
    enable_adverse_selection: bool = True
    adverse_selection_prob: float = Field(default=0.1, ge=0, le=1.0)
    adverse_selection_cost_bps: float = Field(default=2.0, ge=0)
    
    # Queue dynamics
    queue_decay_rate: float = Field(default=0.1, ge=0, le=1.0)
    initial_queue_position: float = Field(default=0.5, ge=0, le=1.0)
    allow_partial_fills: bool = True
    min_fill_ratio: float = Field(default=0.1, ge=0, le=1.0)

class RewardConfig(BaseModel):
    """Reward function configuration"""
    mode: RewardMode = RewardMode.SHAPED
    
    # Dense/Shaped parameters
    lambda_inventory: float = Field(default=0.0005, ge=0)
    lambda_turnover: float = Field(default=0.0, ge=0)
    gamma: float = Field(default=0.99, gt=0, le=1.0)
    
    # Sparse parameters
    sparse_scale: float = Field(default=0.01, gt=0)
    
    # Hybrid parameters
    terminal_bonus_weight: float = Field(default=0.5, ge=0)
    
    # Market making bonuses
    spread_capture_bonus: float = Field(default=0.0, ge=0)
    round_trip_bonus: float = Field(default=0.0, ge=0)
    inventory_revert_bonus: float = Field(default=0.0, ge=0)
    asymmetric_penalty: float = Field(default=0.0, ge=0)
    
    # Scaling
    reward_scale: float = Field(default=1.0, gt=0)

class ObservationConfig(BaseModel):
    """Observation space configuration"""
    include_price: bool = True
    include_inventory: bool = True
    include_time: bool = True
    include_volatility: bool = True
    include_momentum: bool = True
    include_volume: bool = True
    include_inventory_age: bool = True
    
    # Trend features
    include_trend: bool = False
    trend_windows: List[int] = Field(default_factory=lambda: DEFAULT_TREND_WINDOWS)
    
    # Windows
    volatility_windows: List[int] = Field(default_factory=lambda: DEFAULT_VOLATILITY_WINDOWS)
    momentum_windows: List[int] = Field(default_factory=lambda: DEFAULT_MOMENTUM_WINDOWS)

class AdvancedObservationConfig(BaseModel):
    """Advanced observation features"""
    include_order_flow_imbalance: bool = False
    order_flow_window: int = Field(default=20, gt=0)
    
    include_vwap_deviation: bool = False
    vwap_window: int = Field(default=60, gt=0)
    
    include_multi_timeframe_momentum: bool = False
    mtf_windows: List[int] = Field(default_factory=lambda: [15, 60, 240])
    
    include_volatility_forecast: bool = False
    ewma_span: int = Field(default=20, gt=0)
    
    include_microstructure: bool = False

class ActionConfig(BaseModel):
    """Action space configuration"""
    mode: ActionMode = ActionMode.ASYMMETRIC
    allow_no_quote: bool = True
    max_spread_multiplier: float = Field(default=3.0, gt=0)
    min_spread_multiplier: float = Field(default=0.1, gt=0)

class DomainRandomizationConfig(BaseModel):
    """Domain randomization configuration"""
    enabled: bool = False
    
    fee_rate_range: Tuple[float, float] = (0.0003, 0.0005)
    base_spread_range: Tuple[float, float] = (15.0, 35.0)
    volatility_multiplier_range: Tuple[float, float] = (0.8, 1.2)
    fill_probability_noise: float = Field(default=0.1, ge=0)

class PerformanceConfig(BaseModel):
    """Performance optimization configuration"""
    use_numba: bool = True
    precompute_features: bool = True
    feature_cache_size: int = Field(default=DEFAULT_METRICS_BUFFER_SIZE, gt=0)

class RootConfig(BaseModel):
    """Root configuration object"""
    env: EnvConfig
    data_augmentation: DataAugmentationConfig = Field(default_factory=DataAugmentationConfig)
    fill_model: FillModelConfig = Field(default_factory=FillModelConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    observation: ObservationConfig = Field(default_factory=ObservationConfig)
    advanced_metrics: AdvancedObservationConfig = Field(default_factory=AdvancedObservationConfig)
    action: ActionConfig = Field(default_factory=ActionConfig)
    domain_randomization: DomainRandomizationConfig = Field(default_factory=DomainRandomizationConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)

    @model_validator(mode='before')
    @classmethod
    def handle_legacy_keys(cls, data: Any) -> Any:
        """Handle legacy configuration keys or structure differences"""
        if not isinstance(data, dict):
            return data
            
        # If 'advanced_metrics' is missing but 'advanced_observation' exists, rename it
        if 'advanced_observation' in data and 'advanced_metrics' not in data:
            data['advanced_metrics'] = data['advanced_observation']
            
        return data
