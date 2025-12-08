"""MarketMakingEnvV2: 改良版做市強化學習環境。

改進項目：
1. Reward: Potential-based shaping + Sparse reward 選項
2. Observation: 擴展特徵（波動率、動量、時間編碼、庫存年齡）
3. Action: 支援非對稱報價、不報價選項
4. Domain Randomization: 訓練時隨機化環境參數
5. Metrics: 完整的行為與風險指標
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# 導入真實成交模型
from envs.realistic_fill_model import (
    RealisticFillModel, 
    FillModelConfig, 
    MarketData as FillMarketData,
    SIMPLE_FILL_CONFIG,
    MODERATE_FILL_CONFIG,
    REALISTIC_FILL_CONFIG,
)

# 導入常數配置
from envs.constants import (
    REWARD_DEBUG_INTERVAL,
    REWARD_WARNING_THRESHOLD,
    ADVERSE_SELECTION_THRESHOLD,
    DEFAULT_VOLATILITY_WINDOWS,
    DEFAULT_MOMENTUM_WINDOWS,
    DEFAULT_TREND_WINDOWS,
    DEFAULT_VOLUME_MA_WINDOW,
    DEFAULT_METRICS_BUFFER_SIZE,
    INVENTORY_WARNING_RATIO,
    EPSILON,
    MIN_PRICE,
    MIN_STD,
    MINUTES_PER_YEAR,
    VAR_PERCENTILE,
)

# 導入 Numba 優化（可選）
try:
    from utils.numba_optimizations import (
        rolling_std_numba,
        rolling_mean_numba,
        compute_momentum_numba,
        compute_order_flow_imbalance_numba,
        compute_vwap_deviation_numba,
        NUMBA_AVAILABLE,
    )
    logger.info("Numba optimization enabled (10-50x speedup)")
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba not available. Using standard NumPy/Pandas (slower)")
    logger.warning("Install with: pip install numba")


# =============================================================================
# Enums and Config Classes
# =============================================================================

class RewardMode(Enum):
    """Reward 計算模式"""
    DENSE = "dense"              # 每步都給 reward（傳統方式）
    SPARSE = "sparse"            # 只在 episode 結束時給 reward
    SHAPED = "shaped"            # Potential-based reward shaping
    HYBRID = "hybrid"            # 混合模式：shaped + sparse terminal bonus


@dataclass
class RewardConfig:
    """Reward 配置"""
    mode: RewardMode = RewardMode.SHAPED
    
    # Dense/Shaped 模式參數
    lambda_inventory: float = 0.0005      # 庫存懲罰係數（用於 potential function）
    lambda_turnover: float = 0.0          # 刷單懲罰
    gamma: float = 0.99                   # 折扣因子（用於 potential shaping）
    
    # Sparse 模式參數
    sparse_scale: float = 0.01            # Sparse reward 的縮放係數
    
    # Hybrid 模式參數
    terminal_bonus_weight: float = 0.5    # Terminal bonus 權重
    
    # 🆕 做市獎勵參數（鼓勵真正的做市行為）
    spread_capture_bonus: float = 0.0     # 每賺到 $1 spread 的獎勵
    round_trip_bonus: float = 0.0         # 完成買賣配對的獎勵
    inventory_revert_bonus: float = 0.0   # 庫存回歸獎勵
    asymmetric_penalty: float = 0.0       # 不對稱報價懲罰
    
    # 🆕 v3: Reward 縮放（穩定訓練）
    reward_scale: float = 1.0             # 獎勵縮放因子，建議 0.001 將獎勵標準化


@dataclass
class ObservationConfig:
    """Observation 特徵配置"""
    include_price: bool = True            # 價格相關特徵
    include_inventory: bool = True        # 庫存特徵
    include_time: bool = True             # 時間特徵
    include_volatility: bool = True       # 波動率特徵
    include_momentum: bool = True         # 動量特徵
    include_volume: bool = True           # 成交量特徵
    include_inventory_age: bool = True    # 庫存年齡（持倉多久）
    
    # 🆕 趨勢特徵 - 幫助模型適應不同市場狀態
    include_trend: bool = False           # 是否包含趨勢特徵
    trend_windows: List[int] = field(default_factory=lambda: DEFAULT_TREND_WINDOWS)
    
    # 波動率計算窗口 - 使用常數
    volatility_windows: List[int] = field(default_factory=lambda: DEFAULT_VOLATILITY_WINDOWS)
    # 動量計算窗口 - 使用常數
    momentum_windows: List[int] = field(default_factory=lambda: DEFAULT_MOMENTUM_WINDOWS)


@dataclass
class ActionConfig:
    """Action 空間配置"""
    mode: str = "asymmetric"              # "symmetric" | "asymmetric" | "discrete"
    allow_no_quote: bool = True           # 是否允許不報價
    max_spread_multiplier: float = 3.0    # 最大價差倍數
    min_spread_multiplier: float = 0.1    # 最小價差倍數


@dataclass
class DomainRandomizationConfig:
    """Domain Randomization 配置"""
    enabled: bool = False
    
    # 隨機化範圍
    fee_rate_range: Tuple[float, float] = (0.0003, 0.0005)
    base_spread_range: Tuple[float, float] = (15.0, 35.0)
    volatility_multiplier_range: Tuple[float, float] = (0.8, 1.2)
    fill_probability_noise: float = 0.1   # 成交機率的噪聲


@dataclass
class FillModelEnvConfig:
    """成交模型環境配置"""
    enabled: bool = False
    mode: str = "simple"  # "simple", "moderate", "realistic", "custom"
    
    # 自定義配置（當 mode="custom" 時使用）
    enable_queue_position: bool = True
    enable_slippage: bool = True
    slippage_bps: float = 1.0
    max_slippage_bps: float = 5.0         # 🆕 Added
    enable_market_impact: bool = False
    market_impact_factor: float = 0.1     # 🆕 Added
    enable_adverse_selection: bool = True
    adverse_selection_prob: float = 0.1
    adverse_selection_cost_bps: float = 2.0 # 🆕 Added
    
    # Queue dynamics
    queue_decay_rate: float = 0.1         # 🆕 Added
    initial_queue_position: float = 0.5   # 🆕 Added
    allow_partial_fills: bool = True      # 🆕 Added
    min_fill_ratio: float = 0.1           # 🆕 Added


@dataclass
class AdvancedObservationConfig:
    """進階觀察配置 - 擴展特徵"""
    # Order Flow 相關
    include_order_flow_imbalance: bool = False
    order_flow_window: int = 20
    
    # VWAP 相關
    include_vwap_deviation: bool = False
    vwap_window: int = 60
    
    # 多時間框架動量
    include_multi_timeframe_momentum: bool = False
    mtf_windows: List[int] = field(default_factory=lambda: [15, 60, 240])
    
    # 波動率預測（使用 EWMA）
    include_volatility_forecast: bool = False
    ewma_span: int = 20
    
    # 價格微結構
    include_microstructure: bool = False  # High-Low range, True range


@dataclass
class PerformanceConfig:
    """性能優化配置"""
    use_numba: bool = True                              # 使用 Numba JIT 加速（需要安裝 numba）
    precompute_features: bool = True                    # 預計算特徵（建議保持 True）
    feature_cache_size: int = DEFAULT_METRICS_BUFFER_SIZE  # 特徵緩存大小（MetricsTracker）


@dataclass
class FillResult:
    """成交結果"""
    filled: bool
    price: float
    size: float = 1.0
    slippage: float = 0.0
    is_adverse_selection: bool = False


# =============================================================================
# Metrics Tracker
# =============================================================================

class MetricsTracker:
    """追蹤完整的行為與風險指標
    
    🔧 記憶體優化: 使用固定大小緩衝區，避免無限增長
    """
    
    def __init__(self, max_buffer_size: int = 10000):
        """
        Args:
            max_buffer_size: 緩衝區最大大小（默認 10000 步）
                           超過後使用滾動緩衝區（保留最近的資料）
        """
        self.max_buffer_size = max_buffer_size
        self.reset()
    
    def reset(self):
        # 結果指標 - 使用固定大小緩衝區
        self.portfolio_values: List[float] = []
        self.returns: List[float] = []
        
        # 行為指標 - 使用固定大小緩衝區
        self.spreads: List[float] = []
        self.inventory_history: List[float] = []
        
        # 計數器（不佔太多記憶體）
        self.bid_fills: int = 0
        self.ask_fills: int = 0
        self.quote_count: int = 0
        self.no_quote_count: int = 0
        
        # 持有時間（只在平倉時記錄，數量有限）
        self.holding_times: List[int] = []
        
        # 風險指標
        self.max_inventory: float = 0.0
        self.time_at_max_inventory: int = 0
        
        # Drawdown - 使用固定大小緩衝區
        self.drawdowns: List[float] = []
        
        # 逆選擇追蹤
        self.adverse_selection_events: int = 0
        self.post_fill_returns: List[float] = []  # 固定大小緩衝區
        
        # 庫存管理
        self._inventory_entry_step: Dict[int, int] = {}
        self._position_counter: int = 0
        self._current_peak: float = 0.0
        
        # 記憶體使用統計
        self._buffer_overflow_count: int = 0
    
    def update(self, step: int, portfolio_value: float, inventory: float, 
               spread: float, bid_filled: bool, ask_filled: bool,
               quoted: bool, max_inventory: float, mid_price: float,
               prev_mid_price: float):
        """每步更新指標
        
        🔧 記憶體優化: 使用滾動緩衝區，避免記憶體無限增長
        """
        # 使用固定大小緩衝區（滾動更新）
        self._append_to_buffer(self.portfolio_values, portfolio_value)
        self._append_to_buffer(self.inventory_history, inventory)
        self._append_to_buffer(self.spreads, spread)
        
        if len(self.portfolio_values) > 1:
            ret = portfolio_value - self.portfolio_values[-2]
            self._append_to_buffer(self.returns, ret)
        
        if quoted:
            self.quote_count += 1
        else:
            self.no_quote_count += 1
        
        if bid_filled:
            self.bid_fills += 1
            self._position_counter += 1
            self._inventory_entry_step[self._position_counter] = step
            # 追蹤逆選擇
            if prev_mid_price > 0:
                price_change = (mid_price - prev_mid_price) / prev_mid_price
                self.post_fill_returns.append(-price_change)  # 買入後價格下跌是不利的
                if price_change < -0.0001:  # 價格下跌超過 0.01%
                    self.adverse_selection_events += 1
        
        if ask_filled:
            self.ask_fills += 1
            self._position_counter += 1
            self._inventory_entry_step[self._position_counter] = step
            if prev_mid_price > 0:
                price_change = (mid_price - prev_mid_price) / prev_mid_price
                self.post_fill_returns.append(price_change)  # 賣出後價格上漲是不利的
                if price_change > 0.0001:
                    self.adverse_selection_events += 1
        
        # 更新最大庫存
        abs_inv = abs(inventory)
        if abs_inv > self.max_inventory:
            self.max_inventory = abs_inv
        if abs_inv >= max_inventory * 0.9:  # 接近上限
            self.time_at_max_inventory += 1
        
        # 計算 Drawdown - 使用固定大小緩衝區
        if portfolio_value > self._current_peak:
            self._current_peak = portfolio_value
        if self._current_peak > 0:
            dd = (self._current_peak - portfolio_value) / self._current_peak
            self._append_to_buffer(self.drawdowns, dd)
    
    def _append_to_buffer(self, buffer: List[float], value: float):
        """添加值到固定大小緩衝區
        
        當緩衝區達到最大大小時，移除最舊的資料（FIFO）
        """
        buffer.append(value)
        if len(buffer) > self.max_buffer_size:
            buffer.pop(0)  # 移除最舊的資料
            self._buffer_overflow_count += 1
    
    def get_summary(self) -> Dict[str, float]:
        """取得指標摘要"""
        pv = np.array(self.portfolio_values) if self.portfolio_values else np.array([0.0])
        returns = np.array(self.returns) if self.returns else np.array([0.0])
        drawdowns = np.array(self.drawdowns) if self.drawdowns else np.array([0.0])
        
        total_fills = self.bid_fills + self.ask_fills
        total_quotes = self.quote_count + self.no_quote_count
        
        # 計算 VaR 和 ES
        if len(returns) > 10:
            var_95 = np.percentile(returns, 5)  # 5th percentile = 95% VaR
            es_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        else:
            var_95 = 0.0
            es_95 = 0.0
        
        return {
            # 結果指標
            "net_pnl": pv[-1] - pv[0] if len(pv) > 1 else 0.0,
            "sharpe": self._compute_sharpe(returns),
            "max_drawdown": float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0,
            "calmar_ratio": self._compute_calmar(pv, drawdowns),
            
            # 行為指標
            "avg_spread": float(np.mean(self.spreads)) if self.spreads else 0.0,
            "fill_rate": total_fills / max(total_quotes, 1),
            "bid_fill_rate": self.bid_fills / max(self.quote_count, 1),
            "ask_fill_rate": self.ask_fills / max(self.quote_count, 1),
            "quote_rate": self.quote_count / max(total_quotes, 1),
            "inventory_turnover": total_fills / max(len(self.inventory_history), 1),
            
            # 風險指標
            "var_95": var_95,
            "expected_shortfall_95": es_95,
            "max_inventory_reached": self.max_inventory,
            "time_at_max_inventory_pct": self.time_at_max_inventory / max(len(self.inventory_history), 1),
            
            # 逆選擇指標
            "adverse_selection_rate": self.adverse_selection_events / max(total_fills, 1),
            "avg_post_fill_return": float(np.mean(self.post_fill_returns)) if self.post_fill_returns else 0.0,
        }
    
    def _compute_sharpe(self, returns: np.ndarray, risk_free: float = 0.0) -> float:
        if len(returns) < 2:
            return 0.0
        excess = returns - risk_free
        std = np.std(excess)
        if std < 1e-8:
            return 0.0
        # 年化（假設每步 1 分鐘）
        annual_factor = np.sqrt(365 * 24 * 60)
        return float(np.mean(excess) / std * annual_factor)
    
    def _compute_calmar(self, pv: np.ndarray, drawdowns: np.ndarray) -> float:
        if len(pv) < 2 or len(drawdowns) == 0:
            return 0.0
        total_return = (pv[-1] - pv[0]) / max(pv[0], 1e-8)
        max_dd = np.max(drawdowns)
        if max_dd < 1e-8:
            return 0.0
        return float(total_return / max_dd)


# =============================================================================
# Main Environment Class
# =============================================================================

class MarketMakingEnvV2(gym.Env):
    """改良版做市環境"""
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        csv_path: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        episode_length: int = 1000,
        fee_rate: float = 0.0004,
        base_spread: float = 25.0,
        max_inventory: float = 5.0,
        initial_cash: float = 10000.0,
        random_start: bool = True,
        date_range: Optional[Tuple[str, str]] = None,
        seed: Optional[int] = None,
        # 新增配置
        reward_config: Optional[RewardConfig] = None,
        obs_config: Optional[ObservationConfig] = None,
        action_config: Optional[ActionConfig] = None,
        domain_rand_config: Optional[DomainRandomizationConfig] = None,
        # 新增：真實成交模型配置
        fill_model_config: Optional[FillModelEnvConfig] = None,
        # 新增：進階觀察配置
        advanced_obs_config: Optional[AdvancedObservationConfig] = None,
        # 新增：性能優化配置
        performance_config: Optional[PerformanceConfig] = None,
    ):
        super().__init__()
        
        # 驗證輸入
        if csv_path is None and df is None:
            raise ValueError("Must provide either csv_path or df")
        
        if csv_path is not None and df is not None:
            raise ValueError("Cannot provide both csv_path and df. Choose one.")
        
        # 驗證數值參數
        if max_inventory <= 0:
            raise ValueError(f"max_inventory must be positive, got {max_inventory}")
        
        if episode_length <= 0:
            raise ValueError(f"episode_length must be positive, got {episode_length}")
        
        if fee_rate < 0 or fee_rate > 0.1:
            raise ValueError(f"fee_rate must be in [0, 0.1], got {fee_rate}")
        
        if base_spread <= 0:
            raise ValueError(f"base_spread must be positive, got {base_spread}")
        
        if initial_cash <= 0:
            raise ValueError(f"initial_cash must be positive, got {initial_cash}")
        
        # 基本參數
        self.csv_path = csv_path
        self._input_df = df  # 直接傳入的 DataFrame
        self.episode_length = episode_length
        self.base_fee_rate = fee_rate
        self.base_spread = base_spread
        self.max_inventory = max_inventory
        self.initial_cash = initial_cash
        self.random_start = random_start
        self.date_range = date_range
        self._seed = seed
        
        # 配置
        self.reward_cfg = reward_config or RewardConfig()
        self.obs_cfg = obs_config or ObservationConfig()
        self.action_cfg = action_config or ActionConfig()
        self.dr_cfg = domain_rand_config or DomainRandomizationConfig()
        self.fill_model_cfg = fill_model_config or FillModelEnvConfig()
        self.adv_obs_cfg = advanced_obs_config or AdvancedObservationConfig()
        self.perf_cfg = performance_config or PerformanceConfig()
        
        # 檢查 Numba 可用性
        if self.perf_cfg.use_numba and not NUMBA_AVAILABLE:
            logger.warning("Numba requested but not available. Falling back to NumPy/Pandas.")
            self.perf_cfg.use_numba = False
        
        # 初始化真實成交模型
        self._init_fill_model()
        
        # 設定隨機種子
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # 載入資料
        self._load_data()
        
        # 驗證配置
        self._validate_configuration()
        
        # 預計算特徵
        self._precompute_features()
        
        # 預計算進階特徵
        self._precompute_advanced_features()
        
        # 定義空間
        self._setup_spaces()
        
        # 初始化狀態變數
        self._init_state()
        
        # 指標追蹤器 - 使用配置的緩衝區大小
        self.metrics = MetricsTracker(max_buffer_size=self.perf_cfg.feature_cache_size)
    
    def _validate_configuration(self):
        """驗證配置的合理性
        
        檢查配置參數是否在合理範圍內，並發出警告
        """
        # 檢查 reward_scale
        if self.reward_cfg.reward_scale > 1.0:
            logger.warning(f"reward_scale={self.reward_cfg.reward_scale} > 1.0")
            logger.warning(f"This may cause large rewards. Consider using 0.001-0.01")
        
        if self.reward_cfg.reward_scale < 1e-5:
            logger.warning(f"reward_scale={self.reward_cfg.reward_scale} < 1e-5")
            logger.warning(f"Rewards may be too small for learning")
        
        # 檢查 lambda_inventory
        if self.reward_cfg.lambda_inventory > 100:
            logger.warning(f"lambda_inventory={self.reward_cfg.lambda_inventory} is very large")
            logger.warning(f"This may cause excessive inventory penalty")
        
        # 檢查窗口大小
        max_window = max(
            max(self.obs_cfg.volatility_windows) if self.obs_cfg.volatility_windows else 0,
            max(self.obs_cfg.momentum_windows) if self.obs_cfg.momentum_windows else 0,
        )
        
        if max_window > self.episode_length:
            logger.warning(f"max feature window ({max_window}) > episode_length ({self.episode_length})")
            logger.warning(f"Features may not have enough data at episode start")
    
    def _init_fill_model(self):
        """初始化真實成交模型
        
        🔧 加入錯誤處理，確保配置正確
        """
        if not self.fill_model_cfg.enabled:
            self.fill_model = None
            return
        
        mode = self.fill_model_cfg.mode
        
        try:
            if mode == "simple":
                config = SIMPLE_FILL_CONFIG
            elif mode == "moderate":
                config = MODERATE_FILL_CONFIG
            elif mode == "realistic":
                config = REALISTIC_FILL_CONFIG
            elif mode == "custom":
                config = FillModelConfig(
                    enable_queue_position=self.fill_model_cfg.enable_queue_position,
                    queue_decay_rate=self.fill_model_cfg.queue_decay_rate,
                    
                    enable_partial_fills=self.fill_model_cfg.allow_partial_fills,
                    min_fill_ratio=self.fill_model_cfg.min_fill_ratio,
                    
                    enable_slippage=self.fill_model_cfg.enable_slippage,
                    slippage_bps=self.fill_model_cfg.slippage_bps,
                    
                    enable_market_impact=self.fill_model_cfg.enable_market_impact,
                    impact_coefficient=self.fill_model_cfg.market_impact_factor,
                    
                    enable_adverse_selection=self.fill_model_cfg.enable_adverse_selection,
                    adverse_selection_prob=self.fill_model_cfg.adverse_selection_prob,
                )
            else:
                raise ValueError(f"Unknown fill model mode: {mode}. "
                               f"Choose from: simple, moderate, realistic, custom")
            
            self.fill_model = RealisticFillModel(config)
            logger.info(f"Fill model initialized: {mode}")
            
        except Exception as e:
            logger.error(f"Failed to initialize fill model: {e}")
            logger.warning(f"Disabling fill model and using default fill logic")
            self.fill_model = None
            self.fill_model_cfg.enabled = False
    
    def _load_data(self):
        """載入並預處理資料
        
        🔧 加入錯誤處理和資料驗證
        """
        try:
            # 支援直接傳入 DataFrame 或從 CSV 載入
            if self._input_df is not None:
                self.df = self._input_df.copy()
            else:
                self.df = pd.read_csv(self.csv_path)
            
            if self.date_range is not None:
                self.df = self._slice_by_date_range(self.df, self.date_range)
            
            self.df.sort_values("timestamp", inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            
            # 驗證必要欄位
            if "close" not in self.df.columns:
                raise ValueError("CSV must contain 'close' column")
            
            # 檢查資料量
            if len(self.df) < self.episode_length:
                raise ValueError(
                    f"Data length ({len(self.df)}) < episode_length ({self.episode_length}). "
                    f"Need at least {self.episode_length} rows."
                )
            
            logger.info(f"Data loaded: {len(self.df):,} rows")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")
        self.df.reset_index(drop=True, inplace=True)
        
        if "close" not in self.df.columns:
            raise ValueError("CSV 缺少 close 欄位")
        
        # 轉為 numpy 加速
        self.closes = self.df["close"].to_numpy(dtype=np.float64)
        self.highs = self.df.get("high", self.df["close"]).to_numpy(dtype=np.float64)
        self.lows = self.df.get("low", self.df["close"]).to_numpy(dtype=np.float64)
        self.volumes = self.df.get("volume", pd.Series(np.zeros(len(self.df)))).to_numpy(dtype=np.float64)
        self.opens = self.df.get("open", self.df["close"]).to_numpy(dtype=np.float64)
        
        # 時間戳
        if "datetime" in self.df.columns:
            self.timestamps = pd.to_datetime(self.df["datetime"])
        elif "timestamp" in self.df.columns:
            self.timestamps = pd.to_datetime(self.df["timestamp"], unit="ms")
        else:
            self.timestamps = pd.Series(range(len(self.df)))
        
        self.data_len = len(self.closes)
    
    def _precompute_features(self):
        """預計算技術特徵（加速訓練）
        
        🔧 修正: 使用因果計算（Causal Computation）避免資料洩漏
        所有特徵計算只使用「當前時刻及之前」的資料
        
        ⚡ 性能優化: 使用 Numba JIT 加速（如果可用）
        """
        # 收益率
        self.returns = np.zeros(self.data_len)
        self.returns[1:] = (self.closes[1:] - self.closes[:-1]) / self.closes[:-1]
        
        # 波動率（滾動標準差）- 使用 Numba 或 Pandas
        self.volatilities = {}
        for window in self.obs_cfg.volatility_windows:
            if self.perf_cfg.use_numba:
                # 使用 Numba 加速（10-50x faster）
                vol = rolling_std_numba(self.returns, window) * np.sqrt(window)
            else:
                # Fallback to Pandas
                returns_series = pd.Series(self.returns)
                vol = returns_series.rolling(
                    window=window, 
                    min_periods=1
                ).std().shift(1).fillna(0).values * np.sqrt(window)
                
                # 第一個值特殊處理（沒有歷史）
                if len(vol) > 0:
                    vol[0] = 0.0
            
            self.volatilities[window] = vol
        
        # 動量 - 使用 Numba 或 Pandas
        self.momentums = {}
        for window in self.obs_cfg.momentum_windows:
            if self.perf_cfg.use_numba:
                # 使用 Numba 加速
                mom = compute_momentum_numba(self.closes, window)
            else:
                # Fallback to Pandas
                closes_series = pd.Series(self.closes)
                shifted = closes_series.shift(window)
                mom = ((closes_series - shifted) / shifted).fillna(0).values
                mom[:window] = 0.0
            
            self.momentums[window] = mom
        
        # Volume 特徵 - 使用 Numba 或 Pandas
        if self.perf_cfg.use_numba:
            self.volume_ma = rolling_mean_numba(self.volumes, 20)
        else:
            volumes_series = pd.Series(self.volumes)
            self.volume_ma = volumes_series.rolling(
                window=20, 
                min_periods=1
            ).mean().shift(1).fillna(0).values
            
            if len(self.volume_ma) > 0:
                self.volume_ma[0] = self.volumes[0] if len(self.volumes) > 0 else 0.0
        
        # 趨勢特徵
        if self.obs_cfg.include_trend:
            self.trend_sma = {}
            self.trend_direction = {}
            for window in self.obs_cfg.trend_windows:
                if self.perf_cfg.use_numba:
                    # 使用 Numba
                    sma = rolling_mean_numba(self.closes, window)
                else:
                    # Fallback to Pandas
                    closes_series = pd.Series(self.closes)
                    sma = closes_series.rolling(
                        window=window, 
                        min_periods=1
                    ).mean().shift(1).fillna(closes_series.iloc[0] if len(closes_series) > 0 else 0).values
                
                self.trend_sma[window] = sma
                
                # 趨勢方向: (當前價格 - 歷史SMA) / 歷史SMA
                direction = np.where(sma > 0, (self.closes - sma) / sma, 0)
                direction[:window] = 0.0  # 前 window 步設為 0
                
                self.trend_direction[window] = direction
    
    def _precompute_advanced_features(self):
        """預計算進階特徵
        
        🔧 修正: 因果計算，避免使用未來資料
        ⚡ 性能優化: 使用 Numba JIT 加速（如果可用）
        """
        # Order Flow Imbalance - 使用 Numba 或原生 Python
        if self.adv_obs_cfg.include_order_flow_imbalance:
            window = self.adv_obs_cfg.order_flow_window
            if self.perf_cfg.use_numba:
                # 使用 Numba 加速（10-50x faster）
                self.order_flow_imbalance = compute_order_flow_imbalance_numba(
                    self.volumes, self.returns, window
                )
            else:
                # Fallback to native Python
                self.order_flow_imbalance = np.zeros(self.data_len)
                for i in range(window, self.data_len):
                    # 只使用 [i-window, i) 的資料（不包含 i）
                    buy_vol = np.sum(self.volumes[i-window:i] * (self.returns[i-window:i] > 0))
                    sell_vol = np.sum(self.volumes[i-window:i] * (self.returns[i-window:i] < 0))
                    total_vol = buy_vol + sell_vol
                    if total_vol > 0:
                        self.order_flow_imbalance[i] = (buy_vol - sell_vol) / total_vol
        
        # VWAP 偏離 - 使用 Numba 或原生 Python
        if self.adv_obs_cfg.include_vwap_deviation:
            window = self.adv_obs_cfg.vwap_window
            if self.perf_cfg.use_numba:
                # 使用 Numba 加速
                self.vwap_deviation = compute_vwap_deviation_numba(
                    self.closes, self.volumes, window
                )
            else:
                # Fallback to native Python
                self.vwap_deviation = np.zeros(self.data_len)
                for i in range(window, self.data_len):
                    # 只使用歷史窗口 [i-window, i)
                    vol_window = self.volumes[i-window:i]
                    price_window = self.closes[i-window:i]
                    total_vol = np.sum(vol_window)
                    if total_vol > 0:
                        vwap = np.sum(vol_window * price_window) / total_vol
                        # 當前價格與歷史 VWAP 的偏離
                        self.vwap_deviation[i] = (self.closes[i] - vwap) / vwap if vwap > 0 else 0
        
        # 多時間框架動量 - 使用 Numba 或原生 Python
        if self.adv_obs_cfg.include_multi_timeframe_momentum:
            self.mtf_momentums = {}
            for window in self.adv_obs_cfg.mtf_windows:
                if self.perf_cfg.use_numba:
                    # 使用 Numba 加速
                    self.mtf_momentums[window] = compute_momentum_numba(self.closes, window)
                else:
                    # Fallback to native Python
                    mom = np.zeros(self.data_len)
                    for i in range(window, self.data_len):
                        # 當前價格 vs window 步之前的價格
                        if self.closes[i-window] > 0:
                            mom[i] = (self.closes[i] - self.closes[i-window]) / self.closes[i-window]
                    self.mtf_momentums[window] = mom
        
        # 波動率預測 (EWMA) - 已經是因果的（每步只依賴歷史）
        if self.adv_obs_cfg.include_volatility_forecast:
            span = self.adv_obs_cfg.ewma_span
            alpha = 2 / (span + 1)
            self.ewma_volatility = np.zeros(self.data_len)
            sq_returns = self.returns ** 2
            for i in range(1, self.data_len):
                # 使用上一步的 EWMA 和當前的平方收益率
                self.ewma_volatility[i] = alpha * sq_returns[i] + (1 - alpha) * self.ewma_volatility[i-1]
            self.ewma_volatility = np.sqrt(self.ewma_volatility)
        
        # 價格微結構 - 當前 K 線的 High-Low，這是當前資訊，可接受
        if self.adv_obs_cfg.include_microstructure:
            # High-Low Range (正規化) - 使用當前 K 線
            self.hl_range = np.zeros(self.data_len)
            for i in range(self.data_len):
                if self.closes[i] > 0:
                    self.hl_range[i] = (self.highs[i] - self.lows[i]) / self.closes[i]
            
            # True Range - 需要前一根 K 線的收盤價（因果）
            self.true_range = np.zeros(self.data_len)
            for i in range(1, self.data_len):
                tr1 = self.highs[i] - self.lows[i]
                tr2 = abs(self.highs[i] - self.closes[i-1])  # 使用前一根收盤
                tr3 = abs(self.lows[i] - self.closes[i-1])   # 使用前一根收盤
                if self.closes[i-1] > 0:
                    self.true_range[i] = max(tr1, tr2, tr3) / self.closes[i-1]
    
    def _setup_spaces(self):
        """設定 observation 和 action 空間"""
        # 計算 observation 維度
        obs_dim = 0
        if self.obs_cfg.include_price:
            obs_dim += 1  # normalized mid price
        if self.obs_cfg.include_inventory:
            obs_dim += 1  # normalized inventory
        if self.obs_cfg.include_time:
            obs_dim += 3  # time_frac, sin(time), cos(time)
        if self.obs_cfg.include_volatility:
            obs_dim += len(self.obs_cfg.volatility_windows)
        if self.obs_cfg.include_momentum:
            obs_dim += len(self.obs_cfg.momentum_windows)
        if self.obs_cfg.include_volume:
            obs_dim += 2  # log_volume, volume_ratio
        if self.obs_cfg.include_inventory_age:
            obs_dim += 1  # normalized inventory age
        
        # 🆕 趨勢特徵維度
        if self.obs_cfg.include_trend:
            obs_dim += len(self.obs_cfg.trend_windows)
        
        # 進階特徵維度
        if self.adv_obs_cfg.include_order_flow_imbalance:
            obs_dim += 1
        if self.adv_obs_cfg.include_vwap_deviation:
            obs_dim += 1
        if self.adv_obs_cfg.include_multi_timeframe_momentum:
            obs_dim += len(self.adv_obs_cfg.mtf_windows)
        if self.adv_obs_cfg.include_volatility_forecast:
            obs_dim += 1
        if self.adv_obs_cfg.include_microstructure:
            obs_dim += 2  # hl_range, true_range
        
        self.obs_dim = obs_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action 空間
        if self.action_cfg.mode == "symmetric":
            # [spread_action, skew_action]
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(2,), dtype=np.float32
            )
        elif self.action_cfg.mode == "asymmetric":
            # [bid_spread, ask_spread, quote_flag]
            # quote_flag: < 0 = no quote, >= 0 = quote
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(3,), dtype=np.float32
            )
        else:  # discrete
            # 離散動作空間
            self.action_space = gym.spaces.Discrete(7)
    
    def _init_state(self):
        """初始化狀態變數"""
        self.current_step = 0
        self.t = 0
        self.inventory = 0.0
        self.cash = self.initial_cash
        self.last_pv = self.initial_cash
        self.init_mid = 0.0
        self.mid = 0.0
        self.prev_mid = 0.0
        
        # 庫存年齡追蹤
        self.inventory_age = 0  # 當前倉位持有多久
        self.last_inventory = 0.0
        
        # Reward shaping 的 potential
        self.last_potential = 0.0
        
        # 累積統計
        self.cum_gross_pnl = 0.0
        self.cum_fees = 0.0
        
        # Domain Randomization 的實際參數
        self.effective_fee_rate = self.base_fee_rate
        self.effective_base_spread = self.base_spread
        self.volatility_multiplier = 1.0
    
    def _apply_domain_randomization(self):
        """應用 Domain Randomization"""
        if not self.dr_cfg.enabled:
            self.effective_fee_rate = self.base_fee_rate
            self.effective_base_spread = self.base_spread
            self.volatility_multiplier = 1.0
            return
        
        self.effective_fee_rate = np.random.uniform(*self.dr_cfg.fee_rate_range)
        self.effective_base_spread = np.random.uniform(*self.dr_cfg.base_spread_range)
        self.volatility_multiplier = np.random.uniform(*self.dr_cfg.volatility_multiplier_range)
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Domain Randomization
        self._apply_domain_randomization()
        
        # 選擇起始位置
        max_start = self.data_len - self.episode_length - 1
        if max_start <= 0:
            raise ValueError("資料不足以支援 episode_length")
        
        if self.random_start:
            # 確保有足夠的歷史資料計算特徵
            min_start = max(self.obs_cfg.volatility_windows) if self.obs_cfg.volatility_windows else 60
            self.current_step = random.randint(min_start, max_start)
        else:
            self.current_step = max(self.obs_cfg.volatility_windows) if self.obs_cfg.volatility_windows else 60
        
        self.t = 0
        self.inventory = 0.0
        self.cash = self.initial_cash
        self.init_mid = float(self.closes[self.current_step])
        self.mid = self.init_mid
        self.prev_mid = self.mid
        self.last_pv = self.initial_cash
        
        self.inventory_age = 0
        self.last_inventory = 0.0
        
        # 初始化 potential
        self.last_potential = self._compute_potential()
        
        self.cum_gross_pnl = 0.0
        self.cum_fees = 0.0
        
        # 重置指標
        self.metrics.reset()
        
        # 重置真實成交模型
        if self.fill_model is not None:
            self.fill_model.reset()
        
        obs = self._get_obs()
        info = {"domain_rand": {
            "fee_rate": self.effective_fee_rate,
            "base_spread": self.effective_base_spread,
        }}
        
        return obs, info
    
    def _get_obs(self) -> np.ndarray:
        """建構 observation 向量"""
        obs = []
        idx = self.current_step
        
        if self.obs_cfg.include_price:
            mid_norm = (self.mid / self.init_mid) - 1.0
            obs.append(mid_norm)
        
        if self.obs_cfg.include_inventory:
            inv_norm = self.inventory / self.max_inventory
            obs.append(inv_norm)
        
        if self.obs_cfg.include_time:
            time_frac = self.t / self.episode_length
            # 時間的週期性編碼（假設資料是分鐘級）
            minutes_in_day = 24 * 60
            time_of_day = (idx % minutes_in_day) / minutes_in_day
            obs.extend([
                time_frac,
                np.sin(2 * np.pi * time_of_day),
                np.cos(2 * np.pi * time_of_day),
            ])
        
        if self.obs_cfg.include_volatility:
            for window in self.obs_cfg.volatility_windows:
                vol = self.volatilities[window][idx] * self.volatility_multiplier
                # 標準化波動率
                obs.append(np.clip(vol * 100, -5, 5))  # 轉為百分比並裁剪
        
        if self.obs_cfg.include_momentum:
            for window in self.obs_cfg.momentum_windows:
                mom = self.momentums[window][idx]
                obs.append(np.clip(mom * 100, -5, 5))
        
        if self.obs_cfg.include_volume:
            vol_log = np.log1p(self.volumes[idx]) / 10.0
            vol_ratio = self.volumes[idx] / max(self.volume_ma[idx], 1.0) - 1.0
            obs.extend([vol_log, np.clip(vol_ratio, -3, 3)])
        
        if self.obs_cfg.include_inventory_age:
            # 標準化庫存年齡（以 episode 長度為基準）
            age_norm = min(self.inventory_age / 100.0, 1.0)
            obs.append(age_norm)
        
        # 🆕 趨勢特徵 - 幫助模型適應不同市場狀態
        if self.obs_cfg.include_trend:
            for window in self.obs_cfg.trend_windows:
                if hasattr(self, 'trend_direction') and window in self.trend_direction:
                    trend = self.trend_direction[window][idx]
                else:
                    trend = 0.0
                # 趨勢方向: 正值=上漲, 負值=下跌, 0=盤整
                obs.append(np.clip(trend * 100, -10, 10))
        
        # ===== 進階特徵 =====
        if self.adv_obs_cfg.include_order_flow_imbalance:
            ofi = self.order_flow_imbalance[idx] if hasattr(self, 'order_flow_imbalance') else 0.0
            obs.append(np.clip(ofi, -1, 1))
        
        if self.adv_obs_cfg.include_vwap_deviation:
            vwap_dev = self.vwap_deviation[idx] if hasattr(self, 'vwap_deviation') else 0.0
            obs.append(np.clip(vwap_dev * 100, -5, 5))
        
        if self.adv_obs_cfg.include_multi_timeframe_momentum:
            for window in self.adv_obs_cfg.mtf_windows:
                if hasattr(self, 'mtf_momentums') and window in self.mtf_momentums:
                    mom = self.mtf_momentums[window][idx]
                else:
                    mom = 0.0
                obs.append(np.clip(mom * 100, -10, 10))
        
        if self.adv_obs_cfg.include_volatility_forecast:
            ewma_vol = self.ewma_volatility[idx] if hasattr(self, 'ewma_volatility') else 0.0
            obs.append(np.clip(ewma_vol * 100, 0, 10))
        
        if self.adv_obs_cfg.include_microstructure:
            hl = self.hl_range[idx] if hasattr(self, 'hl_range') else 0.0
            tr = self.true_range[idx] if hasattr(self, 'true_range') else 0.0
            obs.extend([np.clip(hl * 100, 0, 5), np.clip(tr * 100, 0, 5)])
        
        return np.array(obs, dtype=np.float32)
    
    def _compute_potential(self) -> float:
        """計算 Potential Function（用於 reward shaping）
        
        改進: 使用非線性懲罰，接近庫存限制時急劇增加
        """
        inv_ratio = abs(self.inventory) / self.max_inventory
        
        # 基礎二次懲罰 + 四次項（接近限制時急劇增加）
        base_penalty = self.inventory ** 2
        limit_penalty = 10.0 * (inv_ratio ** 4) * self.max_inventory ** 2
        
        return -self.reward_cfg.lambda_inventory * (base_penalty + limit_penalty)
    
    def _parse_action(self, action: np.ndarray) -> Tuple[float, float, bool]:
        """解析 action，回傳 (bid_spread, ask_spread, should_quote)"""
        if self.action_cfg.mode == "symmetric":
            a_spread, a_skew = action
            base = self.effective_base_spread
            spread = base * (1.0 + a_spread * (self.action_cfg.max_spread_multiplier - 1))
            spread = max(spread, base * self.action_cfg.min_spread_multiplier)
            
            skew = a_skew * 0.5  # skew 範圍 [-0.5, 0.5]
            bid_spread = spread * (1.0 - skew)
            ask_spread = spread * (1.0 + skew)
            should_quote = True
            
        elif self.action_cfg.mode == "asymmetric":
            a_bid, a_ask, a_quote = action
            base = self.effective_base_spread
            
            # 各自獨立的價差控制
            bid_spread = base * (self.action_cfg.min_spread_multiplier + 
                                 (a_bid + 1) / 2 * (self.action_cfg.max_spread_multiplier - self.action_cfg.min_spread_multiplier))
            ask_spread = base * (self.action_cfg.min_spread_multiplier + 
                                 (a_ask + 1) / 2 * (self.action_cfg.max_spread_multiplier - self.action_cfg.min_spread_multiplier))
            
            should_quote = a_quote >= 0 or not self.action_cfg.allow_no_quote
            
        else:  # discrete
            # 離散動作空間
            discrete_actions = {
                0: (1.0, 1.0, True),    # neutral
                1: (0.5, 0.5, True),    # aggressive (tight spread)
                2: (2.0, 2.0, True),    # defensive (wide spread)
                3: (0.7, 1.3, True),    # skew buy
                4: (1.3, 0.7, True),    # skew sell
                5: (0.3, 0.3, True),    # very aggressive
                6: (1.0, 1.0, False),   # no quote
            }
            mult_bid, mult_ask, should_quote = discrete_actions.get(int(action), (1.0, 1.0, True))
            base = self.effective_base_spread
            bid_spread = base * mult_bid
            ask_spread = base * mult_ask
        
        return bid_spread, ask_spread, should_quote
    
    def _simulate_fill(self, side: str, price: float, mid: float, extreme: float) -> FillResult:
        """模擬成交 - 支援真實成交模型"""
        # 如果啟用真實成交模型
        if self.fill_model is not None:
            idx = self.current_step
            # 建構市場數據
            market_data = FillMarketData(
                mid_price=mid,
                bid_price=mid - self.effective_base_spread / 2,
                ask_price=mid + self.effective_base_spread / 2,
                spread=self.effective_base_spread,
                volume=self.volumes[idx] if idx < len(self.volumes) else 1000,
                volatility=self.volatilities[5][idx] if 5 in self.volatilities else 0.01,
                momentum=self.momentums[5][idx] if 5 in self.momentums else 0.0,
            )
            
            result = self.fill_model.simulate_fill(
                side=side,
                price=price,
                quantity=1.0,
                mid_price=mid,
                market_data=market_data,
            )
            
            # 掛單超出 K 線範圍，降低成交機率（額外檢查）
            if side == "bid" and price < extreme:
                if not (np.random.rand() < 0.1):  # 90% 機率不成交
                    return FillResult(filled=False, price=price)
            if side == "ask" and price > extreme:
                if not (np.random.rand() < 0.1):
                    return FillResult(filled=False, price=price)
            
            return FillResult(
                filled=result.filled,
                price=result.fill_price if result.filled else price,
                slippage=result.slippage,
                is_adverse_selection=result.is_adverse_selection,
            )
        
        # 原始簡化模型
        depth = abs(mid - price)
        k = 1.0 / max(self.effective_base_spread, 1e-6)
        p_fill = math.exp(-k * depth)
        
        # Domain Randomization: 加入噪聲
        if self.dr_cfg.enabled:
            noise = np.random.uniform(-self.dr_cfg.fill_probability_noise, 
                                       self.dr_cfg.fill_probability_noise)
            p_fill = np.clip(p_fill + noise, 0, 1)
        
        # 掛單超出 K 線範圍，降低成交機率
        if side == "bid" and price < extreme:
            p_fill *= 0.1
        if side == "ask" and price > extreme:
            p_fill *= 0.1
        
        filled = np.random.rand() < p_fill
        return FillResult(filled=filled, price=price)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.t += 1
        self.prev_mid = self.mid
        
        # 解析動作
        action = np.clip(action, -1.0, 1.0) if isinstance(action, np.ndarray) else action
        bid_spread, ask_spread, should_quote = self._parse_action(action)
        
        # 取得市場資料
        mid = float(self.closes[self.current_step])
        high = float(self.highs[self.current_step])
        low = float(self.lows[self.current_step])
        
        # 計算報價
        bid = mid - bid_spread
        ask = mid + ask_spread
        
        # 執行交易
        bid_filled = False
        ask_filled = False
        fee_t = 0.0
        trades_count = 0
        
        if should_quote:
            fill_bid = self._simulate_fill("bid", bid, mid, low)
            fill_ask = self._simulate_fill("ask", ask, mid, high)
            
            if fill_bid.filled and self.inventory + 1 <= self.max_inventory:
                self.inventory += 1
                self.cash -= fill_bid.price
                fee = self.effective_fee_rate * abs(fill_bid.price)
                self.cash -= fee
                fee_t += fee
                trades_count += 1
                bid_filled = True
            
            if fill_ask.filled and self.inventory - 1 >= -self.max_inventory:
                self.inventory -= 1
                self.cash += fill_ask.price
                fee = self.effective_fee_rate * abs(fill_ask.price)
                self.cash -= fee
                fee_t += fee
                trades_count += 1
                ask_filled = True
        
        # 更新庫存年齡
        if self.inventory != 0:
            if self.last_inventory == 0:
                self.inventory_age = 1
            elif np.sign(self.inventory) == np.sign(self.last_inventory):
                self.inventory_age += 1
            else:
                self.inventory_age = 1
        else:
            self.inventory_age = 0
        self.last_inventory = self.inventory
        
        # 更新步數和價格
        self.current_step += 1
        terminated = self.current_step >= self.data_len - 1 or self.t >= self.episode_length
        self.mid = float(self.closes[self.current_step])
        
        # 計算 Portfolio Value
        portfolio_value = self.cash + self.inventory * self.mid
        delta_pnl = portfolio_value - self.last_pv
        gross_pnl = delta_pnl + fee_t
        
        self.cum_gross_pnl += gross_pnl
        self.cum_fees += fee_t
        
        # 計算 Reward
        reward = self._compute_reward(delta_pnl, fee_t, trades_count, terminated, portfolio_value)
        
        self.last_pv = portfolio_value
        
        # 更新指標
        self.metrics.update(
            step=self.t,
            portfolio_value=portfolio_value,
            inventory=self.inventory,
            spread=(bid_spread + ask_spread) / 2,
            bid_filled=bid_filled,
            ask_filled=ask_filled,
            quoted=should_quote,
            max_inventory=self.max_inventory,
            mid_price=self.mid,
            prev_mid_price=self.prev_mid,
        )
        
        # 建構 info
        obs = self._get_obs()
        info = {
            "portfolio_value": portfolio_value,
            "inventory": self.inventory,
            "cash": self.cash,
            "spread": (bid_spread + ask_spread) / 2,
            "bid_spread": bid_spread,
            "ask_spread": ask_spread,
            "trades_count": trades_count,
            "quoted": should_quote,
            "step": self.t,
        }
        
        if terminated:
            info["episode_gross_pnl"] = self.cum_gross_pnl
            info["episode_fees"] = self.cum_fees
            info["episode_net_pnl"] = portfolio_value - self.initial_cash
            info["metrics"] = self.metrics.get_summary()
        
        return obs, reward, terminated, False, info
    
    def _compute_reward(self, delta_pnl: float, fee: float, trades: int, 
                        terminated: bool, portfolio_value: float) -> float:
        """根據配置計算 reward
        
        🆕 v3: 增加 Reward 範圍驗證與日誌
        """
        mode = self.reward_cfg.mode
        scale = self.reward_cfg.reward_scale  # 🆕 v3: 獎勵縮放
        
        # Debug: 追蹤 reward 組件 (每 100 步記錄一次)
        _debug_reward = hasattr(self, '_reward_debug_counter')
        if not _debug_reward:
            self._reward_debug_counter = 0
            self._reward_components = []
        
        if mode == RewardMode.DENSE:
            # 傳統方式：即時 reward
            penalty = self.reward_cfg.lambda_inventory * abs(self.inventory)
            turnover_penalty = self.reward_cfg.lambda_turnover * trades
            raw_reward = delta_pnl - penalty - turnover_penalty
            scaled_reward = raw_reward * scale
            
            # Debug logging
            if self._reward_debug_counter % 100 == 0:
                self._reward_components.append({
                    'step': self.t,
                    'mode': 'dense',
                    'raw': raw_reward,
                    'scaled': scaled_reward,
                    'delta_pnl': delta_pnl,
                    'penalty': penalty
                })
            
            return scaled_reward
        
        elif mode == RewardMode.SPARSE:
            # 只在結束時給 reward
            if terminated:
                total_pnl = portfolio_value - self.initial_cash
                raw_reward = total_pnl * self.reward_cfg.sparse_scale
                scaled_reward = raw_reward * scale
                
                self._reward_components.append({
                    'step': self.t,
                    'mode': 'sparse',
                    'raw': raw_reward,
                    'scaled': scaled_reward,
                    'total_pnl': total_pnl
                })
                
                return scaled_reward
            return 0.0
        
        elif mode == RewardMode.SHAPED:
            # Potential-based reward shaping
            current_potential = self._compute_potential()
            shaping = self.reward_cfg.gamma * current_potential - self.last_potential
            self.last_potential = current_potential
            
            # 基礎 reward 是淨損益
            base_reward = delta_pnl - self.reward_cfg.lambda_turnover * trades
            
            # === 庫存回歸獎勵 ===
            revert_bonus = 0.0
            if abs(self.inventory) < abs(self.last_inventory):
                revert_bonus = self.reward_cfg.inventory_revert_bonus if self.reward_cfg.inventory_revert_bonus > 0 else 0.5
            
            # === 庫存方向警告懲罰 ===
            direction_penalty = 0.0
            inv_ratio = abs(self.inventory) / self.max_inventory
            if inv_ratio > 0.6:
                direction_penalty = -0.2 * inv_ratio
            
            # === Spread 捕獲獎勵 ===
            spread_bonus = 0.0
            if self.reward_cfg.spread_capture_bonus > 0 and trades > 0:
                if delta_pnl > 0:
                    spread_bonus = self.reward_cfg.spread_capture_bonus * delta_pnl
            
            # === Round-trip 獎勵 ===
            round_trip_bonus = 0.0
            if self.reward_cfg.round_trip_bonus > 0:
                if abs(self.last_inventory) > 0 and abs(self.inventory) == 0:
                    round_trip_bonus = self.reward_cfg.round_trip_bonus
            
            raw_reward = base_reward + shaping + revert_bonus + direction_penalty + spread_bonus + round_trip_bonus
            scaled_reward = raw_reward * scale
            
            # Debug logging (每 100 步)
            if self._reward_debug_counter % 100 == 0:
                self._reward_components.append({
                    'step': self.t,
                    'mode': 'shaped',
                    'raw': raw_reward,
                    'scaled': scaled_reward,
                    'base': base_reward,
                    'shaping': shaping,
                    'revert': revert_bonus,
                    'direction_pen': direction_penalty,
                    'spread': spread_bonus,
                    'round_trip': round_trip_bonus,
                    'scale_factor': scale
                })
            
            self._reward_debug_counter += 1
            
            # 範圍驗證 (警告異常值)
            if abs(scaled_reward) > 100:
                import warnings
                warnings.warn(f"⚠️  Reward 異常: {scaled_reward:.2f} (raw={raw_reward:.2f}, scale={scale})")
                
                # [新增] 強制截斷獎勵，保護訓練穩定性
                # 將獎勵限制在 [-10, 10] 之間，避免梯度爆炸
                clip_value = 10.0
                scaled_reward = max(min(scaled_reward, clip_value), -clip_value)
            
            return scaled_reward
        
        else:  # HYBRID
            current_potential = self._compute_potential()
            shaping = self.reward_cfg.gamma * current_potential - self.last_potential
            self.last_potential = current_potential
            
            base_reward = delta_pnl - self.reward_cfg.lambda_turnover * trades + shaping
            
            if terminated:
                total_pnl = portfolio_value - self.initial_cash
                terminal_bonus = total_pnl * self.reward_cfg.sparse_scale * self.reward_cfg.terminal_bonus_weight
                raw_reward = base_reward + terminal_bonus
                scaled_reward = raw_reward * scale
                
                self._reward_components.append({
                    'step': self.t,
                    'mode': 'hybrid_terminal',
                    'raw': raw_reward,
                    'scaled': scaled_reward,
                    'base': base_reward,
                    'terminal': terminal_bonus
                })
                
                return scaled_reward
            
            scaled_reward = base_reward * scale
            
            if self._reward_debug_counter % 100 == 0:
                self._reward_components.append({
                    'step': self.t,
                    'mode': 'hybrid',
                    'raw': base_reward,
                    'scaled': scaled_reward
                })
            
            self._reward_debug_counter += 1
            return scaled_reward
    
    def render(self):
        pv = self.cash + self.inventory * self.mid
        logger.info(f"Step {self.t}: mid={self.mid:.2f}, inv={self.inventory:.2f}, "
              f"cash={self.cash:.2f}, PV={pv:.2f}")
    
    def _slice_by_date_range(self, df: pd.DataFrame, date_range: Tuple[str, str]) -> pd.DataFrame:
        """按日期範圍切割資料"""
        start, end = date_range
        if start is None and end is None:
            return df
        
        if "datetime" in df.columns:
            dt_series = pd.to_datetime(df["datetime"])
        elif "timestamp" in df.columns:
            dt_series = pd.to_datetime(df["timestamp"], unit="ms")
        else:
            raise ValueError("缺少 datetime/timestamp 欄位")
        
        tz = getattr(dt_series.dt, "tz", None)
        mask = pd.Series(True, index=df.index)
        
        if start:
            s_ts = pd.to_datetime(start)
            if tz and s_ts.tzinfo is None:
                s_ts = s_ts.tz_localize(tz)
            mask &= dt_series >= s_ts
        
        if end:
            e_ts = pd.to_datetime(end)
            if tz and e_ts.tzinfo is None:
                e_ts = e_ts.tz_localize(tz)
            mask &= dt_series <= e_ts
        
        sliced = df.loc[mask].copy()
        if sliced.empty:
            raise ValueError("日期範圍切割後無資料")
        return sliced


# =============================================================================
# Factory Function
# =============================================================================

def create_env_v2(
    csv_path: str,
    env_config: Optional[Dict[str, Any]] = None,
    reward_mode: str = "shaped",
    action_mode: str = "asymmetric",
    enable_domain_rand: bool = False,
    **kwargs
) -> MarketMakingEnvV2:
    """便利的環境建構函式"""
    
    reward_cfg = RewardConfig(mode=RewardMode(reward_mode))
    obs_cfg = ObservationConfig()
    action_cfg = ActionConfig(mode=action_mode)
    dr_cfg = DomainRandomizationConfig(enabled=enable_domain_rand)
    
    # 從 env_config 覆寫
    if env_config:
        for key, value in env_config.items():
            if hasattr(reward_cfg, key):
                setattr(reward_cfg, key, value)
            elif hasattr(obs_cfg, key):
                setattr(obs_cfg, key, value)
            elif hasattr(action_cfg, key):
                setattr(action_cfg, key, value)
            elif hasattr(dr_cfg, key):
                setattr(dr_cfg, key, value)
    
    return MarketMakingEnvV2(
        csv_path=csv_path,
        reward_config=reward_cfg,
        obs_config=obs_cfg,
        action_config=action_cfg,
        domain_rand_config=dr_cfg,
        **kwargs
    )
