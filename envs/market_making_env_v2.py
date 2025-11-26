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
    
    # 波動率計算窗口
    volatility_windows: List[int] = field(default_factory=lambda: [5, 15, 60])
    # 動量計算窗口
    momentum_windows: List[int] = field(default_factory=lambda: [5, 15])


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
class FillResult:
    """成交結果"""
    filled: bool
    price: float
    size: float = 1.0


# =============================================================================
# Metrics Tracker
# =============================================================================

class MetricsTracker:
    """追蹤完整的行為與風險指標"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        # 結果指標
        self.portfolio_values: List[float] = []
        self.returns: List[float] = []
        
        # 行為指標
        self.spreads: List[float] = []
        self.bid_fills: int = 0
        self.ask_fills: int = 0
        self.quote_count: int = 0
        self.no_quote_count: int = 0
        self.inventory_history: List[float] = []
        self.holding_times: List[int] = []  # 每筆部位的持有時間
        
        # 風險指標
        self.max_inventory: float = 0.0
        self.time_at_max_inventory: int = 0
        self.drawdowns: List[float] = []
        
        # 逆選擇追蹤
        self.adverse_selection_events: int = 0  # 成交後價格不利變動次數
        self.post_fill_returns: List[float] = []  # 成交後的價格變動
        
        # 庫存管理
        self._inventory_entry_step: Dict[int, int] = {}  # position_id -> entry_step
        self._position_counter: int = 0
        self._current_peak: float = 0.0
    
    def update(self, step: int, portfolio_value: float, inventory: float, 
               spread: float, bid_filled: bool, ask_filled: bool,
               quoted: bool, max_inventory: float, mid_price: float,
               prev_mid_price: float):
        """每步更新指標"""
        self.portfolio_values.append(portfolio_value)
        self.inventory_history.append(inventory)
        self.spreads.append(spread)
        
        if len(self.portfolio_values) > 1:
            ret = portfolio_value - self.portfolio_values[-2]
            self.returns.append(ret)
        
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
        
        # 計算 Drawdown
        if portfolio_value > self._current_peak:
            self._current_peak = portfolio_value
        if self._current_peak > 0:
            dd = (self._current_peak - portfolio_value) / self._current_peak
            self.drawdowns.append(dd)
    
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
        csv_path: str,
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
    ):
        super().__init__()
        
        # 基本參數
        self.csv_path = csv_path
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
        
        # 設定隨機種子
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # 載入資料
        self._load_data()
        
        # 預計算特徵
        self._precompute_features()
        
        # 定義空間
        self._setup_spaces()
        
        # 初始化狀態變數
        self._init_state()
        
        # 指標追蹤器
        self.metrics = MetricsTracker()
    
    def _load_data(self):
        """載入並預處理資料"""
        self.df = pd.read_csv(self.csv_path)
        
        if self.date_range is not None:
            self.df = self._slice_by_date_range(self.df, self.date_range)
        
        self.df.sort_values("timestamp", inplace=True)
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
        """預計算技術特徵（加速訓練）"""
        # 收益率
        self.returns = np.zeros(self.data_len)
        self.returns[1:] = (self.closes[1:] - self.closes[:-1]) / self.closes[:-1]
        
        # 波動率（滾動標準差）
        self.volatilities = {}
        for window in self.obs_cfg.volatility_windows:
            vol = np.zeros(self.data_len)
            for i in range(window, self.data_len):
                vol[i] = np.std(self.returns[i-window:i]) * np.sqrt(window)
            self.volatilities[window] = vol
        
        # 動量
        self.momentums = {}
        for window in self.obs_cfg.momentum_windows:
            mom = np.zeros(self.data_len)
            for i in range(window, self.data_len):
                mom[i] = (self.closes[i] - self.closes[i-window]) / self.closes[i-window]
            self.momentums[window] = mom
        
        # Volume 特徵
        self.volume_ma = np.zeros(self.data_len)
        window = 20
        for i in range(window, self.data_len):
            self.volume_ma[i] = np.mean(self.volumes[i-window:i])
    
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
        
        return np.array(obs, dtype=np.float32)
    
    def _compute_potential(self) -> float:
        """計算 Potential Function（用於 reward shaping）"""
        # Potential = -lambda * inventory^2
        # 這是一個凸函數，鼓勵庫存接近 0
        return -self.reward_cfg.lambda_inventory * (self.inventory ** 2)
    
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
        """模擬成交"""
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
        """根據配置計算 reward"""
        mode = self.reward_cfg.mode
        
        if mode == RewardMode.DENSE:
            # 傳統方式：即時 reward
            penalty = self.reward_cfg.lambda_inventory * abs(self.inventory)
            turnover_penalty = self.reward_cfg.lambda_turnover * trades
            return delta_pnl - penalty - turnover_penalty
        
        elif mode == RewardMode.SPARSE:
            # 只在結束時給 reward
            if terminated:
                total_pnl = portfolio_value - self.initial_cash
                return total_pnl * self.reward_cfg.sparse_scale
            return 0.0
        
        elif mode == RewardMode.SHAPED:
            # Potential-based reward shaping
            # F(s, a, s') = gamma * phi(s') - phi(s)
            # 這保證不會改變最優策略（Ng et al., 1999）
            current_potential = self._compute_potential()
            shaping = self.reward_cfg.gamma * current_potential - self.last_potential
            self.last_potential = current_potential
            
            # 基礎 reward 是淨損益
            base_reward = delta_pnl - self.reward_cfg.lambda_turnover * trades
            return base_reward + shaping
        
        else:  # HYBRID
            # 混合：shaped + terminal bonus
            current_potential = self._compute_potential()
            shaping = self.reward_cfg.gamma * current_potential - self.last_potential
            self.last_potential = current_potential
            
            base_reward = delta_pnl - self.reward_cfg.lambda_turnover * trades + shaping
            
            if terminated:
                total_pnl = portfolio_value - self.initial_cash
                terminal_bonus = total_pnl * self.reward_cfg.sparse_scale * self.reward_cfg.terminal_bonus_weight
                return base_reward + terminal_bonus
            
            return base_reward
    
    def render(self):
        pv = self.cash + self.inventory * self.mid
        print(f"Step {self.t}: mid={self.mid:.2f}, inv={self.inventory:.2f}, "
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
