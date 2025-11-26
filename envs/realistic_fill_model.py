"""
envs/realistic_fill_model.py
真實成交模型

模擬真實市場的成交機制:
- Queue Position: 排隊位置影響成交優先權
- Partial Fills: 部分成交
- Slippage: 滑點
- Market Impact: 市場衝擊
- Latency: 延遲

用法:
    from envs.realistic_fill_model import RealisticFillModel, FillModelConfig
    
    config = FillModelConfig(
        enable_queue_position=True,
        enable_partial_fills=True,
        slippage_bps=1.0,
    )
    fill_model = RealisticFillModel(config)
    
    result = fill_model.simulate_fill(
        side="bid",
        price=50000.0,
        quantity=1.0,
        mid_price=50010.0,
        market_data=market_data,
    )
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Enums and Data Classes
# =============================================================================

class OrderSide(Enum):
    BID = "bid"
    ASK = "ask"


@dataclass
class FillModelConfig:
    """成交模型配置"""
    # 排隊位置
    enable_queue_position: bool = True
    avg_queue_depth: float = 100.0       # 平均排隊深度（手）
    queue_decay_rate: float = 0.1        # 排隊位置衰減率
    
    # 部分成交
    enable_partial_fills: bool = True
    min_fill_ratio: float = 0.1          # 最小成交比例
    
    # 滑點
    enable_slippage: bool = True
    slippage_bps: float = 1.0            # 基點滑點 (1 bps = 0.01%)
    slippage_volatility_mult: float = 2.0 # 波動率對滑點的放大係數
    
    # 市場衝擊
    enable_market_impact: bool = True
    impact_coefficient: float = 0.1      # 衝擊係數
    impact_decay_steps: int = 10         # 衝擊衰減步數
    
    # 延遲
    enable_latency: bool = False
    latency_mean_ms: float = 10.0        # 平均延遲（毫秒）
    latency_std_ms: float = 5.0          # 延遲標準差
    
    # 對手方行為
    enable_adverse_selection: bool = True
    adverse_selection_prob: float = 0.1  # 逆選擇機率
    
    # 隨機性
    random_seed: Optional[int] = None


@dataclass
class MarketData:
    """市場數據"""
    mid_price: float
    bid_price: float
    ask_price: float
    spread: float
    volume: float
    volatility: float = 0.01
    momentum: float = 0.0
    # 可選的訂單簿數據
    bid_depth: float = 0.0
    ask_depth: float = 0.0


@dataclass
class FillResult:
    """成交結果"""
    filled: bool
    filled_quantity: float
    fill_price: float
    slippage: float = 0.0
    queue_position: float = 0.0
    market_impact: float = 0.0
    latency_ms: float = 0.0
    is_adverse_selection: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "filled": self.filled,
            "filled_quantity": self.filled_quantity,
            "fill_price": self.fill_price,
            "slippage": self.slippage,
            "queue_position": self.queue_position,
            "market_impact": self.market_impact,
            "latency_ms": self.latency_ms,
            "is_adverse_selection": self.is_adverse_selection,
        }


# =============================================================================
# Realistic Fill Model
# =============================================================================

class RealisticFillModel:
    """真實成交模型"""
    
    def __init__(self, config: FillModelConfig = None):
        self.config = config or FillModelConfig()
        
        # 設定隨機種子
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        # 市場衝擊累積
        self.impact_history: List[Tuple[int, float, str]] = []  # (step, impact, side)
        self.current_step = 0
        
        # 排隊位置追蹤
        self.queue_positions: Dict[str, float] = {"bid": 0.5, "ask": 0.5}
    
    def reset(self):
        """重置模型狀態"""
        self.impact_history = []
        self.current_step = 0
        self.queue_positions = {"bid": 0.5, "ask": 0.5}
    
    def step(self):
        """時間步進"""
        self.current_step += 1
        self._update_queue_positions()
        self._decay_market_impact()
    
    def simulate_fill(
        self,
        side: str,
        price: float,
        quantity: float,
        mid_price: float,
        market_data: MarketData = None,
    ) -> FillResult:
        """
        模擬成交
        
        Args:
            side: "bid" 或 "ask"
            price: 掛單價格
            quantity: 掛單數量
            mid_price: 中間價
            market_data: 市場數據
        
        Returns:
            成交結果
        """
        # 預設市場數據
        if market_data is None:
            market_data = MarketData(
                mid_price=mid_price,
                bid_price=mid_price - 10,
                ask_price=mid_price + 10,
                spread=20,
                volume=1000,
            )
        
        # 計算基礎成交機率
        base_fill_prob = self._compute_base_fill_probability(
            side, price, mid_price, market_data
        )
        
        # 排隊位置調整
        queue_adj = 1.0
        if self.config.enable_queue_position:
            queue_adj = self._compute_queue_adjustment(side, market_data)
        
        # 計算最終成交機率
        fill_prob = base_fill_prob * queue_adj
        
        # 檢查逆選擇
        is_adverse = False
        if self.config.enable_adverse_selection:
            is_adverse = self._check_adverse_selection(side, market_data)
            if is_adverse:
                fill_prob *= 1.5  # 逆選擇時更容易成交（對手想成交）
        
        # 決定是否成交
        filled = np.random.random() < fill_prob
        
        if not filled:
            return FillResult(
                filled=False,
                filled_quantity=0.0,
                fill_price=0.0,
            )
        
        # 計算成交數量（部分成交）
        filled_qty = quantity
        if self.config.enable_partial_fills:
            filled_qty = self._compute_partial_fill(quantity, market_data)
        
        # 計算滑點
        slippage = 0.0
        if self.config.enable_slippage:
            slippage = self._compute_slippage(side, quantity, market_data)
        
        # 計算成交價格
        fill_price = price
        if side == "bid":
            fill_price = price + slippage  # 買入時滑點使價格上升
        else:
            fill_price = price - slippage  # 賣出時滑點使價格下降
        
        # 計算市場衝擊
        impact = 0.0
        if self.config.enable_market_impact:
            impact = self._compute_and_record_impact(side, filled_qty, market_data)
        
        # 計算延遲
        latency = 0.0
        if self.config.enable_latency:
            latency = self._compute_latency()
        
        return FillResult(
            filled=True,
            filled_quantity=filled_qty,
            fill_price=fill_price,
            slippage=slippage,
            queue_position=self.queue_positions.get(side, 0.5),
            market_impact=impact,
            latency_ms=latency,
            is_adverse_selection=is_adverse,
        )
    
    def _compute_base_fill_probability(
        self,
        side: str,
        price: float,
        mid_price: float,
        market_data: MarketData,
    ) -> float:
        """計算基礎成交機率"""
        # 價格距離
        if side == "bid":
            distance = mid_price - price
        else:
            distance = price - mid_price
        
        # 距離越近，成交機率越高
        # 使用指數衰減模型
        half_spread = market_data.spread / 2
        k = 1.0 / max(half_spread, 1e-6)
        
        # 基礎機率
        if distance <= 0:
            # 價格穿過中間價
            base_prob = 0.95
        else:
            base_prob = np.exp(-k * distance)
        
        # 根據成交量調整
        volume_factor = min(market_data.volume / 1000, 1.5)
        
        return base_prob * volume_factor
    
    def _compute_queue_adjustment(
        self,
        side: str,
        market_data: MarketData,
    ) -> float:
        """計算排隊位置調整因子"""
        # 排隊位置 [0, 1]，0 = 隊首，1 = 隊尾
        position = self.queue_positions.get(side, 0.5)
        
        # 使用深度資訊調整
        if side == "bid" and market_data.bid_depth > 0:
            depth = market_data.bid_depth
        elif side == "ask" and market_data.ask_depth > 0:
            depth = market_data.ask_depth
        else:
            depth = self.config.avg_queue_depth
        
        # 排隊位置越前，調整因子越大
        # 使用線性模型：隊首 = 1.0，隊尾 = 0.2
        adjustment = 1.0 - 0.8 * position
        
        # 深度越大，競爭越激烈
        depth_penalty = max(1.0 - depth / (self.config.avg_queue_depth * 2), 0.3)
        
        return adjustment * depth_penalty
    
    def _update_queue_positions(self):
        """更新排隊位置（每步自然前移）"""
        decay = self.config.queue_decay_rate
        for side in ["bid", "ask"]:
            # 位置逐漸前移
            self.queue_positions[side] = max(
                0.0,
                self.queue_positions[side] - decay
            )
    
    def _check_adverse_selection(
        self,
        side: str,
        market_data: MarketData,
    ) -> bool:
        """檢查是否發生逆選擇"""
        if np.random.random() > self.config.adverse_selection_prob:
            return False
        
        # 動量方向與交易方向相反時，更可能是逆選擇
        # 買入時價格正在下跌 = 逆選擇
        # 賣出時價格正在上漲 = 逆選擇
        if side == "bid" and market_data.momentum < -0.001:
            return True
        if side == "ask" and market_data.momentum > 0.001:
            return True
        
        return False
    
    def _compute_partial_fill(
        self,
        quantity: float,
        market_data: MarketData,
    ) -> float:
        """計算部分成交數量"""
        # 根據市場成交量決定成交比例
        volume_ratio = market_data.volume / 1000
        
        # 基礎成交比例
        fill_ratio = min(1.0, volume_ratio + np.random.uniform(0.2, 0.5))
        fill_ratio = max(fill_ratio, self.config.min_fill_ratio)
        
        # 隨機因素
        fill_ratio *= np.random.uniform(0.8, 1.0)
        
        return quantity * fill_ratio
    
    def _compute_slippage(
        self,
        side: str,
        quantity: float,
        market_data: MarketData,
    ) -> float:
        """計算滑點"""
        base_slippage = self.config.slippage_bps * 0.0001 * market_data.mid_price
        
        # 波動率調整
        volatility_adj = 1.0 + market_data.volatility * self.config.slippage_volatility_mult
        
        # 數量調整（大單滑點更大）
        quantity_adj = 1.0 + np.log1p(quantity) * 0.1
        
        # 累積市場衝擊的影響
        total_impact = self._get_current_impact()
        impact_adj = 1.0 + abs(total_impact) * 0.5
        
        slippage = base_slippage * volatility_adj * quantity_adj * impact_adj
        
        # 加入隨機因素
        slippage *= np.random.uniform(0.8, 1.2)
        
        return slippage
    
    def _compute_and_record_impact(
        self,
        side: str,
        quantity: float,
        market_data: MarketData,
    ) -> float:
        """計算並記錄市場衝擊"""
        # 衝擊大小與數量的平方根成正比
        impact = self.config.impact_coefficient * np.sqrt(quantity)
        
        # 方向
        if side == "bid":
            impact = impact  # 買入推升價格
        else:
            impact = -impact  # 賣出壓低價格
        
        # 記錄衝擊
        self.impact_history.append((self.current_step, impact, side))
        
        return impact
    
    def _decay_market_impact(self):
        """衰減市場衝擊"""
        # 移除過期的衝擊
        cutoff_step = self.current_step - self.config.impact_decay_steps
        self.impact_history = [
            (step, impact, side)
            for step, impact, side in self.impact_history
            if step > cutoff_step
        ]
    
    def _get_current_impact(self) -> float:
        """取得當前累積的市場衝擊"""
        if not self.impact_history:
            return 0.0
        
        total_impact = 0.0
        for step, impact, _ in self.impact_history:
            # 時間衰減
            age = self.current_step - step
            decay = np.exp(-age / self.config.impact_decay_steps)
            total_impact += impact * decay
        
        return total_impact
    
    def _compute_latency(self) -> float:
        """計算延遲"""
        latency = np.random.normal(
            self.config.latency_mean_ms,
            self.config.latency_std_ms
        )
        return max(0.0, latency)
    
    def register_order(self, side: str):
        """
        註冊新訂單（設定初始排隊位置）
        
        新訂單從隊尾開始
        """
        self.queue_positions[side] = 1.0


# =============================================================================
# Fill Model Wrapper for Environment
# =============================================================================

class FillModelWrapper:
    """
    成交模型包裝器
    
    用於整合到 MarketMakingEnv
    """
    
    def __init__(self, config: FillModelConfig = None):
        self.model = RealisticFillModel(config)
        self.last_fill_results: Dict[str, FillResult] = {}
    
    def reset(self):
        """重置"""
        self.model.reset()
        self.last_fill_results = {}
    
    def step(self):
        """時間步進"""
        self.model.step()
    
    def simulate_fills(
        self,
        bid_price: float,
        ask_price: float,
        bid_quantity: float,
        ask_quantity: float,
        mid_price: float,
        market_data: MarketData = None,
    ) -> Tuple[FillResult, FillResult]:
        """
        模擬 bid 和 ask 的成交
        
        Returns:
            (bid_result, ask_result)
        """
        # Bid 成交
        if bid_quantity > 0:
            bid_result = self.model.simulate_fill(
                side="bid",
                price=bid_price,
                quantity=bid_quantity,
                mid_price=mid_price,
                market_data=market_data,
            )
        else:
            bid_result = FillResult(filled=False, filled_quantity=0.0, fill_price=0.0)
        
        # Ask 成交
        if ask_quantity > 0:
            ask_result = self.model.simulate_fill(
                side="ask",
                price=ask_price,
                quantity=ask_quantity,
                mid_price=mid_price,
                market_data=market_data,
            )
        else:
            ask_result = FillResult(filled=False, filled_quantity=0.0, fill_price=0.0)
        
        self.last_fill_results = {"bid": bid_result, "ask": ask_result}
        
        return bid_result, ask_result
    
    def get_fill_stats(self) -> Dict[str, Any]:
        """取得成交統計"""
        return {
            "current_impact": self.model._get_current_impact(),
            "bid_queue_position": self.model.queue_positions.get("bid", 0.5),
            "ask_queue_position": self.model.queue_positions.get("ask", 0.5),
        }


# =============================================================================
# 預設配置
# =============================================================================

# 簡化模型（接近原始環境）
SIMPLE_FILL_CONFIG = FillModelConfig(
    enable_queue_position=False,
    enable_partial_fills=False,
    enable_slippage=False,
    enable_market_impact=False,
    enable_latency=False,
    enable_adverse_selection=False,
)

# 中等真實度
MODERATE_FILL_CONFIG = FillModelConfig(
    enable_queue_position=True,
    enable_partial_fills=False,
    enable_slippage=True,
    slippage_bps=0.5,
    enable_market_impact=False,
    enable_latency=False,
    enable_adverse_selection=True,
    adverse_selection_prob=0.05,
)

# 高度真實
REALISTIC_FILL_CONFIG = FillModelConfig(
    enable_queue_position=True,
    avg_queue_depth=100.0,
    enable_partial_fills=True,
    min_fill_ratio=0.3,
    enable_slippage=True,
    slippage_bps=1.0,
    enable_market_impact=True,
    impact_coefficient=0.05,
    enable_latency=True,
    latency_mean_ms=10.0,
    enable_adverse_selection=True,
    adverse_selection_prob=0.1,
)
