"""HistoricalMarketMakingEnv: 使用歷史 1m K 線資料的做市強化學習環境。
此環境目標是讓 agent 學習調整 spread 與 skew，以控制 inventory 並賺取 ΔPnL。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd


def _safe_log_transform(volume: float) -> float:
    """對成交量做 log 標準化，避免極端值。"""
    return math.log1p(max(volume, 0.0)) / 10.0


@dataclass
class FillResult:
    filled: bool
    price: float


class HistoricalMarketMakingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        csv_path: str,
        episode_length: int = 1000,
        fee_rate: float = 0.0004,
        lambda_inv: float = 0.001,
        lambda_turnover: float = 0.0,
        max_inventory: float = 10.0,
        base_spread: float = 0.2,
        alpha: float = 1.0,
        beta: float = 0.5,
        random_start: bool = True,
        date_range: tuple[str | None, str | None] | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.episode_length = episode_length
        self.fee_rate = fee_rate
        self.lambda_inv = lambda_inv
        # lambda_turnover 用來抑制過度刷單的策略，避免手續費壘高
        self.lambda_turnover = lambda_turnover
        self.max_inventory = max_inventory
        self.base_spread = base_spread
        self.alpha = alpha
        self.beta = beta
        self.random_start = random_start
        self.date_range = date_range

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.df = pd.read_csv(csv_path)
        if date_range is not None:
            self.df = self._slice_by_date_range(self.df, date_range)
        self.df.sort_values("timestamp", inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        if "close" not in self.df.columns:
            raise ValueError("CSV 檔缺少 close 欄位，無法建立環境。")
        
        # 優化：將 DataFrame 轉為 Numpy Array 以加速存取
        self.closes = self.df["close"].to_numpy(dtype=np.float32)
        self.highs = (
            self.df["high"].to_numpy(dtype=np.float32)
            if "high" in self.df.columns
            else self.closes
        )
        self.lows = (
            self.df["low"].to_numpy(dtype=np.float32)
            if "low" in self.df.columns
            else self.closes
        )
        self.volumes = (
            self.df["volume"].to_numpy(dtype=np.float32)
            if "volume" in self.df.columns
            else np.zeros_like(self.closes)
        )
        # 若有 datetime 則保留供 render 使用，否則用 index
        self.timestamps = (
            self.df["datetime"].to_numpy()
            if "datetime" in self.df.columns
            else self.df.index.to_numpy()
        )
        self.data_len = len(self.closes)

        self.init_mid = float(self.closes[0])

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32,
        )

        self.current_step = 0
        self.t = 0
        self.inventory = 0.0
        self.cash = 0.0
        self.last_portfolio_value = 0.0
        self.mid = self.init_mid

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        mid = float(self.closes[self.current_step])
        volume = float(self.volumes[self.current_step])
        mid_norm = (mid / self.init_mid) - 1.0
        inventory_norm = self.inventory / self.max_inventory
        volume_norm = _safe_log_transform(volume)
        time_frac = self.t / self.episode_length
        return np.array([mid_norm, inventory_norm, volume_norm, time_frac], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        max_start = self.data_len - self.episode_length - 1
        if max_start <= 0:
            raise ValueError("資料筆數不足以支援 episode_length。")
        if self.random_start:
            self.current_step = random.randint(0, max_start)
        else:
            self.current_step = 0

        self.t = 0
        self.inventory = 0.0
        self.cash = 0.0
        self.last_portfolio_value = 0.0
        self.mid = float(self.closes[self.current_step])

        obs = self._get_obs()
        return obs, {}

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.t += 1
        action = np.clip(action, -1.0, 1.0)
        a_spread, a_skew = action

        mid = float(self.closes[self.current_step])
        high = float(self.highs[self.current_step])
        low = float(self.lows[self.current_step])

        # spread 越小越容易成交，但會犧牲利潤；這裡用 action 線性調整
        spread = max(self.base_spread * (1.0 + self.alpha * a_spread), 0.01)
        # skew 決定偏多還是偏空：正值代表賣方更積極
        skew = self.beta * a_skew
        bid = mid - spread * (1.0 - skew)
        ask = mid + spread * (1.0 + skew)
        bid = min(bid, mid - 0.0001)
        ask = max(ask, mid + 0.0001)

        fill_bid = self._simulate_fill(side="bid", price=bid, mid=mid, extreme=low)
        fill_ask = self._simulate_fill(side="ask", price=ask, mid=mid, extreme=high)
        inventory_change = 0
        trades_count = 0

        # 限制 inventory，避免 agent 無限制累積倉位造成爆倉
        if fill_bid.filled and self.inventory + 1 <= self.max_inventory:
            self.inventory += 1
            self.cash -= fill_bid.price
            self.cash -= self.fee_rate * abs(fill_bid.price)
            inventory_change += 1
            trades_count += 1

        if fill_ask.filled and self.inventory - 1 >= -self.max_inventory:
            self.inventory -= 1
            self.cash += fill_ask.price
            self.cash -= self.fee_rate * abs(fill_ask.price)
            inventory_change -= 1
            trades_count += 1

        self.current_step += 1
        terminated = False
        if self.current_step >= self.data_len - 1 or self.t >= self.episode_length:
            terminated = True
        self.mid = float(self.closes[self.current_step])

        portfolio_value = self.cash + self.inventory * self.mid
        # ΔPnL 代表當前步驟的損益變化，扣除 inventory 懲罰可鼓勵 agent 控制倉位
        delta_pnl = portfolio_value - self.last_portfolio_value
        reward = delta_pnl - self.lambda_inv * abs(self.inventory) - self.lambda_turnover * trades_count
        self.last_portfolio_value = portfolio_value

        obs = self._get_obs()
        info = {
            "portfolio_value": portfolio_value,
            "inventory": self.inventory,
            "cash": self.cash,
            "spread": spread,
            "inventory_change": inventory_change,
            "step": self.t,
            "trades_count": trades_count,
        }
        return obs, reward, terminated, False, info

    # ------------------------------------------------------------------
    def _simulate_fill(self, side: str, price: float, mid: float, extreme: float) -> FillResult:
        """使用簡化的指數衰減模型估計成交機率。"""
        depth = abs(mid - price)
        k = 1.0 / max(self.base_spread, 1e-6)
        p_fill = math.exp(-k * depth)

        # 若掛單超出當前 K 線範圍，視為較難成交
        if side == "bid" and price < extreme:
            p_fill *= 0.1
        if side == "ask" and price > extreme:
            p_fill *= 0.1

        filled = np.random.rand() < p_fill
        return FillResult(filled=filled, price=price)

    def render(self) -> None:
        timestamp = self.timestamps[self.current_step]
        portfolio_value = self.cash + self.inventory * self.mid
        print(
            f"step={self.t}, ts={timestamp}, mid={self.mid:.2f}, "
            f"inventory={self.inventory:.2f}, cash={self.cash:.2f}, PV={portfolio_value:.2f}"
        )

    def _slice_by_date_range(
        self,
        df: pd.DataFrame,
        date_range: tuple[str | None, str | None],
    ) -> pd.DataFrame:
        start, end = date_range
        if start is None and end is None:
            return df
        if "datetime" in df.columns:
            dt_series = pd.to_datetime(df["datetime"])
        elif "timestamp" in df.columns:
            dt_series = pd.to_datetime(df["timestamp"], unit="ms")
        else:
            raise ValueError("CSV 缺少 datetime/timestamp 欄位，無法依日期切割。")
        mask = pd.Series(True, index=df.index)
        if start:
            mask &= dt_series >= pd.to_datetime(start)
        if end:
            mask &= dt_series <= pd.to_datetime(end)
        sliced = df.loc[mask].copy()
        if sliced.empty:
            raise ValueError("日期範圍過窄，切割後沒有資料。")
        return sliced
