"""
utils/curriculum.py
課程學習 (Curriculum Learning) 模組

實作漸進式難度訓練:
- Stage 1: 簡單環境 (低波動、寬價差容忍)
- Stage 2: 中等難度
- Stage 3: 真實市場難度

用法:
    from utils.curriculum import CurriculumScheduler, CurriculumCallback
    
    scheduler = CurriculumScheduler(stages=[...])
    callback = CurriculumCallback(scheduler, env)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


# =============================================================================
# Curriculum Stage Definition
# =============================================================================

@dataclass
class CurriculumStage:
    """課程學習階段定義"""
    name: str
    # 環境參數覆寫
    env_params: Dict[str, Any] = field(default_factory=dict)
    # 進入下一階段的條件
    advancement_threshold: float = 0.0  # 平均獎勵閾值
    min_episodes: int = 100  # 最少經歷的 episode 數
    # 可選的獎勵縮放
    reward_scale: float = 1.0


# =============================================================================
# Predefined Curriculum Stages
# =============================================================================

# 做市策略的預設課程
DEFAULT_MARKET_MAKING_CURRICULUM = [
    CurriculumStage(
        name="easy",
        env_params={
            "fee_rate": 0.0002,           # 低手續費
            "base_spread": 40.0,          # 寬價差（容易賺錢）
            "max_inventory": 3.0,         # 低庫存風險
            "volatility_multiplier": 0.5, # 低波動
        },
        advancement_threshold=50.0,       # 平均獎勵 > 50 進入下一階段
        min_episodes=50,
        reward_scale=1.0,
    ),
    CurriculumStage(
        name="medium",
        env_params={
            "fee_rate": 0.0003,
            "base_spread": 30.0,
            "max_inventory": 5.0,
            "volatility_multiplier": 0.8,
        },
        advancement_threshold=30.0,
        min_episodes=100,
        reward_scale=1.0,
    ),
    CurriculumStage(
        name="hard",
        env_params={
            "fee_rate": 0.0004,
            "base_spread": 25.0,
            "max_inventory": 10.0,
            "volatility_multiplier": 1.0,
        },
        advancement_threshold=0.0,  # 最終階段，不需要進階條件
        min_episodes=0,
        reward_scale=1.0,
    ),
]

# 更激進的課程（快速進階）
AGGRESSIVE_CURRICULUM = [
    CurriculumStage(
        name="warmup",
        env_params={
            "fee_rate": 0.0001,
            "base_spread": 50.0,
            "max_inventory": 2.0,
        },
        advancement_threshold=100.0,
        min_episodes=30,
    ),
    CurriculumStage(
        name="normal",
        env_params={
            "fee_rate": 0.0004,
            "base_spread": 25.0,
            "max_inventory": 10.0,
        },
        advancement_threshold=0.0,
        min_episodes=0,
    ),
]


# =============================================================================
# Curriculum Scheduler
# =============================================================================

class CurriculumScheduler:
    """課程學習調度器"""
    
    def __init__(
        self,
        stages: List[CurriculumStage] = None,
        reward_window: int = 100,
        verbose: int = 1,
    ):
        """
        Args:
            stages: 課程階段列表
            reward_window: 計算平均獎勵的窗口大小
            verbose: 輸出詳細程度
        """
        self.stages = stages or DEFAULT_MARKET_MAKING_CURRICULUM
        self.reward_window = reward_window
        self.verbose = verbose
        
        self.current_stage_idx = 0
        self.episode_rewards: List[float] = []
        self.episodes_in_current_stage = 0
        self.stage_history: List[Dict[str, Any]] = []
    
    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self.current_stage_idx]
    
    @property
    def is_final_stage(self) -> bool:
        return self.current_stage_idx >= len(self.stages) - 1
    
    def record_episode(self, reward: float) -> bool:
        """
        記錄 episode 結果並檢查是否進階
        
        Returns:
            是否進入下一階段
        """
        self.episode_rewards.append(reward)
        self.episodes_in_current_stage += 1
        
        # 檢查是否滿足進階條件
        if self.should_advance():
            self._advance_stage()
            return True
        
        return False
    
    def should_advance(self) -> bool:
        """檢查是否應該進入下一階段"""
        if self.is_final_stage:
            return False
        
        stage = self.current_stage
        
        # 檢查最少 episode 數
        if self.episodes_in_current_stage < stage.min_episodes:
            return False
        
        # 檢查平均獎勵閾值
        if len(self.episode_rewards) < self.reward_window:
            return False
        
        recent_rewards = self.episode_rewards[-self.reward_window:]
        avg_reward = np.mean(recent_rewards)
        
        return avg_reward >= stage.advancement_threshold
    
    def _advance_stage(self):
        """進入下一階段"""
        old_stage = self.current_stage
        
        # 記錄舊階段統計
        self.stage_history.append({
            "stage_name": old_stage.name,
            "episodes": self.episodes_in_current_stage,
            "final_avg_reward": np.mean(self.episode_rewards[-self.reward_window:])
            if len(self.episode_rewards) >= self.reward_window else np.mean(self.episode_rewards),
        })
        
        # 前進
        self.current_stage_idx += 1
        self.episodes_in_current_stage = 0
        
        if self.verbose >= 1:
            new_stage = self.current_stage
            print(f"\n🎓 [Curriculum] Advanced: {old_stage.name} → {new_stage.name}")
            print(f"   Params: {new_stage.env_params}")
    
    def get_env_params(self) -> Dict[str, Any]:
        """取得當前階段的環境參數"""
        return dict(self.current_stage.env_params)
    
    def get_reward_scale(self) -> float:
        """取得當前階段的獎勵縮放"""
        return self.current_stage.reward_scale
    
    def reset(self):
        """重置調度器"""
        self.current_stage_idx = 0
        self.episode_rewards = []
        self.episodes_in_current_stage = 0
        self.stage_history = []
    
    def get_progress(self) -> Dict[str, Any]:
        """取得課程進度"""
        return {
            "current_stage": self.current_stage.name,
            "stage_index": self.current_stage_idx,
            "total_stages": len(self.stages),
            "episodes_in_stage": self.episodes_in_current_stage,
            "min_episodes_required": self.current_stage.min_episodes,
            "advancement_threshold": self.current_stage.advancement_threshold,
            "recent_avg_reward": np.mean(self.episode_rewards[-self.reward_window:])
            if len(self.episode_rewards) >= self.reward_window else None,
        }


# =============================================================================
# Curriculum Environment Wrapper
# =============================================================================

class CurriculumEnvWrapper(gym.Wrapper):
    """
    課程學習環境包裝器
    
    根據 CurriculumScheduler 動態調整環境參數
    """
    
    def __init__(
        self,
        env: gym.Env,
        scheduler: CurriculumScheduler,
        update_on_reset: bool = True,
    ):
        """
        Args:
            env: 原始環境
            scheduler: 課程調度器
            update_on_reset: 是否在 reset 時更新環境參數
        """
        super().__init__(env)
        self.scheduler = scheduler
        self.update_on_reset = update_on_reset
        self.episode_reward = 0.0
    
    def reset(self, **kwargs) -> tuple:
        self.episode_reward = 0.0
        
        if self.update_on_reset:
            self._apply_stage_params()
        
        return self.env.reset(**kwargs)
    
    def step(self, action) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 縮放獎勵
        scaled_reward = reward * self.scheduler.get_reward_scale()
        self.episode_reward += reward  # 記錄原始獎勵
        
        # Episode 結束時更新調度器
        if terminated or truncated:
            advanced = self.scheduler.record_episode(self.episode_reward)
            info["curriculum_advanced"] = advanced
            info["curriculum_stage"] = self.scheduler.current_stage.name
        
        return obs, scaled_reward, terminated, truncated, info
    
    def _apply_stage_params(self):
        """應用當前階段的環境參數"""
        params = self.scheduler.get_env_params()
        
        for key, value in params.items():
            if hasattr(self.env, key):
                setattr(self.env, key, value)
            elif hasattr(self.env.unwrapped, key):
                setattr(self.env.unwrapped, key, value)


# =============================================================================
# Curriculum Callback
# =============================================================================

class CurriculumCallback(BaseCallback):
    """課程學習回調"""
    
    def __init__(
        self,
        scheduler: CurriculumScheduler,
        env: gym.Env = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.scheduler = scheduler
        self.env = env
    
    def _on_step(self) -> bool:
        # 檢查 episode 結束
        if self.locals.get("dones", [False])[0]:
            infos = self.locals.get("infos", [{}])
            if len(infos) > 0 and "episode" in infos[0]:
                ep_reward = infos[0]["episode"]["r"]
                advanced = self.scheduler.record_episode(ep_reward)
                
                # 如果進階，更新環境參數
                if advanced and self.env is not None:
                    self._apply_stage_params()
        
        return True
    
    def _apply_stage_params(self):
        """應用新階段的環境參數"""
        params = self.scheduler.get_env_params()
        env = self.env or self.training_env
        
        if env is None:
            return
        
        # 嘗試設定環境參數
        try:
            for key, value in params.items():
                if hasattr(env, "set_attr"):
                    env.set_attr(key, value)
                elif hasattr(env.unwrapped, key):
                    setattr(env.unwrapped, key, value)
        except Exception as e:
            if self.verbose >= 1:
                print(f"[Curriculum] Warning: Failed to update env params: {e}")


# =============================================================================
# Automatic Curriculum Generator
# =============================================================================

class AutoCurriculumGenerator:
    """
    自動課程生成器
    
    根據環境參數範圍自動生成漸進式課程
    """
    
    def __init__(
        self,
        param_ranges: Dict[str, tuple],
        n_stages: int = 3,
        difficulty_order: Dict[str, str] = None,
    ):
        """
        Args:
            param_ranges: 參數範圍 {"param_name": (easy_value, hard_value)}
            n_stages: 階段數量
            difficulty_order: 難度排序 {"param_name": "asc" 或 "desc"}
                              "asc": 值越大越難
                              "desc": 值越小越難
        """
        self.param_ranges = param_ranges
        self.n_stages = n_stages
        self.difficulty_order = difficulty_order or {}
    
    def generate(
        self,
        base_threshold: float = 50.0,
        threshold_decay: float = 0.6,
        base_min_episodes: int = 50,
    ) -> List[CurriculumStage]:
        """
        生成課程階段
        
        Args:
            base_threshold: 基礎進階閾值
            threshold_decay: 閾值衰減率
            base_min_episodes: 基礎最少 episode 數
        
        Returns:
            課程階段列表
        """
        stages = []
        
        for i in range(self.n_stages):
            progress = i / max(self.n_stages - 1, 1)
            
            # 計算每個參數在當前階段的值
            env_params = {}
            for param, (easy_val, hard_val) in self.param_ranges.items():
                order = self.difficulty_order.get(param, "asc")
                
                if order == "desc":
                    # 值越小越難，從 easy 到 hard
                    value = easy_val + (hard_val - easy_val) * progress
                else:
                    # 值越大越難，從 easy 到 hard
                    value = easy_val + (hard_val - easy_val) * progress
                
                env_params[param] = value
            
            # 計算進階閾值（越後面的階段閾值越低）
            threshold = base_threshold * (threshold_decay ** i) if i < self.n_stages - 1 else 0.0
            
            stage = CurriculumStage(
                name=f"stage_{i+1}",
                env_params=env_params,
                advancement_threshold=threshold,
                min_episodes=base_min_episodes if i < self.n_stages - 1 else 0,
            )
            stages.append(stage)
        
        return stages


# =============================================================================
# 便利函數
# =============================================================================

def create_curriculum_env(
    base_env_factory: Callable[[], gym.Env],
    stages: List[CurriculumStage] = None,
    **scheduler_kwargs,
) -> tuple:
    """
    便利函數：建立課程學習環境
    
    Args:
        base_env_factory: 建立基礎環境的函數
        stages: 課程階段列表
        **scheduler_kwargs: 傳給 CurriculumScheduler 的參數
    
    Returns:
        (wrapped_env, scheduler)
    """
    scheduler = CurriculumScheduler(stages=stages, **scheduler_kwargs)
    env = base_env_factory()
    wrapped_env = CurriculumEnvWrapper(env, scheduler)
    
    return wrapped_env, scheduler


def create_market_making_curriculum(
    difficulty: str = "normal",
) -> List[CurriculumStage]:
    """
    建立做市策略的課程
    
    Args:
        difficulty: 難度等級 ("easy", "normal", "aggressive")
    
    Returns:
        課程階段列表
    """
    if difficulty == "easy":
        return [
            CurriculumStage(
                name="very_easy",
                env_params={"fee_rate": 0.0001, "base_spread": 60.0, "max_inventory": 2.0},
                advancement_threshold=100.0,
                min_episodes=30,
            ),
            CurriculumStage(
                name="easy",
                env_params={"fee_rate": 0.0002, "base_spread": 40.0, "max_inventory": 5.0},
                advancement_threshold=50.0,
                min_episodes=50,
            ),
            CurriculumStage(
                name="normal",
                env_params={"fee_rate": 0.0004, "base_spread": 25.0, "max_inventory": 10.0},
                advancement_threshold=0.0,
                min_episodes=0,
            ),
        ]
    
    elif difficulty == "aggressive":
        return AGGRESSIVE_CURRICULUM
    
    else:  # normal
        return DEFAULT_MARKET_MAKING_CURRICULUM
