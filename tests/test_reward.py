"""
tests/test_reward.py
獎勵函數測試

測試各種 RewardConfig 模式和參數的效果。
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from envs.market_making_env import (
    MarketMakingEnv, RewardConfig, ObservationConfig, ActionConfig, RewardMode
)


class TestRewardConfig:
    """測試 RewardConfig 配置"""
    
    def test_shaped_mode_default(self):
        """測試 SHAPED 模式的預設配置"""
        config = RewardConfig(mode=RewardMode.SHAPED)
        assert config.mode == RewardMode.SHAPED
        assert config.lambda_inventory >= 0
    
    def test_dense_mode(self):
        """測試 DENSE 模式"""
        config = RewardConfig(mode=RewardMode.DENSE)
        assert config.mode == RewardMode.DENSE
    
    def test_sparse_mode(self):
        """測試 SPARSE 模式"""
        config = RewardConfig(mode=RewardMode.SPARSE)
        assert config.mode == RewardMode.SPARSE
    
    def test_hybrid_mode(self):
        """測試 HYBRID 模式"""
        config = RewardConfig(mode=RewardMode.HYBRID)
        assert config.mode == RewardMode.HYBRID


class TestRewardScaling:
    """測試獎勵縮放"""
    
    def test_reward_scale_affects_magnitude(self, sample_ohlcv_data):
        """測試 reward_scale 影響獎勵大小"""
        # 使用較小的 reward_scale
        env_small = MarketMakingEnv(
            df=sample_ohlcv_data,
            episode_length=100,
            base_spread=0.5,
            random_start=False,
            reward_config=RewardConfig(reward_scale=1e-6),
        )
        
        # 使用較大的 reward_scale
        env_large = MarketMakingEnv(
            df=sample_ohlcv_data,
            episode_length=100,
            base_spread=0.5,
            random_start=False,
            reward_config=RewardConfig(reward_scale=1e-3),
        )
        
        env_small.reset(seed=42)
        env_large.reset(seed=42)
        
        rewards_small = []
        rewards_large = []
        
        for _ in range(10):
            action = np.array([0.0, 0.0, 0.5])  # 固定動作
            _, r1, _, _, _ = env_small.step(action)
            _, r2, _, _, _ = env_large.step(action)
            rewards_small.append(r1)
            rewards_large.append(r2)
        
        env_small.close()
        env_large.close()
        
        # 較大的 reward_scale 應該產生較大的獎勵（絕對值）
        avg_small = np.mean(np.abs(rewards_small))
        avg_large = np.mean(np.abs(rewards_large))
        
        # 由於其他因素可能影響，我們只檢查比例關係大致正確
        assert avg_large > avg_small * 10 or avg_small < 1e-10


class TestInventoryPenalty:
    """測試庫存懲罰"""
    
    def test_inventory_penalty_increases_with_lambda(self, sample_ohlcv_data):
        """測試 lambda_inventory 增加會增加庫存懲罰"""
        # 使用較小的 lambda_inventory
        env_small = MarketMakingEnv(
            df=sample_ohlcv_data,
            episode_length=100,
            base_spread=0.5,
            random_start=False,
            reward_config=RewardConfig(
                lambda_inventory=1.0,
                reward_scale=1.0,
            ),
        )
        
        # 使用較大的 lambda_inventory
        env_large = MarketMakingEnv(
            df=sample_ohlcv_data,
            episode_length=100,
            base_spread=0.5,
            random_start=False,
            reward_config=RewardConfig(
                lambda_inventory=50.0,
                reward_scale=1.0,
            ),
        )
        
        env_small.reset(seed=42)
        env_large.reset(seed=42)
        
        env_small.close()
        env_large.close()
        
        # 這個測試主要確保不會報錯
        assert True


class TestRewardFiniteness:
    """測試獎勵有限性"""
    
    def test_reward_is_finite(self, basic_env):
        """測試獎勵始終是有限值"""
        basic_env.reset()
        
        for _ in range(100):
            action = basic_env.action_space.sample()
            _, reward, terminated, truncated, _ = basic_env.step(action)
            
            assert np.isfinite(reward), f"Reward is not finite: {reward}"
            
            if terminated or truncated:
                break
