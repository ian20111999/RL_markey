"""
tests/test_env_basic.py
環境基礎測試

測試 MarketMakingEnv 的基本功能：初始化、reset、step、observation 邊界。
"""

import pytest
import numpy as np
import gymnasium as gym
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestEnvInitialization:
    """測試環境初始化"""
    
    def test_env_creation(self, basic_env):
        """測試環境是否成功創建"""
        assert basic_env is not None
    
    def test_observation_space(self, basic_env):
        """測試 observation space 類型"""
        assert isinstance(basic_env.observation_space, gym.spaces.Box)
        assert basic_env.observation_space.dtype == np.float32
    
    def test_action_space(self, basic_env):
        """測試 action space 類型"""
        assert isinstance(basic_env.action_space, gym.spaces.Box)
        assert basic_env.action_space.shape[0] == 3  # asymmetric mode: [bid, ask, quote_flag]


class TestEnvReset:
    """測試環境 reset"""
    
    def test_reset_returns_observation(self, basic_env):
        """測試 reset 返回正確的 observation"""
        obs, info = basic_env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == basic_env.observation_space.shape
    
    def test_reset_returns_info(self, basic_env):
        """測試 reset 返回 info dict"""
        obs, info = basic_env.reset()
        assert isinstance(info, dict)
    
    def test_reset_observation_finite(self, basic_env):
        """測試 reset 後的 observation 不含 NaN/Inf"""
        obs, _ = basic_env.reset()
        assert not np.any(np.isnan(obs)), "Observation contains NaN"
        assert not np.any(np.isinf(obs)), "Observation contains Inf"
    
    def test_reset_with_seed(self, basic_env):
        """測試帶 seed 的 reset"""
        obs1, _ = basic_env.reset(seed=42)
        obs2, _ = basic_env.reset(seed=42)
        # 相同 seed 應該產生相同結果
        np.testing.assert_array_equal(obs1, obs2)


class TestEnvStep:
    """測試環境 step"""
    
    def test_step_returns_correct_types(self, basic_env):
        """測試 step 返回正確類型"""
        basic_env.reset()
        action = basic_env.action_space.sample()
        obs, reward, terminated, truncated, info = basic_env.step(action)
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_step_observation_shape(self, basic_env):
        """測試 step 後的 observation shape"""
        basic_env.reset()
        action = basic_env.action_space.sample()
        obs, _, _, _, _ = basic_env.step(action)
        
        assert obs.shape == basic_env.observation_space.shape
    
    def test_step_observation_finite(self, basic_env):
        """測試多步後的 observation 仍然有限"""
        basic_env.reset()
        
        for _ in range(50):
            action = basic_env.action_space.sample()
            obs, _, terminated, truncated, _ = basic_env.step(action)
            
            assert not np.any(np.isnan(obs)), "Observation contains NaN"
            assert not np.any(np.isinf(obs)), "Observation contains Inf"
            
            if terminated or truncated:
                break


class TestEnvEpisode:
    """測試完整 episode"""
    
    def test_episode_terminates(self, basic_env):
        """測試 episode 會在 episode_length 內終止"""
        basic_env.reset()
        
        for step in range(basic_env.episode_length + 100):
            action = basic_env.action_space.sample()
            _, _, terminated, truncated, _ = basic_env.step(action)
            
            if terminated or truncated:
                break
        
        # 應該在 episode_length 結束時 truncate
        assert step <= basic_env.episode_length
    
    def test_episode_reward_accumulation(self, basic_env):
        """測試 episode 內獎勵累積正常"""
        basic_env.reset()
        total_reward = 0.0
        
        for _ in range(100):
            action = basic_env.action_space.sample()
            _, reward, terminated, truncated, _ = basic_env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        # 獎勵應該是有限的數值
        assert np.isfinite(total_reward)


class TestEnvInfo:
    """測試環境 info 內容"""
    
    def test_info_contains_portfolio_value(self, basic_env):
        """測試 info 包含 portfolio_value"""
        basic_env.reset()
        action = basic_env.action_space.sample()
        _, _, _, _, info = basic_env.step(action)
        
        assert "portfolio_value" in info
        assert info["portfolio_value"] > 0
    
    def test_info_contains_inventory(self, basic_env):
        """測試 info 包含 inventory"""
        basic_env.reset()
        action = basic_env.action_space.sample()
        _, _, _, _, info = basic_env.step(action)
        
        assert "inventory" in info
