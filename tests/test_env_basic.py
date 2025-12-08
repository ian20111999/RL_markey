import pytest
import numpy as np
import gymnasium as gym
from envs.env_factory import create_env_from_config

@pytest.fixture
def env_config_path():
    return 'configs/env_v3_robust.yaml'

@pytest.fixture
def env(env_config_path):
    env = create_env_from_config(env_config_path)
    yield env
    env.close()

def test_env_initialization(env):
    """測試環境初始化"""
    assert env is not None
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Box)

def test_env_reset(env):
    """測試環境重置"""
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    assert isinstance(info, dict)
    assert "domain_rand" in info

def test_env_step(env):
    """測試環境步進"""
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

def test_observation_bounds(env):
    """測試觀察值是否在合理範圍內 (檢查 NaN/Inf)"""
    obs, _ = env.reset()
    assert not np.any(np.isnan(obs)), "Observation contains NaN"
    assert not np.any(np.isinf(obs)), "Observation contains Inf"
    
    for _ in range(10):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        assert not np.any(np.isnan(obs)), "Observation contains NaN after step"
        assert not np.any(np.isinf(obs)), "Observation contains Inf after step"
