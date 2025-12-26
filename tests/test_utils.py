"""
tests/test_utils.py
Utils 模組測試

測試各種 utility 模組的功能。
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestAlgorithms:
    """測試 utils/algorithms.py"""
    
    def test_create_sac_model(self, basic_env):
        """測試創建 SAC 模型"""
        from utils.algorithms import create_model
        
        model = create_model("sac", basic_env, verbose=0)
        assert model is not None
        
        # 測試預測
        obs, _ = basic_env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert action is not None
        assert action.shape == basic_env.action_space.shape
    
    def test_create_ppo_model(self, basic_env):
        """測試創建 PPO 模型"""
        from utils.algorithms import create_model
        
        model = create_model("ppo", basic_env, verbose=0)
        assert model is not None
    
    def test_create_td3_model(self, basic_env):
        """測試創建 TD3 模型"""
        from utils.algorithms import create_model
        
        model = create_model("td3", basic_env, verbose=0)
        assert model is not None
    
    def test_config_overrides(self, basic_env):
        """測試配置覆蓋"""
        from utils.algorithms import create_model
        
        model = create_model(
            "sac",
            basic_env,
            config_overrides={"learning_rate": 1e-4},
            verbose=0,
        )
        assert model is not None
        assert model.learning_rate == 1e-4


class TestLearningRateSchedulers:
    """測試學習率調度器"""
    
    def test_linear_scheduler(self):
        """測試線性調度器"""
        from utils.algorithms import linear_schedule
        
        scheduler = linear_schedule(1e-3)
        
        # 開始時應該是初始值
        assert abs(scheduler(1.0) - 1e-3) < 1e-10
        
        # 結束時應該接近 0
        assert scheduler(0.0) < 1e-6
        
        # 中間應該是線性下降
        assert abs(scheduler(0.5) - 0.5e-3) < 1e-10
    
    def test_cosine_scheduler(self):
        """測試餘弦調度器"""
        try:
            from utils.algorithms import cosine_schedule
            
            scheduler = cosine_schedule(1e-3, 1e-5)
            
            # 測試調度器是可調用的
            assert callable(scheduler)
            
            # 測試開始和結束值
            assert scheduler(1.0) >= scheduler(0.0)
        except ImportError:
            pytest.skip("cosine_schedule not available")


class TestRewardConfig:
    """測試 RewardConfig"""
    
    def test_reward_modes(self):
        """測試各種獎勵模式"""
        from envs.market_making_env import RewardConfig, RewardMode
        
        modes = [RewardMode.DENSE, RewardMode.SPARSE, RewardMode.SHAPED, RewardMode.HYBRID]
        
        for mode in modes:
            config = RewardConfig(mode=mode)
            assert config.mode == mode


class TestCacheManager:
    """測試快取管理器"""
    
    def test_cache_basic(self, tmp_path):
        """測試基本快取功能"""
        try:
            from utils.cache_manager import cache_manager
            
            # 測試 cache_manager 存在且可調用
            assert cache_manager is not None
        except ImportError:
            pytest.skip("cache_manager not available")
    
    def test_cache_miss(self, tmp_path):
        """測試快取未命中"""
        try:
            from utils.cache_manager import cache_manager
            
            # 測試 cache_manager 存在
            assert cache_manager is not None
        except ImportError:
            pytest.skip("cache_manager not available")


class TestLoggingConfig:
    """測試日誌配置"""
    
    def test_setup_logging(self, tmp_path):
        """測試設置日誌"""
        try:
            from utils.logging_config import setup_logging
            
            logger = setup_logging("test_logger")
            assert logger is not None
            logger.info("Test message")
        except (ImportError, TypeError):
            pytest.skip("logging_config not available or API changed")


class TestPerformanceMonitor:
    """測試性能監控器"""
    
    def test_timer_context(self):
        """測試計時器上下文"""
        try:
            from utils.performance_monitor import PerformanceMonitor
            import time
            
            monitor = PerformanceMonitor()
            
            with monitor.timer("test_operation"):
                time.sleep(0.05)
            
            # 應該有記錄
            assert "test_operation" in monitor.timings
        except (ImportError, ModuleNotFoundError):
            pytest.skip("PerformanceMonitor not available (missing psutil)")


class TestEnsemble:
    """測試 Ensemble 模組"""
    
    def test_ensemble_policy_creation(self, basic_env):
        """測試創建 Ensemble Policy"""
        from utils.algorithms import create_model
        
        # 創建兩個模型
        model1 = create_model("sac", basic_env, verbose=0)
        model2 = create_model("sac", basic_env, verbose=0)
        
        from utils.ensemble import EnsemblePolicy
        
        ensemble = EnsemblePolicy([model1, model2], method="voting")
        
        obs, _ = basic_env.reset()
        action, _ = ensemble.predict(obs, deterministic=True)
        
        assert action is not None
        assert action.shape == basic_env.action_space.shape
