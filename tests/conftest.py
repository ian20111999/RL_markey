"""
tests/conftest.py
Pytest 共用 Fixtures

提供統一的測試設置和清理邏輯，解決資料庫初始化等問題。
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def sample_ohlcv_data():
    """生成模擬的 OHLCV 資料（用於不需要真實數據的測試）"""
    np.random.seed(42)
    n_rows = 5000
    
    # 生成價格序列（隨機 walk）
    returns = np.random.randn(n_rows) * 0.001  # 0.1% 標準差
    price = 100 * np.exp(np.cumsum(returns))
    
    # 生成 OHLCV
    data = {
        "timestamp": pd.date_range("2023-01-01", periods=n_rows, freq="T").astype(int) // 10**6,
        "open": price * (1 + np.random.randn(n_rows) * 0.0001),
        "high": price * (1 + np.abs(np.random.randn(n_rows)) * 0.001),
        "low": price * (1 - np.abs(np.random.randn(n_rows)) * 0.001),
        "close": price,
        "volume": np.random.exponential(1000, n_rows),
    }
    
    df = pd.DataFrame(data)
    # 確保 high >= close >= low
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    
    return df


@pytest.fixture
def sample_data_csv(sample_ohlcv_data, tmp_path):
    """將樣本資料儲存為臨時 CSV 檔案"""
    csv_path = tmp_path / "test_data.csv"
    sample_ohlcv_data.to_csv(csv_path, index=False)
    return str(csv_path)


# =============================================================================
# Environment Fixtures
# =============================================================================

@pytest.fixture
def basic_env(sample_ohlcv_data):
    """創建基本測試環境"""
    from envs.market_making_env import (
        MarketMakingEnv, RewardConfig, ObservationConfig, ActionConfig, RewardMode
    )
    
    env = MarketMakingEnv(
        df=sample_ohlcv_data,
        initial_cash=10000,
        fee_rate=0.0004,
        max_inventory=2.0,
        episode_length=1000,
        base_spread=0.5,  # 適合模擬數據的較小 spread
        random_start=False,  # 測試時使用固定起始點
        reward_config=RewardConfig(
            mode=RewardMode.SHAPED,
            reward_scale=1e-4,
            lambda_inventory=10.0,
        ),
        obs_config=ObservationConfig(
            include_volatility=True,
            include_momentum=True,
        ),
        action_config=ActionConfig(
            mode="asymmetric",
            allow_no_quote=True,
        ),
    )
    
    yield env
    env.close()


@pytest.fixture
def config_path():
    """返回預設配置路徑（如果存在）"""
    path = project_root / "configs" / "default.yaml"
    if not path.exists():
        pytest.skip(f"Config file not found: {path}")
    return str(path)


@pytest.fixture
def real_data_path():
    """返回真實數據路徑（如果存在）"""
    path = project_root / "data" / "btc_usdt_1m_2023.csv"
    if not path.exists():
        pytest.skip(f"Data file not found: {path}. Use sample_ohlcv_data instead.")
    return str(path)


# =============================================================================
# Database Fixtures
# =============================================================================

@pytest.fixture(scope="function")
def test_sqlite_db(tmp_path):
    """創建臨時 SQLite 測試資料庫"""
    db_path = tmp_path / "test_metrics.db"
    
    # 初始化 Schema
    init_sql_path = project_root / "scripts" / "init_db_sqlite.sql"
    
    if init_sql_path.exists():
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        with open(init_sql_path, "r") as f:
            conn.executescript(f.read())
        conn.commit()
        conn.close()
    
    yield str(db_path)
    
    # 清理
    if db_path.exists():
        os.remove(db_path)


@pytest.fixture(scope="function")
def mock_db_connection(test_sqlite_db):
    """提供已初始化的資料庫連接"""
    import sqlite3
    conn = sqlite3.connect(test_sqlite_db)
    yield conn
    conn.close()


# =============================================================================
# Model Fixtures
# =============================================================================

@pytest.fixture
def mock_model_path(tmp_path):
    """創建模擬的模型檔案（用於測試 API）"""
    # 這只是一個佔位符，實際測試可能需要真實模型
    model_path = tmp_path / "mock_model.zip"
    model_path.touch()
    return str(model_path)


# =============================================================================
# Helper Functions
# =============================================================================

def pytest_addoption(parser):
    """添加自訂 pytest 選項"""
    parser.addoption(
        "--use-real-data",
        action="store_true",
        default=False,
        help="Use real market data for tests (requires data files)"
    )
    parser.addoption(
        "--use-real-db",
        action="store_true", 
        default=False,
        help="Use real database for tests (requires PostgreSQL)"
    )


@pytest.fixture
def use_real_data(request):
    """檢查是否使用真實資料"""
    return request.config.getoption("--use-real-data")


@pytest.fixture
def use_real_db(request):
    """檢查是否使用真實資料庫"""
    return request.config.getoption("--use-real-db")


# =============================================================================
# Skip Markers
# =============================================================================

def pytest_configure(config):
    """配置自訂 markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_data: marks tests that require real market data"
    )
    config.addinivalue_line(
        "markers", "requires_db: marks tests that require database connection"
    )
