"""
環境工廠函數

提供統一的環境創建介面，消除重複代碼。
"""
import yaml
from typing import Optional, Dict, Any
import pandas as pd
from pydantic import ValidationError

from envs.market_making_env import (
    MarketMakingEnv,
    RewardConfig,
    ObservationConfig,
    ActionConfig,
    DomainRandomizationConfig,
    FillModelEnvConfig,
    AdvancedObservationConfig,
    PerformanceConfig,
    RewardMode,
)
from envs.config_schema import RootConfig


def create_env_from_config(
    config_path: str,
    csv_path: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    override_params: Optional[Dict[str, Any]] = None,
) -> MarketMakingEnv:
    """從 YAML 配置檔案創建環境
    
    Args:
        config_path: YAML 配置檔案路徑
        csv_path: 市場資料 CSV 路徑（與 df 二選一）
        df: 市場資料 DataFrame（與 csv_path 二選一）
        override_params: 覆蓋配置的參數（例如 {'episode_length': 2000}）
    
    Returns:
        MarketMakingEnv 實例
    
    Example:
        >>> env = create_env_from_config(
        ...     'configs/env_v3_stabilized.yaml',
        ...     csv_path='data/btc_usdt_1m_2023.csv'
        ... )
    """
    # 載入配置
    with open(config_path, 'r', encoding='utf-8') as f:
        raw_config = yaml.safe_load(f)
    
    # 應用覆蓋參數 (在驗證前應用)
    if override_params:
        # 簡單的遞歸更新或直接更新頂層
        # 這裡假設 override_params 結構與 config 匹配
        # 為了簡單起見，我們只處理頂層覆蓋，或者特定的嵌套鍵
        for key, value in override_params.items():
            if key in raw_config:
                if isinstance(raw_config[key], dict) and isinstance(value, dict):
                    raw_config[key].update(value)
                else:
                    raw_config[key] = value
            else:
                # 嘗試在 env 部分查找
                if 'env' in raw_config and key in raw_config['env']:
                    raw_config['env'][key] = value
                else:
                    raw_config[key] = value

    # Pydantic 驗證
    try:
        validated_config = RootConfig(**raw_config)
    except ValidationError as e:
        print(f"❌ Configuration Validation Error in {config_path}:")
        print(e)
        raise

    # 提取配置對象
    # 注意：我們需要將 Pydantic 模型轉換為環境所需的 dataclass
    # 大多數欄位名稱應該是匹配的
    
    # Reward Config
    reward_dict = validated_config.reward.model_dump()
    # 轉換 Enum
    if 'mode' in reward_dict:
        reward_dict['mode'] = RewardMode(reward_dict['mode'])
    reward_config = RewardConfig(**reward_dict)
    
    # Observation Config
    obs_config = ObservationConfig(**validated_config.observation.model_dump())
    
    # Action Config
    action_config = ActionConfig(**validated_config.action.model_dump())
    
    # Domain Randomization Config
    dr_config = DomainRandomizationConfig(**validated_config.domain_randomization.model_dump())
    
    # Fill Model Config
    # FillModelEnvConfig 與 FillModelConfig 略有不同，需要適配
    fm_dump = validated_config.fill_model.model_dump()
    fill_model_config = FillModelEnvConfig(**fm_dump)
    
    # Advanced Observation Config
    adv_obs_config = AdvancedObservationConfig(**validated_config.advanced_metrics.model_dump())
    
    # Performance Config
    perf_config = PerformanceConfig(**validated_config.performance.model_dump())
    
    # 提取基本參數 (從 env 部分)
    env_settings = validated_config.env
    
    # 優先使用傳入的 csv_path/df，否則使用配置中的
    final_csv_path = csv_path if csv_path else env_settings.data_file
    
    env_params = {
        'csv_path': final_csv_path,
        'df': df,
        'episode_length': env_settings.episode_length,
        'fee_rate': env_settings.fee_rate,
        'base_spread': env_settings.base_spread,
        'max_inventory': env_settings.max_inventory,
        'initial_cash': env_settings.initial_cash,
        'random_start': env_settings.random_start,
        # 'date_range': ... # Config schema 中尚未包含 date_range，如果需要可以添加
        'reward_config': reward_config,
        'obs_config': obs_config,
        'action_config': action_config,
        'domain_rand_config': dr_config,
        'fill_model_config': fill_model_config,
        'advanced_obs_config': adv_obs_config,
        'performance_config': perf_config,
    }
    
    # 移除 None 值
    env_params = {k: v for k, v in env_params.items() if v is not None}
    
    return MarketMakingEnv(**env_params)


def create_env_simple(
    csv_path: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    **kwargs
) -> MarketMakingEnv:
    """快速創建環境（使用預設配置）
    
    Args:
        csv_path: 市場資料 CSV 路徑
        df: 市場資料 DataFrame
        **kwargs: 其他環境參數（會覆蓋預設值）
    
    Returns:
        MarketMakingEnv 實例
    
    Example:
        >>> env = create_env_simple(
        ...     csv_path='data/btc_usdt_1m_2023.csv',
        ...     episode_length=2000,
        ...     max_inventory=10.0
        ... )
    """
    return MarketMakingEnv(
        csv_path=csv_path,
        df=df,
        **kwargs
    )


# =============================================================================
# 配置解析輔助函數 (已棄用，改用 Pydantic 驗證)
# =============================================================================



# =============================================================================
# 測試與驗證
# =============================================================================

def validate_config(config_path: str) -> bool:
    """驗證配置檔案是否正確
    
    Args:
        config_path: YAML 配置檔案路徑
    
    Returns:
        True if valid, False otherwise
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 檢查必要欄位
        required_fields = ['episode_length', 'fee_rate', 'base_spread', 'max_inventory']
        for field in required_fields:
            if field not in config:
                print(f"❌ Missing required field: {field}")
                return False
        
        # 檢查數值範圍
        if config['fee_rate'] < 0 or config['fee_rate'] > 0.01:
            print(f"⚠️  Warning: fee_rate {config['fee_rate']} seems unusual")
        
        if config['max_inventory'] <= 0:
            print(f"❌ Invalid max_inventory: {config['max_inventory']}")
            return False
        
        print(f"✅ Config validation passed: {config_path}")
        return True
        
    except Exception as e:
        print(f"❌ Config validation failed: {e}")
        return False


if __name__ == "__main__":
    # 測試配置驗證
    import sys
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        validate_config(config_path)
    else:
        print("Usage: python env_factory.py <config_path>")
        print("Example: python env_factory.py configs/env_v3_stabilized.yaml")
