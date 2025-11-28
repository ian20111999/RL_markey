"""
config.py: Load YAML/JSON config and provide unified interface.
Uses MarketMakingEnvV2 environment.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
import pandas as pd


@dataclass(slots=True)
class ExperimentConfig:
    raw: Dict[str, Any]

    @property
    def data(self) -> Dict[str, Any]:
        return dict(self.raw.get("data", {}))

    @property
    def env(self) -> Dict[str, Any]:
        return dict(self.raw.get("env", {}))

    @property
    def train(self) -> Dict[str, Any]:
        return dict(self.raw.get("train", {}))

    @property
    def data_split(self) -> Dict[str, Any]:
        return dict(self.raw.get("data_split", {}))

    @property
    def run(self) -> Dict[str, Any]:
        return dict(self.raw.get("run", {}))

    @property
    def curriculum(self) -> Dict[str, Any]:
        return dict(self.raw.get("curriculum", {}))

    @property
    def risk_sensitive(self) -> Dict[str, Any]:
        return dict(self.raw.get("risk_sensitive", {}))

    @property
    def backtest(self) -> Dict[str, Any]:
        return dict(self.raw.get("backtest", {}))

    @property
    def explainability(self) -> Dict[str, Any]:
        return dict(self.raw.get("explainability", {}))


def _parse_file(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as f:
        if suffix in (".yaml", ".yml"):
            return yaml.safe_load(f) or {}
        if suffix == ".json":
            return json.load(f)
    raise ValueError(f"Unsupported file format: {path}")


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    raw = _parse_file(config_path)
    return ExperimentConfig(raw=raw)


def export_config(config: ExperimentConfig, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = target_path.suffix.lower()
    data = config.raw
    with target_path.open("w", encoding="utf-8") as f:
        if suffix in (".yaml", ".yml"):
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
        else:
            json.dump(data, f, ensure_ascii=False, indent=2)


def merge_cli_overrides(base: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not overrides:
        return base
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_cli_overrides(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_data(config: ExperimentConfig, root_dir: Path) -> pd.DataFrame:
    data_cfg = config.data
    data_path = data_cfg.get("path", "data/btc_usdt_1m_2023.csv")
    if not Path(data_path).is_absolute():
        data_path = root_dir / data_path
    return pd.read_csv(data_path)


def split_data(data: pd.DataFrame, config: ExperimentConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/valid/test sets.
    
    Supports both ratio-based (0.7) and date-based ('2023-06-30') splitting.
    """
    split_cfg = config.data_split
    n = len(data)
    
    # 檢查是否使用日期範圍（新格式）
    train_start = split_cfg.get("train_start")
    train_end_date = split_cfg.get("train_end")
    valid_start = split_cfg.get("valid_start")
    valid_end_date = split_cfg.get("valid_end")
    test_start = split_cfg.get("test_start")
    test_end_date = split_cfg.get("test_end")
    
    # 如果有 timestamp 列並且使用日期格式
    if isinstance(train_end_date, str) and "timestamp" in data.columns:
        data = data.copy()
        # 處理毫秒時間戳或字串日期
        if data["timestamp"].dtype in ["int64", "float64"]:
            data["_datetime"] = pd.to_datetime(data["timestamp"], unit="ms")
        else:
            data["_datetime"] = pd.to_datetime(data["timestamp"], errors="coerce")
        
        if train_start and train_end_date:
            train_mask = (data["_datetime"] >= train_start) & (data["_datetime"] <= train_end_date)
            train_data = data[train_mask].drop(columns=["_datetime"]).reset_index(drop=True)
        else:
            train_data = pd.DataFrame()
        
        if valid_start and valid_end_date:
            valid_mask = (data["_datetime"] >= valid_start) & (data["_datetime"] <= valid_end_date)
            valid_data = data[valid_mask].drop(columns=["_datetime"]).reset_index(drop=True)
        else:
            valid_data = pd.DataFrame()
        
        if test_start and test_end_date:
            test_mask = (data["_datetime"] >= test_start) & (data["_datetime"] <= test_end_date)
            test_data = data[test_mask].drop(columns=["_datetime"]).reset_index(drop=True)
        else:
            test_data = pd.DataFrame()
        
        return train_data, valid_data, test_data
    
    # 使用比例分割（舊格式）
    train_end = split_cfg.get("train_end", 0.7)
    valid_end = split_cfg.get("valid_end", 0.85)
    
    if isinstance(train_end, float) and 0 < train_end < 1:
        train_end = int(n * train_end)
    if isinstance(valid_end, float) and 0 < valid_end < 1:
        valid_end = int(n * valid_end)
    
    train_data = data.iloc[:train_end].reset_index(drop=True)
    valid_data = data.iloc[train_end:valid_end].reset_index(drop=True)
    test_data = data.iloc[valid_end:].reset_index(drop=True)
    return train_data, valid_data, test_data


def create_env(data: pd.DataFrame, config: ExperimentConfig, seed: Optional[int] = None, enable_domain_rand: Optional[bool] = None):
    from envs.market_making_env_v2 import (
        MarketMakingEnvV2, RewardConfig, ObservationConfig,
        ActionConfig, DomainRandomizationConfig
    )
    env_cfg = config.env
    kwargs = {
        "df": data,  # 直接傳入 DataFrame
        "initial_cash": env_cfg.get("initial_cash", 100000),
        "fee_rate": env_cfg.get("fee_rate", 0.0004),
        "max_inventory": env_cfg.get("max_inventory", 10.0),
        "episode_length": env_cfg.get("episode_length", 1000),
        "base_spread": env_cfg.get("base_spread", 25.0),
        "random_start": env_cfg.get("random_start", True),
    }
    
    # 處理 reward_config
    reward_cfg = env_cfg.get("reward_config", {})
    if reward_cfg:
        kwargs["reward_config"] = RewardConfig(**{
            k: v for k, v in reward_cfg.items()
            if k in RewardConfig.__dataclass_fields__
        })
    
    # 處理 obs_config
    obs_cfg = env_cfg.get("obs_config", {})
    if obs_cfg:
        kwargs["obs_config"] = ObservationConfig(**{
            k: v for k, v in obs_cfg.items()
            if k in ObservationConfig.__dataclass_fields__
        })
    
    # 處理 action_config
    action_cfg = env_cfg.get("action_config", {})
    if action_cfg:
        kwargs["action_config"] = ActionConfig(**{
            k: v for k, v in action_cfg.items()
            if k in ActionConfig.__dataclass_fields__
        })
    
    # 處理 domain_randomization
    dr_cfg = env_cfg.get("domain_randomization", {})
    if enable_domain_rand is not None:
        dr_cfg = dict(dr_cfg)  # 複製以避免修改原始 config
        dr_cfg["enabled"] = enable_domain_rand
    if dr_cfg:
        kwargs["domain_rand_config"] = DomainRandomizationConfig(**{
            k: v for k, v in dr_cfg.items()
            if k in DomainRandomizationConfig.__dataclass_fields__
        })
    
    if seed is not None:
        kwargs["seed"] = seed
    return MarketMakingEnvV2(**kwargs)


def create_env_from_config(
    config: ExperimentConfig,
    root_dir: Path,
    seed: Optional[int] = None,
    date_range: Optional[Tuple[str, str]] = None,
    enable_domain_rand: Optional[bool] = None,
):
    """Create environment from config with optional date filtering.
    
    Args:
        config: Experiment configuration
        root_dir: Project root directory
        seed: Random seed
        date_range: Optional (start_date, end_date) tuple for filtering data
        enable_domain_rand: Whether to enable domain randomization
    
    Returns:
        MarketMakingEnvV2 instance
    """
    data = load_data(config, root_dir)
    
    # Filter by date range if provided
    if date_range and date_range[0] and date_range[1]:
        if "timestamp" in data.columns:
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            mask = (data["timestamp"] >= date_range[0]) & (data["timestamp"] <= date_range[1])
            data = data[mask].reset_index(drop=True)
    
    return create_env(data, config, seed, enable_domain_rand)


def create_env_fn(data: pd.DataFrame, config: ExperimentConfig, seed: Optional[int] = None, enable_domain_rand: Optional[bool] = None):
    def _make_env():
        return create_env(data, config, seed, enable_domain_rand)
    return _make_env


def get_train_params(config: ExperimentConfig) -> Dict[str, Any]:
    train_cfg = config.train
    return {
        "total_timesteps": train_cfg.get("total_timesteps", 100000),
        "learning_rate": train_cfg.get("learning_rate", 3e-4),
        "buffer_size": train_cfg.get("buffer_size", 100000),
        "batch_size": train_cfg.get("batch_size", 256),
        "gamma": train_cfg.get("gamma", 0.99),
        "tau": train_cfg.get("tau", 0.02),
        "train_freq": train_cfg.get("train_freq", 1),
        "gradient_steps": train_cfg.get("gradient_steps", 1),
        "n_envs": train_cfg.get("n_envs", 4),
        "eval_freq": train_cfg.get("eval_freq", 10000),
        "n_eval_episodes": train_cfg.get("n_eval_episodes", 5),
    }


def get_algorithm_params(config: ExperimentConfig, algorithm: str = "SAC") -> Dict[str, Any]:
    train_cfg = config.train
    common_params = {
        "learning_rate": train_cfg.get("learning_rate", 3e-4),
        "batch_size": train_cfg.get("batch_size", 256),
        "gamma": train_cfg.get("gamma", 0.99),
        "verbose": 0,
    }
    if algorithm.upper() == "SAC":
        common_params.update({
            "buffer_size": train_cfg.get("buffer_size", 100000),
            "tau": train_cfg.get("tau", 0.02),
            "train_freq": train_cfg.get("train_freq", 1),
            "gradient_steps": train_cfg.get("gradient_steps", 1),
            "ent_coef": train_cfg.get("ent_coef", "auto"),
        })
    elif algorithm.upper() == "PPO":
        common_params.update({
            "n_steps": train_cfg.get("n_steps", 2048),
            "n_epochs": train_cfg.get("n_epochs", 10),
            "ent_coef": train_cfg.get("ent_coef", 0.01),
            "clip_range": train_cfg.get("clip_range", 0.2),
            "gae_lambda": train_cfg.get("gae_lambda", 0.95),
        })
    elif algorithm.upper() == "TD3":
        common_params.update({
            "buffer_size": train_cfg.get("buffer_size", 100000),
            "tau": train_cfg.get("tau", 0.005),
            "policy_delay": train_cfg.get("policy_delay", 2),
            "target_policy_noise": train_cfg.get("target_policy_noise", 0.2),
        })
    return common_params
