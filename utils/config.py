"""config.py: 負責載入 YAML/JSON 設定，提供統一的字典介面。

此模組的目標是讓訓練、評估、分析等腳本都可以透過單一入口取得
env/train/data_split/run 等區塊，避免到處複製貼上參數。

支援 V1 (HistoricalMarketMakingEnv) 和 V2 (MarketMakingEnvV2) 環境。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml


@dataclass(slots=True)
class ExperimentConfig:
    """包裝後的設定，方便以屬性方式存取。"""

    raw: Dict[str, Any]

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
    def reward(self) -> Dict[str, Any]:
        return dict(self.raw.get("reward", {}))
    
    @property
    def observation(self) -> Dict[str, Any]:
        return dict(self.raw.get("observation", {}))
    
    @property
    def action(self) -> Dict[str, Any]:
        return dict(self.raw.get("action", {}))
    
    @property
    def domain_randomization(self) -> Dict[str, Any]:
        return dict(self.raw.get("domain_randomization", {}))
    
    @property
    def evaluation(self) -> Dict[str, Any]:
        return dict(self.raw.get("evaluation", {}))
    
    @property
    def sanity_criteria(self) -> Dict[str, Any]:
        return dict(self.raw.get("sanity_criteria", {}))
    
    def is_v2_env(self) -> bool:
        """檢查是否使用 V2 環境"""
        env_id = self.env.get("id", "")
        return "V2" in env_id or "v2" in env_id.lower()


def _parse_file(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as f:
        if suffix in (".yaml", ".yml"):
            return yaml.safe_load(f) or {}
        if suffix == ".json":
            return json.load(f)
    raise ValueError(f"無法解析 {path}，僅支援 YAML/JSON。")


def load_config(path: str | Path) -> ExperimentConfig:
    """讀取設定檔並回傳 ExperimentConfig。"""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"找不到設定檔 {config_path}")
    raw = _parse_file(config_path)
    return ExperimentConfig(raw=raw)


def export_config(config: ExperimentConfig, target_path: Path) -> None:
    """將目前設定輸出到 run 目錄，方便追蹤實驗。"""

    target_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = target_path.suffix.lower()
    data = config.raw
    with target_path.open("w", encoding="utf-8") as f:
        if suffix in (".yaml", ".yml"):
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
        else:
            json.dump(data, f, ensure_ascii=False, indent=2)


def merge_cli_overrides(base: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """簡單遞迴合併字典，CLI 參數優先。"""

    if not overrides:
        return base
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_cli_overrides(merged[key], value)
        else:
            merged[key] = value
    return merged


DEFAULT_ENV_PARAMS: Dict[str, Any] = {
    "csv_path": "data/btc_usdt_1m_2023.csv",
    "episode_length": 1000,
    "fee_rate": 0.0004,
    "lambda_inv": 0.001,
    "lambda_turnover": 0.0,
    "max_inventory": 10.0,
    "base_spread": 0.2,
    "alpha": 1.0,
    "beta": 0.5,
    "random_start": True,
}


def build_env_kwargs(
    env_cfg: Dict[str, Any],
    root_dir: Path,
    *,
    seed: int | None = None,
    random_start: bool | None = None,
    date_range: tuple[str | None, str | None] | None = None,
    defaults: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """將 config 中的 env 區塊轉換成 HistoricalMarketMakingEnv 需要的參數。"""

    params: Dict[str, Any] = dict(DEFAULT_ENV_PARAMS)
    if defaults:
        params.update(defaults)
    params.update(env_cfg or {})

    # Handle v2 config structure mapping
    # Map data_file -> csv_path
    if "data_file" in params:
        params["csv_path"] = params["data_file"]
    
    # Map max_episode_steps -> episode_length
    if "max_episode_steps" in params:
        params["episode_length"] = params["max_episode_steps"]
        
    # Map fee -> fee_rate
    if "fee" in params:
        params["fee_rate"] = params["fee"]

    # Map nested features
    if "inventory_features" in params and isinstance(params["inventory_features"], dict):
        inv_feats = params["inventory_features"]
        if "max_inventory" in inv_feats:
            params["max_inventory"] = inv_feats["max_inventory"]
        if "lambda_inventory" in inv_feats:
            params["lambda_inv"] = inv_feats["lambda_inventory"]
            
    if "spread_features" in params and isinstance(params["spread_features"], dict):
        spread_feats = params["spread_features"]
        if "base_spread" in spread_feats:
            params["base_spread"] = spread_feats["base_spread"]

    csv_path = Path(params.get("csv_path", DEFAULT_ENV_PARAMS["csv_path"]))
    if not csv_path.is_absolute():
        csv_path = root_dir / csv_path

    resolved_random_start = random_start if random_start is not None else params.get("random_start", True)

    # Only pass arguments that HistoricalMarketMakingEnv accepts
    # Note: If the Env is updated to support more v2 params, add them here.
    kwargs = {
        "csv_path": str(csv_path),
        "episode_length": int(params.get("episode_length", DEFAULT_ENV_PARAMS["episode_length"])),
        "fee_rate": float(params.get("fee_rate", DEFAULT_ENV_PARAMS["fee_rate"])),
        "lambda_inv": float(params.get("lambda_inv", DEFAULT_ENV_PARAMS["lambda_inv"])),
        "lambda_turnover": float(params.get("lambda_turnover", DEFAULT_ENV_PARAMS["lambda_turnover"])),
        "max_inventory": float(params.get("max_inventory", DEFAULT_ENV_PARAMS["max_inventory"])),
        "base_spread": float(params.get("base_spread", DEFAULT_ENV_PARAMS["base_spread"])),
        "alpha": float(params.get("alpha", DEFAULT_ENV_PARAMS["alpha"])),
        "beta": float(params.get("beta", DEFAULT_ENV_PARAMS["beta"])),
        "random_start": bool(resolved_random_start),
    }
    if seed is not None:
        kwargs["seed"] = seed
    resolved_range = date_range or params.get("date_range")
    if resolved_range is not None:
        kwargs["date_range"] = resolved_range
    return kwargs


# =============================================================================
# V2 Environment Support
# =============================================================================

def build_env_v2_kwargs(
    config: ExperimentConfig,
    root_dir: Path,
    *,
    seed: Optional[int] = None,
    date_range: Optional[Tuple[str, str]] = None,
    enable_domain_rand: Optional[bool] = None,
) -> Dict[str, Any]:
    """為 MarketMakingEnvV2 建構參數。"""
    from envs.market_making_env_v2 import (
        RewardConfig, RewardMode, ObservationConfig, 
        ActionConfig, DomainRandomizationConfig
    )
    
    env_cfg = config.env
    reward_cfg = config.reward
    obs_cfg = config.observation
    action_cfg = config.action
    dr_cfg = config.domain_randomization
    
    # 資料路徑
    csv_path = env_cfg.get("data_file", "data/btc_usdt_1m_2023.csv")
    if not Path(csv_path).is_absolute():
        csv_path = str(root_dir / csv_path)
    
    # 建構 RewardConfig
    reward_mode = RewardMode(reward_cfg.get("mode", "shaped"))
    reward_config = RewardConfig(
        mode=reward_mode,
        lambda_inventory=reward_cfg.get("lambda_inventory", 0.0005),
        lambda_turnover=reward_cfg.get("lambda_turnover", 0.0),
        gamma=reward_cfg.get("gamma", 0.99),
        sparse_scale=reward_cfg.get("sparse_scale", 0.01),
        terminal_bonus_weight=reward_cfg.get("terminal_bonus_weight", 0.5),
    )
    
    # 建構 ObservationConfig
    observation_config = ObservationConfig(
        include_price=obs_cfg.get("include_price", True),
        include_inventory=obs_cfg.get("include_inventory", True),
        include_time=obs_cfg.get("include_time", True),
        include_volatility=obs_cfg.get("include_volatility", True),
        include_momentum=obs_cfg.get("include_momentum", True),
        include_volume=obs_cfg.get("include_volume", True),
        include_inventory_age=obs_cfg.get("include_inventory_age", True),
        volatility_windows=obs_cfg.get("volatility_windows", [5, 15, 60]),
        momentum_windows=obs_cfg.get("momentum_windows", [5, 15]),
    )
    
    # 建構 ActionConfig
    action_config = ActionConfig(
        mode=action_cfg.get("mode", "asymmetric"),
        allow_no_quote=action_cfg.get("allow_no_quote", True),
        max_spread_multiplier=action_cfg.get("max_spread_multiplier", 3.0),
        min_spread_multiplier=action_cfg.get("min_spread_multiplier", 0.1),
    )
    
    # 建構 DomainRandomizationConfig
    dr_enabled = enable_domain_rand if enable_domain_rand is not None else dr_cfg.get("enabled", False)
    domain_rand_config = DomainRandomizationConfig(
        enabled=dr_enabled,
        fee_rate_range=tuple(dr_cfg.get("fee_rate_range", [0.0003, 0.0005])),
        base_spread_range=tuple(dr_cfg.get("base_spread_range", [15.0, 35.0])),
        volatility_multiplier_range=tuple(dr_cfg.get("volatility_multiplier_range", [0.8, 1.2])),
        fill_probability_noise=dr_cfg.get("fill_probability_noise", 0.1),
    )
    
    kwargs = {
        "csv_path": csv_path,
        "episode_length": env_cfg.get("episode_length", 1000),
        "fee_rate": env_cfg.get("fee_rate", 0.0004),
        "base_spread": env_cfg.get("base_spread", 25.0),
        "max_inventory": env_cfg.get("max_inventory", 5.0),
        "initial_cash": env_cfg.get("initial_cash", 10000.0),
        "random_start": env_cfg.get("random_start", True),
        "reward_config": reward_config,
        "obs_config": observation_config,
        "action_config": action_config,
        "domain_rand_config": domain_rand_config,
    }
    
    if seed is not None:
        kwargs["seed"] = seed
    
    if date_range is not None:
        kwargs["date_range"] = date_range
    
    return kwargs


def create_env_from_config(
    config: ExperimentConfig,
    root_dir: Path,
    *,
    seed: Optional[int] = None,
    date_range: Optional[Tuple[str, str]] = None,
    enable_domain_rand: Optional[bool] = None,
):
    """根據 Config 自動建立對應版本的環境。"""
    if config.is_v2_env():
        from envs.market_making_env_v2 import MarketMakingEnvV2
        kwargs = build_env_v2_kwargs(
            config, root_dir,
            seed=seed,
            date_range=date_range,
            enable_domain_rand=enable_domain_rand,
        )
        return MarketMakingEnvV2(**kwargs)
    else:
        from envs.historical_market_making_env import HistoricalMarketMakingEnv
        kwargs = build_env_kwargs(config.env, root_dir, seed=seed, date_range=date_range)
        return HistoricalMarketMakingEnv(**kwargs)

