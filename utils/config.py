"""config.py: 負責載入 YAML/JSON 設定，提供統一的字典介面。

此模組的目標是讓訓練、評估、分析等腳本都可以透過單一入口取得
env/train/data_split/run 等區塊，避免到處複製貼上參數。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

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

    csv_path = Path(params.get("csv_path", DEFAULT_ENV_PARAMS["csv_path"]))
    if not csv_path.is_absolute():
        csv_path = root_dir / csv_path

    resolved_random_start = random_start if random_start is not None else params.get("random_start", True)

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

