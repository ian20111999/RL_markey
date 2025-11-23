"""run_baseline_strategy.py: 以手寫策略檢查環境是否合理，而非追求獲利。

這個腳本提供一個簡單的基準策略（固定 spread、依 inventory 微調 skew），
用來 sanity check：在目前的 fee_rate / lambda_inv / base_spread 設定下，
資金曲線至少要呈現「小賺小賠」而不是一開局就崩盤。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from envs.historical_market_making_env import HistoricalMarketMakingEnv
from utils.config import build_env_kwargs, load_config

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "env_baseline.yaml"
PLOTS_DIR = ROOT / "plots"
BASELINE_LOG_DIR = ROOT / "runs" / "baseline_checks"


def baseline_policy(inventory: float, max_inventory: float) -> np.ndarray:
    """簡單規則：spread 保持為 base_spread，skew 依 inventory 微調。"""

    spread_action = 0.0  # 0 代表使用 config 中的 base_spread
    threshold = 0.2 * max_inventory
    if inventory > threshold:
        skew = min(1.0, inventory / max_inventory)
    elif inventory < -threshold:
        skew = max(-1.0, inventory / max_inventory)
    else:
        skew = 0.0
    return np.array([spread_action, skew], dtype=np.float32)


def run_single_episode(env_kwargs: Dict[str, object], seed: int | None = None) -> pd.DataFrame:
    local_kwargs = dict(env_kwargs)
    local_kwargs["seed"] = seed
    env = HistoricalMarketMakingEnv(**local_kwargs)
    obs, _ = env.reset()
    done = False
    records: List[Dict[str, float]] = []
    while not done:
        action = baseline_policy(env.inventory, env.max_inventory)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        records.append(
            {
                "step": info.get("step", len(records)),
                "mid_price": env.mid,
                "inventory": info.get("inventory", env.inventory),
                "portfolio_value": info.get("portfolio_value", env.cash + env.inventory * env.mid),
                "reward": float(reward),
            }
        )
    env.close()
    return pd.DataFrame(records)


def plot_equity_curve(df: pd.DataFrame, episode_idx: int, plots_dir: Path) -> Path:
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / f"baseline_episode_{episode_idx:02d}.png"
    plt.figure(figsize=(8, 3))
    plt.plot(df["step"], df["portfolio_value"], label="Portfolio Value")
    plt.xlabel("Step")
    plt.ylabel("USD")
    plt.title(f"Baseline Episode {episode_idx}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="執行手寫 baseline 策略做環境 sanity check")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML/JSON 環境設定檔")
    parser.add_argument("--episodes", type=int, default=10, help="要跑幾個 episodes 觀察資金曲線")
    parser.add_argument("--seed", type=int, default=2024, help="baseline 隨機種子，會對每個 episode 偏移")
    parser.add_argument("--log_dir", type=Path, default=BASELINE_LOG_DIR, help="儲存 baseline CSV 的目錄")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    env_kwargs = build_env_kwargs(config.env, root_dir=ROOT)
    env_kwargs["random_start"] = True  # baseline 偏好多個隨機起點

    log_dir = args.log_dir if args.log_dir.is_absolute() else ROOT / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    all_final_values: List[float] = []
    for ep in range(1, args.episodes + 1):
        df = run_single_episode(env_kwargs, seed=args.seed + ep)
        plot_path = plot_equity_curve(df, ep, PLOTS_DIR)
        final_pv = float(df["portfolio_value"].iloc[-1]) if not df.empty else 0.0
        all_final_values.append(final_pv)
        print(f"Episode {ep:02d}: final PV = {final_pv:.2f}, plot -> {plot_path}")
        csv_path = log_dir / f"baseline_episode_{ep:02d}.csv"
        df.to_csv(csv_path, index=False)

    avg_final = np.mean(all_final_values) if all_final_values else 0.0
    std_final = np.std(all_final_values) if all_final_values else 0.0
    print("\n=== Baseline Summary ===")
    print(f"平均最終資金：{avg_final:.2f} USD")
    print(f"標準差：{std_final:.2f} USD")
    print("註：此手寫策略僅供環境檢查，若曲線崩壞代表需要調整 fee 或懲罰參數。")


if __name__ == "__main__":
    main()
