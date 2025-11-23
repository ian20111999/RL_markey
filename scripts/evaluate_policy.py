"""evaluate_policy.py: 針對 train/valid/test 分段評估既有模型，避免只看單次回測。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC

from envs.historical_market_making_env import HistoricalMarketMakingEnv
from utils.config import build_env_kwargs, load_config
from utils.metrics import max_drawdown, sharpe_ratio

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "env_baseline.yaml"
PLOTS_DIR = ROOT / "plots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分段評估 SAC 做市模型")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML/JSON 設定檔")
    parser.add_argument("--model_path", type=Path, required=True, help="要評估的 model.zip 路徑")
    parser.add_argument("--episodes", type=int, default=3, help="每個時間段重複 episode 次數")
    parser.add_argument("--device", type=str, default="auto", help="模型推論裝置 (auto/cpu/cuda/mps)")
    parser.add_argument("--output_dir", type=Path, default=None, help="評估結果輸出目錄，預設為模型目錄/evaluation")
    return parser.parse_args()


def extract_segments(split_cfg: Dict[str, str]) -> Dict[str, Tuple[str | None, str | None]]:
    mapping = {
        "train": (split_cfg.get("train_start"), split_cfg.get("train_end")),
        "valid": (split_cfg.get("valid_start"), split_cfg.get("valid_end")),
        "test": (split_cfg.get("test_start"), split_cfg.get("test_end")),
    }
    return {k: v for k, v in mapping.items() if v[0] or v[1]}


def rollout_episode(model: SAC, env_kwargs: Dict[str, object], seed: int) -> Tuple[List[int], List[float], List[float]]:
    local_kwargs = dict(env_kwargs)
    local_kwargs["seed"] = seed
    env = HistoricalMarketMakingEnv(**local_kwargs)
    obs, _ = env.reset()
    done = False
    steps: List[int] = []
    pvs: List[float] = []
    rewards: List[float] = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps.append(int(info.get("step", len(steps))))
        pvs.append(float(info.get("portfolio_value", env.cash + env.inventory * env.mid)))
        rewards.append(float(reward))
    env.close()
    return steps, pvs, rewards


def summarize_segment(pv_histories: List[List[float]], reward_histories: List[List[float]]) -> Dict[str, float]:
    finals = [pv[-1] for pv in pv_histories if pv]
    concat_returns = np.concatenate(
        [np.diff(np.array(pv), prepend=pv[0]) for pv in pv_histories if len(pv) > 0]
    ) if pv_histories else np.array([])
    mean_return = float(np.mean(concat_returns)) if concat_returns.size else 0.0
    return_std = float(np.std(concat_returns)) if concat_returns.size else 0.0
    sharpe = sharpe_ratio(concat_returns) if concat_returns.size else 0.0
    dd_list = [max_drawdown(pv) for pv in pv_histories if pv]
    mean_reward = float(np.mean([np.sum(r) for r in reward_histories])) if reward_histories else 0.0
    return {
        "mean_final_pv": float(np.mean(finals)) if finals else 0.0,
        "mean_return": mean_return,
        "return_std": return_std,
        "sharpe": sharpe,
        "max_drawdown": float(max(dd_list)) if dd_list else 0.0,
        "mean_episode_reward": mean_reward,
    }


def save_test_plot(test_histories: List[Tuple[List[int], List[float]]], model_name: str) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOTS_DIR / f"test_equity_{model_name}.png"
    plt.figure(figsize=(9, 4))
    for idx, (_, pv) in enumerate(test_histories, start=1):
        plt.plot(pv, label=f"Episode {idx}")
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value")
    plt.title("Test 段資金曲線（多次重複）")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_path = args.model_path
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型 {model_path}")
    model = SAC.load(model_path, device=args.device)

    output_dir = args.output_dir or model_path.parent / "evaluation"
    output_dir = output_dir if output_dir.is_absolute() else ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    segments = extract_segments(config.data_split)
    if not segments:
        raise ValueError("config.data_split 未提供 train/valid/test 日期，無法分段評估。")

    segment_metrics: Dict[str, Dict[str, float]] = {}
    test_histories: List[Tuple[List[int], List[float]]] = []
    for idx, (seg_name, date_range) in enumerate(segments.items()):
        # 多次重複不同 seed，避免被單一幸運樣本誤導
        seg_kwargs = build_env_kwargs(config.env, root_dir=ROOT, random_start=True, date_range=date_range)
        pv_histories: List[List[float]] = []
        reward_histories: List[List[float]] = []
        for ep in range(args.episodes):
            seed = 10_000 * (idx + 1) + ep
            steps, pvs, rewards = rollout_episode(model, seg_kwargs, seed=seed)
            pv_histories.append(pvs)
            reward_histories.append(rewards)
            if seg_name == "test":
                test_histories.append((steps, pvs))
            csv_path = output_dir / f"{seg_name}_episode_{ep+1:02d}.csv"
            with csv_path.open("w", encoding="utf-8") as f:
                f.write("step,portfolio_value\n")
                for step, pv in zip(steps, pvs):
                    f.write(f"{step},{pv}\n")
        metrics = summarize_segment(pv_histories, reward_histories)
        segment_metrics[seg_name] = metrics
        print(f"\n[{seg_name.upper()}] metrics: {json.dumps(metrics, ensure_ascii=False)}")

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(segment_metrics, f, ensure_ascii=False, indent=2)

    if test_histories:
        plot_path = save_test_plot(test_histories, model_path.parent.name)
        csv_path = output_dir / "test_equity_curve.csv"
        with csv_path.open("w", encoding="utf-8") as f:
            f.write("episode,step,portfolio_value\n")
            for ep_idx, (steps, pvs) in enumerate(test_histories, start=1):
                for step, pv in zip(steps, pvs):
                    f.write(f"{ep_idx},{step},{pv}\n")
        print(f"Test 段資金曲線已輸出：{plot_path}, {csv_path}")

    print(f"完整評估輸出：{output_dir}")


if __name__ == "__main__":
    main()
