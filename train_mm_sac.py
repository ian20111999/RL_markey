"""使用 Stable-Baselines3 SAC 訓練歷史做市 Agent，支援統一 config 與實驗輸出。"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from envs.historical_market_making_env import HistoricalMarketMakingEnv
from utils.config import ExperimentConfig, build_env_kwargs, export_config, load_config
from utils.metrics import max_drawdown, sharpe_ratio

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_CSV = DATA_DIR / "btc_usdt_1m_2023.csv"
DEFAULT_RUNS_DIR = ROOT / "runs"


class TrainLoggerCallback(BaseCallback):
    """每隔固定步數紀錄 reward 與投資組合價值，協助後續分析。"""

    def __init__(self, log_path: Path, log_interval: int) -> None:
        super().__init__()
        self.log_path = log_path
        self.log_interval = max(1, log_interval)
        self._last_step = 0
        self._csv_file = None
        self._writer: csv.DictWriter[str] | None = None

    def _on_training_start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = self.log_path.open("w", newline="", encoding="utf-8")
        fieldnames = ["timesteps", "reward_mean", "portfolio_value", "inventory", "trades_count"]
        self._writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
        self._writer.writeheader()

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_step < self.log_interval:
            return True
        infos = self.locals.get("infos", [{}])
        info = infos[0] if infos else {}
        rewards = self.locals.get("rewards")
        reward_mean = float(np.mean(rewards)) if rewards is not None else 0.0
        row = {
            "timesteps": int(self.num_timesteps),
            "reward_mean": reward_mean,
            "portfolio_value": float(info.get("portfolio_value", np.nan)),
            "inventory": float(info.get("inventory", np.nan)),
            "trades_count": float(info.get("trades_count", np.nan)),
        }
        if self._writer and self._csv_file:
            self._writer.writerow(row)
            self._csv_file.flush()
        self._last_step = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
        if self._csv_file:
            self._csv_file.close()


def build_default_config(args: argparse.Namespace) -> ExperimentConfig:
    raw = {
        "env": {
            "csv_path": str(args.csv_path),
            "episode_length": args.episode_length,
            "fee_rate": args.fee_rate,
            "lambda_inv": args.lambda_inv,
            "lambda_turnover": args.lambda_turnover,
            "max_inventory": args.max_inventory,
            "base_spread": args.base_spread,
            "alpha": args.alpha,
            "beta": args.beta,
            "random_start": not args.deterministic_start,
        },
        "train": {
            "algo": "SAC",
            "total_timesteps": args.total_timesteps,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "batch_size": args.batch_size,
            "tau": args.tau,
            "train_freq": args.train_freq,
            "gradient_steps": args.gradient_steps,
            "buffer_size": args.buffer_size,
            "eval_freq": args.eval_freq,
            "net_arch": args.net_arch,
            "seed": args.seed,
            "deterministic_eval": True,
            "log_interval": args.log_interval,
        },
        "run": {
            "short_name": args.run_name or "manual",
            "notes": args.notes,
        },
        "data_split": {},
    }
    return ExperimentConfig(raw=raw)


def resolve_config(args: argparse.Namespace) -> ExperimentConfig:
    if args.config:
        return load_config(args.config)
    return build_default_config(args)


def ensure_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = ROOT / path
    return path


def make_env_factory(env_kwargs: Dict[str, Any], seed: int | None = None):
    def _init() -> Monitor:
        local_kwargs = dict(env_kwargs)
        if seed is not None:
            local_kwargs["seed"] = seed
        env = HistoricalMarketMakingEnv(**local_kwargs)
        return Monitor(env)

    return _init


def create_run_directory(train_cfg: Dict[str, Any], run_cfg: Dict[str, Any], runs_root: Path) -> Path:
    algo = str(train_cfg.get("algo", "SAC")).upper()
    short_name = run_cfg.get("short_name") or "exp"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / algo / f"{timestamp}_{short_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def rollout_episode(model: SAC, env_kwargs: Dict[str, Any], deterministic: bool = True) -> Tuple[List[int], List[float], List[float]]:
    eval_kwargs = dict(env_kwargs)
    eval_kwargs["random_start"] = False
    env = HistoricalMarketMakingEnv(**eval_kwargs)
    obs, _ = env.reset()
    done = False
    steps: List[int] = []
    pvs: List[float] = []
    rewards: List[float] = []
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps.append(int(info.get("step", len(steps))))
        pvs.append(float(info.get("portfolio_value", env.cash + env.inventory * env.mid)))
        rewards.append(float(reward))
    env.close()
    return steps, pvs, rewards


def save_equity_curve(steps: Iterable[int], pvs: Iterable[float], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "portfolio_value"])
        for step, pv in zip(steps, pvs):
            writer.writerow([step, pv])


def plot_equity_curve(pvs: List[float], plot_path: Path) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(pvs, label="Portfolio Value")
    plt.xlabel("Step")
    plt.ylabel("USD")
    plt.title("SAC 做市 Agent 資金曲線（單次測試）")
    plt.legend()
    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)
    plt.close()


def summarize_metrics(pvs: List[float], rewards: List[float]) -> Dict[str, float]:
    if not pvs:
        return {
            "final_portfolio_value": 0.0,
            "mean_reward": 0.0,
            "mean_return": 0.0,
            "return_std": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }
    pv_array = np.array(pvs, dtype=np.float64)
    returns = np.diff(pv_array, prepend=pv_array[0])
    mean_return = float(np.mean(returns))
    return_std = float(np.std(returns))
    sharpe = sharpe_ratio(returns)
    return {
        "final_portfolio_value": float(pv_array[-1]),
        "mean_reward": float(np.mean(rewards) if rewards else 0.0),
        "mean_return": mean_return,
        "return_std": return_std,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(pv_array),
    }


def train(args: argparse.Namespace) -> None:
    config = resolve_config(args)
    env_cfg = config.env
    train_cfg = config.train
    run_cfg = config.run

    # 若 command line 有指定 run_name，強制覆蓋 config
    if args.run_name:
        run_cfg["short_name"] = args.run_name

    # 若有指定 params_path，則讀取並覆蓋 train_cfg
    if args.params_path and args.params_path.exists():
        print(f"正在載入最佳參數: {args.params_path}")
        with open(args.params_path, "r", encoding="utf-8") as f:
            best_params = json.load(f)
        
        # 覆蓋關鍵超參數
        for key in ["learning_rate", "gamma", "batch_size", "tau", "buffer_size", "train_freq", "gradient_steps"]:
            if key in best_params:
                val = best_params[key]
                print(f"  Overriding {key}: {train_cfg.get(key)} -> {val}")
                train_cfg[key] = val
        
        # net_arch 特殊處理 (JSON 中可能是 list 或 string "256x2")
        if "net_arch" in best_params:
            na = best_params["net_arch"]
            if isinstance(na, str) and "x" in na:
                # 處理 "256x2" 這種格式 (雖然 tune_mm_sac 似乎存的是 list，但以防萬一)
                units, layers = map(int, na.split("x"))
                na = [units] * layers
            print(f"  Overriding net_arch: {train_cfg.get('net_arch')} -> {na}")
            train_cfg["net_arch"] = na

    env_kwargs = build_env_kwargs(env_cfg, root_dir=ROOT, seed=train_cfg.get("seed"))
    csv_path = Path(env_kwargs["csv_path"])
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 {csv_path}，請先下載資料或更新 config。")

    runs_root = ensure_path(args.runs_dir or DEFAULT_RUNS_DIR)
    run_dir = create_run_directory(train_cfg, run_cfg, runs_root)
    export_config(config, run_dir / "config.yaml")

    # 優化：使用 SubprocVecEnv 加速訓練
    n_envs = 4
    base_seed = train_cfg.get("seed")
    if base_seed is None:
        base_seed = np.random.randint(0, 10000)

    train_env_fns = [
        make_env_factory(env_kwargs, seed=base_seed + i)
        for i in range(n_envs)
    ]
    train_env = SubprocVecEnv(train_env_fns)
    
    # 評估環境保持單一即可
    eval_env = DummyVecEnv([make_env_factory(env_kwargs, seed=base_seed + 100)])

    policy_kwargs: Dict[str, Any] = {}
    net_arch = train_cfg.get("net_arch")
    if net_arch:
        policy_kwargs["net_arch"] = list(net_arch)

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=float(train_cfg.get("learning_rate", args.learning_rate)),
        batch_size=int(train_cfg.get("batch_size", args.batch_size)),
        gamma=float(train_cfg.get("gamma", args.gamma)),
        tau=float(train_cfg.get("tau", args.tau)),
        buffer_size=int(train_cfg.get("buffer_size", args.buffer_size)),
        train_freq=int(train_cfg.get("train_freq", args.train_freq)),
        gradient_steps=int(train_cfg.get("gradient_steps", args.gradient_steps)),
        device=args.device,
        verbose=1,
        policy_kwargs=policy_kwargs or None,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "checkpoints"),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=int(train_cfg.get("eval_freq", args.eval_freq)),
        deterministic=bool(train_cfg.get("deterministic_eval", True)),
        render=False,
    )
    train_logger = TrainLoggerCallback(
        log_path=run_dir / "train_log.csv",
        log_interval=int(train_cfg.get("log_interval", args.log_interval)),
    )

    total_timesteps = int(train_cfg.get("total_timesteps", args.total_timesteps))
    callback = CallbackList([eval_callback, train_logger])
    model.learn(total_timesteps=total_timesteps, callback=callback)

    model_path = run_dir / "model"
    model.save(model_path)

    steps, pvs, rewards = rollout_episode(model, env_kwargs)
    save_equity_curve(steps, pvs, run_dir / "test_equity_curve.csv")
    plot_equity_curve(pvs, run_dir / "test_equity_curve.png")

    metrics = summarize_metrics(pvs, rewards)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"訓練完成，輸出位於 {run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="訓練歷史做市 SAC agent（config 版）")
    parser.add_argument("--config", type=Path, default=None, help="YAML/JSON 實驗設定檔路徑")
    parser.add_argument("--params_path", type=Path, default=None, help="最佳超參數 JSON 檔案路徑 (若提供將覆蓋 config 設定)")
    parser.add_argument("--run_name", type=str, default=None, help="若無 config.run.short_name，可自訂名稱")
    parser.add_argument("--notes", type=str, default="", help="此實驗的備註說明")
    parser.add_argument("--runs_dir", type=Path, default=DEFAULT_RUNS_DIR, help="實驗輸出根目錄")
    parser.add_argument("--device", type=str, default="auto", help="Stable-Baselines3 裝置 (auto/cpu/cuda/mps)")

    # env fallback
    parser.add_argument("--csv_path", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--episode_length", type=int, default=1000)
    parser.add_argument("--fee_rate", type=float, default=0.0004)
    parser.add_argument("--lambda_inv", type=float, default=0.001)
    parser.add_argument("--lambda_turnover", type=float, default=0.0)
    parser.add_argument("--max_inventory", type=float, default=10.0)
    parser.add_argument("--base_spread", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--deterministic_start", action="store_true", help="是否固定從資料起點開始 episode")

    # train fallback
    parser.add_argument("--total_timesteps", type=int, default=200_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--train_freq", type=int, default=1)
    parser.add_argument("--gradient_steps", type=int, default=1)
    parser.add_argument("--buffer_size", type=int, default=100_000)
    parser.add_argument("--eval_freq", type=int, default=10_000)
    parser.add_argument("--net_arch", nargs="*", type=int, default=[256, 256])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=5_000)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
