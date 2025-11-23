"""使用 Optuna 搜尋 SAC 做市 agent 的超參數組合。
透過自動化 tuning 減少手動調整 learning rate / net_arch 的時間。
"""
from __future__ import annotations

import argparse
import datetime
import json
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import optuna
from optuna.exceptions import TrialPruned
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.callbacks import BaseCallback

from envs.historical_market_making_env import HistoricalMarketMakingEnv
from utils.config import build_env_kwargs, load_config

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DEFAULT_CSV = DATA_DIR / "btc_usdt_1m_2023.csv"
BEST_PARAMS_PATH = MODELS_DIR / "best_sac_params.json"

NetArch = List[int]
NET_ARCH_LIBRARY: Dict[str, NetArch] = {
    "64x2": [64, 64],
    "128x2": [128, 128],
    "256x2": [256, 256],
}


def env_section_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    csv_path = Path(args.csv_path)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    return {
        "csv_path": str(csv_path),
        "episode_length": int(args.episode_length),
        "fee_rate": float(args.fee_rate),
        "lambda_inv": float(args.lambda_inv),
        "lambda_turnover": float(args.lambda_turnover),
        "max_inventory": float(args.max_inventory),
        "base_spread": float(args.base_spread),
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "random_start": bool(args.random_start),
    }


def apply_config_overrides(args: argparse.Namespace) -> None:
    if not args.config:
        return
    cfg = load_config(args.config)
    env_cfg = cfg.env
    for key in [
        "csv_path",
        "episode_length",
        "fee_rate",
        "lambda_inv",
        "lambda_turnover",
        "max_inventory",
        "base_spread",
        "alpha",
        "beta",
        "random_start",
    ]:
        if key in env_cfg:
            value = env_cfg[key]
            if key == "csv_path":
                value = Path(value)
            setattr(args, key, value)
    train_cfg = cfg.train
    mapping = {
        "total_timesteps": "train_timesteps",
        "buffer_size": "buffer_size",
        "train_freq": "train_freq",
        "gradient_steps": "gradient_steps",
    }
    for src, dest in mapping.items():
        if src in train_cfg:
            setattr(args, dest, train_cfg[src])
def make_env(env_kwargs: Dict[str, Any], seed: int | None = None, random_start: bool = True) -> Callable[[], HistoricalMarketMakingEnv]:
    """封裝環境建立函式，方便 DummyVecEnv 呼叫。"""

    def _init() -> HistoricalMarketMakingEnv:
        local_kwargs = dict(env_kwargs)
        local_kwargs["random_start"] = random_start
        local_kwargs["seed"] = seed
        env = HistoricalMarketMakingEnv(**local_kwargs)
        return Monitor(env)

    return _init


def evaluate_model(
    model: SAC,
    eval_env: VecEnv,
    n_episodes: int,
    metric: str,
) -> float:
    """
    在獨立驗證環境上回測，回傳平均績效。
    優化：使用傳入的 VecEnv，支援平行評估，且不需重複建立環境。
    """
    obs = eval_env.reset()
    n_envs = eval_env.num_envs
    dones = np.array([False] * n_envs)
    episode_rewards = np.zeros(n_envs)
    final_values = np.zeros(n_envs)
    
    # 假設 n_episodes == n_envs，一次跑完
    while not all(dones):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones_step, infos = eval_env.step(action)
        
        for i in range(n_envs):
            if not dones[i]:
                episode_rewards[i] += rewards[i]
                if dones_step[i]:
                    dones[i] = True
                    if metric == "portfolio":
                        final_values[i] = infos[i].get("portfolio_value", 0.0)
                    else:
                        final_values[i] = episode_rewards[i]
    
    return float(np.mean(final_values))


def suggest_hyperparams(trial: optuna.Trial) -> Dict[str, object]:
    """定義 Optuna 搜尋空間。"""

    net_arch_label = trial.suggest_categorical("net_arch", list(NET_ARCH_LIBRARY.keys()))
    net_arch_choice = NET_ARCH_LIBRARY[net_arch_label]
    params: Dict[str, object] = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.98, 0.999),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "tau": trial.suggest_float("tau", 0.01, 0.1),
        "policy_kwargs": {"net_arch": list(net_arch_choice)},
    }
    return params


def decode_net_arch(value: object) -> NetArch | None:
    if isinstance(value, str) and value in NET_ARCH_LIBRARY:
        return NET_ARCH_LIBRARY[value]
    if isinstance(value, list):
        return value
    return None


class PruningCallback(BaseCallback):
    """
    Optuna Pruning Callback:
    每隔 eval_freq 步數，執行一次評估。
    若評估結果不佳（由 Optuna 判斷），則拋出 TrialPruned 中斷訓練。
    """
    def __init__(
        self,
        trial: optuna.Trial,
        eval_env: VecEnv,
        eval_freq: int = 20000,
        n_eval_episodes: int = 3,
        metric: str = "portfolio",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.trial = trial
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.metric = metric

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # 執行評估
            score = evaluate_model(
                model=self.model,
                eval_env=self.eval_env,
                n_episodes=self.n_eval_episodes,
                metric=self.metric,
            )
            
            # 回報給 Optuna
            self.trial.report(score, self.n_calls)
            
            # 檢查是否需要剪枝
            if self.trial.should_prune():
                raise TrialPruned()
                
        return True


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """單一 trial：取樣參數 -> 訓練 -> 驗證並回傳績效。"""

    hyperparams = suggest_hyperparams(trial)
    env_cfg = env_section_from_args(args)
    env_kwargs = build_env_kwargs(env_cfg, root_dir=ROOT)
    
    # 優化：使用 SubprocVecEnv 進行多進程並行採樣
    # 根據 CPU 核心數決定並行數量，這裡保守設為 4
    n_envs = 4
    env_fns = [
        make_env(
            env_kwargs=env_kwargs,
            seed=trial.number * n_envs + i,
            random_start=args.random_start,
        )
        for i in range(n_envs)
    ]
    train_env = SubprocVecEnv(env_fns)

    # 優化：建立持久化評估環境 (平行化)
    n_eval_envs = args.eval_episodes
    eval_kwargs = dict(env_kwargs)
    eval_kwargs["episode_length"] = args.eval_episode_length
    
    eval_env_fns = [
        make_env(
            env_kwargs=eval_kwargs,
            seed=trial.number * 200 + i,
            random_start=True
        )
        for i in range(n_eval_envs)
    ]
    eval_env = SubprocVecEnv(eval_env_fns)

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=hyperparams["learning_rate"],
        gamma=hyperparams["gamma"],
        batch_size=hyperparams["batch_size"],
        tau=hyperparams["tau"],
        buffer_size=args.buffer_size,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        policy_kwargs=hyperparams["policy_kwargs"],
        device=args.device,
        verbose=0,
    )

    # 設定 Pruning Callback
    # 頻率設為總步數的 1/4，即評估 4 次
    eval_freq = max(args.train_timesteps // 4, 1000)
    pruning_callback = PruningCallback(
        trial=trial,
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_envs,
        metric=args.metric,
    )

    try:
        model.learn(total_timesteps=args.train_timesteps, callback=pruning_callback)
    except TrialPruned:
        train_env.close()
        eval_env.close()
        raise
    
    train_env.close()

    score = evaluate_model(
        model=model,
        eval_env=eval_env,
        n_episodes=n_eval_envs,
        metric=args.metric,
    )
    eval_env.close()
    return score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 Optuna 搜尋 SAC 超參數")
    parser.add_argument("--config", type=Path, default=None, help="YAML/JSON 設定檔，用於覆寫 env/train 參數")
    parser.add_argument("--csv_path", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--episode_length", type=int, default=600, help="tuning 訓練時的 episode 長度")
    parser.add_argument("--fee_rate", type=float, default=0.0004)
    parser.add_argument("--lambda_inv", type=float, default=0.001)
    parser.add_argument("--lambda_turnover", type=float, default=0.0)
    parser.add_argument("--max_inventory", type=float, default=10.0)
    parser.add_argument("--base_spread", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--random_start", dest="random_start", action="store_true", help="預設從隨機位置開始 episode")
    parser.add_argument("--fixed_start", dest="random_start", action="store_false", help="固定從資料開頭開始 episode")
    parser.set_defaults(random_start=True)
    parser.add_argument("--train_timesteps", type=int, default=50_000)
    parser.add_argument("--buffer_size", type=int, default=50_000)
    parser.add_argument("--train_freq", type=int, default=1)
    parser.add_argument("--gradient_steps", type=int, default=1)
    parser.add_argument("--eval_episode_length", type=int, default=600)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--metric", choices=["portfolio", "reward"], default="portfolio")
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--study_name", type=str, default="mm_sac_optuna")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage，例如 sqlite:///optuna.db")
    parser.add_argument("--save_best_params", action="store_true")
    parser.add_argument("--best_params_path", type=Path, default=BEST_PARAMS_PATH)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Stable-Baselines3 裝置，常見值如 auto / cpu / cuda / mps",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_config_overrides(args)
    csv_path = Path(args.csv_path)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到資料檔案 {csv_path}，請先執行資料下載腳本。")
    args.csv_path = csv_path

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=args.storage is not None,
    )
    study.optimize(partial(objective, args=args), n_trials=args.n_trials)

    best_trial = study.best_trial
    print("=== Optuna 最佳結果 ===")
    print(f"score = {best_trial.value:.4f}")
    print("params =")
    for k, v in best_trial.params.items():
        readable = decode_net_arch(v) if k == "net_arch" else v
        print(f"  {k}: {readable}")

    if args.save_best_params:
        best_path = args.best_params_path
        best_path.parent.mkdir(parents=True, exist_ok=True)
        export_params = dict(best_trial.params)
        decoded_arch = decode_net_arch(export_params.get("net_arch"))
        if decoded_arch is not None:
            export_params["net_arch"] = decoded_arch
        with best_path.open("w", encoding="utf-8") as f:
            json.dump(export_params, f, indent=2, ensure_ascii=False)
        print(f"已將最佳超參數寫入 {best_path}")

    # 新增結束提示
    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*40}")
    print(f"✅ Tuning Complete: {args.study_name}")
    print(f"🕒 End Time: {end_time}")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    main()
