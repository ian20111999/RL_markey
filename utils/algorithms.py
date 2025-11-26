"""
utils/algorithms.py
多演算法支援模組

支援演算法:
- SAC (Soft Actor-Critic): 最大熵 RL，適合連續動作
- PPO (Proximal Policy Optimization): 更穩定，樣本效率較低
- TD3 (Twin Delayed DDPG): SAC 的確定性版本

用法:
    from utils.algorithms import create_model, ALGO_CONFIGS
    
    model = create_model(
        algo="sac",
        env=env,
        learning_rate=3e-4,
        tensorboard_log="./logs/"
    )
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Type, Union
import gymnasium as gym

from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import numpy as np


# =============================================================================
# 預設超參數配置
# =============================================================================

ALGO_CONFIGS: Dict[str, Dict[str, Any]] = {
    "sac": {
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "buffer_size": 100_000,
        "learning_starts": 1000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto",
        "target_update_interval": 1,
        "target_entropy": "auto",
        "policy_kwargs": {
            "net_arch": [256, 256],
        },
    },
    "ppo": {
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "normalize_advantage": True,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "policy_kwargs": {
            "net_arch": dict(pi=[256, 256], vf=[256, 256]),
        },
    },
    "td3": {
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "buffer_size": 100_000,
        "learning_starts": 1000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "policy_delay": 2,
        "target_policy_noise": 0.2,
        "target_noise_clip": 0.5,
        "policy_kwargs": {
            "net_arch": [256, 256],
        },
    },
}

# 演算法類別映射
ALGO_CLASSES: Dict[str, Type[BaseAlgorithm]] = {
    "sac": SAC,
    "ppo": PPO,
    "td3": TD3,
}


# =============================================================================
# 工廠函數
# =============================================================================

def get_algo_class(algo_name: str) -> Type[BaseAlgorithm]:
    """取得演算法類別"""
    algo_name = algo_name.lower()
    if algo_name not in ALGO_CLASSES:
        raise ValueError(f"不支援的演算法: {algo_name}. 支援: {list(ALGO_CLASSES.keys())}")
    return ALGO_CLASSES[algo_name]


def get_default_config(algo_name: str) -> Dict[str, Any]:
    """取得演算法預設配置"""
    algo_name = algo_name.lower()
    if algo_name not in ALGO_CONFIGS:
        raise ValueError(f"不支援的演算法: {algo_name}")
    return dict(ALGO_CONFIGS[algo_name])


def create_model(
    algo: str,
    env: gym.Env,
    *,
    config_overrides: Optional[Dict[str, Any]] = None,
    tensorboard_log: Optional[str] = None,
    verbose: int = 1,
    seed: Optional[int] = None,
    device: str = "auto",
) -> BaseAlgorithm:
    """
    建立 RL 模型
    
    Args:
        algo: 演算法名稱 ("sac", "ppo", "td3")
        env: Gymnasium 環境
        config_overrides: 覆寫預設配置的字典
        tensorboard_log: TensorBoard 日誌路徑
        verbose: 輸出詳細程度
        seed: 隨機種子
        device: 運算裝置 ("auto", "cpu", "cuda")
    
    Returns:
        初始化的 RL 模型
    """
    algo_name = algo.lower()
    algo_class = get_algo_class(algo_name)
    config = get_default_config(algo_name)
    
    # 合併覆寫配置
    if config_overrides:
        for key, value in config_overrides.items():
            if key == "policy_kwargs" and "policy_kwargs" in config:
                config["policy_kwargs"].update(value)
            else:
                config[key] = value
    
    # 提取 policy
    policy = config.pop("policy", "MlpPolicy")
    
    # TD3 需要 action noise
    action_noise = None
    if algo_name == "td3":
        n_actions = env.action_space.shape[0]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )
        config["action_noise"] = action_noise
    
    # 建立模型
    model = algo_class(
        policy=policy,
        env=env,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        seed=seed,
        device=device,
        **config
    )
    
    return model


def load_model(
    algo: str,
    path: str,
    env: Optional[gym.Env] = None,
    device: str = "auto",
) -> BaseAlgorithm:
    """載入已訓練的模型"""
    algo_class = get_algo_class(algo)
    return algo_class.load(path, env=env, device=device)


# =============================================================================
# 演算法比較工具
# =============================================================================

class AlgorithmComparator:
    """比較不同演算法的工具類別"""
    
    def __init__(
        self,
        env_factory,
        algos: list[str] = ["sac", "ppo", "td3"],
        n_seeds: int = 3,
        total_timesteps: int = 100_000,
    ):
        """
        Args:
            env_factory: 建立環境的函數 () -> gym.Env
            algos: 要比較的演算法列表
            n_seeds: 每個演算法訓練的種子數
            total_timesteps: 總訓練步數
        """
        self.env_factory = env_factory
        self.algos = algos
        self.n_seeds = n_seeds
        self.total_timesteps = total_timesteps
        self.results: Dict[str, list] = {algo: [] for algo in algos}
    
    def run_comparison(
        self,
        eval_env_factory=None,
        n_eval_episodes: int = 10,
        eval_freq: int = 10_000,
        output_dir: str = "algo_comparison",
    ) -> Dict[str, Any]:
        """
        執行演算法比較
        
        Returns:
            包含各演算法結果的字典
        """
        from pathlib import Path
        from stable_baselines3.common.evaluation import evaluate_policy
        import json
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        eval_env_factory = eval_env_factory or self.env_factory
        
        all_results = {}
        
        for algo in self.algos:
            algo_results = []
            print(f"\n{'='*50}")
            print(f"Training {algo.upper()}")
            print(f"{'='*50}")
            
            for seed in range(self.n_seeds):
                print(f"\n--- Seed {seed + 1}/{self.n_seeds} ---")
                
                # 建立環境和模型
                env = self.env_factory()
                model = create_model(
                    algo=algo,
                    env=env,
                    seed=seed,
                    tensorboard_log=str(output_path / "logs"),
                    verbose=0,
                )
                
                # 訓練
                model.learn(
                    total_timesteps=self.total_timesteps,
                    progress_bar=True,
                )
                
                # 評估
                eval_env = eval_env_factory()
                mean_reward, std_reward = evaluate_policy(
                    model, eval_env, n_eval_episodes=n_eval_episodes
                )
                
                result = {
                    "seed": seed,
                    "mean_reward": float(mean_reward),
                    "std_reward": float(std_reward),
                }
                algo_results.append(result)
                
                print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
                
                # 儲存模型
                model.save(output_path / f"{algo}_seed{seed}")
                
                env.close()
                eval_env.close()
            
            # 彙總該演算法結果
            rewards = [r["mean_reward"] for r in algo_results]
            all_results[algo] = {
                "seeds": algo_results,
                "mean": float(np.mean(rewards)),
                "std": float(np.std(rewards)),
                "max": float(np.max(rewards)),
                "min": float(np.min(rewards)),
            }
            
            self.results[algo] = algo_results
        
        # 儲存結果
        with open(output_path / "comparison_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        # 打印比較結果
        print("\n" + "=" * 60)
        print("ALGORITHM COMPARISON RESULTS")
        print("=" * 60)
        for algo, res in all_results.items():
            print(f"{algo.upper():>6}: {res['mean']:>10.2f} ± {res['std']:<8.2f} "
                  f"(min: {res['min']:.2f}, max: {res['max']:.2f})")
        
        return all_results


# =============================================================================
# 自訂 Policy 網路
# =============================================================================

def create_custom_policy_kwargs(
    net_arch: list[int] = [256, 256],
    activation_fn: str = "relu",
    use_sde: bool = False,
    log_std_init: float = -3.0,
) -> Dict[str, Any]:
    """
    建立自訂 Policy 網路配置
    
    Args:
        net_arch: 網路架構 (隱藏層大小列表)
        activation_fn: 激活函數 ("relu", "tanh", "elu")
        use_sde: 是否使用 State-Dependent Exploration
        log_std_init: 初始 log 標準差 (僅用於 SAC)
    
    Returns:
        policy_kwargs 字典
    """
    import torch.nn as nn
    
    activation_map = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }
    
    if activation_fn.lower() not in activation_map:
        raise ValueError(f"不支援的激活函數: {activation_fn}")
    
    kwargs = {
        "net_arch": net_arch,
        "activation_fn": activation_map[activation_fn.lower()],
    }
    
    if use_sde:
        kwargs["use_sde"] = True
        kwargs["log_std_init"] = log_std_init
    
    return kwargs


# =============================================================================
# 學習率調度器
# =============================================================================

def linear_schedule(initial_value: float):
    """
    線性學習率調度器
    
    Args:
        initial_value: 初始學習率
    
    Returns:
        學習率函數
    """
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return schedule


def cosine_schedule(initial_value: float, min_value: float = 1e-6):
    """
    餘弦學習率調度器
    
    Args:
        initial_value: 初始學習率
        min_value: 最小學習率
    
    Returns:
        學習率函數
    """
    import math
    
    def schedule(progress_remaining: float) -> float:
        return min_value + 0.5 * (initial_value - min_value) * (
            1 + math.cos(math.pi * (1 - progress_remaining))
        )
    return schedule


def step_schedule(initial_value: float, milestones: list[float], gamma: float = 0.1):
    """
    階梯學習率調度器
    
    Args:
        initial_value: 初始學習率
        milestones: 降低學習率的進度點列表 (0.0 - 1.0)
        gamma: 學習率衰減因子
    
    Returns:
        學習率函數
    """
    def schedule(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining
        lr = initial_value
        for milestone in milestones:
            if progress >= milestone:
                lr *= gamma
        return lr
    return schedule
