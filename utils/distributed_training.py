# =============================================================================
# 分散式訓練工具
# 支援多環境並行、多種子驗證、超參數搜尋
# =============================================================================

import os
import json
import time
import pickle
import logging
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd

try:
    from stable_baselines3 import SAC, PPO, TD3
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

try:
    import optuna
    from optuna.trial import Trial
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

logger = logging.getLogger(__name__)


# =============================================================================
# 資料類別
# =============================================================================

@dataclass
class TrainingResult:
    """單次訓練結果"""
    seed: int
    mean_reward: float
    std_reward: float
    episodes: int
    training_time: float
    final_model_path: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'seed': self.seed,
            'mean_reward': self.mean_reward,
            'std_reward': self.std_reward,
            'episodes': self.episodes,
            'training_time': self.training_time,
            'final_model_path': self.final_model_path,
            'metrics': self.metrics,
            'config': self.config
        }


@dataclass
class HyperparameterSearchResult:
    """超參數搜尋結果"""
    best_params: Dict[str, Any]
    best_value: float
    all_trials: List[Dict]
    study_name: str
    n_trials: int
    search_time: float
    
    def to_dict(self) -> Dict:
        return {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'all_trials': self.all_trials,
            'study_name': self.study_name,
            'n_trials': self.n_trials,
            'search_time': self.search_time
        }


@dataclass
class MultiSeedResult:
    """多種子驗證結果"""
    seeds: List[int]
    results: List[TrainingResult]
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    
    def compute_aggregates(self):
        """計算聚合指標"""
        rewards = [r.mean_reward for r in self.results]
        self.aggregate_metrics = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'median_reward': np.median(rewards),
            'cv': np.std(rewards) / (np.mean(rewards) + 1e-8),  # Coefficient of Variation
            'n_seeds': len(self.seeds),
            'total_training_time': sum(r.training_time for r in self.results)
        }
        return self.aggregate_metrics


# =============================================================================
# 多環境並行訓練
# =============================================================================

class ParallelEnvTrainer:
    """
    多環境並行訓練器
    使用 SubprocVecEnv 實現真正的並行環境採樣
    """
    
    def __init__(
        self,
        env_fn: Callable,
        n_envs: int = 4,
        algorithm: str = "SAC",
        device: str = "auto"
    ):
        self.env_fn = env_fn
        self.n_envs = n_envs
        self.algorithm = algorithm.upper()
        self.device = device
        self.vec_env = None
        self.model = None
    
    def _create_vec_env(self, use_subprocess: bool = True) -> Union[SubprocVecEnv, DummyVecEnv]:
        """建立向量化環境"""
        env_fns = [self.env_fn for _ in range(self.n_envs)]
        
        if use_subprocess and self.n_envs > 1:
            return SubprocVecEnv(env_fns, start_method='spawn')
        else:
            return DummyVecEnv(env_fns)
    
    def _get_algorithm_class(self):
        """取得演算法類別"""
        if not HAS_SB3:
            raise ImportError("stable_baselines3 is required for training")
        
        algo_map = {
            'SAC': SAC,
            'PPO': PPO,
            'TD3': TD3
        }
        
        if self.algorithm not in algo_map:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        return algo_map[self.algorithm]
    
    def train(
        self,
        total_timesteps: int,
        hyperparams: Optional[Dict] = None,
        callbacks: Optional[List[BaseCallback]] = None,
        eval_env: Any = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        save_path: Optional[str] = None,
        use_subprocess: bool = True
    ) -> TrainingResult:
        """
        執行訓練
        
        Args:
            total_timesteps: 總訓練步數
            hyperparams: 超參數字典
            callbacks: 回調列表
            eval_env: 評估環境
            eval_freq: 評估頻率
            n_eval_episodes: 每次評估的回合數
            save_path: 模型保存路徑
            use_subprocess: 是否使用子進程
        
        Returns:
            TrainingResult: 訓練結果
        """
        start_time = time.time()
        
        # 建立環境
        self.vec_env = self._create_vec_env(use_subprocess)
        
        # 預設超參數
        default_params = {
            'learning_rate': 3e-4,
            'buffer_size': 100000,
            'batch_size': 256,
            'gamma': 0.99,
            'policy': 'MlpPolicy',
            'verbose': 0
        }
        
        if hyperparams:
            default_params.update(hyperparams)
        
        # 建立模型
        algo_class = self._get_algorithm_class()
        self.model = algo_class(
            default_params.pop('policy', 'MlpPolicy'),
            self.vec_env,
            device=self.device,
            **default_params
        )
        
        # 設定回調
        all_callbacks = callbacks or []
        
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=save_path,
                log_path=save_path,
                eval_freq=eval_freq // self.n_envs,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False
            )
            all_callbacks.append(eval_callback)
        
        # 訓練
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=all_callbacks if all_callbacks else None,
            progress_bar=True
        )
        
        # 評估
        if eval_env is not None:
            mean_reward, std_reward = evaluate_policy(
                self.model, eval_env, n_eval_episodes=n_eval_episodes
            )
        else:
            mean_reward, std_reward = 0.0, 0.0
        
        training_time = time.time() - start_time
        
        # 保存模型
        final_model_path = None
        if save_path:
            final_model_path = os.path.join(save_path, "final_model")
            self.model.save(final_model_path)
        
        # 清理
        self.vec_env.close()
        
        return TrainingResult(
            seed=0,
            mean_reward=mean_reward,
            std_reward=std_reward,
            episodes=n_eval_episodes,
            training_time=training_time,
            final_model_path=final_model_path,
            metrics={
                'total_timesteps': total_timesteps,
                'n_envs': self.n_envs,
                'algorithm': self.algorithm
            },
            config=default_params
        )


# =============================================================================
# 多種子驗證
# =============================================================================

def _train_single_seed(args: Tuple) -> TrainingResult:
    """
    單種子訓練函數（用於並行執行）
    """
    seed, env_fn, hyperparams, total_timesteps, n_eval_episodes, save_dir = args
    
    if not HAS_SB3:
        raise ImportError("stable_baselines3 is required")
    
    import random
    import torch
    
    # 設定種子
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    start_time = time.time()
    
    # 建立環境
    env = env_fn()
    eval_env = env_fn()
    
    # 建立模型
    algorithm = hyperparams.pop('algorithm', 'SAC')
    algo_class = {'SAC': SAC, 'PPO': PPO, 'TD3': TD3}[algorithm]
    
    model = algo_class(
        'MlpPolicy',
        env,
        seed=seed,
        verbose=0,
        **hyperparams
    )
    
    # 訓練
    model.learn(total_timesteps=total_timesteps)
    
    # 評估
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=n_eval_episodes
    )
    
    training_time = time.time() - start_time
    
    # 保存模型
    model_path = None
    if save_dir:
        model_path = os.path.join(save_dir, f"model_seed_{seed}")
        model.save(model_path)
    
    # 清理
    env.close()
    eval_env.close()
    
    return TrainingResult(
        seed=seed,
        mean_reward=mean_reward,
        std_reward=std_reward,
        episodes=n_eval_episodes,
        training_time=training_time,
        final_model_path=model_path,
        metrics={'algorithm': algorithm}
    )


class MultiSeedValidator:
    """
    多種子驗證器
    用於評估訓練結果的穩定性
    """
    
    def __init__(
        self,
        env_fn: Callable,
        seeds: Optional[List[int]] = None,
        n_seeds: int = 5
    ):
        """
        Args:
            env_fn: 環境工廠函數
            seeds: 種子列表，如果為 None 則自動生成
            n_seeds: 如果 seeds 為 None，生成的種子數量
        """
        self.env_fn = env_fn
        self.seeds = seeds if seeds is not None else list(range(42, 42 + n_seeds))
    
    def validate(
        self,
        hyperparams: Dict,
        total_timesteps: int,
        n_eval_episodes: int = 10,
        n_workers: int = 1,
        save_dir: Optional[str] = None
    ) -> MultiSeedResult:
        """
        執行多種子驗證
        
        Args:
            hyperparams: 超參數
            total_timesteps: 每個種子的訓練步數
            n_eval_episodes: 評估回合數
            n_workers: 並行工作數
            save_dir: 保存目錄
        
        Returns:
            MultiSeedResult: 驗證結果
        """
        results = []
        
        if n_workers > 1:
            # 並行訓練
            args_list = [
                (seed, self.env_fn, hyperparams.copy(), total_timesteps, 
                 n_eval_episodes, save_dir)
                for seed in self.seeds
            ]
            
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(_train_single_seed, args) 
                          for args in args_list]
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                        logger.info(f"Seed {result.seed}: reward = {result.mean_reward:.2f}")
                    except Exception as e:
                        logger.error(f"Training failed: {e}")
        else:
            # 序列訓練
            for seed in self.seeds:
                args = (seed, self.env_fn, hyperparams.copy(), total_timesteps,
                       n_eval_episodes, save_dir)
                result = _train_single_seed(args)
                results.append(result)
                logger.info(f"Seed {result.seed}: reward = {result.mean_reward:.2f}")
        
        # 按種子排序
        results.sort(key=lambda x: x.seed)
        
        multi_result = MultiSeedResult(
            seeds=self.seeds,
            results=results
        )
        multi_result.compute_aggregates()
        
        return multi_result
    
    def select_best_seed(
        self,
        result: MultiSeedResult,
        criterion: str = 'mean'
    ) -> Tuple[int, TrainingResult]:
        """
        選擇最佳種子
        
        Args:
            result: 多種子結果
            criterion: 選擇標準 ('mean', 'robust')
        
        Returns:
            (best_seed, best_result)
        """
        if criterion == 'mean':
            # 選擇平均獎勵最高的
            best_result = max(result.results, key=lambda x: x.mean_reward)
        elif criterion == 'robust':
            # 選擇 Sharpe-like ratio 最高的
            best_result = max(
                result.results,
                key=lambda x: x.mean_reward / (x.std_reward + 1e-8)
            )
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        return best_result.seed, best_result


# =============================================================================
# 超參數搜尋
# =============================================================================

class HyperparameterSearch:
    """
    超參數搜尋
    使用 Optuna 進行貝葉斯優化
    """
    
    def __init__(
        self,
        env_fn: Callable,
        algorithm: str = "SAC",
        n_eval_episodes: int = 5,
        eval_timesteps: int = 50000
    ):
        """
        Args:
            env_fn: 環境工廠函數
            algorithm: 演算法名稱
            n_eval_episodes: 評估回合數
            eval_timesteps: 每次試驗的訓練步數
        """
        if not HAS_OPTUNA:
            raise ImportError("optuna is required for hyperparameter search")
        
        self.env_fn = env_fn
        self.algorithm = algorithm.upper()
        self.n_eval_episodes = n_eval_episodes
        self.eval_timesteps = eval_timesteps
    
    def _get_search_space(self, trial: Trial) -> Dict:
        """定義搜尋空間"""
        common_params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'gamma': trial.suggest_float('gamma', 0.95, 0.999),
            'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
        }
        
        if self.algorithm == 'SAC':
            common_params.update({
                'buffer_size': trial.suggest_categorical('buffer_size', [50000, 100000, 200000]),
                'tau': trial.suggest_float('tau', 0.005, 0.05),
                'train_freq': trial.suggest_categorical('train_freq', [1, 4, 8]),
                'gradient_steps': trial.suggest_categorical('gradient_steps', [1, 2, 4]),
            })
        elif self.algorithm == 'PPO':
            common_params.update({
                'n_steps': trial.suggest_categorical('n_steps', [256, 512, 1024, 2048]),
                'n_epochs': trial.suggest_int('n_epochs', 3, 20),
                'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.1),
                'clip_range': trial.suggest_float('clip_range', 0.1, 0.3),
                'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.99),
            })
        elif self.algorithm == 'TD3':
            common_params.update({
                'buffer_size': trial.suggest_categorical('buffer_size', [50000, 100000, 200000]),
                'tau': trial.suggest_float('tau', 0.005, 0.05),
                'policy_delay': trial.suggest_int('policy_delay', 1, 4),
                'target_policy_noise': trial.suggest_float('target_policy_noise', 0.1, 0.3),
            })
        
        return common_params
    
    def _objective(self, trial: Trial) -> float:
        """優化目標函數"""
        if not HAS_SB3:
            raise ImportError("stable_baselines3 is required")
        
        params = self._get_search_space(trial)
        
        # 建立環境
        env = self.env_fn()
        eval_env = self.env_fn()
        
        try:
            # 建立模型
            algo_class = {'SAC': SAC, 'PPO': PPO, 'TD3': TD3}[self.algorithm]
            model = algo_class('MlpPolicy', env, verbose=0, **params)
            
            # 訓練
            model.learn(total_timesteps=self.eval_timesteps)
            
            # 評估
            mean_reward, _ = evaluate_policy(
                model, eval_env, n_eval_episodes=self.n_eval_episodes
            )
            
            return mean_reward
            
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return float('-inf')
            
        finally:
            env.close()
            eval_env.close()
    
    def search(
        self,
        n_trials: int = 50,
        n_jobs: int = 1,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        timeout: Optional[int] = None,
        show_progress_bar: bool = True
    ) -> HyperparameterSearchResult:
        """
        執行超參數搜尋
        
        Args:
            n_trials: 試驗次數
            n_jobs: 並行任務數
            study_name: 研究名稱
            storage: Optuna 資料庫路徑
            timeout: 超時時間（秒）
            show_progress_bar: 是否顯示進度條
        
        Returns:
            HyperparameterSearchResult: 搜尋結果
        """
        start_time = time.time()
        
        study_name = study_name or f"{self.algorithm}_tuning_{int(time.time())}"
        
        # 建立 Optuna study
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction='maximize',
            sampler=sampler,
            load_if_exists=True
        )
        
        # 執行優化
        study.optimize(
            self._objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=timeout,
            show_progress_bar=show_progress_bar
        )
        
        search_time = time.time() - start_time
        
        # 整理結果
        all_trials = []
        for trial in study.trials:
            all_trials.append({
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': str(trial.state)
            })
        
        return HyperparameterSearchResult(
            best_params=study.best_params,
            best_value=study.best_value,
            all_trials=all_trials,
            study_name=study_name,
            n_trials=len(study.trials),
            search_time=search_time
        )
    
    def resume_search(
        self,
        study_name: str,
        storage: str,
        additional_trials: int = 20
    ) -> HyperparameterSearchResult:
        """
        繼續先前的搜尋
        """
        study = optuna.load_study(
            study_name=study_name,
            storage=storage
        )
        
        start_time = time.time()
        
        study.optimize(
            self._objective,
            n_trials=additional_trials,
            show_progress_bar=True
        )
        
        search_time = time.time() - start_time
        
        all_trials = []
        for trial in study.trials:
            all_trials.append({
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': str(trial.state)
            })
        
        return HyperparameterSearchResult(
            best_params=study.best_params,
            best_value=study.best_value,
            all_trials=all_trials,
            study_name=study_name,
            n_trials=len(study.trials),
            search_time=search_time
        )


# =============================================================================
# 分散式訓練管理器
# =============================================================================

class DistributedTrainingManager:
    """
    分散式訓練管理器
    整合所有分散式訓練功能
    """
    
    def __init__(
        self,
        env_fn: Callable,
        algorithm: str = "SAC",
        output_dir: str = "./distributed_runs"
    ):
        self.env_fn = env_fn
        self.algorithm = algorithm
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_full_pipeline(
        self,
        # 超參數搜尋
        n_hp_trials: int = 30,
        hp_timesteps: int = 50000,
        # 多種子驗證
        validation_seeds: List[int] = [42, 43, 44, 45, 46],
        validation_timesteps: int = 100000,
        # 最終訓練
        final_timesteps: int = 500000,
        final_n_envs: int = 4,
        # 通用參數
        n_eval_episodes: int = 10
    ) -> Dict[str, Any]:
        """
        執行完整的分散式訓練流程
        
        1. 超參數搜尋
        2. 多種子驗證
        3. 選擇最佳配置
        4. 最終訓練
        
        Returns:
            包含所有結果的字典
        """
        results = {}
        
        # === Phase 1: 超參數搜尋 ===
        logger.info("Phase 1: Hyperparameter Search")
        hp_search = HyperparameterSearch(
            env_fn=self.env_fn,
            algorithm=self.algorithm,
            n_eval_episodes=n_eval_episodes,
            eval_timesteps=hp_timesteps
        )
        
        hp_result = hp_search.search(n_trials=n_hp_trials)
        results['hyperparameter_search'] = hp_result.to_dict()
        
        # 保存超參數搜尋結果
        with open(self.output_dir / "hp_search_result.json", 'w') as f:
            json.dump(results['hyperparameter_search'], f, indent=2)
        
        logger.info(f"Best hyperparameters: {hp_result.best_params}")
        logger.info(f"Best value: {hp_result.best_value:.2f}")
        
        # === Phase 2: 多種子驗證 ===
        logger.info("Phase 2: Multi-Seed Validation")
        validator = MultiSeedValidator(
            env_fn=self.env_fn,
            seeds=validation_seeds
        )
        
        best_params = hp_result.best_params.copy()
        best_params['algorithm'] = self.algorithm
        
        multi_seed_result = validator.validate(
            hyperparams=best_params,
            total_timesteps=validation_timesteps,
            n_eval_episodes=n_eval_episodes,
            save_dir=str(self.output_dir / "seed_validation")
        )
        
        results['multi_seed_validation'] = {
            'aggregate_metrics': multi_seed_result.aggregate_metrics,
            'individual_results': [r.to_dict() for r in multi_seed_result.results]
        }
        
        logger.info(f"Aggregate metrics: {multi_seed_result.aggregate_metrics}")
        
        # === Phase 3: 選擇最佳配置 ===
        best_seed, best_seed_result = validator.select_best_seed(
            multi_seed_result, criterion='robust'
        )
        results['best_seed'] = best_seed
        
        logger.info(f"Best seed: {best_seed} (reward: {best_seed_result.mean_reward:.2f})")
        
        # === Phase 4: 最終訓練 ===
        logger.info("Phase 4: Final Training")
        final_trainer = ParallelEnvTrainer(
            env_fn=self.env_fn,
            n_envs=final_n_envs,
            algorithm=self.algorithm
        )
        
        # 設定種子
        import random
        import torch
        np.random.seed(best_seed)
        random.seed(best_seed)
        torch.manual_seed(best_seed)
        
        eval_env = self.env_fn()
        
        final_result = final_trainer.train(
            total_timesteps=final_timesteps,
            hyperparams=hp_result.best_params,
            eval_env=eval_env,
            eval_freq=10000,
            n_eval_episodes=n_eval_episodes,
            save_path=str(self.output_dir / "final_model")
        )
        
        results['final_training'] = final_result.to_dict()
        
        eval_env.close()
        
        # 保存完整結果
        with open(self.output_dir / "full_pipeline_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Pipeline complete. Results saved to {self.output_dir}")
        
        return results
    
    def run_hyperparameter_search_only(
        self,
        n_trials: int = 50,
        timesteps_per_trial: int = 50000,
        n_eval_episodes: int = 5
    ) -> HyperparameterSearchResult:
        """只執行超參數搜尋"""
        hp_search = HyperparameterSearch(
            env_fn=self.env_fn,
            algorithm=self.algorithm,
            n_eval_episodes=n_eval_episodes,
            eval_timesteps=timesteps_per_trial
        )
        
        return hp_search.search(n_trials=n_trials)
    
    def run_multi_seed_validation_only(
        self,
        hyperparams: Dict,
        seeds: Optional[List[int]] = None,
        total_timesteps: int = 100000,
        n_eval_episodes: int = 10
    ) -> MultiSeedResult:
        """只執行多種子驗證"""
        validator = MultiSeedValidator(
            env_fn=self.env_fn,
            seeds=seeds
        )
        
        hyperparams = hyperparams.copy()
        hyperparams['algorithm'] = self.algorithm
        
        return validator.validate(
            hyperparams=hyperparams,
            total_timesteps=total_timesteps,
            n_eval_episodes=n_eval_episodes,
            save_dir=str(self.output_dir / "seed_validation")
        )


# =============================================================================
# 便利函數
# =============================================================================

def train_with_multiple_seeds(
    env_fn: Callable,
    hyperparams: Dict,
    seeds: List[int] = [42, 43, 44],
    total_timesteps: int = 100000,
    algorithm: str = "SAC"
) -> MultiSeedResult:
    """
    便利函數：使用多個種子訓練模型
    """
    validator = MultiSeedValidator(env_fn=env_fn, seeds=seeds)
    hyperparams['algorithm'] = algorithm
    return validator.validate(
        hyperparams=hyperparams,
        total_timesteps=total_timesteps
    )


def search_hyperparameters(
    env_fn: Callable,
    algorithm: str = "SAC",
    n_trials: int = 30,
    timesteps_per_trial: int = 50000
) -> HyperparameterSearchResult:
    """
    便利函數：搜尋最佳超參數
    """
    searcher = HyperparameterSearch(
        env_fn=env_fn,
        algorithm=algorithm,
        eval_timesteps=timesteps_per_trial
    )
    return searcher.search(n_trials=n_trials)


# =============================================================================
# 主程式（測試用）
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 測試用簡單環境
    import gymnasium as gym
    
    def make_env():
        return gym.make("Pendulum-v1")
    
    # 測試多種子驗證
    print("Testing MultiSeedValidator...")
    validator = MultiSeedValidator(make_env, seeds=[42, 43])
    result = validator.validate(
        hyperparams={'learning_rate': 3e-4, 'algorithm': 'SAC'},
        total_timesteps=1000,
        n_eval_episodes=2
    )
    print(f"Aggregate metrics: {result.aggregate_metrics}")
    
    print("\nDistributed training utilities ready!")
