"""
utils/ensemble.py
集成學習 (Ensemble Methods) 模組

實作多模型組合策略:
- Voting: 簡單平均投票
- Weighted: 加權平均（根據驗證表現）
- Stacking: 堆疊模型

用法:
    from utils.ensemble import EnsemblePolicy, create_ensemble
    
    ensemble = create_ensemble(
        model_paths=["model1.zip", "model2.zip", "model3.zip"],
        algo="sac",
        method="weighted",
    )
    action = ensemble.predict(obs)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import gymnasium as gym
from stable_baselines3.common.base_class import BaseAlgorithm


# =============================================================================
# Ensemble Policy
# =============================================================================

class EnsemblePolicy:
    """
    集成策略
    
    組合多個 RL 模型的預測
    """
    
    def __init__(
        self,
        models: List[BaseAlgorithm],
        method: str = "voting",
        weights: Optional[List[float]] = None,
    ):
        """
        Args:
            models: RL 模型列表
            method: 集成方法 ("voting", "weighted", "max_confidence")
            weights: 模型權重（僅用於 weighted 方法）
        """
        self.models = models
        self.method = method
        
        # 設定權重
        if weights is None:
            self.weights = np.ones(len(models)) / len(models)
        else:
            self.weights = np.array(weights)
            self.weights = self.weights / self.weights.sum()
        
        self.n_models = len(models)
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, None]:
        """
        預測動作
        
        Args:
            observation: 觀察值
            deterministic: 是否使用確定性策略
        
        Returns:
            (action, state)
        """
        # 收集所有模型的預測
        actions = []
        for model in self.models:
            action, _ = model.predict(observation, deterministic=deterministic)
            actions.append(action)
        
        actions = np.array(actions)
        
        # 根據方法組合
        if self.method == "voting":
            # 簡單平均
            ensemble_action = np.mean(actions, axis=0)
        
        elif self.method == "weighted":
            # 加權平均
            ensemble_action = np.average(actions, axis=0, weights=self.weights)
        
        elif self.method == "max_confidence":
            # 選擇動作方差最小的模型
            variances = np.var(actions, axis=1)
            best_idx = np.argmin(variances)
            ensemble_action = actions[best_idx]
        
        elif self.method == "median":
            # 中位數（對異常值更魯棒）
            ensemble_action = np.median(actions, axis=0)
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
        
        return ensemble_action, None
    
    def predict_with_uncertainty(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        預測動作並估計不確定性
        
        Returns:
            (mean_action, action_std)
        """
        actions = []
        for model in self.models:
            action, _ = model.predict(observation, deterministic=deterministic)
            actions.append(action)
        
        actions = np.array(actions)
        mean_action = np.mean(actions, axis=0)
        std_action = np.std(actions, axis=0)
        
        return mean_action, std_action
    
    def set_weights(self, weights: List[float]):
        """設定模型權重"""
        self.weights = np.array(weights)
        self.weights = self.weights / self.weights.sum()


# =============================================================================
# Ensemble Builder
# =============================================================================

class EnsembleBuilder:
    """集成模型建構器"""
    
    def __init__(
        self,
        algo: str = "sac",
        device: str = "auto",
    ):
        """
        Args:
            algo: 演算法名稱
            device: 運算裝置
        """
        self.algo = algo
        self.device = device
        self.models: List[BaseAlgorithm] = []
        self.model_scores: List[float] = []
    
    def add_model(
        self,
        model_path: Union[str, Path],
        score: float = None,
        env: gym.Env = None,
    ):
        """
        添加模型
        
        Args:
            model_path: 模型檔案路徑
            score: 模型驗證分數（用於加權）
            env: 可選的環境（用於載入）
        """
        from utils.algorithms import load_model
        
        model = load_model(self.algo, str(model_path), env=env, device=self.device)
        self.models.append(model)
        self.model_scores.append(score if score is not None else 1.0)
    
    def add_models_from_dir(
        self,
        directory: Union[str, Path],
        pattern: str = "*.zip",
        env: gym.Env = None,
    ):
        """
        從目錄載入多個模型
        
        Args:
            directory: 目錄路徑
            pattern: 檔案匹配模式
            env: 可選的環境
        """
        dir_path = Path(directory)
        for model_path in sorted(dir_path.glob(pattern)):
            self.add_model(model_path, env=env)
    
    def build(
        self,
        method: str = "voting",
        use_scores_as_weights: bool = True,
    ) -> EnsemblePolicy:
        """
        建構集成策略
        
        Args:
            method: 集成方法
            use_scores_as_weights: 是否使用分數作為權重
        
        Returns:
            集成策略
        """
        if len(self.models) == 0:
            raise ValueError("No models added to ensemble")
        
        weights = None
        if use_scores_as_weights and method == "weighted":
            weights = self.model_scores
        
        return EnsemblePolicy(
            models=self.models,
            method=method,
            weights=weights,
        )
    
    def evaluate_and_weight(
        self,
        eval_env: gym.Env,
        n_episodes: int = 10,
    ):
        """
        評估各模型並設定權重
        
        Args:
            eval_env: 評估環境
            n_episodes: 評估 episode 數
        """
        from stable_baselines3.common.evaluation import evaluate_policy
        
        self.model_scores = []
        for model in self.models:
            mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=n_episodes)
            self.model_scores.append(max(mean_reward, 0.01))  # 確保正數
        
        print(f"Model scores: {self.model_scores}")


# =============================================================================
# Stacking Ensemble
# =============================================================================

class StackingEnsemble:
    """
    堆疊集成
    
    使用元學習器組合基礎模型的預測
    """
    
    def __init__(
        self,
        base_models: List[BaseAlgorithm],
        meta_model: Any = None,
    ):
        """
        Args:
            base_models: 基礎模型列表
            meta_model: 元模型（組合基礎模型輸出）
        """
        self.base_models = base_models
        self.meta_model = meta_model
        
        # 如果沒有提供元模型，使用簡單的線性組合
        if meta_model is None:
            self.use_simple_meta = True
            self.meta_weights = np.ones(len(base_models)) / len(base_models)
        else:
            self.use_simple_meta = False
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, None]:
        """預測動作"""
        # 收集基礎模型預測
        base_predictions = []
        for model in self.base_models:
            action, _ = model.predict(observation, deterministic=deterministic)
            base_predictions.append(action)
        
        base_predictions = np.array(base_predictions)
        
        if self.use_simple_meta:
            # 簡單加權平均
            ensemble_action = np.average(base_predictions, axis=0, weights=self.meta_weights)
        else:
            # 使用元模型
            meta_input = base_predictions.flatten()
            ensemble_action = self.meta_model.predict(meta_input.reshape(1, -1))[0]
        
        return ensemble_action, None
    
    def train_meta_model(
        self,
        env: gym.Env,
        n_episodes: int = 100,
    ):
        """
        訓練元模型
        
        使用基礎模型在環境中的表現來訓練元模型
        """
        from sklearn.linear_model import Ridge
        
        X_meta = []  # 基礎模型預測
        y_meta = []  # 實際最佳動作（用獎勵評估）
        
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            
            while not done:
                # 收集基礎模型預測
                base_predictions = []
                for model in self.base_models:
                    action, _ = model.predict(obs, deterministic=True)
                    base_predictions.append(action)
                
                # 評估每個動作
                best_reward = -np.inf
                best_action = None
                
                for i, model in enumerate(self.base_models):
                    action = base_predictions[i]
                    # 模擬執行（需要環境支援）
                    # 這裡簡化為選擇一個模型
                    if i == 0:
                        obs, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated
                        best_reward = reward
                        best_action = action
                
                X_meta.append(np.array(base_predictions).flatten())
                y_meta.append(best_action)
        
        X_meta = np.array(X_meta)
        y_meta = np.array(y_meta)
        
        # 訓練元模型
        self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(X_meta, y_meta)
        self.use_simple_meta = False


# =============================================================================
# Diversity-Weighted Ensemble
# =============================================================================

class DiversityEnsemble:
    """
    多樣性加權集成
    
    根據模型的多樣性（預測差異）動態調整權重
    """
    
    def __init__(
        self,
        models: List[BaseAlgorithm],
        diversity_weight: float = 0.5,
    ):
        """
        Args:
            models: 模型列表
            diversity_weight: 多樣性權重因子
        """
        self.models = models
        self.diversity_weight = diversity_weight
        self.performance_weights = np.ones(len(models)) / len(models)
        self.diversity_history: List[np.ndarray] = []
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, None]:
        """預測動作"""
        actions = []
        for model in self.models:
            action, _ = model.predict(observation, deterministic=deterministic)
            actions.append(action)
        
        actions = np.array(actions)
        
        # 計算動態權重
        weights = self._compute_dynamic_weights(actions)
        
        # 加權平均
        ensemble_action = np.average(actions, axis=0, weights=weights)
        
        return ensemble_action, None
    
    def _compute_dynamic_weights(self, actions: np.ndarray) -> np.ndarray:
        """
        計算動態權重
        
        結合表現和多樣性
        """
        n_models = len(self.models)
        
        # 計算每個模型與平均預測的距離
        mean_action = np.mean(actions, axis=0)
        distances = np.array([
            np.linalg.norm(actions[i] - mean_action)
            for i in range(n_models)
        ])
        
        # 距離越遠，多樣性貢獻越大
        if np.max(distances) > 0:
            diversity_scores = distances / np.max(distances)
        else:
            diversity_scores = np.ones(n_models) / n_models
        
        # 組合表現權重和多樣性分數
        weights = (1 - self.diversity_weight) * self.performance_weights + \
                  self.diversity_weight * diversity_scores
        
        # 正規化
        weights = weights / weights.sum()
        
        return weights
    
    def update_performance_weights(self, rewards: List[float]):
        """根據獎勵更新表現權重"""
        rewards = np.array(rewards)
        if np.sum(rewards) > 0:
            self.performance_weights = rewards / np.sum(rewards)
        else:
            self.performance_weights = np.ones(len(self.models)) / len(self.models)


# =============================================================================
# 便利函數
# =============================================================================

def create_ensemble(
    model_paths: List[Union[str, Path]],
    algo: str = "sac",
    method: str = "voting",
    weights: Optional[List[float]] = None,
    env: gym.Env = None,
    device: str = "auto",
) -> EnsemblePolicy:
    """
    便利函數：建立集成策略
    
    Args:
        model_paths: 模型檔案路徑列表
        algo: 演算法名稱
        method: 集成方法
        weights: 可選的權重
        env: 可選的環境
        device: 運算裝置
    
    Returns:
        集成策略
    """
    builder = EnsembleBuilder(algo=algo, device=device)
    
    for path in model_paths:
        builder.add_model(path, env=env)
    
    return builder.build(method=method, use_scores_as_weights=weights is None)


def train_diverse_ensemble(
    env_factory,
    algo: str = "sac",
    n_models: int = 3,
    total_timesteps: int = 100_000,
    diversity_seeds: List[int] = None,
    output_dir: str = "ensemble_models",
) -> EnsemblePolicy:
    """
    訓練多樣化集成
    
    使用不同種子訓練多個模型以增加多樣性
    
    Args:
        env_factory: 建立環境的函數
        algo: 演算法名稱
        n_models: 模型數量
        total_timesteps: 每個模型的訓練步數
        diversity_seeds: 各模型的隨機種子
        output_dir: 輸出目錄
    
    Returns:
        集成策略
    """
    from utils.algorithms import create_model
    
    if diversity_seeds is None:
        diversity_seeds = list(range(n_models))
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    models = []
    
    for i, seed in enumerate(diversity_seeds[:n_models]):
        print(f"\n Training model {i+1}/{n_models} (seed={seed})")
        
        env = env_factory()
        model = create_model(algo=algo, env=env, seed=seed, verbose=0)
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
        
        # 儲存
        model_path = output_path / f"model_{i}_seed{seed}.zip"
        model.save(str(model_path))
        
        models.append(model)
        env.close()
    
    return EnsemblePolicy(models=models, method="voting")
