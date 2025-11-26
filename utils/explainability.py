"""
utils/explainability.py
策略可解釋性模組

提供 RL 策略的解釋工具:
- Feature Importance: 特徵重要性分析
- Action Distribution: 動作分佈分析
- State-Action Mapping: 狀態-動作映射視覺化
- Decision Boundary: 決策邊界分析

用法:
    from utils.explainability import PolicyExplainer
    
    explainer = PolicyExplainer(model, env, feature_names)
    importance = explainer.compute_feature_importance()
    explainer.plot_action_distribution(observations)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from stable_baselines3.common.base_class import BaseAlgorithm


# =============================================================================
# Policy Explainer
# =============================================================================

class PolicyExplainer:
    """策略解釋器"""
    
    def __init__(
        self,
        model: BaseAlgorithm,
        env: gym.Env = None,
        feature_names: List[str] = None,
    ):
        """
        Args:
            model: RL 模型
            env: 環境（用於採樣）
            feature_names: 特徵名稱列表
        """
        self.model = model
        self.env = env
        self.feature_names = feature_names
        
        # 嘗試從環境取得 observation 維度
        if env is not None:
            self.obs_dim = env.observation_space.shape[0]
            self.action_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else 1
        else:
            self.obs_dim = None
            self.action_dim = None
        
        # 預設特徵名稱
        if self.feature_names is None and self.obs_dim is not None:
            self.feature_names = [f"feature_{i}" for i in range(self.obs_dim)]
    
    def compute_feature_importance(
        self,
        observations: np.ndarray = None,
        n_samples: int = 1000,
        method: str = "permutation",
    ) -> Dict[str, float]:
        """
        計算特徵重要性
        
        Args:
            observations: 觀察值樣本 (n_samples, obs_dim)
            n_samples: 採樣數量（如果沒有提供 observations）
            method: 計算方法 ("permutation", "gradient", "noise")
        
        Returns:
            特徵重要性字典
        """
        # 取得觀察值樣本
        if observations is None:
            observations = self._sample_observations(n_samples)
        
        if method == "permutation":
            importance = self._permutation_importance(observations)
        elif method == "noise":
            importance = self._noise_importance(observations)
        elif method == "gradient":
            importance = self._gradient_importance(observations)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # 建立結果字典
        result = {}
        for i, imp in enumerate(importance):
            name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            result[name] = float(imp)
        
        return result
    
    def _sample_observations(self, n_samples: int) -> np.ndarray:
        """從環境採樣觀察值"""
        if self.env is None:
            raise ValueError("Environment is required for sampling")
        
        observations = []
        obs, _ = self.env.reset()
        
        for _ in range(n_samples):
            observations.append(obs)
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = self.env.step(action)
            
            if terminated or truncated:
                obs, _ = self.env.reset()
        
        return np.array(observations)
    
    def _permutation_importance(self, observations: np.ndarray) -> np.ndarray:
        """
        排列重要性
        
        隨機打亂每個特徵，觀察動作變化
        """
        n_samples, n_features = observations.shape
        importance = np.zeros(n_features)
        
        # 基礎預測
        base_actions, _ = self.model.predict(observations, deterministic=True)
        if len(base_actions.shape) == 1:
            base_actions = base_actions.reshape(-1, 1)
        
        for i in range(n_features):
            # 打亂第 i 個特徵
            permuted_obs = observations.copy()
            np.random.shuffle(permuted_obs[:, i])
            
            # 新預測
            perm_actions, _ = self.model.predict(permuted_obs, deterministic=True)
            if len(perm_actions.shape) == 1:
                perm_actions = perm_actions.reshape(-1, 1)
            
            # 計算動作變化量
            diff = np.mean(np.abs(base_actions - perm_actions))
            importance[i] = diff
        
        # 正規化
        if np.sum(importance) > 0:
            importance = importance / np.sum(importance)
        
        return importance
    
    def _noise_importance(self, observations: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
        """
        噪聲敏感度
        
        對每個特徵加入噪聲，觀察動作變化
        """
        n_samples, n_features = observations.shape
        importance = np.zeros(n_features)
        
        # 基礎預測
        base_actions, _ = self.model.predict(observations, deterministic=True)
        if len(base_actions.shape) == 1:
            base_actions = base_actions.reshape(-1, 1)
        
        for i in range(n_features):
            # 計算特徵的標準差
            feature_std = np.std(observations[:, i])
            
            # 加入噪聲
            noisy_obs = observations.copy()
            noisy_obs[:, i] += np.random.normal(0, feature_std * noise_scale, n_samples)
            
            # 新預測
            noisy_actions, _ = self.model.predict(noisy_obs, deterministic=True)
            if len(noisy_actions.shape) == 1:
                noisy_actions = noisy_actions.reshape(-1, 1)
            
            # 計算動作變化量
            diff = np.mean(np.abs(base_actions - noisy_actions))
            importance[i] = diff
        
        # 正規化
        if np.sum(importance) > 0:
            importance = importance / np.sum(importance)
        
        return importance
    
    def _gradient_importance(self, observations: np.ndarray) -> np.ndarray:
        """
        梯度重要性（使用數值梯度近似）
        
        計算動作對每個特徵的梯度
        """
        n_samples, n_features = observations.shape
        importance = np.zeros(n_features)
        eps = 1e-5
        
        for i in range(n_features):
            # 正向擾動
            obs_plus = observations.copy()
            obs_plus[:, i] += eps
            
            # 負向擾動
            obs_minus = observations.copy()
            obs_minus[:, i] -= eps
            
            # 預測
            actions_plus, _ = self.model.predict(obs_plus, deterministic=True)
            actions_minus, _ = self.model.predict(obs_minus, deterministic=True)
            
            if len(actions_plus.shape) == 1:
                actions_plus = actions_plus.reshape(-1, 1)
                actions_minus = actions_minus.reshape(-1, 1)
            
            # 數值梯度
            grad = np.mean(np.abs(actions_plus - actions_minus)) / (2 * eps)
            importance[i] = grad
        
        # 正規化
        if np.sum(importance) > 0:
            importance = importance / np.sum(importance)
        
        return importance
    
    def analyze_action_distribution(
        self,
        observations: np.ndarray = None,
        n_samples: int = 1000,
    ) -> Dict[str, Any]:
        """
        分析動作分佈
        
        Returns:
            動作統計資訊
        """
        if observations is None:
            observations = self._sample_observations(n_samples)
        
        # 預測動作
        actions, _ = self.model.predict(observations, deterministic=True)
        if len(actions.shape) == 1:
            actions = actions.reshape(-1, 1)
        
        n_actions = actions.shape[1]
        
        stats = {
            "n_samples": len(actions),
            "action_dims": n_actions,
        }
        
        for i in range(n_actions):
            action_i = actions[:, i]
            stats[f"action_{i}"] = {
                "mean": float(np.mean(action_i)),
                "std": float(np.std(action_i)),
                "min": float(np.min(action_i)),
                "max": float(np.max(action_i)),
                "median": float(np.median(action_i)),
                "q25": float(np.percentile(action_i, 25)),
                "q75": float(np.percentile(action_i, 75)),
            }
        
        return stats
    
    def analyze_state_action_correlation(
        self,
        observations: np.ndarray = None,
        n_samples: int = 1000,
    ) -> np.ndarray:
        """
        分析狀態-動作相關性
        
        Returns:
            相關性矩陣 (n_features x n_actions)
        """
        if observations is None:
            observations = self._sample_observations(n_samples)
        
        # 預測動作
        actions, _ = self.model.predict(observations, deterministic=True)
        if len(actions.shape) == 1:
            actions = actions.reshape(-1, 1)
        
        n_features = observations.shape[1]
        n_actions = actions.shape[1]
        
        # 計算相關係數
        correlation = np.zeros((n_features, n_actions))
        
        for i in range(n_features):
            for j in range(n_actions):
                corr = np.corrcoef(observations[:, i], actions[:, j])[0, 1]
                correlation[i, j] = corr if not np.isnan(corr) else 0.0
        
        return correlation
    
    def analyze_policy_behavior(
        self,
        n_episodes: int = 10,
    ) -> Dict[str, Any]:
        """
        分析策略行為
        
        Returns:
            行為統計
        """
        if self.env is None:
            raise ValueError("Environment is required")
        
        episode_stats = []
        
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            
            ep_actions = []
            ep_rewards = []
            ep_observations = []
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                ep_observations.append(obs)
                ep_actions.append(action)
                
                obs, reward, terminated, truncated, _ = self.env.step(action)
                ep_rewards.append(reward)
                done = terminated or truncated
            
            episode_stats.append({
                "total_reward": sum(ep_rewards),
                "n_steps": len(ep_actions),
                "avg_action": np.mean(ep_actions, axis=0).tolist(),
                "action_std": np.std(ep_actions, axis=0).tolist(),
            })
        
        # 彙總
        total_rewards = [ep["total_reward"] for ep in episode_stats]
        n_steps = [ep["n_steps"] for ep in episode_stats]
        
        return {
            "n_episodes": n_episodes,
            "avg_reward": float(np.mean(total_rewards)),
            "std_reward": float(np.std(total_rewards)),
            "avg_episode_length": float(np.mean(n_steps)),
            "episode_details": episode_stats,
        }


# =============================================================================
# Feature Importance Visualizer
# =============================================================================

class FeatureImportanceVisualizer:
    """特徵重要性視覺化"""
    
    @staticmethod
    def plot_importance(
        importance: Dict[str, float],
        title: str = "Feature Importance",
        output_path: str = None,
    ):
        """
        繪製特徵重要性條形圖
        
        Args:
            importance: 特徵重要性字典
            title: 圖表標題
            output_path: 輸出路徑（可選）
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed, skipping plot")
            return
        
        # 排序
        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        names = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.3)))
        
        y_pos = np.arange(len(names))
        ax.barh(y_pos, values, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {output_path}")
        
        plt.close()
    
    @staticmethod
    def plot_correlation_heatmap(
        correlation: np.ndarray,
        feature_names: List[str],
        action_names: List[str] = None,
        title: str = "State-Action Correlation",
        output_path: str = None,
    ):
        """
        繪製狀態-動作相關性熱圖
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed, skipping plot")
            return
        
        if action_names is None:
            action_names = [f"action_{i}" for i in range(correlation.shape[1])]
        
        fig, ax = plt.subplots(figsize=(8, max(6, len(feature_names) * 0.3)))
        
        im = ax.imshow(correlation, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
        
        ax.set_xticks(np.arange(len(action_names)))
        ax.set_yticks(np.arange(len(feature_names)))
        ax.set_xticklabels(action_names)
        ax.set_yticklabels(feature_names)
        
        plt.colorbar(im, ax=ax, label='Correlation')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {output_path}")
        
        plt.close()


# =============================================================================
# SHAP-like Explainer (Simplified)
# =============================================================================

class SHAPLikeExplainer:
    """
    類 SHAP 解釋器（簡化版本）
    
    使用採樣方法近似 Shapley 值
    """
    
    def __init__(
        self,
        model: BaseAlgorithm,
        background_data: np.ndarray,
        n_samples: int = 100,
    ):
        """
        Args:
            model: RL 模型
            background_data: 背景數據（用於計算期望）
            n_samples: 採樣數量
        """
        self.model = model
        self.background_data = background_data
        self.n_samples = n_samples
        self.n_features = background_data.shape[1]
    
    def explain(self, observation: np.ndarray) -> np.ndarray:
        """
        計算單個觀察值的 SHAP 值
        
        Args:
            observation: 觀察值 (1D array)
        
        Returns:
            SHAP 值 (n_features,)
        """
        shap_values = np.zeros(self.n_features)
        
        # 基礎預測
        base_action, _ = self.model.predict(observation.reshape(1, -1), deterministic=True)
        base_action = np.mean(base_action)
        
        # 背景預測
        bg_actions, _ = self.model.predict(self.background_data, deterministic=True)
        expected_action = np.mean(bg_actions)
        
        # 對每個特徵計算邊際貢獻
        for i in range(self.n_features):
            contribution = 0.0
            
            for _ in range(self.n_samples):
                # 隨機選擇背景樣本
                bg_idx = np.random.randint(len(self.background_data))
                bg_sample = self.background_data[bg_idx].copy()
                
                # 創建兩個版本：有/沒有當前特徵
                with_feature = bg_sample.copy()
                with_feature[i] = observation[i]
                
                without_feature = bg_sample.copy()
                # without_feature[i] 保持背景值
                
                # 預測
                action_with, _ = self.model.predict(with_feature.reshape(1, -1), deterministic=True)
                action_without, _ = self.model.predict(without_feature.reshape(1, -1), deterministic=True)
                
                # 邊際貢獻
                contribution += (np.mean(action_with) - np.mean(action_without))
            
            shap_values[i] = contribution / self.n_samples
        
        return shap_values
    
    def explain_batch(
        self,
        observations: np.ndarray,
        progress: bool = True,
    ) -> np.ndarray:
        """
        計算多個觀察值的 SHAP 值
        
        Returns:
            SHAP 值矩陣 (n_samples, n_features)
        """
        n_obs = len(observations)
        shap_values = np.zeros((n_obs, self.n_features))
        
        for i, obs in enumerate(observations):
            if progress and (i + 1) % 10 == 0:
                print(f"Explaining {i + 1}/{n_obs}...")
            shap_values[i] = self.explain(obs)
        
        return shap_values


# =============================================================================
# 便利函數
# =============================================================================

def explain_policy(
    model: BaseAlgorithm,
    env: gym.Env,
    n_samples: int = 500,
    feature_names: List[str] = None,
    output_dir: str = None,
) -> Dict[str, Any]:
    """
    便利函數：完整策略解釋
    
    Returns:
        包含所有解釋結果的字典
    """
    explainer = PolicyExplainer(model, env, feature_names)
    
    # 採樣觀察值
    observations = explainer._sample_observations(n_samples)
    
    # 計算各種解釋
    results = {
        "feature_importance": explainer.compute_feature_importance(observations, method="permutation"),
        "noise_sensitivity": explainer.compute_feature_importance(observations, method="noise"),
        "action_distribution": explainer.analyze_action_distribution(observations),
        "state_action_correlation": explainer.analyze_state_action_correlation(observations).tolist(),
        "policy_behavior": explainer.analyze_policy_behavior(n_episodes=10),
    }
    
    # 視覺化
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        FeatureImportanceVisualizer.plot_importance(
            results["feature_importance"],
            title="Feature Importance (Permutation)",
            output_path=str(output_path / "feature_importance.png"),
        )
        
        if feature_names:
            FeatureImportanceVisualizer.plot_correlation_heatmap(
                np.array(results["state_action_correlation"]),
                feature_names,
                title="State-Action Correlation",
                output_path=str(output_path / "correlation_heatmap.png"),
            )
    
    return results
