"""
utils/lstm_features.py
LSTM 特徵提取器用於強化學習

實作自定義特徵提取器，捕捉時序依賴性
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Type

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LSTMFeaturesExtractor(BaseFeaturesExtractor):
    """
    LSTM 特徵提取器
    
    將觀察歷史轉換為 LSTM 特徵向量
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 1,
        sequence_length: int = 20,
    ):
        """
        Args:
            observation_space: 觀察空間
            features_dim: 輸出特徵維度
            lstm_hidden_size: LSTM 隱藏層大小
            lstm_num_layers: LSTM 層數
            sequence_length: 序列長度
        """
        super().__init__(observation_space, features_dim)
        
        self.obs_dim = observation_space.shape[0]
        self.sequence_length = sequence_length
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        
        # LSTM 層
        self.lstm = nn.LSTM(
            input_size=self.obs_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=0.1 if lstm_num_layers > 1 else 0,
        )
        
        # 全連接層
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, features_dim),
            nn.ReLU(),
        )
        
        # 觀察歷史緩存
        self._obs_history: List[np.ndarray] = []
        self._batch_history: Dict[int, List[np.ndarray]] = {}
    
    def reset_history(self):
        """重置歷史"""
        self._obs_history = []
        self._batch_history = {}
    
    def _update_history(self, obs: torch.Tensor) -> torch.Tensor:
        """
        更新歷史並返回序列
        
        Args:
            obs: 當前觀察 (batch_size, obs_dim)
        
        Returns:
            序列張量 (batch_size, sequence_length, obs_dim)
        """
        batch_size = obs.shape[0]
        device = obs.device
        
        # 如果是單一環境
        if batch_size == 1:
            obs_np = obs.detach().cpu().numpy()[0]
            self._obs_history.append(obs_np)
            
            # 保持固定長度
            if len(self._obs_history) > self.sequence_length:
                self._obs_history = self._obs_history[-self.sequence_length:]
            
            # 填充或截取
            if len(self._obs_history) < self.sequence_length:
                # 用第一個觀察填充
                padding = [self._obs_history[0]] * (self.sequence_length - len(self._obs_history))
                sequence = padding + self._obs_history
            else:
                sequence = self._obs_history[-self.sequence_length:]
            
            return torch.tensor(np.array(sequence), dtype=torch.float32, device=device).unsqueeze(0)
        
        # 批量處理（用於訓練）
        # 這裡簡化處理：每個觀察重複 sequence_length 次
        # 實際應用中應該維護每個環境的歷史
        sequences = obs.unsqueeze(1).repeat(1, self.sequence_length, 1)
        return sequences
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            observations: 觀察張量
        
        Returns:
            特徵張量
        """
        # 建構序列
        sequence = self._update_history(observations)
        
        # LSTM 處理
        lstm_out, (h_n, c_n) = self.lstm(sequence)
        
        # 取最後一個時間步的輸出
        last_output = lstm_out[:, -1, :]
        
        # 全連接層
        features = self.fc(last_output)
        
        return features


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """
    Transformer 特徵提取器
    
    使用 self-attention 機制處理時序數據
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        sequence_length: int = 20,
    ):
        super().__init__(observation_space, features_dim)
        
        self.obs_dim = observation_space.shape[0]
        self.sequence_length = sequence_length
        self.d_model = d_model
        
        # 輸入嵌入
        self.input_embedding = nn.Linear(self.obs_dim, d_model)
        
        # 位置編碼
        self.pos_encoding = self._create_positional_encoding(sequence_length, d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 輸出層
        self.fc = nn.Sequential(
            nn.Linear(d_model, features_dim),
            nn.ReLU(),
        )
        
        # 觀察歷史
        self._obs_history: List[np.ndarray] = []
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """建立位置編碼"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def reset_history(self):
        """重置歷史"""
        self._obs_history = []
    
    def _update_history(self, obs: torch.Tensor) -> torch.Tensor:
        """更新歷史並返回序列"""
        batch_size = obs.shape[0]
        device = obs.device
        
        if batch_size == 1:
            obs_np = obs.detach().cpu().numpy()[0]
            self._obs_history.append(obs_np)
            
            if len(self._obs_history) > self.sequence_length:
                self._obs_history = self._obs_history[-self.sequence_length:]
            
            if len(self._obs_history) < self.sequence_length:
                padding = [self._obs_history[0]] * (self.sequence_length - len(self._obs_history))
                sequence = padding + self._obs_history
            else:
                sequence = self._obs_history[-self.sequence_length:]
            
            return torch.tensor(np.array(sequence), dtype=torch.float32, device=device).unsqueeze(0)
        
        # 批量處理
        sequences = obs.unsqueeze(1).repeat(1, self.sequence_length, 1)
        return sequences
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """前向傳播"""
        device = observations.device
        
        # 建構序列
        sequence = self._update_history(observations)
        
        # 輸入嵌入
        embedded = self.input_embedding(sequence)
        
        # 加入位置編碼
        pos_encoding = self.pos_encoding[:, :sequence.shape[1], :].to(device)
        embedded = embedded + pos_encoding
        
        # Transformer 處理
        transformer_out = self.transformer(embedded)
        
        # 取 CLS token (第一個位置) 或平均
        cls_output = transformer_out[:, 0, :]  # 使用第一個位置
        # cls_output = transformer_out.mean(dim=1)  # 或使用平均
        
        # 輸出層
        features = self.fc(cls_output)
        
        return features


class CNNLSTMFeaturesExtractor(BaseFeaturesExtractor):
    """
    CNN-LSTM 混合特徵提取器
    
    CNN 提取局部特徵，LSTM 捕捉時序依賴
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
        cnn_channels: List[int] = None,
        lstm_hidden_size: int = 64,
        sequence_length: int = 20,
    ):
        super().__init__(observation_space, features_dim)
        
        self.obs_dim = observation_space.shape[0]
        self.sequence_length = sequence_length
        
        if cnn_channels is None:
            cnn_channels = [16, 32]
        
        # 1D CNN 層
        cnn_layers = []
        in_channels = 1
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # 計算 CNN 輸出維度
        with torch.no_grad():
            sample = torch.zeros(1, 1, self.obs_dim)
            cnn_out = self.cnn(sample)
            self.cnn_out_dim = cnn_out.shape[1] * cnn_out.shape[2]
        
        # LSTM 層
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_dim,
            hidden_size=lstm_hidden_size,
            batch_first=True,
        )
        
        # 輸出層
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, features_dim),
            nn.ReLU(),
        )
        
        self._obs_history: List[np.ndarray] = []
    
    def reset_history(self):
        self._obs_history = []
    
    def _update_history(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        device = obs.device
        
        if batch_size == 1:
            obs_np = obs.detach().cpu().numpy()[0]
            self._obs_history.append(obs_np)
            
            if len(self._obs_history) > self.sequence_length:
                self._obs_history = self._obs_history[-self.sequence_length:]
            
            if len(self._obs_history) < self.sequence_length:
                padding = [self._obs_history[0]] * (self.sequence_length - len(self._obs_history))
                sequence = padding + self._obs_history
            else:
                sequence = self._obs_history[-self.sequence_length:]
            
            return torch.tensor(np.array(sequence), dtype=torch.float32, device=device).unsqueeze(0)
        
        sequences = obs.unsqueeze(1).repeat(1, self.sequence_length, 1)
        return sequences
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """前向傳播"""
        batch_size = observations.shape[0]
        
        # 建構序列
        sequence = self._update_history(observations)
        
        # CNN 處理每個時間步
        cnn_features = []
        for t in range(sequence.shape[1]):
            step_obs = sequence[:, t, :].unsqueeze(1)  # (batch, 1, obs_dim)
            cnn_out = self.cnn(step_obs)  # (batch, channels, reduced_dim)
            cnn_out = cnn_out.view(batch_size, -1)  # (batch, cnn_out_dim)
            cnn_features.append(cnn_out)
        
        cnn_features = torch.stack(cnn_features, dim=1)  # (batch, seq, cnn_out_dim)
        
        # LSTM 處理
        lstm_out, _ = self.lstm(cnn_features)
        
        # 取最後輸出
        last_output = lstm_out[:, -1, :]
        
        # 全連接層
        features = self.fc(last_output)
        
        return features


# =============================================================================
# 策略網路包裝器
# =============================================================================

def create_policy_kwargs(
    extractor_type: str = "lstm",
    features_dim: int = 128,
    sequence_length: int = 20,
    **kwargs
) -> Dict:
    """
    建立策略網路參數
    
    Args:
        extractor_type: 特徵提取器類型 ("mlp", "lstm", "transformer", "cnn_lstm")
        features_dim: 特徵維度
        sequence_length: 序列長度
        **kwargs: 其他參數
    
    Returns:
        policy_kwargs 字典
    """
    if extractor_type == "mlp":
        # 使用預設 MLP
        return {
            "net_arch": kwargs.get("net_arch", [256, 256]),
        }
    
    elif extractor_type == "lstm":
        return {
            "features_extractor_class": LSTMFeaturesExtractor,
            "features_extractor_kwargs": {
                "features_dim": features_dim,
                "lstm_hidden_size": kwargs.get("lstm_hidden_size", 64),
                "lstm_num_layers": kwargs.get("lstm_num_layers", 1),
                "sequence_length": sequence_length,
            },
            "net_arch": kwargs.get("net_arch", [128]),
        }
    
    elif extractor_type == "transformer":
        return {
            "features_extractor_class": TransformerFeaturesExtractor,
            "features_extractor_kwargs": {
                "features_dim": features_dim,
                "d_model": kwargs.get("d_model", 64),
                "nhead": kwargs.get("nhead", 4),
                "num_layers": kwargs.get("num_layers", 2),
                "sequence_length": sequence_length,
            },
            "net_arch": kwargs.get("net_arch", [128]),
        }
    
    elif extractor_type == "cnn_lstm":
        return {
            "features_extractor_class": CNNLSTMFeaturesExtractor,
            "features_extractor_kwargs": {
                "features_dim": features_dim,
                "cnn_channels": kwargs.get("cnn_channels", [16, 32]),
                "lstm_hidden_size": kwargs.get("lstm_hidden_size", 64),
                "sequence_length": sequence_length,
            },
            "net_arch": kwargs.get("net_arch", [128]),
        }
    
    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}")


# =============================================================================
# 環境包裝器 - 維護觀察歷史
# =============================================================================

class SequenceObservationWrapper(gym.ObservationWrapper):
    """
    序列觀察包裝器
    
    將單一觀察轉換為包含歷史的序列
    """
    
    def __init__(
        self,
        env: gym.Env,
        sequence_length: int = 20,
    ):
        super().__init__(env)
        
        self.sequence_length = sequence_length
        self.obs_dim = env.observation_space.shape[0]
        
        # 更新觀察空間
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(sequence_length, self.obs_dim),
            dtype=np.float32,
        )
        
        self._obs_history: List[np.ndarray] = []
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._obs_history = [obs.copy()]
        return self.observation(obs), info
    
    def observation(self, obs: np.ndarray) -> np.ndarray:
        """將觀察轉換為序列"""
        self._obs_history.append(obs.copy())
        
        # 保持固定長度
        if len(self._obs_history) > self.sequence_length:
            self._obs_history = self._obs_history[-self.sequence_length:]
        
        # 填充
        if len(self._obs_history) < self.sequence_length:
            padding = [self._obs_history[0]] * (self.sequence_length - len(self._obs_history))
            sequence = padding + self._obs_history
        else:
            sequence = self._obs_history
        
        return np.array(sequence, dtype=np.float32)
