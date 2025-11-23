"""metrics.py: 集中放置績效統計函式，讓訓練與評估腳本共用。"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def max_drawdown(values: Sequence[float]) -> float:
    """計算最大回撤（結果為正值，代表相對高點的最大百分比下跌）。"""

    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(arr)
    denom = np.where(running_max == 0, 1.0, running_max)
    drawdowns = (running_max - arr) / denom
    return float(np.max(drawdowns))


def sharpe_ratio(returns: Sequence[float]) -> float:
    """簡化版 Sharpe（假設無風險利率為 0）。"""

    arr = np.array(list(returns), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std < 1e-8:
        return 0.0
    return mean / std