"""
性能監控模組
實時追蹤系統資源使用、訓練速度、內存佔用等關鍵指標

特性:
- CPU/GPU/內存監控
- 訓練速度追蹤（steps/sec）
- 自動性能瓶頸檢測
- 資源使用警報
- 性能報告生成
"""

import time
import psutil
import threading
from collections import deque
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class PerformanceMetrics:
    """性能指標數據類"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    gpu_memory_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    steps_per_sec: Optional[float] = None
    
    def to_dict(self):
        return {
            'timestamp': self.timestamp,
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_mb': self.memory_mb,
            'gpu_memory_mb': self.gpu_memory_mb,
            'gpu_utilization': self.gpu_utilization,
            'steps_per_sec': self.steps_per_sec
        }


class PerformanceMonitor:
    """性能監控器"""
    
    def __init__(
        self,
        sample_interval: float = 5.0,
        history_size: int = 720,  # 1小時數據（5秒間隔）
        enable_gpu: bool = True
    ):
        """
        初始化監控器
        
        Args:
            sample_interval: 採樣間隔（秒）
            history_size: 保留的歷史記錄數量
            enable_gpu: 是否監控 GPU
        """
        self.sample_interval = sample_interval
        self.history_size = history_size
        self.enable_gpu = enable_gpu
        
        # 性能指標歷史
        self.metrics_history: deque = deque(maxlen=history_size)
        
        # 訓練速度追蹤
        self.step_counter = 0
        self.last_step_time = time.time()
        self.step_times: deque = deque(maxlen=100)
        
        # GPU 監控
        self.gpu_available = False
        if enable_gpu:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.gpu_available = True
            except:
                pass
        
        # 監控線程
        self.monitoring = False
        self.monitor_thread = None
        
        # 進程信息
        self.process = psutil.Process()
    
    def start_monitoring(self):
        """開始後台監控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止監控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
    
    def _monitor_loop(self):
        """監控循環（在後台線程中運行）"""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
            except Exception as e:
                pass  # 靜默失敗，不影響訓練
            
            time.sleep(self.sample_interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """收集當前性能指標"""
        # CPU 和內存
        cpu_percent = self.process.cpu_percent(interval=0.1)
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        memory_percent = self.process.memory_percent()
        
        # GPU 指標
        gpu_memory_mb = None
        gpu_utilization = None
        
        if self.gpu_available:
            try:
                import pynvml
                gpu_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_memory_mb = gpu_info.used / 1024 / 1024
                
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_utilization = gpu_util.gpu
            except:
                pass
        
        # 訓練速度
        steps_per_sec = self._calculate_steps_per_sec()
        
        return PerformanceMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_mb=memory_mb,
            gpu_memory_mb=gpu_memory_mb,
            gpu_utilization=gpu_utilization,
            steps_per_sec=steps_per_sec
        )
    
    def update_step_count(self, steps: int = 1):
        """更新訓練步數（用於計算速度）"""
        current_time = time.time()
        elapsed = current_time - self.last_step_time
        
        if elapsed > 0:
            self.step_times.append(steps / elapsed)
        
        self.step_counter += steps
        self.last_step_time = current_time
    
    def _calculate_steps_per_sec(self) -> Optional[float]:
        """計算平均訓練速度"""
        if len(self.step_times) == 0:
            return None
        return sum(self.step_times) / len(self.step_times)
    
    def get_current_metrics(self) -> Dict:
        """獲取當前性能指標"""
        metrics = self._collect_metrics()
        return metrics.to_dict()
    
    def get_summary(self) -> Dict:
        """獲取性能摘要統計"""
        if not self.metrics_history:
            return {}
        
        cpu_values = [m.cpu_percent for m in self.metrics_history]
        memory_values = [m.memory_mb for m in self.metrics_history]
        
        summary = {
            'cpu': {
                'current': cpu_values[-1],
                'avg': sum(cpu_values) / len(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values)
            },
            'memory_mb': {
                'current': memory_values[-1],
                'avg': sum(memory_values) / len(memory_values),
                'max': max(memory_values),
                'min': min(memory_values)
            },
            'total_steps': self.step_counter,
            'samples': len(self.metrics_history)
        }
        
        # GPU 統計
        if self.gpu_available:
            gpu_memory_values = [m.gpu_memory_mb for m in self.metrics_history if m.gpu_memory_mb]
            gpu_util_values = [m.gpu_utilization for m in self.metrics_history if m.gpu_utilization]
            
            if gpu_memory_values:
                summary['gpu_memory_mb'] = {
                    'current': gpu_memory_values[-1],
                    'avg': sum(gpu_memory_values) / len(gpu_memory_values),
                    'max': max(gpu_memory_values)
                }
            
            if gpu_util_values:
                summary['gpu_utilization'] = {
                    'current': gpu_util_values[-1],
                    'avg': sum(gpu_util_values) / len(gpu_util_values),
                    'max': max(gpu_util_values)
                }
        
        # 訓練速度
        if self.step_times:
            summary['steps_per_sec'] = {
                'current': self.step_times[-1],
                'avg': sum(self.step_times) / len(self.step_times)
            }
        
        return summary
    
    def check_bottlenecks(self) -> List[str]:
        """檢測性能瓶頸"""
        warnings = []
        
        if not self.metrics_history:
            return warnings
        
        summary = self.get_summary()
        
        # CPU 瓶頸
        if summary['cpu']['avg'] > 90:
            warnings.append("CPU 使用率過高（>90%），可能影響訓練速度")
        
        # 內存瓶頸
        if summary['memory_mb']['current'] > 8000:  # 8GB
            warnings.append("內存使用超過 8GB，建議增加系統內存")
        
        # GPU 瓶頸
        if self.gpu_available and 'gpu_utilization' in summary:
            if summary['gpu_utilization']['avg'] < 30:
                warnings.append("GPU 利用率過低（<30%），可能存在數據加載瓶頸")
        
        # 訓練速度警告
        if 'steps_per_sec' in summary:
            if summary['steps_per_sec']['avg'] < 10:
                warnings.append("訓練速度較慢（<10 steps/sec），考慮優化環境或增加批次大小")
        
        return warnings
    
    def export_metrics(self, filepath: str):
        """導出性能指標到文件"""
        data = {
            'summary': self.get_summary(),
            'history': [m.to_dict() for m in self.metrics_history],
            'bottlenecks': self.check_bottlenecks(),
            'exported_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def __enter__(self):
        """上下文管理器：開始監控"""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器：停止監控並生成報告"""
        self.stop_monitoring()
        
        # 打印摘要
        summary = self.get_summary()
        print("\n" + "="*60)
        print("性能監控摘要")
        print("="*60)
        print(f"總訓練步數: {summary.get('total_steps', 0)}")
        
        if 'cpu' in summary:
            print(f"CPU 使用率: 平均 {summary['cpu']['avg']:.1f}% | 最大 {summary['cpu']['max']:.1f}%")
        
        if 'memory_mb' in summary:
            print(f"內存使用: 平均 {summary['memory_mb']['avg']:.0f}MB | 最大 {summary['memory_mb']['max']:.0f}MB")
        
        if 'steps_per_sec' in summary:
            print(f"訓練速度: {summary['steps_per_sec']['avg']:.1f} steps/sec")
        
        # 瓶頸警告
        bottlenecks = self.check_bottlenecks()
        if bottlenecks:
            print("\n⚠️  性能警告:")
            for warning in bottlenecks:
                print(f"  - {warning}")
        
        print("="*60 + "\n")


# 全局監控器實例（單例模式）
_global_monitor: Optional[PerformanceMonitor] = None


def get_monitor() -> PerformanceMonitor:
    """獲取全局監控器實例"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


if __name__ == "__main__":
    # 測試監控器
    monitor = PerformanceMonitor(sample_interval=1.0)
    
    print("開始性能監控測試（5秒）...")
    with monitor:
        for i in range(5):
            time.sleep(1)
            monitor.update_step_count(100)
            print(f"步數: {monitor.step_counter}")
    
    print("\n導出性能報告...")
    monitor.export_metrics("performance_report.json")
    print("完成！")
