#!/usr/bin/env python3
"""
訓練監控腳本
自動追蹤訓練進度，記錄關鍵指標，並在異常時發出警告
"""

import time
import subprocess
import re
from pathlib import Path
from datetime import datetime

def check_process():
    """檢查訓練進程是否仍在運行"""
    try:
        result = subprocess.run(
            ["ps", "aux"], 
            capture_output=True, 
            text=True
        )
        for line in result.stdout.split('\n'):
            if ('continue_training.py' in line or 'run_stabilization.py' in line) and 'grep' not in line:
                # 提取 CPU 使用率
                parts = line.split()
                cpu = parts[2] if len(parts) > 2 else 'N/A'
                mem = parts[3] if len(parts) > 3 else 'N/A'
                return True, cpu, mem
        return False, None, None
    except Exception as e:
        print(f"Error checking process: {e}")
        return False, None, None

def parse_log(log_path, last_position=0):
    """解析訓練日誌，提取最新指標"""
    try:
        with open(log_path, 'r') as f:
            f.seek(last_position)
            new_content = f.read()
            new_position = f.tell()
            
        if not new_content.strip():
            return None, new_position
        
        # 提取關鍵指標
        metrics = {}
        
        # 提取 timesteps
        timesteps_matches = re.findall(r'total_timesteps\s+\|\s+(\d+)', new_content)
        if timesteps_matches:
            metrics['timesteps'] = int(timesteps_matches[-1])
        
        # 提取 episodes
        episodes_matches = re.findall(r'episodes\s+\|\s+(\d+)', new_content)
        if episodes_matches:
            metrics['episodes'] = int(episodes_matches[-1])
        
        # 提取 actor_loss
        actor_loss_matches = re.findall(r'actor_loss\s+\|\s+([-\d.]+)', new_content)
        if actor_loss_matches:
            metrics['actor_loss'] = float(actor_loss_matches[-1])
        
        # 提取 critic_loss
        critic_loss_matches = re.findall(r'critic_loss\s+\|\s+([\d.]+)', new_content)
        if critic_loss_matches:
            metrics['critic_loss'] = float(critic_loss_matches[-1])
        
        # 提取 FPS
        fps_matches = re.findall(r'fps\s+\|\s+(\d+)', new_content)
        if fps_matches:
            metrics['fps'] = int(fps_matches[-1])
        
        # 提取 eval reward (如果有)
        eval_reward_match = re.search(r'eval.*mean_reward.*?([-\d.]+)', new_content, re.IGNORECASE)
        if eval_reward_match:
            metrics['eval_reward'] = float(eval_reward_match.group(1))
        
        return metrics, new_position
    except FileNotFoundError:
        print(f"Log file not found: {log_path}")
        return None, last_position
    except Exception as e:
        print(f"Error parsing log: {e}")
        return None, last_position

def create_progress_bar(current, total, width=20):
    """建立文字進度條"""
    percent = min(1.0, current / total)
    filled = int(width * percent)
    bar = "=" * filled + ">" + "." * (width - filled - 1)
    if filled == width:
        bar = "=" * width
    return f"[{bar}]"

def format_metrics(metrics):
    """格式化指標輸出"""
    if not metrics:
        return "No new metrics"
    
    lines = []
    if 'timesteps' in metrics:
        total_steps = 300000
        # 如果是 stabilization run，目標可能是 200000，但這裡先寫死或根據上下文判斷
        # 為了通用，我們假設目標是 300k，或者從 metrics 推斷（如果有的話）
        # 這裡簡單處理，使用 300k
        progress = (metrics['timesteps'] / total_steps) * 100
        bar = create_progress_bar(metrics['timesteps'], total_steps)
        lines.append(f"{bar} {progress:.1f}% ({metrics['timesteps']:,}/{total_steps:,})")
    if 'episodes' in metrics:
        lines.append(f"Eps: {metrics['episodes']}")
    if 'actor_loss' in metrics:
        lines.append(f"Actor: {metrics['actor_loss']:.2f}")
    if 'critic_loss' in metrics:
        lines.append(f"Critic: {metrics['critic_loss']:.2f}")
    if 'fps' in metrics:
        lines.append(f"FPS: {metrics['fps']}")
    if 'eval_reward' in metrics:
        lines.append(f"⭐ Eval Reward: {metrics['eval_reward']:.2f}")
    
    return " | ".join(lines)

def check_anomalies(metrics, history):
    """檢查異常狀況"""
    warnings = []
    
    if metrics:
        # 檢查 critic loss 爆炸
        if 'critic_loss' in metrics and metrics['critic_loss'] > 1000:
            warnings.append(f"⚠️  Critic Loss 異常高: {metrics['critic_loss']:.1f}")
        
        # 檢查 FPS 異常低
        if 'fps' in metrics and metrics['fps'] < 50:
            warnings.append(f"⚠️  FPS 異常低: {metrics['fps']}")
        
        # 檢查進度停滯
        if history and len(history) > 3:
            recent_steps = [h.get('timesteps', 0) for h in history[-3:]]
            if len(set(recent_steps)) == 1 and recent_steps[0] > 0:
                warnings.append("⚠️  訓練可能停滯 (timesteps 未增加)")
    
    return warnings

def main():
    log_path = Path("stabilization_log.txt")
    monitor_log = Path("training_monitor.log")
    
    print("=" * 60)
    print("訓練監控啟動")
    print(f"目標: 200,000 steps (Stabilization)")
    print(f"日誌: {log_path}")
    print(f"監控記錄: {monitor_log}")
    print("=" * 60)
    print()
    
    last_position = 0
    history = []
    check_interval = 60  # 每 60 秒檢查一次
    
    with open(monitor_log, 'a') as log_file:
        log_file.write(f"\n{'='*60}\n")
        log_file.write(f"監控開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"{'='*60}\n\n")
    
    try:
        while True:
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            # 檢查進程
            is_running, cpu, mem = check_process()
            
            if not is_running:
                msg = f"[{timestamp}] ⚠️  訓練進程已停止"
                print(msg)
                with open(monitor_log, 'a') as log_file:
                    log_file.write(f"{msg}\n")
                break
            
            # 解析日誌
            metrics, last_position = parse_log(log_path, last_position)
            
            if metrics:
                history.append(metrics)
                
                # 檢查異常
                warnings = check_anomalies(metrics, history)
                
                # 輸出狀態
                status = f"[{timestamp}] {format_metrics(metrics)}"
                if cpu and mem:
                    status += f" | CPU: {cpu}% | MEM: {mem}%"
                
                print(status)
                
                # 寫入日誌
                with open(monitor_log, 'a') as log_file:
                    log_file.write(f"{status}\n")
                    for warning in warnings:
                        print(warning)
                        log_file.write(f"{warning}\n")
                
                # 檢查是否完成
                if 'timesteps' in metrics and metrics['timesteps'] >= 300000:
                    msg = f"[{timestamp}] ✅ 訓練完成！"
                    print("\n" + "=" * 60)
                    print(msg)
                    print("=" * 60)
                    with open(monitor_log, 'a') as log_file:
                        log_file.write(f"\n{msg}\n")
                    break
            else:
                status = f"[{timestamp}] Waiting for new logs..."
                if cpu and mem:
                    status += f" | CPU: {cpu}% | MEM: {mem}%"
                print(status)
            
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n監控已停止")
        with open(monitor_log, 'a') as log_file:
            log_file.write(f"\n監控停止: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

if __name__ == "__main__":
    main()
