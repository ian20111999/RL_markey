#!/usr/bin/env python3
"""
導入歷史訓練運行到 metrics.db
"""
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from utils.metrics_db import MetricsDatabase


def import_run(db: MetricsDatabase, run_dir: Path):
    """導入單個運行"""
    try:
        # 讀取配置
        config_path = run_dir / "config.yaml"
        if not config_path.exists():
            print(f"⚠️  跳過 {run_dir.name}: 沒有 config.yaml")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 提取運行資訊
        run_id = run_dir.name
        
        # 從 data_file 提取 symbol
        data_file = config.get('env', {}).get('data_file', '')
        if 'btc' in data_file.lower():
            symbol = 'BTCUSDT'
        elif 'eth' in data_file.lower():
            symbol = 'ETHUSDT'
        else:
            symbol = config.get('symbol', 'UNKNOWN')
        
        algorithm = config.get('algorithm', 'SAC')
        
        # 解析時間戳
        try:
            timestamp_str = run_id.split('_')[2]
            timestamp = int(timestamp_str)
        except:
            timestamp = int(run_dir.stat().st_mtime)
        
        # 讀取評估結果
        eval_path = run_dir / "evaluation_results.json"
        if eval_path.exists():
            with open(eval_path, 'r') as f:
                eval_results = json.load(f)
            
            mean_pnl = eval_results.get('mean_pnl', 0)
            std_pnl = eval_results.get('std_pnl', 0)
            mean_trades = eval_results.get('mean_trades', 0)
            win_rate = eval_results.get('win_rate', 0)
            sharpe_ratio = mean_pnl / std_pnl if std_pnl > 0 else 0
            
            # 檢查是否可接受
            is_acceptable = (
                mean_pnl > 0 and
                win_rate >= 0.45 and
                mean_trades >= 50
            )
            
            status = 'completed'
        else:
            mean_pnl = None
            std_pnl = None
            mean_trades = None
            win_rate = None
            sharpe_ratio = None
            is_acceptable = False
            status = 'started'
        
        # 記錄到資料庫
        run_info = {
            'run_id': run_id,
            'symbol': symbol,
            'timestamp': datetime.fromtimestamp(timestamp).isoformat(),
            'attempt': 1,
            'seed': config.get('seed', 42),
            'status': status,
            'config': config
        }
        db.add_training_run(run_info)
        
        if status == 'completed':
            # 計算複合分數
            composite_score = (
                mean_pnl * 0.4 +
                win_rate * 5000 * 0.3 +
                sharpe_ratio * 1000 * 0.2 +
                (mean_trades / 100) * 500 * 0.1
            )
            
            results = {
                'mean_pnl': mean_pnl,
                'std_pnl': std_pnl,
                'total_trades': int(mean_trades) if mean_trades else 0,
                'win_rate': win_rate,
                'sharpe_ratio': sharpe_ratio,
                'composite_score': composite_score,
                'is_acceptable': is_acceptable
            }
            db.add_results(run_id, results)
        
        pnl_str = f"${mean_pnl:.2f}" if mean_pnl is not None else "$0.00"
        print(f"✅ 導入: {run_id[:30]}... - {symbol} - PnL: {pnl_str}")
        return True
        
    except Exception as e:
        print(f"❌ 錯誤導入 {run_dir.name}: {e}")
        return False


def main():
    print("🔄 導入歷史訓練運行到資料庫")
    print("="*60)
    
    # 初始化資料庫
    db_path = Path("logs/metrics.db")
    db = MetricsDatabase(db_path)
    
    # 掃描 runs 目錄
    runs_dir = Path("runs")
    if not runs_dir.exists():
        print("❌ runs 目錄不存在")
        return 1
    
    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith('run_')])
    
    print(f"找到 {len(run_dirs)} 個運行目錄")
    print()
    
    success_count = 0
    for run_dir in run_dirs:
        if import_run(db, run_dir):
            success_count += 1
    
    print()
    print("="*60)
    print(f"✅ 成功導入: {success_count}/{len(run_dirs)}")
    
    # 顯示統計
    recent = db.get_recent_runs(limit=100)
    print(f"📊 資料庫中現有運行數: {len(recent)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
