#!/usr/bin/env python3
"""
項目清理和優化腳本
自動清理無用文件、優化項目結構

功能:
- 清理日誌文件
- 清理 Python 緩存
- 清理舊訓練記錄
- 優化數據庫
- 生成清理報告
"""

import os
import shutil
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import json


class ProjectCleaner:
    """項目清理工具"""
    
    def __init__(self, project_root: str = "."):
        self.root = Path(project_root).resolve()
        self.stats = {
            'files_removed': 0,
            'dirs_removed': 0,
            'space_freed_mb': 0,
            'actions': []
        }
    
    def log_action(self, action: str, details: str = ""):
        """記錄清理動作"""
        self.stats['actions'].append({
            'action': action,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
        print(f"✓ {action}: {details}")
    
    def get_dir_size(self, path: Path) -> int:
        """計算目錄大小（字節）"""
        total = 0
        try:
            for entry in path.rglob('*'):
                if entry.is_file():
                    total += entry.stat().st_size
        except Exception:
            pass
        return total
    
    def clean_pycache(self):
        """清理 Python 緩存文件"""
        print("\n🧹 清理 Python 緩存...")
        
        patterns = ['__pycache__', '*.pyc', '*.pyo', '*.pyd']
        freed_bytes = 0
        
        for pattern in patterns:
            if pattern.startswith('__'):
                # 目錄
                for cache_dir in self.root.rglob(pattern):
                    if cache_dir.is_dir():
                        size = self.get_dir_size(cache_dir)
                        shutil.rmtree(cache_dir, ignore_errors=True)
                        self.stats['dirs_removed'] += 1
                        freed_bytes += size
            else:
                # 文件
                for cache_file in self.root.rglob(pattern):
                    if cache_file.is_file():
                        size = cache_file.stat().st_size
                        cache_file.unlink()
                        self.stats['files_removed'] += 1
                        freed_bytes += size
        
        self.stats['space_freed_mb'] += freed_bytes / 1024 / 1024
        self.log_action("清理 Python 緩存", f"釋放 {freed_bytes / 1024 / 1024:.2f}MB")
    
    def clean_logs(self, keep_days: int = 7):
        """清理舊日誌文件"""
        print(f"\n🧹 清理 {keep_days} 天前的日誌...")
        
        logs_dir = self.root / "logs"
        if not logs_dir.exists():
            return
        
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        freed_bytes = 0
        
        for log_file in logs_dir.glob("*.log"):
            if log_file.is_file():
                mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                if mtime < cutoff_date:
                    size = log_file.stat().st_size
                    log_file.unlink()
                    self.stats['files_removed'] += 1
                    freed_bytes += size
        
        self.stats['space_freed_mb'] += freed_bytes / 1024 / 1024
        self.log_action("清理舊日誌", f"釋放 {freed_bytes / 1024 / 1024:.2f}MB")
    
    def clean_old_runs(self, keep_count: int = 10):
        """清理舊訓練記錄（保留最新 N 個）"""
        print(f"\n🧹 清理訓練記錄（保留最新 {keep_count} 個）...")
        
        runs_dir = self.root / "runs"
        if not runs_dir.exists():
            return
        
        # 獲取所有 run 目錄並按時間排序
        run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
        run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        freed_bytes = 0
        
        # 刪除舊的
        for old_run in run_dirs[keep_count:]:
            size = self.get_dir_size(old_run)
            shutil.rmtree(old_run, ignore_errors=True)
            self.stats['dirs_removed'] += 1
            freed_bytes += size
        
        self.stats['space_freed_mb'] += freed_bytes / 1024 / 1024
        self.log_action("清理舊訓練記錄", f"刪除 {len(run_dirs) - keep_count} 個，釋放 {freed_bytes / 1024 / 1024:.2f}MB")
    
    def clean_checkpoints(self, keep_best: bool = True):
        """清理訓練檢查點（可選保留最佳）"""
        print("\n🧹 清理訓練檢查點...")
        
        models_dir = self.root / "models"
        if not models_dir.exists():
            return
        
        checkpoint_dirs = [
            models_dir / "checkpoints",
            models_dir / "temp"
        ]
        
        freed_bytes = 0
        
        for checkpoint_dir in checkpoint_dirs:
            if checkpoint_dir.exists():
                # 保留 best_model
                for item in checkpoint_dir.rglob('*'):
                    if item.is_file():
                        if keep_best and 'best' in item.name.lower():
                            continue
                        
                        size = item.stat().st_size
                        item.unlink()
                        self.stats['files_removed'] += 1
                        freed_bytes += size
        
        self.stats['space_freed_mb'] += freed_bytes / 1024 / 1024
        self.log_action("清理檢查點", f"釋放 {freed_bytes / 1024 / 1024:.2f}MB")
    
    def optimize_database(self):
        """優化 SQLite 數據庫"""
        print("\n🧹 優化 SQLite 數據庫...")
        
        db_files = list(self.root.rglob("*.db"))
        
        for db_file in db_files:
            try:
                original_size = db_file.stat().st_size
                
                # 連接並優化
                conn = sqlite3.connect(str(db_file))
                cursor = conn.cursor()
                
                # VACUUM 釋放未使用空間
                cursor.execute("VACUUM")
                
                # ANALYZE 更新統計信息
                cursor.execute("ANALYZE")
                
                conn.commit()
                conn.close()
                
                new_size = db_file.stat().st_size
                saved = (original_size - new_size) / 1024 / 1024
                
                self.log_action(f"優化數據庫 {db_file.name}", f"節省 {saved:.2f}MB")
                self.stats['space_freed_mb'] += saved
                
            except Exception as e:
                print(f"  ⚠️  無法優化 {db_file.name}: {e}")
    
    def clean_cache_dirs(self):
        """清理緩存目錄"""
        print("\n🧹 清理緩存目錄...")
        
        cache_dirs = [
            self.root / ".cache",
            self.root / ".pytest_cache",
            self.root / "htmlcov"
        ]
        
        freed_bytes = 0
        
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                size = self.get_dir_size(cache_dir)
                shutil.rmtree(cache_dir, ignore_errors=True)
                self.stats['dirs_removed'] += 1
                freed_bytes += size
        
        self.stats['space_freed_mb'] += freed_bytes / 1024 / 1024
        self.log_action("清理緩存目錄", f"釋放 {freed_bytes / 1024 / 1024:.2f}MB")
    
    def clean_temp_files(self):
        """清理臨時文件"""
        print("\n🧹 清理臨時文件...")
        
        patterns = ['*.tmp', '*.temp', '*.bak', '*~', '.DS_Store']
        freed_bytes = 0
        
        for pattern in patterns:
            for temp_file in self.root.rglob(pattern):
                if temp_file.is_file():
                    size = temp_file.stat().st_size
                    temp_file.unlink()
                    self.stats['files_removed'] += 1
                    freed_bytes += size
        
        self.stats['space_freed_mb'] += freed_bytes / 1024 / 1024
        self.log_action("清理臨時文件", f"釋放 {freed_bytes / 1024 / 1024:.2f}MB")
    
    def generate_report(self):
        """生成清理報告"""
        report = {
            'summary': {
                'files_removed': self.stats['files_removed'],
                'dirs_removed': self.stats['dirs_removed'],
                'space_freed_mb': round(self.stats['space_freed_mb'], 2),
                'timestamp': datetime.now().isoformat()
            },
            'actions': self.stats['actions']
        }
        
        report_file = self.root / "logs" / "cleanup_report.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def run_all(
        self,
        clean_cache: bool = True,
        clean_logs_days: int = 7,
        clean_runs_keep: int = 10,
        clean_checkpoints: bool = True,
        optimize_db: bool = True
    ):
        """執行所有清理操作"""
        print("=" * 60)
        print("🚀 開始項目清理...")
        print("=" * 60)
        
        if clean_cache:
            self.clean_pycache()
            self.clean_cache_dirs()
            self.clean_temp_files()
        
        if clean_logs_days > 0:
            self.clean_logs(keep_days=clean_logs_days)
        
        if clean_runs_keep > 0:
            self.clean_old_runs(keep_count=clean_runs_keep)
        
        if clean_checkpoints:
            self.clean_checkpoints(keep_best=True)
        
        if optimize_db:
            self.optimize_database()
        
        # 生成報告
        report = self.generate_report()
        
        # 打印摘要
        print("\n" + "=" * 60)
        print("✅ 清理完成！")
        print("=" * 60)
        print(f"📁 刪除文件: {self.stats['files_removed']}")
        print(f"📂 刪除目錄: {self.stats['dirs_removed']}")
        print(f"💾 釋放空間: {self.stats['space_freed_mb']:.2f}MB")
        print(f"📊 詳細報告: logs/cleanup_report.json")
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="項目清理和優化工具")
    parser.add_argument("--project-root", default=".", help="項目根目錄")
    parser.add_argument("--keep-logs-days", type=int, default=7, help="保留多少天的日誌")
    parser.add_argument("--keep-runs", type=int, default=10, help="保留多少個訓練記錄")
    parser.add_argument("--no-cache-clean", action="store_true", help="不清理緩存")
    parser.add_argument("--no-checkpoint-clean", action="store_true", help="不清理檢查點")
    parser.add_argument("--no-db-optimize", action="store_true", help="不優化數據庫")
    
    args = parser.parse_args()
    
    cleaner = ProjectCleaner(args.project_root)
    cleaner.run_all(
        clean_cache=not args.no_cache_clean,
        clean_logs_days=args.keep_logs_days,
        clean_runs_keep=args.keep_runs,
        clean_checkpoints=not args.no_checkpoint_clean,
        optimize_db=not args.no_db_optimize
    )


if __name__ == "__main__":
    main()
