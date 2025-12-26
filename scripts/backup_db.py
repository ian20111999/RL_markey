#!/usr/bin/env python3
"""
資料庫備份腳本

備份 PostgreSQL 資料庫到本地檔案，或將 SQLite 資料庫複製備份。
"""

import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

# 添加專案根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))


def backup_postgres():
    """備份 PostgreSQL 資料庫"""
    host = os.getenv('POSTGRES_HOST', 'localhost')
    port = os.getenv('POSTGRES_PORT', '5432')
    user = os.getenv('POSTGRES_USER', 'rl_user')
    db = os.getenv('POSTGRES_DB', 'rl_market')
    password = os.getenv('POSTGRES_PASSWORD', '')
    
    # 建立備份目錄
    backup_dir = Path('backups')
    backup_dir.mkdir(exist_ok=True)
    
    # 生成備份檔案名稱
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = backup_dir / f'postgres_backup_{timestamp}.sql'
    
    print(f"開始備份 PostgreSQL 資料庫到 {backup_file}...")
    
    # 使用 pg_dump 備份
    env = os.environ.copy()
    env['PGPASSWORD'] = password
    
    try:
        subprocess.run([
            'pg_dump',
            '-h', host,
            '-p', port,
            '-U', user,
            '-d', db,
            '-f', str(backup_file),
            '--verbose'
        ], env=env, check=True)
        
        print(f"✅ 備份成功: {backup_file}")
        print(f"檔案大小: {backup_file.stat().st_size / 1024:.2f} KB")
        
        # 壓縮備份
        compressed_file = f"{backup_file}.gz"
        subprocess.run(['gzip', str(backup_file)], check=True)
        print(f"✅ 已壓縮: {compressed_file}")
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 備份失敗: {e}")
        return 1
    except FileNotFoundError:
        print("❌ 錯誤: 找不到 pg_dump 命令")
        print("請確保已安裝 PostgreSQL 客戶端工具")
        return 1


def backup_sqlite():
    """備份 SQLite 資料庫"""
    import shutil
    
    db_path = Path(os.getenv('SQLITE_DB_PATH', 'logs/metrics.db'))
    
    if not db_path.exists():
        print(f"❌ SQLite 資料庫不存在: {db_path}")
        return 1
    
    # 建立備份目錄
    backup_dir = Path('backups')
    backup_dir.mkdir(exist_ok=True)
    
    # 生成備份檔案名稱
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = backup_dir / f'sqlite_backup_{timestamp}.db'
    
    print(f"開始備份 SQLite 資料庫到 {backup_file}...")
    
    try:
        shutil.copy2(db_path, backup_file)
        print(f"✅ 備份成功: {backup_file}")
        print(f"檔案大小: {backup_file.stat().st_size / 1024:.2f} KB")
        return 0
        
    except Exception as e:
        print(f"❌ 備份失敗: {e}")
        return 1


def main():
    """主函數"""
    db_type = os.getenv('DB_TYPE', 'sqlite').lower()
    
    print("=" * 60)
    print("資料庫備份工具")
    print("=" * 60)
    print(f"資料庫類型: {db_type}")
    print()
    
    if db_type == 'postgresql':
        return backup_postgres()
    elif db_type == 'sqlite':
        return backup_sqlite()
    else:
        print(f"❌ 不支援的資料庫類型: {db_type}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
