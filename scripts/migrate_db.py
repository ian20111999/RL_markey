#!/usr/bin/env python3
"""
資料庫遷移腳本：從 SQLite 遷移到 PostgreSQL

這個腳本會：
1. 讀取 SQLite 資料庫的所有資料
2. 將資料轉換為 PostgreSQL 兼容格式
3. 寫入 PostgreSQL 資料庫
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# 添加專案根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.sqlite_db import SQLiteDatabase
from utils.postgres_db import PostgresDatabase

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_json_fields(data: dict, json_fields: list) -> dict:
    """
    將 JSON 字串欄位轉換為字典
    
    Args:
        data: 資料字典
        json_fields: 需要轉換的欄位列表
    
    Returns:
        dict: 轉換後的資料
    """
    result = data.copy()
    for field in json_fields:
        if field in result and result[field]:
            if isinstance(result[field], str):
                try:
                    result[field] = json.loads(result[field])
                except json.JSONDecodeError:
                    logger.warning(f"無法解析 JSON 欄位 {field}: {result[field]}")
                    result[field] = {}
    return result


def migrate_training_runs(sqlite_db: SQLiteDatabase, postgres_db: PostgresDatabase) -> int:
    """
    遷移 training_runs 表
    
    Returns:
        int: 遷移的記錄數
    """
    logger.info("開始遷移 training_runs 表...")
    
    # 從 SQLite 讀取所有訓練執行
    runs = sqlite_db.fetchall("""
        SELECT run_id, symbol, start_time, end_time, status, config,
               best_reward, final_pnl, total_episodes, created_at, updated_at
        FROM training_runs
        ORDER BY start_time
    """)
    
    count = 0
    for run in runs:
        try:
            # 轉換 JSON 欄位
            run_data = convert_json_fields(run, ['config'])
            
            # 插入到 PostgreSQL
            postgres_db.execute("""
                INSERT INTO training_runs 
                (run_id, symbol, start_time, end_time, status, config,
                 best_reward, final_pnl, total_episodes, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id) DO UPDATE SET
                    end_time = EXCLUDED.end_time,
                    status = EXCLUDED.status,
                    best_reward = EXCLUDED.best_reward,
                    final_pnl = EXCLUDED.final_pnl,
                    total_episodes = EXCLUDED.total_episodes,
                    updated_at = EXCLUDED.updated_at
            """, (
                run_data['run_id'],
                run_data['symbol'],
                run_data['start_time'],
                run_data.get('end_time'),
                run_data['status'],
                json.dumps(run_data.get('config')) if run_data.get('config') else None,
                run_data.get('best_reward'),
                run_data.get('final_pnl'),
                run_data.get('total_episodes'),
                run_data.get('created_at', datetime.now()),
                run_data.get('updated_at', datetime.now())
            ))
            count += 1
            
        except Exception as e:
            logger.error(f"遷移 training_run {run['run_id']} 失敗: {e}")
            continue
    
    postgres_db.commit()
    logger.info(f"成功遷移 {count} 條 training_runs 記錄")
    return count


def migrate_episodes(sqlite_db: SQLiteDatabase, postgres_db: PostgresDatabase) -> int:
    """
    遷移 episodes 表
    
    Returns:
        int: 遷移的記錄數
    """
    logger.info("開始遷移 episodes 表...")
    
    # 從 SQLite 讀取所有 episodes
    episodes = sqlite_db.fetchall("""
        SELECT run_id, episode_num, timestamp, episode_reward, episode_length,
               win_rate, sharpe_ratio, max_drawdown, total_pnl, total_trades,
               metrics, created_at
        FROM episodes
        ORDER BY timestamp
    """)
    
    count = 0
    for episode in episodes:
        try:
            # 轉換 JSON 欄位
            episode_data = convert_json_fields(episode, ['metrics'])
            
            # 插入到 PostgreSQL
            postgres_db.execute("""
                INSERT INTO episodes 
                (run_id, episode_num, timestamp, episode_reward, episode_length,
                 win_rate, sharpe_ratio, max_drawdown, total_pnl, total_trades,
                 metrics, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id, episode_num) DO UPDATE SET
                    episode_reward = EXCLUDED.episode_reward,
                    episode_length = EXCLUDED.episode_length,
                    win_rate = EXCLUDED.win_rate,
                    sharpe_ratio = EXCLUDED.sharpe_ratio,
                    max_drawdown = EXCLUDED.max_drawdown,
                    total_pnl = EXCLUDED.total_pnl,
                    total_trades = EXCLUDED.total_trades,
                    metrics = EXCLUDED.metrics
            """, (
                episode_data['run_id'],
                episode_data['episode_num'],
                episode_data['timestamp'],
                episode_data.get('episode_reward'),
                episode_data.get('episode_length'),
                episode_data.get('win_rate'),
                episode_data.get('sharpe_ratio'),
                episode_data.get('max_drawdown'),
                episode_data.get('total_pnl'),
                episode_data.get('total_trades'),
                json.dumps(episode_data.get('metrics')) if episode_data.get('metrics') else None,
                episode_data.get('created_at', datetime.now())
            ))
            count += 1
            
            # 每 100 筆提交一次
            if count % 100 == 0:
                postgres_db.commit()
                logger.info(f"已遷移 {count} 條 episodes 記錄...")
            
        except Exception as e:
            logger.error(f"遷移 episode {episode['run_id']}-{episode['episode_num']} 失敗: {e}")
            continue
    
    postgres_db.commit()
    logger.info(f"成功遷移 {count} 條 episodes 記錄")
    return count


def migrate_models(sqlite_db: SQLiteDatabase, postgres_db: PostgresDatabase) -> int:
    """
    遷移 models 表
    
    Returns:
        int: 遷移的記錄數
    """
    logger.info("開始遷移 models 表...")
    
    # 從 SQLite 讀取所有模型
    models = sqlite_db.fetchall("""
        SELECT run_id, symbol, model_path, performance_metrics, config,
               created_at, is_best
        FROM models
        ORDER BY created_at
    """)
    
    count = 0
    for model in models:
        try:
            # 轉換 JSON 欄位
            model_data = convert_json_fields(model, ['performance_metrics', 'config'])
            
            # 插入到 PostgreSQL
            postgres_db.execute("""
                INSERT INTO models 
                (run_id, symbol, model_path, performance_metrics, config,
                 created_at, is_best)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                model_data['run_id'],
                model_data['symbol'],
                model_data['model_path'],
                json.dumps(model_data.get('performance_metrics')) if model_data.get('performance_metrics') else None,
                json.dumps(model_data.get('config')) if model_data.get('config') else None,
                model_data.get('created_at', datetime.now()),
                model_data.get('is_best', False)
            ))
            count += 1
            
        except Exception as e:
            logger.error(f"遷移 model {model['run_id']} 失敗: {e}")
            continue
    
    postgres_db.commit()
    logger.info(f"成功遷移 {count} 條 models 記錄")
    return count


def main():
    """主函數"""
    try:
        # 從環境變數讀取資料庫配置
        sqlite_path = os.getenv('SQLITE_DB_PATH', 'logs/metrics.db')
        
        # 檢查 SQLite 資料庫是否存在
        if not os.path.exists(sqlite_path):
            logger.error(f"SQLite 資料庫不存在: {sqlite_path}")
            return 1
        
        logger.info(f"開始從 SQLite ({sqlite_path}) 遷移到 PostgreSQL...")
        
        # 建立資料庫連接
        sqlite_db = SQLiteDatabase(sqlite_path)
        postgres_db = PostgresDatabase()
        
        sqlite_db.connect()
        postgres_db.connect()
        
        # 初始化 PostgreSQL 資料庫結構
        logger.info("初始化 PostgreSQL 資料庫結構...")
        postgres_db.init_schema()
        
        # 執行遷移
        runs_count = migrate_training_runs(sqlite_db, postgres_db)
        episodes_count = migrate_episodes(sqlite_db, postgres_db)
        models_count = migrate_models(sqlite_db, postgres_db)
        
        # 關閉連接
        sqlite_db.close()
        postgres_db.close()
        
        logger.info("=" * 60)
        logger.info("遷移完成！")
        logger.info(f"  - Training Runs: {runs_count} 條")
        logger.info(f"  - Episodes: {episodes_count} 條")
        logger.info(f"  - Models: {models_count} 條")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"遷移過程發生錯誤: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
