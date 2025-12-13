"""
Metrics Database for tracking all training runs
"""
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class MetricsDatabase:
    """Store and query training metrics"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Training runs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_runs (
                run_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                attempt INTEGER,
                seed INTEGER,
                status TEXT,
                duration_seconds REAL,
                config_json TEXT,
                data_metrics_json TEXT
            )
        ''')
        
        # Results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_results (
                run_id TEXT PRIMARY KEY,
                mean_pnl REAL,
                std_pnl REAL,
                win_rate REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                total_trades INTEGER,
                composite_score REAL,
                is_acceptable BOOLEAN,
                validation_message TEXT,
                FOREIGN KEY (run_id) REFERENCES training_runs(run_id)
            )
        ''')
        
        # Best models table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS best_models (
                symbol TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                model_path TEXT,
                composite_score REAL,
                mean_pnl REAL,
                sharpe_ratio REAL,
                saved_at TEXT,
                FOREIGN KEY (run_id) REFERENCES training_runs(run_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_training_run(self, run_info: Dict):
        """Add a new training run"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO training_runs 
            (run_id, symbol, timestamp, attempt, seed, status, duration_seconds, config_json, data_metrics_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            run_info['run_id'],
            run_info['symbol'],
            run_info['timestamp'],
            run_info.get('attempt', 1),
            run_info.get('seed', 0),
            run_info.get('status', 'started'),
            run_info.get('duration_seconds', 0),
            json.dumps(run_info.get('config', {})),
            json.dumps(run_info.get('data_metrics', {}))
        ))
        
        conn.commit()
        conn.close()
    
    def update_run_status(self, run_id: str, status: str, duration: float = None):
        """Update run status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if duration is not None:
            cursor.execute('''
                UPDATE training_runs 
                SET status = ?, duration_seconds = ?
                WHERE run_id = ?
            ''', (status, duration, run_id))
        else:
            cursor.execute('''
                UPDATE training_runs 
                SET status = ?
                WHERE run_id = ?
            ''', (status, run_id))
        
        conn.commit()
        conn.close()
    
    def add_results(self, run_id: str, results: Dict):
        """Add training results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO training_results 
            (run_id, mean_pnl, std_pnl, win_rate, sharpe_ratio, max_drawdown, 
             total_trades, composite_score, is_acceptable, validation_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            run_id,
            results.get('mean_pnl', 0),
            results.get('std_pnl', 0),
            results.get('win_rate', 0),
            results.get('sharpe_ratio', None),
            results.get('max_drawdown', None),
            results.get('total_trades', 0),
            results.get('composite_score', 0),
            results.get('is_acceptable', False),
            results.get('validation_message', '')
        ))
        
        conn.commit()
        conn.close()
    
    def update_best_model(self, symbol: str, run_id: str, model_path: str, score: float, pnl: float, sharpe: float = None):
        """Update best model for symbol"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO best_models 
            (symbol, run_id, model_path, composite_score, mean_pnl, sharpe_ratio, saved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol,
            run_id,
            model_path,
            score,
            pnl,
            sharpe,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_best_run_for_symbol(self, symbol: str) -> Optional[Dict]:
        """Get best training run for a symbol"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT r.run_id, r.mean_pnl, r.sharpe_ratio, r.composite_score, t.timestamp
            FROM training_results r
            JOIN training_runs t ON r.run_id = t.run_id
            WHERE t.symbol = ? AND r.is_acceptable = 1
            ORDER BY r.composite_score DESC
            LIMIT 1
        ''', (symbol,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'run_id': row[0],
                'mean_pnl': row[1],
                'sharpe_ratio': row[2],
                'composite_score': row[3],
                'timestamp': row[4]
            }
        return None
    
    def get_recent_runs(self, symbol: str = None, limit: int = 10) -> List[Dict]:
        """Get recent training runs"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if symbol:
            cursor.execute('''
                SELECT t.run_id, t.symbol, t.timestamp, t.status, t.duration_seconds,
                       r.mean_pnl, r.composite_score, r.is_acceptable
                FROM training_runs t
                LEFT JOIN training_results r ON t.run_id = r.run_id
                WHERE t.symbol = ?
                ORDER BY t.timestamp DESC
                LIMIT ?
            ''', (symbol, limit))
        else:
            cursor.execute('''
                SELECT t.run_id, t.symbol, t.timestamp, t.status, t.duration_seconds,
                       r.mean_pnl, r.composite_score, r.is_acceptable
                FROM training_runs t
                LEFT JOIN training_results r ON t.run_id = r.run_id
                ORDER BY t.timestamp DESC
                LIMIT ?
            ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            results.append({
                'run_id': row[0],
                'symbol': row[1],
                'timestamp': row[2],
                'status': row[3],
                'duration_seconds': row[4],
                'mean_pnl': row[5],
                'composite_score': row[6],
                'is_acceptable': bool(row[7]) if row[7] is not None else None
            })
        
        return results
    
    def export_to_json(self, output_path: Path):
        """Export all data to JSON for frontend"""
        conn = sqlite3.connect(self.db_path)
        
        # Get all data
        runs_df = pd.read_sql_query("SELECT * FROM training_runs", conn)
        results_df = pd.read_sql_query("SELECT * FROM training_results", conn)
        best_df = pd.read_sql_query("SELECT * FROM best_models", conn)
        
        conn.close()
        
        data = {
            'training_runs': runs_df.to_dict('records'),
            'results': results_df.to_dict('records'),
            'best_models': best_df.to_dict('records'),
            'exported_at': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported metrics to {output_path}")


# Import pandas only when needed for export
try:
    import pandas as pd
except ImportError:
    pd = None
