"""
Complete Database Testing Suite
測試 PostgreSQL 和 SQLite 資料庫功能
"""
import pytest
import os
import sys
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.database import get_database


class TestDatabaseConnection:
    """測試資料庫連接"""
    
    def test_sqlite_connection(self):
        """測試 SQLite 連接"""
        os.environ['DB_TYPE'] = 'sqlite'
        test_db_path = project_root / 'logs' / 'test_metrics.db'
        os.environ['SQLITE_DB_PATH'] = str(test_db_path)
        
        # Initialize schema if needed
        if not test_db_path.exists():
            import sqlite3
            conn = sqlite3.connect(str(test_db_path))
            schema_file = project_root / 'scripts' / 'init_db_sqlite.sql'
            if schema_file.exists():
                with open(schema_file, 'r') as f:
                    conn.executescript(f.read())
            conn.close()
        
        db = get_database()
        assert db is not None
        # SQLiteDatabase doesn't have db_type attribute, just check it's the right class
        from utils.sqlite_db import SQLiteDatabase
        assert isinstance(db, SQLiteDatabase)
        db.close()
        
    def test_postgresql_connection(self):
        """測試 PostgreSQL 連接"""
        os.environ['DB_TYPE'] = 'postgresql'
        os.environ['POSTGRES_HOST'] = 'localhost'
        os.environ['POSTGRES_PORT'] = '5432'
        os.environ['POSTGRES_DB'] = 'rl_market'
        os.environ['POSTGRES_USER'] = 'rl_user'
        os.environ['POSTGRES_PASSWORD'] = 'rl_password'
        
        try:
            db = get_database()
            assert db is not None
            # 驗證連接有效 - 執行簡單查詢
            result = db.fetchone("SELECT 1 as test")
            assert result is not None
            db.close()
        except Exception as e:
            pytest.skip(f"PostgreSQL not available: {e}")


class TestSymbolOperations:
    """測試交易對操作"""
    
    @pytest.fixture
    def db(self):
        """Setup test database"""
        os.environ['DB_TYPE'] = 'sqlite'
        test_db_path = project_root / 'logs' / 'test_symbols.db'
        os.environ['SQLITE_DB_PATH'] = str(test_db_path)
        
        # Initialize schema
        import sqlite3
        conn = sqlite3.connect(str(test_db_path))
        schema_file = project_root / 'scripts' / 'init_db_sqlite.sql'
        if schema_file.exists():
            with open(schema_file, 'r') as f:
                conn.executescript(f.read())
        conn.close()
        
        db = get_database()
        
        # Clean up test data
        db.execute("DELETE FROM symbols WHERE symbol LIKE 'TEST%'")
        db.commit()
        
        yield db
        
        # Cleanup
        db.execute("DELETE FROM symbols WHERE symbol LIKE 'TEST%'")
        db.commit()
        db.close()
        
        # Remove test db
        if test_db_path.exists():
            test_db_path.unlink()
    
    def test_insert_symbol(self, db):
        """測試插入交易對"""
        db.execute("""
            INSERT INTO symbols (symbol, base_currency, quote_currency, is_active)
            VALUES (?, ?, ?, ?)
        """, ('TESTBTCUSDT', 'TEST', 'USDT', True))
        db.commit()
        
        result = db.fetchone("SELECT * FROM symbols WHERE symbol = ?", ('TESTBTCUSDT',))
        assert result is not None
        assert result['symbol'] == 'TESTBTCUSDT'
        assert result['base_currency'] == 'TEST'
    
    def test_update_symbol(self, db):
        """測試更新交易對"""
        # Insert
        db.execute("""
            INSERT INTO symbols (symbol, base_currency, quote_currency, is_active)
            VALUES (?, ?, ?, ?)
        """, ('TESTETHUSDT', 'ETH', 'USDT', True))
        db.commit()
        
        # Update
        db.execute("""
            UPDATE symbols SET is_active = ? WHERE symbol = ?
        """, (False, 'TESTETHUSDT'))
        db.commit()
        
        result = db.fetchone("SELECT * FROM symbols WHERE symbol = ?", ('TESTETHUSDT',))
        assert result['is_active'] == False
    
    def test_list_symbols(self, db):
        """測試列出交易對"""
        # Insert multiple
        symbols = [
            ('TESTBTCUSDT', 'BTC', 'USDT', True),
            ('TESTETHUSDT', 'ETH', 'USDT', True),
            ('TESTBNBUSDT', 'BNB', 'USDT', False)
        ]
        
        for symbol_data in symbols:
            db.execute("""
                INSERT INTO symbols (symbol, base_currency, quote_currency, is_active)
                VALUES (?, ?, ?, ?)
            """, symbol_data)
        db.commit()
        
        results = db.fetchall("SELECT * FROM symbols WHERE symbol LIKE 'TEST%'")
        assert len(results) >= 3


class TestTrainingRunOperations:
    """測試訓練記錄操作"""
    
    @pytest.fixture
    def db(self):
        """Setup test database"""
        os.environ['DB_TYPE'] = 'sqlite'
        os.environ['SQLITE_DB_PATH'] = 'logs/test_metrics.db'
        db = get_database()
        
        # Ensure symbol exists
        db.execute("""
            INSERT OR IGNORE INTO symbols (symbol, base_currency, quote_currency)
            VALUES (?, ?, ?)
        """, ('TESTBTC', 'TEST', 'USDT'))
        db.commit()
        
        # Clean up test data
        db.execute("DELETE FROM training_runs WHERE run_id LIKE 'TEST%'")
        db.commit()
        
        yield db
        
        # Cleanup
        db.execute("DELETE FROM training_runs WHERE run_id LIKE 'TEST%'")
        db.commit()
        db.close()
    
    def test_create_training_run(self, db):
        """測試創建訓練記錄"""
        run_id = 'TEST_RUN_001'
        
        db.execute("""
            INSERT INTO training_runs (
                run_id, symbol, algorithm, start_time, status, config
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            'TESTBTC',
            'SAC',
            datetime.now().isoformat(),
            'running',
            json.dumps({'learning_rate': 0.0003})
        ))
        db.commit()
        
        result = db.fetchone("SELECT * FROM training_runs WHERE run_id = ?", (run_id,))
        assert result is not None
        assert result['algorithm'] == 'SAC'
        assert result['status'] == 'running'
    
    def test_update_training_run(self, db):
        """測試更新訓練記錄"""
        run_id = 'TEST_RUN_002'
        
        # Create
        db.execute("""
            INSERT INTO training_runs (
                run_id, symbol, algorithm, start_time, status
            ) VALUES (?, ?, ?, ?, ?)
        """, (run_id, 'TESTBTC', 'PPO', datetime.now().isoformat(), 'running'))
        db.commit()
        
        # Update
        db.execute("""
            UPDATE training_runs 
            SET status = ?, end_time = ?, final_pnl = ?
            WHERE run_id = ?
        """, ('completed', datetime.now().isoformat(), 150.5, run_id))
        db.commit()
        
        result = db.fetchone("SELECT * FROM training_runs WHERE run_id = ?", (run_id,))
        assert result['status'] == 'completed'
        assert result['final_pnl'] == 150.5


class TestModelOperations:
    """測試模型操作"""
    
    @pytest.fixture
    def db(self):
        """Setup test database"""
        os.environ['DB_TYPE'] = 'sqlite'
        os.environ['SQLITE_DB_PATH'] = 'logs/test_metrics.db'
        db = get_database()
        
        # Ensure symbol exists
        db.execute("""
            INSERT OR IGNORE INTO symbols (symbol, base_currency, quote_currency)
            VALUES (?, ?, ?)
        """, ('TESTBTC', 'TEST', 'USDT'))
        db.commit()
        
        # Clean up test data
        db.execute("DELETE FROM models WHERE model_name LIKE 'TEST%'")
        db.commit()
        
        yield db
        
        # Cleanup
        db.execute("DELETE FROM models WHERE model_name LIKE 'TEST%'")
        db.commit()
        db.close()
    
    def test_register_model(self, db):
        """測試註冊模型"""
        model_name = 'TEST_MODEL_001'
        run_id = 'TEST_RUN_001'
        
        # First create a training run
        db.execute("""
            INSERT INTO training_runs (
                run_id, symbol, algorithm, start_time, status
            ) VALUES (?, ?, ?, ?, ?)
        """, (run_id, 'TESTBTC', 'SAC', datetime.now().isoformat(), 'completed'))
        
        db.execute("""
            INSERT INTO models (
                run_id, model_name, symbol, model_type, model_path,
                performance_metrics, created_at, is_deployed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            model_name,
            'TESTBTC',
            'SAC',
            '/models/test_model.zip',
            json.dumps({'sharpe_ratio': 2.5, 'pnl': 200}),
            datetime.now().isoformat(),
            1
        ))
        db.commit()
        
        result = db.fetchone("SELECT * FROM models WHERE model_name = ?", (model_name,))
        assert result is not None
        assert result['is_deployed'] == 1
    
    def test_get_best_model(self, db):
        """測試獲取最佳模型"""
        # Create training run first with unique ID
        import time
        run_id = f'TEST_RUN_BEST_{int(time.time() * 1000)}'
        db.execute("""
            INSERT INTO training_runs (
                run_id, symbol, algorithm, start_time, status
            ) VALUES (?, ?, ?, ?, ?)
        """, (run_id, 'TESTBTC', 'SAC', datetime.now().isoformat(), 'completed'))
        
        # Insert multiple models
        models = [
            ('TEST_MODEL_A', 'SAC', json.dumps({'sharpe_ratio': 1.5})),
            ('TEST_MODEL_B', 'PPO', json.dumps({'sharpe_ratio': 2.5})),
            ('TEST_MODEL_C', 'SAC', json.dumps({'sharpe_ratio': 2.0}))
        ]
        
        for model_name, model_type, metrics in models:
            db.execute("""
                INSERT INTO models (
                    run_id, model_name, symbol, model_type, model_path,
                    performance_metrics, created_at, is_deployed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (run_id, model_name, 'TESTBTC', model_type, '/models/test.zip', metrics, datetime.now().isoformat(), 1))
        db.commit()
        
        # Query best model (highest sharpe_ratio)
        result = db.fetchone("""
            SELECT model_name, 
                   json_extract(performance_metrics, '$.sharpe_ratio') as sharpe
            FROM models 
            WHERE symbol = ? AND model_name LIKE 'TEST%'
            ORDER BY sharpe DESC 
            LIMIT 1
        """, ('TESTBTC',))
        
        assert result is not None
        assert result['model_name'] == 'TEST_MODEL_B'


class TestMarketDataOperations:
    """測試市場資料操作"""
    
    @pytest.fixture
    def db(self):
        """Setup test database"""
        os.environ['DB_TYPE'] = 'sqlite'
        os.environ['SQLITE_DB_PATH'] = 'logs/test_metrics.db'
        db = get_database()
        
        # Ensure symbol exists and get ID
        db.execute("""
            INSERT OR IGNORE INTO symbols (symbol, base_currency, quote_currency)
            VALUES (?, ?, ?)
        """, ('TESTBTC', 'TEST', 'USDT'))
        db.commit()
        
        # Clean up test data
        symbol_result = db.fetchone("SELECT id FROM symbols WHERE symbol = ?", ('TESTBTC',))
        if symbol_result:
            db.execute("DELETE FROM market_data WHERE symbol_id = ?", (symbol_result['id'],))
            db.commit()
        
        yield db
        
        # Cleanup
        symbol_result = db.fetchone("SELECT id FROM symbols WHERE symbol = ?", ('TESTBTC',))
        if symbol_result:
            db.execute("DELETE FROM market_data WHERE symbol_id = ?", (symbol_result['id'],))
            db.commit()
        db.close()
    
    def test_insert_market_data(self, db):
        """測試插入市場資料"""
        # Get symbol_id
        symbol_result = db.fetchone("SELECT id FROM symbols WHERE symbol = ?", ('TESTBTC',))
        symbol_id = symbol_result['id']
        
        # Insert market data
        db.execute("""
            INSERT INTO market_data (
                symbol_id, timestamp, timeframe, open, high, low, close, volume
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (symbol_id, datetime.now().isoformat(), '1m', 50000, 50100, 49900, 50050, 100.5))
        db.commit()
        
        result = db.fetchone("""
            SELECT * FROM market_data WHERE symbol_id = ?
        """, (symbol_id,))
        
        assert result is not None
        assert result['close'] == 50050
    
    def test_query_market_data_range(self, db):
        """測試查詢市場資料範圍"""
        symbol_result = db.fetchone("SELECT id FROM symbols WHERE symbol = ?", ('TESTBTC',))
        symbol_id = symbol_result['id']
        
        # Insert multiple records
        for i in range(10):
            db.execute("""
                INSERT INTO market_data (
                    symbol_id, timestamp, timeframe, open, high, low, close, volume
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (symbol_id, datetime.now().isoformat(), '1m', 50000+i, 50100+i, 49900+i, 50050+i, 100+i))
        db.commit()
        
        results = db.fetchall("""
            SELECT * FROM market_data WHERE symbol_id = ? LIMIT 5
        """, (symbol_id,))
        
        assert len(results) >= 5


class TestDatabaseViews:
    """測試資料庫視圖"""
    
    @pytest.fixture
    def db(self):
        """Setup test database"""
        os.environ['DB_TYPE'] = 'sqlite'
        os.environ['SQLITE_DB_PATH'] = 'logs/test_metrics.db'
        return get_database()
    
    def test_query_views(self, db):
        """測試查詢視圖（如果存在）"""
        # Try to query views if they exist
        try:
            result = db.fetchall("SELECT * FROM v_latest_training_runs LIMIT 5")
            assert isinstance(result, list)
        except:
            pytest.skip("Views not available in test database")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
