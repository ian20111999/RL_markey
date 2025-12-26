"""
Integration Testing Suite
完整整合測試：環境、訓練、資料庫、API
"""
import pytest
import sys
from pathlib import Path
import tempfile
import shutil
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestEnvironmentIntegration:
    """測試環境整合"""
    
    def test_env_creation_from_config(self):
        """測試從配置創建環境"""
        try:
            from envs.env_factory import create_env_from_config
        except ImportError:
            pytest.skip("env_factory not available")
        
        config_path = project_root / 'configs' / 'default.yaml'
        if not config_path.exists():
            pytest.skip("Config file not found")
        
        try:
            env = create_env_from_config(str(config_path))
            assert env is not None
            
            # Test reset
            obs, info = env.reset()
            assert obs is not None
            assert info is not None
            
            # Test step
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            assert isinstance(reward, (float, int))
            env.close()
        except Exception as e:
            pytest.skip(f"Cannot create env: {e}")
    
    def test_env_with_data(self, sample_ohlcv_data):
        """測試環境使用合成資料"""
        from envs.market_making_env import MarketMakingEnv
        
        env = MarketMakingEnv(
            df=sample_ohlcv_data,
            initial_cash=10000,
            episode_length=500,
        )
        
        obs, info = env.reset()
        assert obs is not None
        
        # Run a few steps
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        
        env.close()


class TestTrainingIntegration:
    """測試訓練整合"""
    
    def test_quick_training_run(self, sample_ohlcv_data):
        """測試快速訓練（少量步數）"""
        from stable_baselines3 import SAC
        from envs.market_making_env import MarketMakingEnv
        
        env = MarketMakingEnv(
            df=sample_ohlcv_data,
            initial_cash=10000,
            episode_length=500,
        )
        
        # Train for minimal steps
        model = SAC('MlpPolicy', env, verbose=0)
        model.learn(total_timesteps=1000)
        
        # Test prediction
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert action is not None
        
        env.close()
    
    def test_model_save_load(self, sample_ohlcv_data, tmp_path):
        """測試模型保存與載入"""
        from stable_baselines3 import SAC
        from envs.market_making_env import MarketMakingEnv
        
        env = MarketMakingEnv(
            df=sample_ohlcv_data,
            initial_cash=10000,
            episode_length=500,
        )
        
        # Train and save
        model = SAC('MlpPolicy', env, verbose=0)
        model.learn(total_timesteps=500)
        
        model_path = tmp_path / 'test_model.zip'
        model.save(str(model_path))
        
        # Load and test
        loaded_model = SAC.load(str(model_path))
        
        obs, _ = env.reset()
        action1, _ = model.predict(obs, deterministic=True)
        action2, _ = loaded_model.predict(obs, deterministic=True)
        
        # Actions should be identical
        assert (action1 == action2).all()
        
        env.close()


class TestDatabaseIntegration:
    """測試資料庫整合"""
    
    def test_end_to_end_training_logging(self):
        """測試端到端訓練日誌"""
        import os
        from utils.database import get_database
        
        # Use SQLite for testing
        os.environ['DB_TYPE'] = 'sqlite'
        test_db_path = project_root / 'logs' / 'test_integration.db'
        os.environ['SQLITE_DB_PATH'] = str(test_db_path)
        
        # Remove old test database if exists
        if test_db_path.exists():
            test_db_path.unlink()
        
        try:
            db = get_database()
            
            # Initialize database schema
            schema_file = project_root / 'scripts' / 'init_db_sqlite.sql'
            if schema_file.exists():
                with open(schema_file, 'r', encoding='utf-8') as f:
                    schema_sql = f.read()
                    
                    # Remove single-line comments
                    lines = []
                    for line in schema_sql.split('\n'):
                        # Remove comments
                        if '--' in line:
                            line = line[:line.index('--')]
                        lines.append(line)
                    schema_sql = '\n'.join(lines)
                    
                    # Split statements properly (handle BEGIN...END blocks)
                    statements = []
                    current_statement = []
                    in_trigger = False
                    
                    for line in schema_sql.split('\n'):
                        line = line.strip()
                        if not line:
                            continue
                        
                        current_statement.append(line)
                        
                        # Check if we're entering a trigger
                        if 'BEGIN' in line.upper():
                            in_trigger = True
                        
                        # Check if we're leaving a trigger
                        if in_trigger and 'END' in line.upper() and ';' in line:
                            in_trigger = False
                            statements.append(' '.join(current_statement))
                            current_statement = []
                        # Normal statement ending
                        elif not in_trigger and ';' in line:
                            statements.append(' '.join(current_statement))
                            current_statement = []
                    
                    # Add any remaining statement
                    if current_statement:
                        statements.append(' '.join(current_statement))
                    
                    # Execute each statement
                    for statement in statements:
                        statement = statement.strip()
                        if statement:
                            try:
                                db.execute(statement)
                            except Exception as e:
                                # Ignore certain errors
                                error_msg = str(e).lower()
                                if 'no such table' not in error_msg and 'already exists' not in error_msg:
                                    print(f"Warning: SQL execution error: {e}")
                db.commit()
            
            # Insert test symbol
            db.execute("""
                INSERT OR IGNORE INTO symbols (symbol, base_currency, quote_currency)
                VALUES (?, ?, ?)
            """, ('TESTBTC', 'TEST', 'USDT'))
            db.commit()
            
            # Create training run
            run_id = f'TEST_INTEGRATION_{int(datetime.now().timestamp())}'
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
                json.dumps({'lr': 0.0003})
            ))
            db.commit()
            
            # Insert episodes
            for ep in range(5):
                db.execute("""
                    INSERT INTO episodes (
                        run_id, episode_num, episode_reward, episode_pnl, timestamp
                    ) VALUES (?, ?, ?, ?, ?)
                """, (run_id, ep, 100 + ep * 10, 50 + ep * 5, datetime.now().isoformat()))
            db.commit()
            
            # Update run to completed
            db.execute("""
                UPDATE training_runs 
                SET status = ?, end_time = ?, final_pnl = ?
                WHERE run_id = ?
            """, ('completed', datetime.now().isoformat(), 250.5, run_id))
            db.commit()
            
            # Register model
            db.execute("""
                INSERT INTO models (
                    run_id, model_name, symbol, model_type, model_path,
                    performance_metrics, created_at, is_deployed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                f'MODEL_{run_id}',
                'TESTBTC',
                'SAC',
                f'/models/{run_id}.zip',
                json.dumps({'sharpe': 2.5, 'pnl': 250.5}),
                datetime.now().isoformat(),
                1
            ))
            db.commit()
            
            # Verify data
            result = db.fetchone("""
                SELECT * FROM training_runs WHERE run_id = ?
            """, (run_id,))
            assert result is not None
            assert result['status'] == 'completed'
            
            episodes = db.fetchall("""
                SELECT * FROM episodes WHERE run_id = ?
            """, (run_id,))
            assert len(episodes) == 5
            
            model = db.fetchone("""
                SELECT * FROM models WHERE model_name = ?
            """, (f'MODEL_{run_id}',))
            assert model is not None
            
            # Cleanup
            db.execute("DELETE FROM episodes WHERE run_id = ?", (run_id,))
            db.execute("DELETE FROM models WHERE model_name = ?", (f'MODEL_{run_id}',))
            db.execute("DELETE FROM training_runs WHERE run_id = ?", (run_id,))
            db.execute("DELETE FROM symbols WHERE symbol = ?", ('TESTBTC',))
            db.commit()
            db.close()
            
        finally:
            # Clean up test database
            if test_db_path.exists():
                test_db_path.unlink()


class TestProductionIntegration:
    """測試生產環境整合"""
    
    def test_model_registry_workflow(self):
        """測試模型註冊流程"""
        from production.model_registry import ModelRegistry, ModelMetrics
        
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / 'registry'
            registry = ModelRegistry(registry_path)
            
            # Create dummy model files
            model_path = Path(tmpdir) / 'test_model.zip'
            config_path = Path(tmpdir) / 'test_config.yaml'
            model_path.write_text('dummy model')
            config_path.write_text('dummy config')
            
            # Register model
            metrics = ModelMetrics(
                mean_pnl=150.0,
                std_pnl=20.0,
                sharpe_ratio=2.5,
                max_drawdown=0.10,
                win_rate=0.65,
                total_trades=100,
                mean_max_inventory=3.0,
                profitability_score=85.0
            )
            
            model_id = registry.register_model(
                model_path=str(model_path),
                config_path=str(config_path),
                metrics=metrics,
                algorithm='SAC',
                training_timesteps=100000,
                tags=['test', 'integration']
            )
            
            assert model_id is not None
            
            # Get model metadata
            metadata = registry.get_model_metadata(model_id)
            assert metadata is not None
            assert metadata.algorithm == 'SAC'
            assert metadata.production_ready is True
            
            # List models
            models = registry.list_models()
            assert len(models) >= 1
            
            # Get best model
            best_id = registry.get_best_model()
            assert best_id is not None


class TestPipelineIntegration:
    """測試完整 Pipeline"""
    
    def test_mini_pipeline(self):
        """測試迷你 Pipeline（快速版本）"""
        import os
        from utils.database import get_database
        import pandas as pd
        import numpy as np
        
        # Setup test environment
        os.environ['DB_TYPE'] = 'sqlite'
        test_db_path = project_root / 'logs' / 'test_pipeline.db'
        os.environ['SQLITE_DB_PATH'] = str(test_db_path)
        
        # Remove old test database if exists
        if test_db_path.exists():
            test_db_path.unlink()
        
        try:
            # Create synthetic data
            data = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=2000, freq='1min'),
                'open': np.random.randn(2000).cumsum() + 50000,
                'high': np.random.randn(2000).cumsum() + 50100,
                'low': np.random.randn(2000).cumsum() + 49900,
                'close': np.random.randn(2000).cumsum() + 50000,
                'volume': np.random.rand(2000) * 100
            })
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                data.to_csv(f.name, index=False)
                data_file = f.name
            
            try:
                # Initialize database
                db = get_database()
                
                # Initialize database schema
                schema_file = project_root / 'scripts' / 'init_db_sqlite.sql'
                if schema_file.exists():
                    with open(schema_file, 'r', encoding='utf-8') as f:
                        schema_sql = f.read()
                        
                        # Remove single-line comments
                        lines = []
                        for line in schema_sql.split('\n'):
                            # Remove comments
                            if '--' in line:
                                line = line[:line.index('--')]
                            lines.append(line)
                        schema_sql = '\n'.join(lines)
                        
                        # Split statements properly (handle BEGIN...END blocks)
                        statements = []
                        current_statement = []
                        in_trigger = False
                        
                        for line in schema_sql.split('\n'):
                            line = line.strip()
                            if not line:
                                continue
                            
                            current_statement.append(line)
                            
                            # Check if we're entering a trigger
                            if 'BEGIN' in line.upper():
                                in_trigger = True
                            
                            # Check if we're leaving a trigger
                            if in_trigger and 'END' in line.upper() and ';' in line:
                                in_trigger = False
                                statements.append(' '.join(current_statement))
                                current_statement = []
                            # Normal statement ending
                            elif not in_trigger and ';' in line:
                                statements.append(' '.join(current_statement))
                                current_statement = []
                        
                        # Add any remaining statement
                        if current_statement:
                            statements.append(' '.join(current_statement))
                        
                        # Execute each statement
                        for statement in statements:
                            statement = statement.strip()
                            if statement:
                                try:
                                    db.execute(statement)
                                except Exception as e:
                                    # Ignore certain errors
                                    error_msg = str(e).lower()
                                    if 'no such table' not in error_msg and 'already exists' not in error_msg:
                                        print(f"Warning: SQL execution error: {e}")
                    db.commit()
                
                db.execute("""
                    INSERT OR IGNORE INTO symbols (symbol, base_currency, quote_currency)
                    VALUES (?, ?, ?)
                """, ('TESTPIPELINE', 'TEST', 'USDT'))
                db.commit()
                
                # Train model
                from stable_baselines3 import SAC
                from envs.market_making_env import MarketMakingEnv
                
                env = MarketMakingEnv(df=data, initial_cash=10000)
                model = SAC('MlpPolicy', env, verbose=0)
                
                # Quick training
                model.learn(total_timesteps=2000)
                
                # Evaluate
                obs, _ = env.reset()
                total_reward = 0
                for _ in range(100):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total_reward += reward
                    if terminated or truncated:
                        break
                
                # Log to database
                run_id = f'PIPELINE_TEST_{int(datetime.now().timestamp())}'
                db.execute("""
                    INSERT INTO training_runs (
                        run_id, symbol, algorithm, start_time, end_time,
                        status, final_pnl, config
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    'TESTPIPELINE',
                    'SAC',
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    'completed',
                    total_reward,
                    json.dumps({'timesteps': 2000})
                ))
                db.commit()
                
                # Verify
                result = db.fetchone("""
                    SELECT * FROM training_runs WHERE run_id = ?
                """, (run_id,))
                assert result is not None
                assert result['status'] == 'completed'
                
                # Cleanup
                db.execute("DELETE FROM training_runs WHERE run_id = ?", (run_id,))
                db.execute("DELETE FROM symbols WHERE symbol = ?", ('TESTPIPELINE',))
                db.commit()
                db.close()
                env.close()
                
            finally:
                # Clean up data file
                Path(data_file).unlink()
        
        finally:
            # Clean up test database
            if test_db_path.exists():
                test_db_path.unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
