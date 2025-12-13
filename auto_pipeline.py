"""
Enhanced Automated Training Pipeline with Quality Checks
Fully automated pipeline that ensures production-ready models
"""
import argparse
import subprocess
import sys
import shutil
import json
import yaml
import time
import random
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.validators import DataQualityChecker, EnvironmentHealthChecker, TrainingResultValidator
from utils.production_checker import ProductionReadinessChecker
from utils.metrics_db import MetricsDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class AutomatedPipeline:
    """
    Fully automated training pipeline with:
    - Data quality validation
    - Environment health checks
    - Auto-retry with adaptive tuning
    - Production readiness validation
    - Comprehensive metrics tracking
    """
    
    def __init__(self, config_path: Path = None):
        """Initialize pipeline with configuration"""
        if config_path is None:
            config_path = project_root / "configs" / "pipeline_config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        data_val_cfg = self.config['data_validation']
        self.data_checker = DataQualityChecker(
            min_samples=data_val_cfg['min_samples'],
            max_missing_ratio=data_val_cfg['max_missing_ratio']
        )
        
        self.env_checker = EnvironmentHealthChecker()
        
        train_acc_cfg = self.config['training_acceptance']
        self.result_validator = TrainingResultValidator(
            min_pnl=train_acc_cfg['min_pnl'],
            min_win_rate=train_acc_cfg['min_win_rate'],
            min_sharpe=train_acc_cfg['min_sharpe'],
            max_drawdown=train_acc_cfg['max_drawdown']
        )
        
        prod_cfg = self.config['production_readiness']
        self.prod_checker = ProductionReadinessChecker(
            min_sharpe=prod_cfg['min_sharpe'],
            max_drawdown_threshold=prod_cfg['max_drawdown'],
            min_win_rate=prod_cfg['min_win_rate'],
            min_trades=prod_cfg['min_trades'],
            min_profit_factor=prod_cfg['min_profit_factor'],
            max_volatility_ratio=prod_cfg['max_volatility_ratio']
        )
        
        # Initialize metrics database
        if self.config['metrics']['enabled']:
            db_path = project_root / self.config['metrics']['db_path']
            self.metrics_db = MetricsDatabase(db_path)
        else:
            self.metrics_db = None
        
        # Setup logging
        if self.config['logging']['save_logs']:
            log_dir = project_root / self.config['logging']['log_dir']
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(file_handler)
    
    def find_or_fetch_data(self, symbol: str, data_dir: str = "data") -> Optional[Path]:
        """Find data file or fetch if not available"""
        logger.info(f"Looking for data for {symbol}...")
        
        data_path = project_root / data_dir
        candidates = list(data_path.glob(f"*{symbol.lower()}*.csv"))
        
        if candidates:
            logger.info(f"✅ Data found: {candidates[0].name}")
            return candidates[0]
        
        logger.warning(f"⚠️  Data file for {symbol} not found. Attempting to download...")
        
        # Try to fetch data
        binance_symbol = f"{symbol.upper()}USDT"
        fetch_cmd = [
            sys.executable, str(project_root / "scripts" / "fetch_data.py"),
            "--symbol", binance_symbol,
            "--interval", "1m",
            "--year", "2023",
            "--output_dir", str(data_path)
        ]
        
        try:
            subprocess.run(fetch_cmd, check=True, timeout=300)
            candidates = list(data_path.glob(f"*{symbol.lower()}*.csv"))
            if candidates:
                logger.info(f"✅ Data downloaded: {candidates[0].name}")
                return candidates[0]
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.error(f"❌ Failed to download data: {e}")
        
        return None
    
    def analyze_data(self, data_path: Path) -> Tuple[bool, Dict]:
        """Validate data quality and extract metrics"""
        logger.info(f"🔍 Analyzing data quality: {data_path.name}")
        
        is_valid, metrics, message = self.data_checker.validate(data_path)
        
        if is_valid:
            logger.info(f"✅ Data quality check passed")
            logger.info(f"   Samples: {metrics['total_samples']}, Avg Price: {metrics['avg_price']:.2f}")
            logger.info(f"   Volatility: {metrics['volatility']:.4f}, Returns: {metrics['mean_return']:.6f}")
        else:
            logger.error(f"❌ Data quality check failed: {message}")
        
        return is_valid, metrics
    
    def create_adaptive_config(self, data_path: Path, data_metrics: Dict, base_config_path: Path = None) -> Dict:
        """Create adaptive configuration based on data characteristics"""
        logger.info("⚙️  Creating adaptive configuration...")
        
        if base_config_path is None:
            base_config_path = project_root / "configs" / "default.yaml"
        
        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update data file
        config['env']['data_file'] = str(data_path.relative_to(project_root))
        
        # Auto-tune parameters if enabled
        if self.config['auto_tuning']['enabled']:
            avg_price = data_metrics['avg_price']
            
            # Base spread
            spread_ratio = self.config['auto_tuning']['spread_ratio']
            base_spread = avg_price * spread_ratio
            config['env']['base_spread'] = float(base_spread)
            
            # Initial cash
            cash_multiplier = self.config['auto_tuning']['cash_multiplier']
            initial_cash = avg_price * cash_multiplier
            config['env']['initial_cash'] = float(initial_cash)
            
            # Reward scale
            ref_price = self.config['auto_tuning']['reward_scale_reference_price']
            scale_base = self.config['auto_tuning']['reward_scale_base']
            reward_scale = scale_base * (ref_price / avg_price)
            
            if 'reward' not in config:
                config['reward'] = {}
            config['reward']['reward_scale'] = float(reward_scale)
            
            logger.info(f"   Auto-tuned params:")
            logger.info(f"     Base Spread: {base_spread:.4f} ({spread_ratio:.4%} of price)")
            logger.info(f"     Initial Cash: {initial_cash:.0f}")
            logger.info(f"     Reward Scale: {reward_scale:.2e}")
        
        # Validate environment config
        is_healthy, warnings = self.env_checker.validate(config, data_metrics)
        if not is_healthy:
            logger.warning("⚠️  Environment health warnings:")
            for warning in warnings:
                logger.warning(f"     {warning}")
        else:
            logger.info("✅ Environment configuration healthy")
        
        return config
    
    def train_model(self, config_path: Path, run_dir: Path, seed: int) -> bool:
        """Execute training"""
        logger.info("🏋️  Starting training...")
        
        train_cmd = [
            sys.executable, str(project_root / "scripts" / "train.py"),
            "--config", str(config_path),
            "--output_dir", str(run_dir),
            "--seed", str(seed),
            "--total_timesteps", str(self.config['pipeline']['total_timesteps'])
        ]
        
        try:
            timeout = self.config['pipeline']['training_timeout']
            subprocess.run(train_cmd, check=True, timeout=timeout)
            logger.info("✅ Training completed successfully")
            return True
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Training timeout after {timeout}s")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Training failed: {e}")
            return False
    
    def evaluate_model(self, run_dir: Path) -> Optional[Dict]:
        """Execute evaluation"""
        logger.info("🔬 Evaluating model...")
        
        eval_cmd = [
            sys.executable, str(project_root / "scripts" / "evaluate.py"),
            "--run_folder", str(run_dir),
            "--episodes", str(self.config['pipeline']['eval_episodes'])
        ]
        
        try:
            timeout = self.config['pipeline']['evaluation_timeout']
            subprocess.run(eval_cmd, check=True, timeout=timeout)
            
            # Load results
            results_path = run_dir / "evaluation_results.json"
            if results_path.exists():
                with open(results_path, 'r') as f:
                    results = json.load(f)
                logger.info("✅ Evaluation completed")
                return results
            else:
                logger.error("❌ Evaluation results not found")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Evaluation timeout")
            return None
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Evaluation failed: {e}")
            return None
    
    def validate_results(self, results: Dict) -> Tuple[bool, float, str]:
        """Validate training results"""
        logger.info("📊 Validating training results...")
        
        is_acceptable, score, reason = self.result_validator.validate(results)
        
        pnl = results.get('mean_pnl', 0)
        win_rate = results.get('win_rate', 0)
        sharpe = results.get('sharpe_ratio', 'N/A')
        
        logger.info(f"   PnL: ${pnl:.2f}, Win Rate: {win_rate:.2%}, Sharpe: {sharpe}")
        logger.info(f"   Composite Score: {score:.1f}/100")
        
        if is_acceptable:
            logger.info(f"✅ Results acceptable: {reason}")
        else:
            logger.warning(f"⚠️  Results not acceptable: {reason}")
        
        return is_acceptable, score, reason
    
    def check_production_readiness(self, results: Dict, episode_pnls: List[float] = None) -> bool:
        """Check if model is production ready"""
        logger.info("🔒 Checking production readiness...")
        
        is_ready, warnings, details = self.prod_checker.check(results, episode_pnls)
        
        report = self.prod_checker.generate_report(is_ready, warnings, details)
        logger.info("\n" + report)
        
        return is_ready
    
    def save_best_model(self, symbol: str, run_dir: Path, score: float, results: Dict):
        """Save best model to models directory"""
        logger.info("💾 Saving best model...")
        
        models_dir = project_root / self.config['model_saving']['models_dir']
        models_dir.mkdir(exist_ok=True)
        
        # Find source model
        source_model = run_dir / "best_model" / "best_model.zip"
        if not source_model.exists():
            source_model = run_dir / "final_model.zip"
        
        if not source_model.exists():
            logger.error("❌ Model file not found")
            return
        
        # Copy model and config
        target_model = models_dir / f"{symbol}_best_model.zip"
        target_config = models_dir / f"{symbol}_best_config.yaml"
        
        shutil.copy(source_model, target_model)
        shutil.copy(run_dir / "config.yaml", target_config)
        
        logger.info(f"✅ Saved model: {target_model}")
        
        # Update database
        if self.metrics_db:
            pnl = results.get('mean_pnl', 0)
            sharpe = results.get('sharpe_ratio', None)
            self.metrics_db.update_best_model(
                symbol=symbol,
                run_id=run_dir.name,
                model_path=str(target_model),
                score=score,
                pnl=pnl,
                sharpe=sharpe
            )
    
    def run(self, symbol: str, data_dir: str = "data") -> bool:
        """
        Execute complete automated pipeline
        
        Returns: True if successful production-ready model found
        """
        logger.info("=" * 80)
        logger.info(f"🚀 AUTOMATED PIPELINE STARTED FOR {symbol.upper()}")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Step 1: Find/fetch data
        data_path = self.find_or_fetch_data(symbol, data_dir)
        if data_path is None:
            logger.error("❌ Pipeline failed: No data available")
            return False
        
        # Step 2: Validate data quality
        is_valid, data_metrics = self.analyze_data(data_path)
        if not is_valid:
            logger.error("❌ Pipeline failed: Data quality check failed")
            return False
        
        # Step 3: Training loop with retries
        max_retries = self.config['pipeline']['max_retries']
        best_score = -float('inf')
        best_run_dir = None
        best_results = None
        production_ready_found = False
        
        for attempt in range(1, max_retries + 1):
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"🔄 ATTEMPT {attempt}/{max_retries}")
            logger.info("=" * 80)
            
            # Generate unique run ID
            seed = random.randint(1, 10000)
            run_id = f"run_{symbol}_{int(time.time())}_v{attempt}"
            run_dir = project_root / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Run ID: {run_id}")
            logger.info(f"Seed: {seed}")
            
            attempt_start = time.time()
            
            # Record in database
            if self.metrics_db:
                self.metrics_db.add_training_run({
                    'run_id': run_id,
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'attempt': attempt,
                    'seed': seed,
                    'status': 'started',
                    'data_metrics': data_metrics
                })
            
            # Create adaptive config
            config = self.create_adaptive_config(data_path, data_metrics)
            config_path = run_dir / "config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            if self.metrics_db:
                self.metrics_db.add_training_run({
                    'run_id': run_id,
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'attempt': attempt,
                    'seed': seed,
                    'status': 'training',
                    'config': config,
                    'data_metrics': data_metrics
                })
            
            # Train
            train_success = self.train_model(config_path, run_dir, seed)
            if not train_success:
                if self.metrics_db:
                    duration = time.time() - attempt_start
                    self.metrics_db.update_run_status(run_id, 'train_failed', duration)
                if self.config['pipeline']['cleanup_failed_runs']:
                    shutil.rmtree(run_dir)
                continue
            
            # Evaluate
            results = self.evaluate_model(run_dir)
            if results is None:
                if self.metrics_db:
                    duration = time.time() - attempt_start
                    self.metrics_db.update_run_status(run_id, 'eval_failed', duration)
                if self.config['pipeline']['cleanup_failed_runs']:
                    shutil.rmtree(run_dir)
                continue
            
            # Validate results
            is_acceptable, score, reason = self.validate_results(results)
            
            # Store results in database
            if self.metrics_db:
                duration = time.time() - attempt_start
                results['composite_score'] = score
                results['is_acceptable'] = is_acceptable
                results['validation_message'] = reason
                self.metrics_db.add_results(run_id, results)
                self.metrics_db.update_run_status(run_id, 'completed', duration)
            
            # Track best run
            if score > best_score:
                best_score = score
                best_run_dir = run_dir
                best_results = results
                logger.info(f"🌟 New best score: {score:.1f}")
            
            # Check if production ready
            if is_acceptable:
                episode_pnls = results.get('episode_pnls', None)
                is_prod_ready = self.check_production_readiness(results, episode_pnls)
                
                if is_prod_ready:
                    logger.info("🎉 Production-ready model found!")
                    production_ready_found = True
                    
                    if self.config['pipeline']['early_stopping']:
                        logger.info("Early stopping enabled - stopping search")
                        break
            
            # Cleanup if not best run
            if run_dir != best_run_dir and self.config['pipeline']['cleanup_failed_runs']:
                shutil.rmtree(run_dir)
        
        # Finalize
        logger.info("")
        logger.info("=" * 80)
        logger.info("📋 PIPELINE SUMMARY")
        logger.info("=" * 80)
        
        total_duration = time.time() - start_time
        logger.info(f"Total Duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
        
        if best_run_dir and best_results:
            logger.info(f"🏆 Best Run: {best_run_dir.name}")
            logger.info(f"   Score: {best_score:.1f}/100")
            logger.info(f"   PnL: ${best_results.get('mean_pnl', 0):.2f}")
            logger.info(f"   Win Rate: {best_results.get('win_rate', 0):.2%}")
            
            # Save best model
            self.save_best_model(symbol, best_run_dir, best_score, best_results)
            
            if production_ready_found:
                logger.info("")
                logger.info("✅ SUCCESS: Production-ready model created")
                logger.info("=" * 80)
                return True
            else:
                logger.warning("")
                logger.warning("⚠️  WARNING: Best model found but not production-ready")
                logger.warning("Consider running with more retries or adjusting criteria")
                logger.info("=" * 80)
                return False
        else:
            logger.error("")
            logger.error("❌ FAILED: No acceptable model found")
            logger.error("=" * 80)
            return False


def main():
    parser = argparse.ArgumentParser(description="Automated Training Pipeline")
    parser.add_argument("--symbol", type=str, required=True, help="Trading symbol (e.g., btc, eth)")
    parser.add_argument("--config", type=str, help="Pipeline config path", default=None)
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    
    args = parser.parse_args()
    
    config_path = Path(args.config) if args.config else None
    
    pipeline = AutomatedPipeline(config_path)
    success = pipeline.run(args.symbol, args.data_dir)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
