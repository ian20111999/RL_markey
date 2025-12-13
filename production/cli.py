#!/usr/bin/env python3
"""
Production CLI Tool for RL Market Making

A simplified, user-friendly command-line interface for:
- Training profitable models
- Validating model performance
- Deploying models to production
- Managing model registry
"""

import argparse
import sys
import subprocess
from pathlib import Path
import yaml
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from production.model_registry import ModelRegistry, ModelMetrics, calculate_profitability_score


class ProductionCLI:
    """Production-ready CLI for RL Market Making"""
    
    def __init__(self):
        self.registry = ModelRegistry()
        self.project_root = Path(__file__).parent.parent
    
    def train_profitable_model(
        self,
        symbol: str = "btc",
        algorithm: str = "SAC",
        max_attempts: int = 3,
        timesteps: int = 200000,
        config: str = "configs/default.yaml"
    ):
        """
        Train a model until it's profitable or max attempts reached
        
        Args:
            symbol: Trading symbol (btc, eth, etc.)
            algorithm: RL algorithm (SAC, PPO, TD3)
            max_attempts: Maximum training attempts
            timesteps: Training timesteps per attempt
            config: Configuration file path
        """
        print("=" * 80)
        print("🚀 PRODUCTION MODEL TRAINING")
        print("=" * 80)
        print(f"Symbol: {symbol.upper()}")
        print(f"Algorithm: {algorithm}")
        print(f"Max Attempts: {max_attempts}")
        print(f"Timesteps per Attempt: {timesteps:,}")
        print("=" * 80)
        print()
        
        best_model = None
        best_score = -float('inf')
        
        for attempt in range(1, max_attempts + 1):
            print(f"\n📍 ATTEMPT {attempt}/{max_attempts}")
            print("-" * 80)
            
            # Create unique run directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = self.project_root / "runs" / f"{algorithm.lower()}_{symbol}_{timestamp}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare config
            config_path = self.project_root / config
            run_config = run_dir / "config.yaml"
            
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            
            # Ensure data file path is correct
            data_file = self.project_root / cfg['env']['data_file']
            if not data_file.exists():
                print(f"❌ Data file not found: {data_file}")
                print("   Please download data first: python scripts/fetch_binance_ohlcv.py")
                return None
            
            with open(run_config, 'w') as f:
                yaml.dump(cfg, f)
            
            # Train
            print("🏋️  Training model...")
            train_cmd = [
                sys.executable,
                "scripts/train.py",
                "--config", str(run_config),
                "--output_dir", str(run_dir),
                "--total_timesteps", str(timesteps),
                "--seed", str(42 + attempt)
            ]
            
            try:
                result = subprocess.run(
                    train_cmd,
                    cwd=str(self.project_root),
                    capture_output=False,
                    check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"❌ Training failed: {e}")
                continue
            
            # Evaluate
            print("\n🔬 Evaluating model...")
            eval_cmd = [
                sys.executable,
                "scripts/evaluate.py",
                "--run_folder", str(run_dir),
                "--episodes", "30"
            ]
            
            try:
                subprocess.run(
                    eval_cmd,
                    cwd=str(self.project_root),
                    capture_output=False,
                    check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"❌ Evaluation failed: {e}")
                continue
            
            # Check results
            results_file = run_dir / "evaluation_results.json"
            if not results_file.exists():
                print("❌ No evaluation results found")
                continue
            
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            # Calculate comprehensive metrics
            metrics = self._calculate_metrics(results)
            score = metrics.profitability_score
            
            print("\n📊 RESULTS:")
            print(f"   Mean PnL:           {metrics.mean_pnl:+.2f}")
            print(f"   Win Rate:           {metrics.win_rate*100:.1f}%")
            print(f"   Sharpe Ratio:       {metrics.sharpe_ratio:.2f}")
            print(f"   Max Drawdown:       {metrics.max_drawdown*100:.1f}%")
            print(f"   Profitability Score: {score:.1f}/100")
            
            # Check if profitable
            is_profitable = metrics.mean_pnl > 0 and metrics.win_rate > 0.5
            
            if is_profitable:
                print("\n✅ PROFITABLE MODEL FOUND!")
            else:
                print("\n⚠️  Model not profitable enough")
            
            # Track best
            if score > best_score:
                best_score = score
                best_model = {
                    'run_dir': run_dir,
                    'metrics': metrics,
                    'algorithm': algorithm
                }
            
            # Stop if we found a good model
            if is_profitable and score >= 70:
                print("\n🎉 Excellent model achieved! Stopping early.")
                break
        
        print("\n" + "=" * 80)
        
        if best_model:
            print("🏆 BEST MODEL SUMMARY")
            print("=" * 80)
            print(f"Algorithm: {best_model['algorithm']}")
            print(f"Profitability Score: {best_model['metrics'].profitability_score:.1f}/100")
            print(f"Mean PnL: {best_model['metrics'].mean_pnl:+.2f}")
            print(f"Win Rate: {best_model['metrics'].win_rate*100:.1f}%")
            
            # Register model
            print("\n📝 Registering model...")
            model_path = best_model['run_dir'] / "best_model" / "best_model.zip"
            if not model_path.exists():
                model_path = best_model['run_dir'] / "final_model.zip"
            
            config_path = best_model['run_dir'] / "config.yaml"
            
            if model_path.exists():
                model_id = self.registry.register_model(
                    model_path=str(model_path),
                    config_path=str(config_path),
                    metrics=best_model['metrics'],
                    algorithm=best_model['algorithm'],
                    training_timesteps=timesteps,
                    data_source=str(data_file),
                    tags=["auto-trained", symbol],
                    notes=f"Auto-trained model (attempt {attempt})"
                )
                print(f"✅ Model registered with ID: {model_id}")
                return model_id
            else:
                print("❌ Model file not found")
        else:
            print("❌ No successful training run")
        
        return None
    
    def _calculate_metrics(self, results: dict) -> ModelMetrics:
        """Calculate comprehensive metrics from evaluation results"""
        mean_pnl = results.get('mean_pnl', 0)
        std_pnl = results.get('std_pnl', 0)
        win_rate = results.get('win_rate', 0)
        mean_trades = results.get('mean_trades', 0)
        mean_max_inv = results.get('mean_max_inv', 0)
        
        # Estimate Sharpe ratio (assuming daily episodes)
        sharpe_ratio = mean_pnl / std_pnl if std_pnl > 0 else 0
        
        # Estimate max drawdown (conservative)
        MAX_DRAWDOWN_ESTIMATE = 0.5
        MIN_PNL_DENOMINATOR = 1.0
        max_drawdown = min(MAX_DRAWDOWN_ESTIMATE, std_pnl / max(abs(mean_pnl), MIN_PNL_DENOMINATOR))
        
        metrics = ModelMetrics(
            mean_pnl=mean_pnl,
            std_pnl=std_pnl,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=int(mean_trades),
            mean_max_inventory=mean_max_inv,
            profitability_score=0  # Will be calculated
        )
        
        # Calculate profitability score
        metrics.profitability_score = calculate_profitability_score(metrics)
        
        return metrics
    
    def list_models(self, production_only: bool = False):
        """List all registered models"""
        print("=" * 80)
        print("📋 REGISTERED MODELS")
        print("=" * 80)
        
        models = self.registry.list_models(production_ready_only=production_only)
        
        if not models:
            print("No models found in registry")
            return
        
        print(f"{'Model ID':<30} {'Algo':<6} {'Score':>6} {'PnL':>10} {'Win%':>6} {'Prod':>5}")
        print("-" * 80)
        
        for m in models:
            model_id = m['model_id'][:28]
            algo = m['algorithm']
            score = m['metrics']['profitability_score']
            pnl = m['metrics']['mean_pnl']
            win_rate = m['metrics']['win_rate'] * 100
            prod = "✅" if m['production_ready'] else "❌"
            
            print(f"{model_id:<30} {algo:<6} {score:>6.1f} {pnl:>10.2f} {win_rate:>6.1f} {prod:>5}")
        
        print("-" * 80)
        print(f"Total: {len(models)} models")
    
    def get_best_model(self):
        """Show best model information"""
        model_id = self.registry.get_best_model()
        
        if not model_id:
            print("❌ No production-ready models available")
            return
        
        metadata = self.registry.get_model_metadata(model_id)
        
        print("=" * 80)
        print("🏆 BEST MODEL")
        print("=" * 80)
        print(f"Model ID:            {metadata.model_id}")
        print(f"Algorithm:           {metadata.algorithm}")
        print(f"Version:             {metadata.version}")
        print(f"Created:             {metadata.created_at}")
        print(f"Production Ready:    {'✅ Yes' if metadata.production_ready else '❌ No'}")
        print()
        print("PERFORMANCE METRICS:")
        print(f"  Profitability Score: {metadata.metrics.profitability_score:.1f}/100")
        print(f"  Mean PnL:            {metadata.metrics.mean_pnl:+.2f}")
        print(f"  Win Rate:            {metadata.metrics.win_rate*100:.1f}%")
        print(f"  Sharpe Ratio:        {metadata.metrics.sharpe_ratio:.2f}")
        print(f"  Max Drawdown:        {metadata.metrics.max_drawdown*100:.1f}%")
        print("=" * 80)
    
    def export_leaderboard(self):
        """Export leaderboard to CSV"""
        output_path = self.project_root / "models" / "leaderboard.csv"
        self.registry.export_leaderboard(str(output_path))
    
    def start_api_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the production API server"""
        print("=" * 80)
        print("🚀 STARTING PRODUCTION API SERVER")
        print("=" * 80)
        print(f"Host: {host}")
        print(f"Port: {port}")
        print(f"Docs: http://{host}:{port}/docs")
        print("=" * 80)
        print()
        
        cmd = [
            sys.executable,
            "-m", "uvicorn",
            "production.api:app",
            "--host", host,
            "--port", str(port),
            "--reload"
        ]
        
        subprocess.run(cmd, cwd=str(self.project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Production CLI for RL Market Making",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a profitable model
  python production/cli.py train --symbol btc --attempts 3
  
  # List all models
  python production/cli.py list
  
  # Show best model
  python production/cli.py best
  
  # Start API server
  python production/cli.py serve --port 8000
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a profitable model')
    train_parser.add_argument('--symbol', type=str, default='btc', help='Trading symbol')
    train_parser.add_argument('--algorithm', type=str, default='SAC', choices=['SAC', 'PPO', 'TD3'])
    train_parser.add_argument('--attempts', type=int, default=3, help='Max training attempts')
    train_parser.add_argument('--timesteps', type=int, default=200000, help='Training timesteps')
    train_parser.add_argument('--config', type=str, default='configs/default.yaml', help='Config file')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List registered models')
    list_parser.add_argument('--production-only', action='store_true', help='Show only production-ready models')
    
    # Best command
    subparsers.add_parser('best', help='Show best model information')
    
    # Leaderboard command
    subparsers.add_parser('leaderboard', help='Export model leaderboard')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start API server')
    serve_parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind')
    serve_parser.add_argument('--port', type=int, default=8000, help='Port to bind')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = ProductionCLI()
    
    if args.command == 'train':
        cli.train_profitable_model(
            symbol=args.symbol,
            algorithm=args.algorithm,
            max_attempts=args.attempts,
            timesteps=args.timesteps,
            config=args.config
        )
    elif args.command == 'list':
        cli.list_models(production_only=args.production_only)
    elif args.command == 'best':
        cli.get_best_model()
    elif args.command == 'leaderboard':
        cli.export_leaderboard()
    elif args.command == 'serve':
        cli.start_api_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
