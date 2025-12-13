#!/usr/bin/env python3
"""
Integrated Automated Pipeline with Multi-Symbol Support
One-click solution for automated training, validation, and deployment
"""
import argparse
import sys
import threading
import queue
import time
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import yaml
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from auto_pipeline import AutomatedPipeline
from utils.metrics_db import MetricsDatabase


class IntegratedPipeline:
    """
    Integrated multi-symbol automated training pipeline
    Supports training multiple symbols in parallel or sequentially
    """
    
    def __init__(self, config_path: Path = None):
        """Initialize integrated pipeline"""
        if config_path is None:
            config_path = project_root / "configs" / "pipeline_config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.pipeline = AutomatedPipeline(config_path)
        
        # Initialize metrics database
        if self.config['metrics']['enabled']:
            db_path = project_root / self.config['metrics']['db_path']
            self.metrics_db = MetricsDatabase(db_path)
        else:
            self.metrics_db = None
        
        # Progress tracking
        self.progress_queue = queue.Queue()
        self.results = {}
    
    def run_single_symbol(self, symbol: str, data_dir: str = "data") -> Dict:
        """Run pipeline for a single symbol"""
        print(f"\n{'='*80}")
        print(f"🎯 Starting training for {symbol.upper()}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        success = self.pipeline.run(symbol, data_dir)
        duration = time.time() - start_time
        
        result = {
            'symbol': symbol,
            'success': success,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        }
        
        # Get best model info if available
        if self.metrics_db:
            best_run = self.metrics_db.get_best_run_for_symbol(symbol)
            if best_run:
                result['best_run'] = best_run
        
        return result
    
    def run_multiple_symbols(
        self, 
        symbols: List[str], 
        data_dir: str = "data",
        parallel: bool = False
    ) -> Dict[str, Dict]:
        """
        Run pipeline for multiple symbols
        
        Args:
            symbols: List of trading symbols
            data_dir: Data directory
            parallel: Run in parallel (not recommended for limited resources)
        
        Returns:
            Dictionary of results keyed by symbol
        """
        print(f"\n{'='*100}")
        print(f"🚀 INTEGRATED MULTI-SYMBOL PIPELINE")
        print(f"{'='*100}")
        print(f"\nTraining {len(symbols)} symbols: {', '.join(s.upper() for s in symbols)}")
        print(f"Mode: {'Parallel' if parallel else 'Sequential'}")
        print(f"{'='*100}\n")
        
        results = {}
        
        if parallel:
            # Parallel execution (use with caution - high resource usage)
            threads = []
            for symbol in symbols:
                thread = threading.Thread(
                    target=lambda s: results.update({s: self.run_single_symbol(s, data_dir)}),
                    args=(symbol,)
                )
                threads.append(thread)
                thread.start()
            
            # Wait for all threads
            for thread in threads:
                thread.join()
        else:
            # Sequential execution (recommended)
            for i, symbol in enumerate(symbols, 1):
                print(f"\n[{i}/{len(symbols)}] Processing {symbol.upper()}...")
                result = self.run_single_symbol(symbol, data_dir)
                results[symbol] = result
                
                # Print intermediate summary
                print(f"\n📊 Progress: {i}/{len(symbols)} completed")
                successful = sum(1 for r in results.values() if r['success'])
                print(f"   Success Rate: {successful}/{i} ({successful/i*100:.1f}%)")
        
        # Final summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict[str, Dict]):
        """Print summary of all results"""
        print(f"\n\n{'='*100}")
        print("📊 TRAINING SUMMARY")
        print(f"{'='*100}\n")
        
        successful = [r for r in results.values() if r['success']]
        failed = [r for r in results.values() if not r['success']]
        
        print(f"Total Symbols: {len(results)}")
        print(f"Successful: {len(successful)} ✅")
        print(f"Failed: {len(failed)} ❌")
        print(f"Success Rate: {len(successful)/len(results)*100:.1f}%")
        
        total_time = sum(r['duration'] for r in results.values())
        avg_time = total_time / len(results) if results else 0
        print(f"\nTotal Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Average Time per Symbol: {avg_time:.1f}s ({avg_time/60:.1f} minutes)")
        
        # Detailed results table
        print(f"\n{'-'*100}")
        print(f"{'Symbol':<12} {'Status':<15} {'Duration':<15} {'Best PnL':<15} {'Score':<10}")
        print(f"{'-'*100}")
        
        for symbol, result in results.items():
            status = "✅ Success" if result['success'] else "❌ Failed"
            duration = f"{result['duration']:.1f}s"
            
            if 'best_run' in result and result['best_run']:
                best_pnl = f"${result['best_run']['mean_pnl']:.2f}"
                score = f"{result['best_run']['composite_score']:.1f}"
            else:
                best_pnl = "N/A"
                score = "N/A"
            
            print(f"{symbol.upper():<12} {status:<15} {duration:<15} {best_pnl:<15} {score:<10}")
        
        print(f"{'-'*100}")
        
        # Production-ready models
        if successful:
            print(f"\n✅ Production-Ready Models:")
            for result in successful:
                symbol = result['symbol']
                model_path = project_root / "models" / f"{symbol}_best_model.zip"
                if model_path.exists():
                    print(f"   - {symbol.upper()}: {model_path}")
        
        print(f"\n{'='*100}\n")
    
    def auto_discover_and_train(self, data_dir: str = "data") -> Dict[str, Dict]:
        """
        Auto-discover available data and train all symbols
        """
        data_path = project_root / data_dir
        
        if not data_path.exists():
            print(f"❌ Data directory not found: {data_path}")
            return {}
        
        # Find all CSV files
        csv_files = list(data_path.glob("*.csv"))
        
        if not csv_files:
            print(f"❌ No CSV files found in {data_path}")
            return {}
        
        # Extract symbols from filenames
        symbols = []
        for csv_file in csv_files:
            # Try to extract symbol from filename
            # Common patterns: BTCUSDT_2023.csv, btc_data.csv, etc.
            name = csv_file.stem.lower()
            for common in ['usdt', 'usd', 'busd', '_2023', '_2024', '_data']:
                name = name.replace(common.lower(), '')
            name = name.strip('_')
            
            if name and len(name) <= 10:  # Reasonable symbol length
                symbols.append(name)
        
        if not symbols:
            print(f"❌ Could not extract symbols from CSV filenames")
            return {}
        
        # Remove duplicates
        symbols = list(set(symbols))
        
        print(f"🔍 Auto-discovered {len(symbols)} symbols: {', '.join(s.upper() for s in symbols)}")
        print()
        
        # Ask for confirmation
        response = input("Proceed with training all symbols? (y/N): ")
        if response.lower() != 'y':
            print("❌ Training cancelled")
            return {}
        
        return self.run_multiple_symbols(symbols, data_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Integrated Automated Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train single symbol
  python integrated_pipeline.py --symbol btc
  
  # Train multiple symbols
  python integrated_pipeline.py --symbols btc eth bnb
  
  # Auto-discover and train all available data
  python integrated_pipeline.py --auto-discover
  
  # Train with custom config
  python integrated_pipeline.py --symbols btc eth --config configs/custom.yaml
        """
    )
    
    # Symbol selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--symbol",
        type=str,
        help="Single trading symbol to train"
    )
    group.add_argument(
        "--symbols",
        type=str,
        nargs='+',
        help="Multiple trading symbols to train"
    )
    group.add_argument(
        "--auto-discover",
        action='store_true',
        help="Auto-discover and train all available data"
    )
    
    # Options
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Custom pipeline config path"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory"
    )
    
    parser.add_argument(
        "--parallel",
        action='store_true',
        help="Run multiple symbols in parallel (high resource usage)"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    config_path = Path(args.config) if args.config else None
    pipeline = IntegratedPipeline(config_path)
    
    # Run based on mode
    if args.symbol:
        # Single symbol
        result = pipeline.run_single_symbol(args.symbol, args.data_dir)
        success = result['success']
    elif args.symbols:
        # Multiple symbols
        results = pipeline.run_multiple_symbols(
            args.symbols, 
            args.data_dir,
            args.parallel
        )
        success = any(r['success'] for r in results.values())
    else:
        # Auto-discover
        results = pipeline.auto_discover_and_train(args.data_dir)
        success = any(r['success'] for r in results.values()) if results else False
    
    # Next steps
    print("\n📌 Next Steps:")
    print("   1. View training metrics: python monitoring_dashboard.py")
    print("   2. Start web dashboard: python monitoring_dashboard.py --mode server")
    print("   3. Open dashboard.html in browser (after starting server)")
    print("   4. Deploy production models from models/ directory")
    print()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
