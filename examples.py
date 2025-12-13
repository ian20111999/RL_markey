#!/usr/bin/env python3
"""
Example Usage of Integrated Pipeline
Demonstrates various usage scenarios
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from integrated_pipeline import IntegratedPipeline


def example_1_single_symbol():
    """Example 1: Train a single symbol with custom retries"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Symbol Training")
    print("="*80 + "\n")
    
    pipeline = IntegratedPipeline()
    
    # Train BTC
    result = pipeline.run_single_symbol('btc', data_dir='data')
    
    if result['success']:
        print(f"\n✅ Success! BTC model trained in {result['duration']:.1f}s")
        if 'best_run' in result:
            print(f"   PnL: ${result['best_run']['mean_pnl']:.2f}")
            print(f"   Score: {result['best_run']['composite_score']:.1f}")
    else:
        print(f"\n❌ Training failed after {result['duration']:.1f}s")


def example_2_multiple_symbols():
    """Example 2: Train multiple symbols sequentially"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Multiple Symbols Training")
    print("="*80 + "\n")
    
    pipeline = IntegratedPipeline()
    
    # Train multiple symbols
    symbols = ['btc', 'eth', 'bnb']
    results = pipeline.run_multiple_symbols(symbols, data_dir='data', parallel=False)
    
    # Print summary
    print(f"\n📊 Training Summary:")
    for symbol, result in results.items():
        status = "✅" if result['success'] else "❌"
        print(f"   {status} {symbol.upper()}: {result['duration']:.1f}s")


def example_3_custom_config():
    """Example 3: Use custom configuration"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Custom Configuration")
    print("="*80 + "\n")
    
    # Create custom config path
    custom_config = project_root / "configs" / "pipeline_config.yaml"
    
    # Initialize with custom config
    pipeline = IntegratedPipeline(config_path=custom_config)
    
    # Override max retries
    pipeline.config['pipeline']['max_retries'] = 10
    print(f"Using custom config with {pipeline.config['pipeline']['max_retries']} max retries")
    
    # Train
    result = pipeline.run_single_symbol('btc', data_dir='data')
    print(f"\nResult: {'Success' if result['success'] else 'Failed'}")


def example_4_auto_discover():
    """Example 4: Auto-discover and train all available data"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Auto-Discover Training")
    print("="*80 + "\n")
    
    pipeline = IntegratedPipeline()
    
    # This will scan data/ directory and train all found symbols
    results = pipeline.auto_discover_and_train(data_dir='data')
    
    if results:
        print(f"\n✅ Trained {len(results)} symbols")
    else:
        print("\n❌ No data found or training cancelled")


def example_5_monitoring():
    """Example 5: Access training metrics programmatically"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Monitoring Metrics")
    print("="*80 + "\n")
    
    from utils.metrics_db import MetricsDatabase
    
    db_path = project_root / "logs" / "metrics.db"
    
    if not db_path.exists():
        print("⚠️  No metrics database found. Run training first.")
        return
    
    db = MetricsDatabase(db_path)
    
    # Get recent runs
    recent_runs = db.get_recent_runs(limit=5)
    
    print("Recent Training Runs:")
    print("-" * 80)
    for run in recent_runs:
        print(f"Symbol: {run['symbol']}")
        print(f"Status: {run['status']}")
        if run.get('mean_pnl') is not None:
            print(f"PnL: ${run['mean_pnl']:.2f}")
        print("-" * 80)
    
    # Get best runs per symbol
    print("\nBest Models:")
    print("-" * 80)
    symbols = set(r['symbol'] for r in recent_runs)
    for symbol in symbols:
        best = db.get_best_run_for_symbol(symbol)
        if best:
            print(f"{symbol.upper()}: PnL=${best['mean_pnl']:.2f}, Score={best['composite_score']:.1f}")


def main():
    """Run examples based on user input"""
    print("\n" + "="*80)
    print("  🎓 RL Market Making - Usage Examples")
    print("="*80 + "\n")
    
    examples = {
        '1': ('Single Symbol Training', example_1_single_symbol),
        '2': ('Multiple Symbols Training', example_2_multiple_symbols),
        '3': ('Custom Configuration', example_3_custom_config),
        '4': ('Auto-Discover Training', example_4_auto_discover),
        '5': ('View Metrics', example_5_monitoring),
    }
    
    print("Available Examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}) {name}")
    print()
    
    choice = input("Select example (1-5) or 'all' to run all: ").strip()
    
    if choice.lower() == 'all':
        for name, func in examples.values():
            try:
                func()
            except Exception as e:
                print(f"\n❌ Error in {name}: {e}")
    elif choice in examples:
        name, func = examples[choice]
        try:
            func()
        except Exception as e:
            print(f"\n❌ Error: {e}")
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()
