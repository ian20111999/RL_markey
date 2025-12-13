#!/usr/bin/env python3
"""
One-Click Pipeline Launcher
Simple interface to start the automated training pipeline
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from auto_pipeline import AutomatedPipeline


def main():
    print("=" * 80)
    print("   🚀 RL Market Making - Automated Training Pipeline")
    print("=" * 80)
    print()
    
    parser = argparse.ArgumentParser(
        description="One-click automated training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_pipeline.py --symbol btc
  python start_pipeline.py --symbol eth --retries 10
  python start_pipeline.py --symbol bnb --config configs/custom_pipeline.yaml
        """
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Trading symbol (e.g., btc, eth, bnb)"
    )
    
    parser.add_argument(
        "--retries",
        type=int,
        default=None,
        help="Max retry attempts (overrides config)"
    )
    
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
    
    args = parser.parse_args()
    
    # Initialize pipeline
    config_path = Path(args.config) if args.config else None
    pipeline = AutomatedPipeline(config_path)
    
    # Override retries if specified
    if args.retries:
        pipeline.config['pipeline']['max_retries'] = args.retries
        print(f"🔧 Max retries set to: {args.retries}")
        print()
    
    # Run pipeline
    print(f"Starting automated pipeline for {args.symbol.upper()}...")
    print("This will automatically:")
    print("  ✓ Validate data quality")
    print("  ✓ Create optimal environment configuration")
    print("  ✓ Train models with auto-retry")
    print("  ✓ Evaluate and validate results")
    print("  ✓ Check production readiness")
    print("  ✓ Save best model for deployment")
    print()
    print("Please wait... this may take a while.")
    print("=" * 80)
    print()
    
    success = pipeline.run(args.symbol, args.data_dir)
    
    print()
    if success:
        print("🎉 Pipeline completed successfully!")
        print(f"Production-ready model saved in models/{args.symbol}_best_model.zip")
        print()
        print("Next steps:")
        print(f"  1. Review model at: models/{args.symbol}_best_model.zip")
        print("  2. Check metrics at: logs/metrics.db or logs/metrics.json")
        print("  3. Start monitoring dashboard: python monitoring_dashboard.py")
        print("  4. Deploy to production environment")
        sys.exit(0)
    else:
        print("⚠️  Pipeline completed but no production-ready model found.")
        print("Consider:")
        print("  1. Running again with more retries: --retries 10")
        print("  2. Checking data quality")
        print("  3. Adjusting acceptance criteria in configs/pipeline_config.yaml")
        print("  4. Reviewing logs in logs/pipeline/")
        sys.exit(1)


if __name__ == "__main__":
    main()
