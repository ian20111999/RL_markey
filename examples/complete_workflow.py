#!/usr/bin/env python3
"""
Complete Example: Train and Deploy a Profitable Model

This script demonstrates the full workflow:
1. Train a model automatically
2. Validate it meets profitability criteria
3. Register it in the model registry
4. Deploy via API
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from production.cli import ProductionCLI

def main():
    print("=" * 80)
    print("🎯 COMPLETE PRODUCTION WORKFLOW EXAMPLE")
    print("=" * 80)
    print()
    print("This example will:")
    print("  1. Train a model with automatic retries")
    print("  2. Validate profitability criteria")
    print("  3. Register the best model")
    print("  4. Show how to use it via API")
    print()
    print("=" * 80)
    print()
    
    # Initialize CLI
    cli = ProductionCLI()
    
    # Step 1: Train a profitable model
    print("📍 STEP 1: Training Model")
    print("-" * 80)
    print("Training with automatic retry until profitable...")
    print()
    
    model_id = cli.train_profitable_model(
        symbol="btc",
        algorithm="SAC",
        max_attempts=3,
        timesteps=200000,
        config="configs/default.yaml"
    )
    
    if not model_id:
        print("\n❌ Failed to train a profitable model.")
        print("   Try increasing --attempts or --timesteps")
        return
    
    print()
    print("=" * 80)
    
    # Step 2: Show model details
    print("📍 STEP 2: Model Details")
    print("-" * 80)
    print()
    
    metadata = cli.registry.get_model_metadata(model_id)
    if metadata:
        print(f"✅ Model Registered Successfully!")
        print()
        print(f"Model ID:            {metadata.model_id}")
        print(f"Algorithm:           {metadata.algorithm}")
        print(f"Production Ready:    {'✅ Yes' if metadata.production_ready else '❌ No'}")
        print()
        print("Performance Metrics:")
        print(f"  Profitability Score: {metadata.metrics.profitability_score:.1f}/100")
        print(f"  Mean PnL:            {metadata.metrics.mean_pnl:+.2f}")
        print(f"  Win Rate:            {metadata.metrics.win_rate*100:.1f}%")
        print(f"  Sharpe Ratio:        {metadata.metrics.sharpe_ratio:.2f}")
    
    print()
    print("=" * 80)
    
    # Step 3: Show API usage
    print("📍 STEP 3: API Deployment")
    print("-" * 80)
    print()
    print("To deploy this model via API:")
    print()
    print("1. Start the API server:")
    print("   python production/cli.py serve --port 8000")
    print()
    print("2. Make predictions:")
    print("   curl -X POST http://localhost:8000/predict \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"observation\": [0.5, 0.2, ...], \"deterministic\": true}'")
    print()
    print("3. Or use the best model automatically:")
    print("   curl http://localhost:8000/models/best/current")
    print()
    print("4. View all models:")
    print("   curl http://localhost:8000/models")
    print()
    
    print("=" * 80)
    
    # Step 4: Show dashboard
    print("📍 STEP 4: Monitoring Dashboard")
    print("-" * 80)
    print()
    print("To monitor models in a web interface:")
    print()
    print("1. Start the dashboard:")
    print("   python production/dashboard.py")
    print()
    print("2. Open browser:")
    print("   http://localhost:8080")
    print()
    
    print("=" * 80)
    print()
    print("✅ WORKFLOW COMPLETE!")
    print()
    print("Your model is now:")
    print("  ✅ Trained and validated")
    print("  ✅ Registered in the model registry")
    print("  ✅ Ready for production deployment")
    print()
    print("Next steps:")
    print("  • Deploy via Docker: docker-compose up -d")
    print("  • Start API: python production/cli.py serve")
    print("  • View docs: See docs/PRODUCTION_GUIDE.md")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
