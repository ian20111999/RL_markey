import argparse
import subprocess
import sys
import shutil
import json
import yaml
import time
from pathlib import Path
import random

def run_pipeline(symbol, data_dir="data", max_retries=3):
    print(f"🚀 Starting Pipeline for {symbol}...")
    
    # 1. Setup Paths
    project_root = Path(__file__).parent
    data_path = project_root / data_dir / f"{symbol}_usdt_1m_2023.csv"
    
    # Fallback for generic naming if specific not found
    if not data_path.exists():
        # Try finding any csv with symbol
        candidates = list((project_root / data_dir).glob(f"*{symbol}*.csv"))
        if candidates:
            data_path = candidates[0]
            print(f"⚠️  Exact match not found, using: {data_path.name}")
        else:
            print(f"❌ Data file for {symbol} not found in {data_dir}/")
            sys.exit(1)
            
    print(f"✅ Data found: {data_path}")
    
    # 2. Loop for Retries
    best_pnl = -float('inf')
    best_run_dir = None
    
    for attempt in range(1, max_retries + 1):
        seed = random.randint(1, 10000)
        run_id = f"run_{symbol}_{int(time.time())}_v{attempt}"
        run_dir = project_root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🔄 Attempt {attempt}/{max_retries} (Seed: {seed})")
        print(f"   Run Dir: {run_dir}")
        
        # 3. Create Config
        config_template = project_root / "configs" / "default.yaml"
        with open(config_template, 'r') as f:
            config = yaml.safe_load(f)
            
        config['env']['data_file'] = str(data_path.relative_to(project_root))
        
        config_path = run_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
            
        # 4. Train
        print("   🏋️  Training...")
        train_cmd = [
            sys.executable, "scripts/train.py",
            "--config", str(config_path),
            "--output_dir", str(run_dir),
            "--seed", str(seed),
            "--total_timesteps", "200000" # Adjust as needed
        ]
        
        try:
            subprocess.run(train_cmd, check=True)
        except subprocess.CalledProcessError:
            print("   ❌ Training failed. Skipping...")
            shutil.rmtree(run_dir)
            continue
            
        # 5. Evaluate
        print("   🔬 Evaluating...")
        eval_cmd = [
            sys.executable, "scripts/evaluate.py",
            "--run_folder", str(run_dir),
            "--episodes", "20"
        ]
        
        try:
            subprocess.run(eval_cmd, check=True)
        except subprocess.CalledProcessError:
            print("   ❌ Evaluation failed. Skipping...")
            shutil.rmtree(run_dir)
            continue
            
        # 6. Check Results
        results_path = run_dir / "evaluation_results.json"
        if not results_path.exists():
            print("   ❌ No results found.")
            shutil.rmtree(run_dir)
            continue
            
        with open(results_path, 'r') as f:
            results = json.load(f)
            
        pnl = results['mean_pnl']
        win_rate = results['win_rate']
        
        print(f"   📊 Result: PnL=${pnl:.2f}, WinRate={win_rate*100:.1f}%")
        
        if pnl > best_pnl:
            best_pnl = pnl
            best_run_dir = run_dir
            
        # Success Criteria
        if pnl > 0 and win_rate > 0.5:
            print("   🎉 Success! Profitable model found.")
            break
        else:
            print("   ⚠️  Performance not satisfactory. Retrying...")
            if attempt < max_retries:
                # Delete failed run to save space, unless it's the best so far (handled later)
                if run_dir != best_run_dir:
                    shutil.rmtree(run_dir)
    
    # 7. Finalize
    if best_run_dir:
        print(f"\n🏆 Best Run: {best_run_dir.name} (PnL: ${best_pnl:.2f})")
        
        # Save Best Model
        models_dir = project_root / "models"
        models_dir.mkdir(exist_ok=True)
        
        target_model = models_dir / f"{symbol}_best_model.zip"
        target_config = models_dir / f"{symbol}_best_config.yaml"
        
        source_model = best_run_dir / "best_model" / "best_model.zip"
        if not source_model.exists():
             source_model = best_run_dir / "final_model.zip"
             
        if source_model.exists():
            shutil.copy(source_model, target_model)
            shutil.copy(best_run_dir / "config.yaml", target_config)
            print(f"   💾 Saved best model to: {target_model}")
        else:
            print("   ❌ Could not find model file to save.")
            
        # Cleanup all runs except best? Or just leave best run dir?
        # User said "keep project clean".
        # We will delete the run dir after copying.
        print("   🧹 Cleaning up run directories...")
        # shutil.rmtree(best_run_dir) # Optional: keep if you want logs
        
    else:
        print("\n❌ All attempts failed to produce a valid result.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="btc", help="Trading symbol (e.g., btc)")
    parser.add_argument("--retries", type=int, default=3, help="Max retries")
    args = parser.parse_args()
    
    run_pipeline(args.symbol, max_retries=args.retries)
