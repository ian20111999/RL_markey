import argparse
import subprocess
import sys
import shutil
import json
import yaml
import time
import random
import pandas as pd
from pathlib import Path

def analyze_data_and_get_params(data_path):
    """
    智能分析數據，回傳適合該幣種的參數
    """
    print(f"   🔍 Analyzing data: {data_path.name}...")
    try:
        # 只讀前 10000 行來估算價格，節省時間
        df = pd.read_csv(data_path, nrows=10000)
        
        # 假設 CSV 有 'close' 或 'Close' 欄位
        price_col = 'close' if 'close' in df.columns else 'Close'
        if price_col not in df.columns:
            # 嘗試用第 4 欄 (通常是 Close)
            avg_price = df.iloc[:, 4].mean()
        else:
            avg_price = df[price_col].mean()
            
        print(f"   💡 Average Price: {avg_price:.2f}")
        
        # === 自動參數計算邏輯 ===
        
        # 1. Base Spread: 設定為價格的 0.05% (5 bps)
        # BTC(100k) -> 50, ETH(3k) -> 1.5
        base_spread = avg_price * 0.0005
        
        # 2. Initial Cash: 足夠買 10 顆的資金
        initial_cash = avg_price * 10.0
        
        # 3. Reward Scale: 根據價格動態調整，目標讓 PnL 獎勵落在合理範圍
        # 價格越高，PnL 數字越大，所以 scale 要越小
        # 基準：BTC(100k) 用 1e-6
        reward_scale = 1.0e-6 * (100000.0 / avg_price)
        
        params = {
            'base_spread': float(base_spread),
            'initial_cash': float(initial_cash),
            'reward_scale': float(reward_scale)
        }
        
        print(f"   ⚙️  Auto-Tuned Params: Spread={base_spread:.4f}, Cash={initial_cash:.0f}, Scale={reward_scale:.2e}")
        return params
        
    except Exception as e:
        print(f"   ⚠️  Analysis failed: {e}. Using default params.")
        return {}

def run_pipeline(symbol, data_dir="data", max_retries=3):
    print(f"🚀 Starting Pipeline for {symbol}...")
    
    # 1. Setup Paths
    project_root = Path(__file__).parent
    
    # Check if data exists, if not, try to fetch it
    candidates = list((project_root / data_dir).glob(f"*{symbol}*.csv"))
    
    if not candidates:
        print(f"⚠️  Data file for {symbol} not found in {data_dir}/. Attempting to download...")
        
        # Construct fetch command
        # Assuming symbol is like 'btc' or 'eth', convert to 'BTCUSDT' for Binance
        binance_symbol = f"{symbol.upper()}USDT"
        fetch_cmd = [
            sys.executable, "scripts/fetch_data.py",
            "--symbol", binance_symbol,
            "--interval", "1m",
            "--year", "2023",
            "--output_dir", str(project_root / data_dir)
        ]
        
        try:
            subprocess.run(fetch_cmd, check=True)
            # Re-check for file
            candidates = list((project_root / data_dir).glob(f"*{symbol}*.csv"))
            if not candidates:
                print(f"❌ Failed to download data for {symbol}.")
                sys.exit(1)
        except subprocess.CalledProcessError:
            print(f"❌ Error executing data fetch script.")
            sys.exit(1)
            
    data_path = candidates[0]
    print(f"✅ Data found: {data_path.name}")
    
    # === 新增：智能分析數據 ===
    auto_params = analyze_data_and_get_params(data_path)
    
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
            
        # === 注入動態參數 ===
        config['env']['data_file'] = str(data_path.relative_to(project_root))
        
        if auto_params:
            config['env']['base_spread'] = auto_params['base_spread']
            config['env']['initial_cash'] = auto_params['initial_cash']
            # 如果 config 裡有 reward 設定，也更新它
            if 'reward' in config:
                config['reward']['reward_scale'] = auto_params['reward_scale']
        
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
