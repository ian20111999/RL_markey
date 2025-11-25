"""analyze_experiments.py: 批次彙整 runs/ 底下的 config 與 metrics，協助挑選穩定策略。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from utils.config import load_config

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = ROOT / "runs"
PLOTS_DIR = ROOT / "plots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="整理 run 結果並做簡單比較")
    parser.add_argument("--runs_dir", type=Path, default=DEFAULT_RUNS_DIR, help="實驗輸出根目錄")
    parser.add_argument("--sort_by", type=str, default="test_mean_final_pv", help="排序欄位")
    parser.add_argument("--top_k", type=int, default=5, help="終端列印前幾名實驗")
    parser.add_argument("--plot_param", type=str, default="lambda_inv", help="要畫散佈圖的環境/訓練參數名稱")
    parser.add_argument("--plot_metric", type=str, default="test_sharpe", help="散佈圖的績效指標欄位")
    return parser.parse_args()


def find_config_file(run_dir: Path) -> Path | None:
    for name in ("config.yaml", "config.yml", "config.json"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    return None


def load_metrics(metrics_path: Path) -> Dict[str, float]:
    with metrics_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "test" in data and isinstance(data["test"], dict):
        return data["test"]
    return data


def collect_runs(runs_dir: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    if not runs_dir.exists():
        return records
    for algo_dir in runs_dir.iterdir():
        if not algo_dir.is_dir():
            continue
        for run_dir in algo_dir.iterdir():
            if not run_dir.is_dir():
                continue
            config_path = find_config_file(run_dir)
            
            # 優先讀取 evaluation/metrics.json，若無則讀取根目錄 metrics.json
            metrics_path = run_dir / "evaluation" / "metrics.json"
            if not metrics_path.exists():
                metrics_path = run_dir / "metrics.json"
            
            if not config_path or not metrics_path.exists():
                continue
            cfg = load_config(config_path)
            metrics = load_metrics(metrics_path)
            env_cfg = cfg.env
            train_cfg = cfg.train
            record = {
                "run_path": str(run_dir),
                "algo": train_cfg.get("algo", "SAC"),
                "run_name": run_dir.name,
                "fee_rate": env_cfg.get("fee_rate"),
                "lambda_inv": env_cfg.get("lambda_inv"),
                "lambda_turnover": env_cfg.get("lambda_turnover"),
                "base_spread": env_cfg.get("base_spread"),
                "max_inventory": env_cfg.get("max_inventory"),
                "learning_rate": train_cfg.get("learning_rate"),
                "gamma": train_cfg.get("gamma"),
                "batch_size": train_cfg.get("batch_size"),
                "net_arch": "-".join(map(str, train_cfg.get("net_arch", []))) if train_cfg.get("net_arch") else "",
                "seed": train_cfg.get("seed"),
                "test_mean_final_pv": metrics.get("mean_final_pv", metrics.get("final_portfolio_value")),
                "test_sharpe": metrics.get("sharpe"),
                "test_max_drawdown": metrics.get("max_drawdown"),
            }
            records.append(record)
    return records


def save_summary(df: pd.DataFrame, runs_dir: Path) -> Path:
    csv_path = runs_dir / "runs_summary.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def plot_scatter(df: pd.DataFrame, param: str, metric: str) -> Path | None:
    if param not in df.columns or metric not in df.columns:
        return None
    valid_df = df[[param, metric]].dropna()
    if valid_df.empty:
        return None
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOTS_DIR / f"analysis_{param}_vs_{metric}.png"
    plt.figure(figsize=(6, 4))
    plt.scatter(valid_df[param], valid_df[metric], alpha=0.7)
    plt.xlabel(param)
    plt.ylabel(metric)
    plt.title(f"{param} vs {metric}")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def main() -> None:
    args = parse_args()
    runs_dir = args.runs_dir if args.runs_dir.is_absolute() else ROOT / args.runs_dir
    records = collect_runs(runs_dir)
    if not records:
        print("找不到任何含 config/metrics 的 run，請先執行訓練。")
        return
    df = pd.DataFrame(records)
    csv_path = save_summary(df, runs_dir)
    sort_col = args.sort_by if args.sort_by in df.columns else "test_mean_final_pv"
    sorted_df = df.sort_values(by=sort_col, ascending=False)
    print(f"\n=== Top {args.top_k} experiments by {sort_col} ===")
    print(sorted_df.head(args.top_k)[["run_name", sort_col, "test_sharpe", "test_max_drawdown"]])
    plot_path = plot_scatter(df, args.plot_param, args.plot_metric)
    if plot_path:
        print(f"散佈圖輸出到 {plot_path}")
    print(f"完整彙整寫入 {csv_path}")


if __name__ == "__main__":
    main()
