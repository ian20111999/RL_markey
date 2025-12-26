#!/usr/bin/env python3
"""
Enhanced Web Dashboard API with Real-time Updates
Provides REST API for interactive web frontend
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from flask import Flask, jsonify, request, send_from_directory
    from flask_cors import CORS
except ImportError:
    print("❌ Flask not installed. Install with: pip install flask flask-cors")
    sys.exit(1)

from utils.metrics_db import MetricsDatabase


app = Flask(__name__, 
            template_folder='frontend/dashboards',
            static_folder='frontend/static')
CORS(app)

# Global database instance
db = None


@app.route('/')
def index():
    """Serve the main dashboard HTML"""
    return send_from_directory('frontend/dashboards', 'basic.html')


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0'
    })


@app.route('/api/dashboard', methods=['GET'])
def get_dashboard():
    """Get complete dashboard data"""
    try:
        # Get recent runs for all symbols
        recent_runs = db.get_recent_runs(limit=100)
        
        # Group by symbol
        symbols_data = {}
        for run in recent_runs:
            symbol = run['symbol']
            if symbol not in symbols_data:
                symbols_data[symbol] = {
                    'symbol': symbol,
                    'runs': [],
                    'best_run': None,
                    'total_runs': 0,
                    'successful_runs': 0,
                    'failed_runs': 0,
                    'in_progress': 0
                }
            
            symbols_data[symbol]['runs'].append(run)
            symbols_data[symbol]['total_runs'] += 1
            
            if run.get('status') == 'completed' and run.get('is_acceptable'):
                symbols_data[symbol]['successful_runs'] += 1
            elif run.get('status') in ['started', 'training']:
                symbols_data[symbol]['in_progress'] += 1
            elif run.get('status') in ['train_failed', 'eval_failed']:
                symbols_data[symbol]['failed_runs'] += 1
        
        # Get best run for each symbol
        for symbol, data in symbols_data.items():
            best_run = db.get_best_run_for_symbol(symbol)
            data['best_run'] = best_run
        
        # Calculate global statistics
        total_runs = len(recent_runs)
        completed_runs = [r for r in recent_runs if r.get('status') == 'completed']
        successful_runs = [r for r in completed_runs if r.get('is_acceptable')]
        
        dashboard_data = {
            'symbols': list(symbols_data.values()),
            'total_symbols': len(symbols_data),
            'total_runs': total_runs,
            'successful_runs': len(successful_runs),
            'failed_runs': total_runs - len(completed_runs),
            'success_rate': len(successful_runs) / total_runs if total_runs > 0 else 0,
            'recent_runs': recent_runs[:20],
            'generated_at': datetime.now().isoformat()
        }
        
        return jsonify(dashboard_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/symbol/<symbol>', methods=['GET'])
def get_symbol_details(symbol: str):
    """Get detailed information for a specific symbol"""
    try:
        limit = request.args.get('limit', 100, type=int)
        runs = db.get_recent_runs(symbol=symbol, limit=limit)
        best_run = db.get_best_run_for_symbol(symbol)
        
        # Calculate statistics
        total_runs = len(runs)
        completed = [r for r in runs if r.get('status') == 'completed']
        successful_runs = [r for r in completed if r.get('is_acceptable')]
        
        pnls = [r['mean_pnl'] for r in completed if r.get('mean_pnl') is not None]
        scores = [r['composite_score'] for r in completed if r.get('composite_score') is not None]
        
        details = {
            'symbol': symbol,
            'total_runs': total_runs,
            'successful_runs': len(successful_runs),
            'success_rate': len(successful_runs) / total_runs if total_runs > 0 else 0,
            'best_run': best_run,
            'recent_runs': runs,
            'statistics': {
                'total_pnls': len(pnls),
                'avg_pnl': sum(pnls) / len(pnls) if pnls else 0,
                'max_pnl': max(pnls) if pnls else 0,
                'min_pnl': min(pnls) if pnls else 0,
                'avg_score': sum(scores) / len(scores) if scores else 0,
                'max_score': max(scores) if scores else 0
            },
            'chart_data': {
                'pnl_history': [
                    {'run': i+1, 'pnl': pnl} 
                    for i, pnl in enumerate(pnls)
                ],
                'score_history': [
                    {'run': i+1, 'score': score}
                    for i, score in enumerate(scores)
                ]
            },
            'generated_at': datetime.now().isoformat()
        }
        
        return jsonify(details)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models', methods=['GET'])
def get_available_models():
    """Get list of available trained models"""
    try:
        models_dir = project_root / "models"
        
        if not models_dir.exists():
            return jsonify({'models': []})
        
        models = []
        for model_file in models_dir.glob("*_best_model.zip"):
            symbol = model_file.stem.replace('_best_model', '')
            config_file = models_dir / f"{symbol}_best_config.yaml"
            
            model_info = {
                'symbol': symbol,
                'model_path': str(model_file),
                'config_path': str(config_file) if config_file.exists() else None,
                'size': model_file.stat().st_size,
                'created_at': datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
            }
            
            # Get model metrics from database
            best_run = db.get_best_run_for_symbol(symbol)
            if best_run:
                model_info['metrics'] = {
                    'pnl': best_run.get('mean_pnl'),
                    'win_rate': best_run.get('win_rate'),
                    'sharpe': best_run.get('sharpe_ratio'),
                    'score': best_run.get('composite_score')
                }
            
            models.append(model_info)
        
        return jsonify({
            'models': models,
            'total': len(models),
            'generated_at': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_global_stats():
    """Get global statistics across all symbols"""
    try:
        all_runs = db.get_recent_runs(limit=1000)
        
        # Filter by time period if specified
        days = request.args.get('days', None, type=int)
        if days:
            from datetime import timedelta
            cutoff = datetime.now() - timedelta(days=days)
            all_runs = [r for r in all_runs if datetime.fromisoformat(r['timestamp']) > cutoff]
        
        completed = [r for r in all_runs if r.get('status') == 'completed']
        successful = [r for r in completed if r.get('is_acceptable')]
        
        # Get unique symbols
        symbols = set(r['symbol'] for r in all_runs)
        
        # Calculate PnL statistics
        pnls = [r['mean_pnl'] for r in completed if r.get('mean_pnl') is not None]
        
        stats = {
            'overview': {
                'total_symbols': len(symbols),
                'total_runs': len(all_runs),
                'completed_runs': len(completed),
                'successful_runs': len(successful),
                'success_rate': len(successful) / len(completed) if completed else 0
            },
            'pnl': {
                'total_trades': len(pnls),
                'average': sum(pnls) / len(pnls) if pnls else 0,
                'maximum': max(pnls) if pnls else 0,
                'minimum': min(pnls) if pnls else 0,
                'positive_rate': sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0
            },
            'time_period': f"Last {days} days" if days else "All time",
            'generated_at': datetime.now().isoformat()
        }
        
        return jsonify(stats)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/compare', methods=['POST'])
def compare_symbols():
    """Compare multiple symbols"""
    try:
        data = request.get_json()
        symbols = data.get('symbols', [])
        
        if not symbols:
            return jsonify({'error': 'No symbols provided'}), 400
        
        comparison = []
        for symbol in symbols:
            best_run = db.get_best_run_for_symbol(symbol)
            runs = db.get_recent_runs(symbol=symbol, limit=100)
            
            completed = [r for r in runs if r.get('status') == 'completed']
            
            comparison.append({
                'symbol': symbol,
                'best_run': best_run,
                'total_runs': len(runs),
                'completed_runs': len(completed),
                'has_model': best_run is not None
            })
        
        return jsonify({
            'comparison': comparison,
            'generated_at': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def start_server(host: str = "0.0.0.0", port: int = 5000, db_path: Path = None):
    """Start the web dashboard server"""
    global db
    
    # Initialize database
    if db_path is None:
        db_path = project_root / "logs" / "metrics.db"
    
    if not db_path.exists():
        print(f"⚠️  Warning: Metrics database not found at {db_path}")
        print("Creating new database. Run training pipeline to populate it.")
        db_path.parent.mkdir(parents=True, exist_ok=True)
    
    db = MetricsDatabase(db_path)
    
    print("=" * 80)
    print("   🚀 RL Market Making - Web Dashboard Server")
    print("=" * 80)
    print()
    print(f"   Server: http://{host}:{port}")
    print(f"   Dashboard: http://localhost:{port}/")
    print(f"   API: http://localhost:{port}/api/dashboard")
    print()
    print("   Available Endpoints:")
    print(f"   - GET  /api/health         - Health check")
    print(f"   - GET  /api/dashboard      - Dashboard data")
    print(f"   - GET  /api/symbol/<name>  - Symbol details")
    print(f"   - GET  /api/models         - Available models")
    print(f"   - GET  /api/stats          - Global statistics")
    print(f"   - POST /api/compare        - Compare symbols")
    print()
    print("   Press Ctrl+C to stop")
    print("=" * 80)
    print()
    
    app.run(host=host, port=port, debug=False, threaded=True)


def main():
    parser = argparse.ArgumentParser(description="Web Dashboard Server")
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="Server port"
    )
    
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to metrics database"
    )
    
    args = parser.parse_args()
    
    db_path = Path(args.db) if args.db else None
    start_server(args.host, args.port, db_path)


if __name__ == "__main__":
    main()
