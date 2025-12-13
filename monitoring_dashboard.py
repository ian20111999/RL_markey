#!/usr/bin/env python3
"""
Monitoring Dashboard API
Provides endpoints for tracking training progress and viewing results
Can be used as backend for web frontend
"""
import sys
from pathlib import Path
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.metrics_db import MetricsDatabase


class MonitoringDashboard:
    """Dashboard for monitoring training pipeline"""
    
    def __init__(self, db_path: Path = None):
        if db_path is None:
            db_path = project_root / "logs" / "metrics.db"
        
        if not db_path.exists():
            print(f"⚠️  Metrics database not found at {db_path}")
            print("Run training pipeline first to generate metrics.")
            sys.exit(1)
        
        self.db = MetricsDatabase(db_path)
    
    def get_dashboard_data(self) -> Dict:
        """Get all dashboard data"""
        # Get recent runs for all symbols
        recent_runs = self.db.get_recent_runs(limit=50)
        
        # Group by symbol
        symbols = {}
        for run in recent_runs:
            symbol = run['symbol']
            if symbol not in symbols:
                symbols[symbol] = {
                    'symbol': symbol,
                    'runs': [],
                    'best_run': None,
                    'total_runs': 0,
                    'successful_runs': 0
                }
            
            symbols[symbol]['runs'].append(run)
            symbols[symbol]['total_runs'] += 1
            if run.get('is_acceptable'):
                symbols[symbol]['successful_runs'] += 1
        
        # Get best run for each symbol
        for symbol_data in symbols.values():
            best_run = self.db.get_best_run_for_symbol(symbol_data['symbol'])
            symbol_data['best_run'] = best_run
        
        dashboard_data = {
            'symbols': list(symbols.values()),
            'total_symbols': len(symbols),
            'recent_runs': recent_runs[:10],
            'generated_at': datetime.now().isoformat()
        }
        
        return dashboard_data
    
    def get_symbol_details(self, symbol: str) -> Dict:
        """Get detailed information for a specific symbol"""
        runs = self.db.get_recent_runs(symbol=symbol, limit=100)
        best_run = self.db.get_best_run_for_symbol(symbol)
        
        # Calculate statistics
        total_runs = len(runs)
        successful_runs = sum(1 for r in runs if r.get('is_acceptable'))
        
        pnls = [r['mean_pnl'] for r in runs if r['mean_pnl'] is not None]
        scores = [r['composite_score'] for r in runs if r['composite_score'] is not None]
        
        details = {
            'symbol': symbol,
            'total_runs': total_runs,
            'successful_runs': successful_runs,
            'success_rate': successful_runs / total_runs if total_runs > 0 else 0,
            'best_run': best_run,
            'recent_runs': runs,
            'statistics': {
                'avg_pnl': sum(pnls) / len(pnls) if pnls else 0,
                'max_pnl': max(pnls) if pnls else 0,
                'min_pnl': min(pnls) if pnls else 0,
                'avg_score': sum(scores) / len(scores) if scores else 0,
                'max_score': max(scores) if scores else 0
            },
            'generated_at': datetime.now().isoformat()
        }
        
        return details
    
    def print_dashboard(self):
        """Print dashboard to console"""
        data = self.get_dashboard_data()
        
        print("=" * 100)
        print("   📊 TRAINING PIPELINE MONITORING DASHBOARD")
        print("=" * 100)
        print()
        
        print(f"Total Symbols Trained: {data['total_symbols']}")
        print()
        
        # Symbol summary
        print("-" * 100)
        print(f"{'Symbol':<10} {'Total Runs':<12} {'Success Rate':<15} {'Best PnL':<12} {'Best Score':<12} {'Status':<15}")
        print("-" * 100)
        
        for symbol_data in data['symbols']:
            symbol = symbol_data['symbol']
            total = symbol_data['total_runs']
            success = symbol_data['successful_runs']
            success_rate = success / total if total > 0 else 0
            
            best = symbol_data['best_run']
            if best:
                best_pnl = f"${best['mean_pnl']:.2f}"
                best_score = f"{best['composite_score']:.1f}"
                status = "✅ Ready"
            else:
                best_pnl = "N/A"
                best_score = "N/A"
                status = "❌ No Model"
            
            print(f"{symbol:<10} {total:<12} {success_rate:<15.1%} {best_pnl:<12} {best_score:<12} {status:<15}")
        
        print("-" * 100)
        print()
        
        # Recent runs
        print("Recent Runs:")
        print("-" * 100)
        print(f"{'Run ID':<30} {'Symbol':<10} {'Status':<15} {'PnL':<12} {'Score':<10}")
        print("-" * 100)
        
        for run in data['recent_runs']:
            run_id = run['run_id'][:28] + "..." if len(run['run_id']) > 28 else run['run_id']
            symbol = run['symbol']
            status = run['status']
            pnl = f"${run['mean_pnl']:.2f}" if run['mean_pnl'] is not None else "N/A"
            score = f"{run['composite_score']:.1f}" if run['composite_score'] is not None else "N/A"
            
            print(f"{run_id:<30} {symbol:<10} {status:<15} {pnl:<12} {score:<10}")
        
        print("-" * 100)
        print()
        print(f"Generated at: {data['generated_at']}")
        print("=" * 100)
    
    def export_json(self, output_path: Path):
        """Export dashboard data to JSON for web frontend"""
        data = self.get_dashboard_data()
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Dashboard data exported to {output_path}")
    
    def start_api_server(self, host: str = "0.0.0.0", port: int = 5556):
        """Start REST API server for frontend (requires Flask)"""
        try:
            from flask import Flask, jsonify, request, send_from_directory
            from flask_cors import CORS
        except ImportError:
            print("❌ Flask not installed. Install with: pip install flask flask-cors")
            sys.exit(1)
        
        app = Flask(__name__, 
                    template_folder='frontend/dashboards',
                    static_folder='frontend/static')
        CORS(app)
        
        @app.route('/')
        def index():
            """Serve the enhanced dashboard HTML"""
            return send_from_directory('frontend/dashboards', 'enhanced.html')
        
        @app.route('/api/dashboard', methods=['GET'])
        def get_dashboard():
            return jsonify(self.get_dashboard_data())
        
        @app.route('/api/symbol/<symbol>', methods=['GET'])
        def get_symbol(symbol):
            return jsonify(self.get_symbol_details(symbol))
        
        @app.route('/api/health', methods=['GET'])
        def health():
            return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})
        
        print(f"🚀 Starting monitoring API server on {host}:{port}")
        print(f"   Dashboard: http://{host}:{port}/")
        print(f"   API: http://{host}:{port}/api/dashboard")
        print(f"   Symbol details: http://{host}:{port}/api/symbol/<symbol>")
        print()
        
        app.run(host=host, port=port, debug=False)


def main():
    parser = argparse.ArgumentParser(description="Training Monitoring Dashboard")
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=['console', 'export', 'server'],
        default='console',
        help="Dashboard mode"
    )
    
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to metrics database"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="logs/dashboard.json",
        help="Output path for JSON export"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="API server host"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=5556,
        help="API server port"
    )
    
    args = parser.parse_args()
    
    db_path = Path(args.db) if args.db else None
    dashboard = MonitoringDashboard(db_path)
    
    if args.mode == 'console':
        dashboard.print_dashboard()
    elif args.mode == 'export':
        output_path = project_root / args.output
        dashboard.export_json(output_path)
    elif args.mode == 'server':
        dashboard.start_api_server(args.host, args.port)


if __name__ == "__main__":
    main()
