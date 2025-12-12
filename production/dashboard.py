"""
Simple Web Dashboard for RL Market Making Production System

Provides a web interface for:
- Viewing model performance
- Monitoring system health
- Managing models
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from production.model_registry import ModelRegistry

# Create dashboard app
dashboard = FastAPI(title="RL Market Making Dashboard")

# Setup templates
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))

# Initialize registry
registry = ModelRegistry()


@dashboard.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Dashboard home page"""
    
    # Get models
    models = registry.list_models()
    production_models = [m for m in models if m['production_ready']]
    best_model_id = registry.get_best_model()
    best_model = None
    
    if best_model_id:
        best_model = registry.get_model_metadata(best_model_id)
    
    # Statistics
    stats = {
        'total_models': len(models),
        'production_models': len(production_models),
        'avg_profitability': sum(m['metrics']['profitability_score'] for m in models) / len(models) if models else 0,
        'best_score': max((m['metrics']['profitability_score'] for m in models), default=0)
    }
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "models": models[:10],  # Top 10
        "best_model": best_model,
        "stats": stats
    })


# Create a simple HTML template
template_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RL Market Making Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        
        .header h1 {
            color: #333;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            color: #666;
            font-size: 1.1em;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .stat-card h3 {
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }
        
        .stat-card .value {
            color: #333;
            font-size: 2.5em;
            font-weight: bold;
        }
        
        .best-model {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        
        .best-model h2 {
            color: #333;
            margin-bottom: 20px;
        }
        
        .best-model .metric {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }
        
        .best-model .metric:last-child {
            border-bottom: none;
        }
        
        .best-model .metric .label {
            color: #666;
            font-weight: 500;
        }
        
        .best-model .metric .value {
            color: #333;
            font-weight: bold;
        }
        
        .models-table {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            overflow-x: auto;
        }
        
        .models-table h2 {
            color: #333;
            margin-bottom: 20px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        th {
            background: #f8f9fa;
            padding: 15px;
            text-align: left;
            color: #666;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }
        
        td {
            padding: 15px;
            border-bottom: 1px solid #eee;
            color: #333;
        }
        
        tr:last-child td {
            border-bottom: none;
        }
        
        tr:hover {
            background: #f8f9fa;
        }
        
        .badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
        }
        
        .badge-success {
            background: #d4edda;
            color: #155724;
        }
        
        .badge-danger {
            background: #f8d7da;
            color: #721c24;
        }
        
        .score {
            font-weight: bold;
            padding: 5px 10px;
            border-radius: 5px;
        }
        
        .score-excellent {
            background: #d4edda;
            color: #155724;
        }
        
        .score-good {
            background: #fff3cd;
            color: #856404;
        }
        
        .score-poor {
            background: #f8d7da;
            color: #721c24;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 RL Market Making Dashboard</h1>
            <p>Production Model Management & Monitoring</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <h3>Total Models</h3>
                <div class="value">{{ stats.total_models }}</div>
            </div>
            <div class="stat-card">
                <h3>Production Ready</h3>
                <div class="value">{{ stats.production_models }}</div>
            </div>
            <div class="stat-card">
                <h3>Avg Score</h3>
                <div class="value">{{ "%.1f"|format(stats.avg_profitability) }}</div>
            </div>
            <div class="stat-card">
                <h3>Best Score</h3>
                <div class="value">{{ "%.1f"|format(stats.best_score) }}</div>
            </div>
        </div>
        
        {% if best_model %}
        <div class="best-model">
            <h2>🏆 Best Model: {{ best_model.model_id }}</h2>
            <div class="metric">
                <span class="label">Algorithm</span>
                <span class="value">{{ best_model.algorithm }}</span>
            </div>
            <div class="metric">
                <span class="label">Profitability Score</span>
                <span class="value">{{ "%.1f"|format(best_model.metrics.profitability_score) }}/100</span>
            </div>
            <div class="metric">
                <span class="label">Mean PnL</span>
                <span class="value">{{ "%+.2f"|format(best_model.metrics.mean_pnl) }}</span>
            </div>
            <div class="metric">
                <span class="label">Win Rate</span>
                <span class="value">{{ "%.1f"|format(best_model.metrics.win_rate * 100) }}%</span>
            </div>
            <div class="metric">
                <span class="label">Sharpe Ratio</span>
                <span class="value">{{ "%.2f"|format(best_model.metrics.sharpe_ratio) }}</span>
            </div>
        </div>
        {% endif %}
        
        <div class="models-table">
            <h2>📋 Model Leaderboard</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Model ID</th>
                        <th>Algorithm</th>
                        <th>Score</th>
                        <th>Mean PnL</th>
                        <th>Win Rate</th>
                        <th>Production</th>
                    </tr>
                </thead>
                <tbody>
                    {% for model in models %}
                    <tr>
                        <td>{{ loop.index }}</td>
                        <td>{{ model.model_id[:20] }}...</td>
                        <td>{{ model.algorithm }}</td>
                        <td>
                            {% set score = model.metrics.profitability_score %}
                            {% if score >= 70 %}
                            <span class="score score-excellent">{{ "%.1f"|format(score) }}</span>
                            {% elif score >= 50 %}
                            <span class="score score-good">{{ "%.1f"|format(score) }}</span>
                            {% else %}
                            <span class="score score-poor">{{ "%.1f"|format(score) }}</span>
                            {% endif %}
                        </td>
                        <td>{{ "%+.2f"|format(model.metrics.mean_pnl) }}</td>
                        <td>{{ "%.1f"|format(model.metrics.win_rate * 100) }}%</td>
                        <td>
                            {% if model.production_ready %}
                            <span class="badge badge-success">✓ Ready</span>
                            {% else %}
                            <span class="badge badge-danger">✗ Not Ready</span>
                            {% endif %}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""

# Save template
(templates_dir / "dashboard.html").write_text(template_html)


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Dashboard at http://localhost:8080")
    uvicorn.run(dashboard, host="0.0.0.0", port=8080)
