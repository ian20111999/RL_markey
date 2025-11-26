# =============================================================================
# 自動化報告生成器
# 生成 HTML/PDF 訓練與評估報告
# =============================================================================

import os
import json
import base64
from io import BytesIO
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import logging

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from jinja2 import Template, Environment, FileSystemLoader
    HAS_JINJA = True
except ImportError:
    HAS_JINJA = False

logger = logging.getLogger(__name__)


# =============================================================================
# 資料類別
# =============================================================================

@dataclass
class ReportSection:
    """報告區段"""
    title: str
    content: str
    order: int = 0
    section_type: str = "text"  # text, table, figure, metrics


@dataclass
class ReportFigure:
    """報告圖表"""
    title: str
    figure_data: str  # Base64 encoded
    caption: Optional[str] = None
    width: str = "100%"


@dataclass
class ReportMetric:
    """報告指標"""
    name: str
    value: Union[float, int, str]
    unit: Optional[str] = None
    description: Optional[str] = None
    is_better_higher: bool = True  # 用於顯示顏色


@dataclass
class ReportConfig:
    """報告配置"""
    title: str = "RL Training Report"
    author: str = "Auto-generated"
    theme: str = "light"  # light, dark
    include_code: bool = False
    language: str = "zh-TW"
    date_format: str = "%Y-%m-%d %H:%M:%S"


# =============================================================================
# 圖表生成器
# =============================================================================

class ChartGenerator:
    """
    生成各種分析圖表
    """
    
    def __init__(self, style: str = "seaborn"):
        if HAS_MATPLOTLIB:
            plt.style.use('seaborn-v0_8-darkgrid')
            self.colors = sns.color_palette("husl", 8)
    
    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """將圖表轉換為 Base64"""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    
    def create_equity_curve(
        self,
        equity_data: pd.DataFrame,
        title: str = "Equity Curve"
    ) -> ReportFigure:
        """建立權益曲線圖"""
        if not HAS_MATPLOTLIB:
            return ReportFigure(title=title, figure_data="", caption="Matplotlib not available")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if 'timestamp' in equity_data.columns:
            x = equity_data['timestamp']
        else:
            x = range(len(equity_data))
        
        ax.plot(x, equity_data['equity'], color=self.colors[0], linewidth=1.5, label='Equity')
        
        if 'drawdown' in equity_data.columns:
            ax2 = ax.twinx()
            ax2.fill_between(x, 0, equity_data['drawdown'] * 100, 
                           alpha=0.3, color='red', label='Drawdown')
            ax2.set_ylabel('Drawdown (%)', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Equity ($)')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        return ReportFigure(
            title=title,
            figure_data=self._fig_to_base64(fig),
            caption="Portfolio equity over time with drawdown"
        )
    
    def create_reward_curve(
        self,
        rewards: List[float],
        window: int = 100,
        title: str = "Training Reward"
    ) -> ReportFigure:
        """建立訓練獎勵曲線"""
        if not HAS_MATPLOTLIB:
            return ReportFigure(title=title, figure_data="", caption="Matplotlib not available")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        episodes = range(len(rewards))
        ax.plot(episodes, rewards, alpha=0.3, color=self.colors[0], label='Episode Reward')
        
        # 滑動平均
        if len(rewards) >= window:
            ma = pd.Series(rewards).rolling(window=window).mean()
            ax.plot(episodes, ma, color=self.colors[1], linewidth=2, 
                   label=f'{window}-Episode MA')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ReportFigure(
            title=title,
            figure_data=self._fig_to_base64(fig),
            caption=f"Training rewards with {window}-episode moving average"
        )
    
    def create_action_distribution(
        self,
        actions: np.ndarray,
        action_names: Optional[List[str]] = None,
        title: str = "Action Distribution"
    ) -> ReportFigure:
        """建立動作分佈圖"""
        if not HAS_MATPLOTLIB:
            return ReportFigure(title=title, figure_data="", caption="Matplotlib not available")
        
        n_actions = actions.shape[1] if len(actions.shape) > 1 else 1
        
        if n_actions == 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(actions.flatten(), bins=50, color=self.colors[0], alpha=0.7)
            ax.set_xlabel('Action Value')
            ax.set_ylabel('Frequency')
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            action_names = action_names or [f'Action {i}' for i in range(n_actions)]
            
            for i in range(min(n_actions, 4)):
                axes[i].hist(actions[:, i], bins=50, color=self.colors[i], alpha=0.7)
                axes[i].set_xlabel(action_names[i])
                axes[i].set_ylabel('Frequency')
                axes[i].set_title(action_names[i])
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return ReportFigure(
            title=title,
            figure_data=self._fig_to_base64(fig),
            caption="Distribution of actions taken by the policy"
        )
    
    def create_comparison_bar(
        self,
        data: Dict[str, float],
        title: str = "Strategy Comparison",
        ylabel: str = "Value"
    ) -> ReportFigure:
        """建立比較柱狀圖"""
        if not HAS_MATPLOTLIB:
            return ReportFigure(title=title, figure_data="", caption="Matplotlib not available")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        names = list(data.keys())
        values = list(data.values())
        colors = [self.colors[i % len(self.colors)] for i in range(len(names))]
        
        bars = ax.bar(names, values, color=colors, alpha=0.8)
        
        # 添加數值標籤
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        return ReportFigure(
            title=title,
            figure_data=self._fig_to_base64(fig),
            caption=f"Comparison of {ylabel.lower()} across strategies"
        )
    
    def create_heatmap(
        self,
        data: pd.DataFrame,
        title: str = "Correlation Matrix"
    ) -> ReportFigure:
        """建立熱力圖"""
        if not HAS_MATPLOTLIB:
            return ReportFigure(title=title, figure_data="", caption="Matplotlib not available")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(data, annot=True, fmt='.2f', cmap='RdYlGn', 
                   center=0, ax=ax, square=True)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return ReportFigure(
            title=title,
            figure_data=self._fig_to_base64(fig),
            caption="Correlation heatmap"
        )
    
    def create_multi_seed_boxplot(
        self,
        seed_results: Dict[int, List[float]],
        title: str = "Multi-Seed Performance"
    ) -> ReportFigure:
        """建立多種子結果箱形圖"""
        if not HAS_MATPLOTLIB:
            return ReportFigure(title=title, figure_data="", caption="Matplotlib not available")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data = [seed_results[seed] for seed in sorted(seed_results.keys())]
        labels = [f'Seed {seed}' for seed in sorted(seed_results.keys())]
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(self.colors[i % len(self.colors)])
            box.set_alpha(0.7)
        
        ax.set_ylabel('Reward')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        return ReportFigure(
            title=title,
            figure_data=self._fig_to_base64(fig),
            caption="Performance distribution across different random seeds"
        )


# =============================================================================
# 報告生成器
# =============================================================================

class ReportGenerator:
    """
    自動化報告生成器
    """
    
    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
        self.chart_gen = ChartGenerator()
        self.sections: List[ReportSection] = []
        self.figures: List[ReportFigure] = []
        self.metrics: List[ReportMetric] = []
        self.tables: Dict[str, pd.DataFrame] = {}
    
    def add_section(self, title: str, content: str, order: int = None):
        """添加文字區段"""
        if order is None:
            order = len(self.sections)
        self.sections.append(ReportSection(title=title, content=content, order=order))
    
    def add_figure(self, figure: ReportFigure):
        """添加圖表"""
        self.figures.append(figure)
    
    def add_metric(
        self,
        name: str,
        value: Union[float, int, str],
        unit: str = None,
        description: str = None,
        is_better_higher: bool = True
    ):
        """添加指標"""
        self.metrics.append(ReportMetric(
            name=name, value=value, unit=unit,
            description=description, is_better_higher=is_better_higher
        ))
    
    def add_table(self, name: str, df: pd.DataFrame):
        """添加表格"""
        self.tables[name] = df
    
    def add_training_summary(
        self,
        train_log_path: str,
        config: Dict = None
    ):
        """從訓練日誌添加摘要"""
        if os.path.exists(train_log_path):
            train_log = pd.read_csv(train_log_path)
            
            # 添加訓練獎勵曲線
            if 'episode_reward' in train_log.columns:
                fig = self.chart_gen.create_reward_curve(
                    train_log['episode_reward'].tolist()
                )
                self.add_figure(fig)
            
            # 添加訓練指標
            if 'episode_reward' in train_log.columns:
                self.add_metric("Final Reward", train_log['episode_reward'].iloc[-1], 
                              description="Last episode reward")
                self.add_metric("Best Reward", train_log['episode_reward'].max(),
                              description="Maximum episode reward")
                self.add_metric("Total Episodes", len(train_log),
                              description="Total training episodes")
        
        if config:
            self.add_section(
                "Training Configuration",
                f"```json\n{json.dumps(config, indent=2)}\n```",
                order=1
            )
    
    def add_evaluation_results(
        self,
        eval_results: Dict,
        baseline_results: Dict = None
    ):
        """添加評估結果"""
        # 主要指標
        for key, value in eval_results.items():
            if isinstance(value, (int, float)):
                self.add_metric(key, value)
        
        # 策略比較
        if baseline_results:
            comparison_data = {
                'RL Policy': eval_results.get('mean_reward', 0),
                'Baseline': baseline_results.get('mean_reward', 0)
            }
            
            if 'random' in baseline_results:
                comparison_data['Random'] = baseline_results['random'].get('mean_reward', 0)
            
            fig = self.chart_gen.create_comparison_bar(
                comparison_data,
                title="Strategy Comparison",
                ylabel="Mean Reward"
            )
            self.add_figure(fig)
    
    def add_equity_curve(
        self,
        equity_path: str,
        title: str = "Equity Curve"
    ):
        """添加權益曲線"""
        if os.path.exists(equity_path):
            equity_data = pd.read_csv(equity_path)
            fig = self.chart_gen.create_equity_curve(equity_data, title)
            self.add_figure(fig)
    
    def add_multi_seed_results(
        self,
        seed_results: Dict[int, Dict]
    ):
        """添加多種子驗證結果"""
        # 提取獎勵
        seed_rewards = {
            seed: [result.get('mean_reward', 0)]
            for seed, result in seed_results.items()
        }
        
        fig = self.chart_gen.create_multi_seed_boxplot(seed_rewards)
        self.add_figure(fig)
        
        # 聚合指標
        all_rewards = [r.get('mean_reward', 0) for r in seed_results.values()]
        self.add_metric("Mean Reward (all seeds)", np.mean(all_rewards))
        self.add_metric("Std Reward (all seeds)", np.std(all_rewards))
        self.add_metric("CV (Coefficient of Variation)", 
                       np.std(all_rewards) / (np.mean(all_rewards) + 1e-8),
                       is_better_higher=False)
    
    def _generate_html(self) -> str:
        """生成 HTML 報告"""
        # 基本 HTML 模板
        template = """
<!DOCTYPE html>
<html lang="{{ language }}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        :root {
            --bg-color: {{ '#ffffff' if theme == 'light' else '#1a1a2e' }};
            --text-color: {{ '#333333' if theme == 'light' else '#eaeaea' }};
            --card-bg: {{ '#f8f9fa' if theme == 'light' else '#16213e' }};
            --border-color: {{ '#dee2e6' if theme == 'light' else '#0f3460' }};
            --accent-color: #4CAF50;
            --warning-color: #ff9800;
            --danger-color: #f44336;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            padding: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            padding: 40px 20px;
            border-bottom: 2px solid var(--border-color);
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header .meta {
            color: #666;
            font-size: 0.9em;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .metric-card {
            background: var(--card-bg);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            border: 1px solid var(--border-color);
            transition: transform 0.2s;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
        }
        
        .metric-card .value {
            font-size: 2em;
            font-weight: bold;
            color: var(--accent-color);
        }
        
        .metric-card .name {
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
        }
        
        .metric-card .unit {
            font-size: 0.8em;
            color: #999;
        }
        
        .section {
            margin-bottom: 40px;
        }
        
        .section h2 {
            font-size: 1.5em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--accent-color);
        }
        
        .figure {
            background: var(--card-bg);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
            border: 1px solid var(--border-color);
        }
        
        .figure img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }
        
        .figure .caption {
            text-align: center;
            font-size: 0.9em;
            color: #666;
            margin-top: 10px;
        }
        
        .figure h3 {
            margin-bottom: 15px;
            text-align: center;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        
        table th, table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        
        table th {
            background: var(--card-bg);
            font-weight: bold;
        }
        
        table tr:hover {
            background: var(--card-bg);
        }
        
        pre {
            background: {{ '#f4f4f4' if theme == 'light' else '#0a0a1a' }};
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-size: 0.9em;
        }
        
        code {
            font-family: 'Monaco', 'Consolas', monospace;
        }
        
        .footer {
            text-align: center;
            padding: 20px;
            margin-top: 40px;
            border-top: 1px solid var(--border-color);
            color: #666;
            font-size: 0.9em;
        }
        
        @media print {
            body {
                padding: 0;
            }
            .metric-card:hover {
                transform: none;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <div class="meta">
            <p>Generated: {{ generated_time }}</p>
            <p>Author: {{ author }}</p>
        </div>
    </div>
    
    {% if metrics %}
    <div class="section">
        <h2>📊 Key Metrics</h2>
        <div class="metrics-grid">
            {% for metric in metrics %}
            <div class="metric-card">
                <div class="value">
                    {% if metric.value is number %}
                        {{ "%.4f"|format(metric.value) if metric.value < 1 and metric.value > -1 else "%.2f"|format(metric.value) }}
                    {% else %}
                        {{ metric.value }}
                    {% endif %}
                    {% if metric.unit %}<span class="unit">{{ metric.unit }}</span>{% endif %}
                </div>
                <div class="name">{{ metric.name }}</div>
                {% if metric.description %}<div class="unit">{{ metric.description }}</div>{% endif %}
            </div>
            {% endfor %}
        </div>
    </div>
    {% endif %}
    
    {% for section in sections %}
    <div class="section">
        <h2>{{ section.title }}</h2>
        <div>{{ section.content | safe }}</div>
    </div>
    {% endfor %}
    
    {% if figures %}
    <div class="section">
        <h2>📈 Charts & Visualizations</h2>
        {% for figure in figures %}
        <div class="figure">
            <h3>{{ figure.title }}</h3>
            {% if figure.figure_data %}
            <img src="data:image/png;base64,{{ figure.figure_data }}" alt="{{ figure.title }}" style="width: {{ figure.width }}">
            {% endif %}
            {% if figure.caption %}<p class="caption">{{ figure.caption }}</p>{% endif %}
        </div>
        {% endfor %}
    </div>
    {% endif %}
    
    {% for table_name, table_html in tables.items() %}
    <div class="section">
        <h2>📋 {{ table_name }}</h2>
        {{ table_html | safe }}
    </div>
    {% endfor %}
    
    <div class="footer">
        <p>Generated by RL Market Making Pipeline | {{ generated_time }}</p>
    </div>
</body>
</html>
        """
        
        if HAS_JINJA:
            tmpl = Template(template)
        else:
            # 簡單的字串替換
            return template.replace("{{ title }}", self.config.title)
        
        # 準備表格 HTML
        tables_html = {}
        for name, df in self.tables.items():
            tables_html[name] = df.to_html(classes='dataframe', index=True)
        
        # 渲染模板
        html = tmpl.render(
            title=self.config.title,
            author=self.config.author,
            language=self.config.language,
            theme=self.config.theme,
            generated_time=datetime.now().strftime(self.config.date_format),
            metrics=[m.__dict__ for m in self.metrics],
            sections=[s.__dict__ for s in sorted(self.sections, key=lambda x: x.order)],
            figures=[f.__dict__ for f in self.figures],
            tables=tables_html
        )
        
        return html
    
    def generate(
        self,
        output_path: str,
        format: str = "html"
    ) -> str:
        """
        生成報告
        
        Args:
            output_path: 輸出路徑
            format: 輸出格式 ('html' 或 'pdf')
        
        Returns:
            輸出文件路徑
        """
        html_content = self._generate_html()
        
        if format.lower() == "html":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            return output_path
        
        elif format.lower() == "pdf":
            # 嘗試使用 weasyprint 或 pdfkit
            try:
                from weasyprint import HTML
                html_path = output_path.replace('.pdf', '.html')
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                HTML(html_path).write_pdf(output_path)
                return output_path
            except ImportError:
                try:
                    import pdfkit
                    pdfkit.from_string(html_content, output_path)
                    return output_path
                except ImportError:
                    logger.warning("PDF generation requires weasyprint or pdfkit. Saving as HTML.")
                    html_output = output_path.replace('.pdf', '.html')
                    with open(html_output, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    return html_output
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def clear(self):
        """清除所有內容"""
        self.sections = []
        self.figures = []
        self.metrics = []
        self.tables = {}


# =============================================================================
# 完整報告生成函數
# =============================================================================

def generate_training_report(
    run_dir: str,
    output_path: Optional[str] = None,
    title: str = "Training Report"
) -> str:
    """
    從訓練目錄生成完整報告
    
    Args:
        run_dir: 訓練運行目錄
        output_path: 輸出路徑（預設為 run_dir/report.html）
        title: 報告標題
    
    Returns:
        報告文件路徑
    """
    run_dir = Path(run_dir)
    output_path = output_path or str(run_dir / "report.html")
    
    generator = ReportGenerator(ReportConfig(title=title))
    
    # 載入配置
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        generator.add_section(
            "Configuration",
            f"<pre><code>{json.dumps(config, indent=2)}</code></pre>",
            order=0
        )
    
    # 載入訓練日誌
    train_log_path = run_dir / "train_log.csv"
    if train_log_path.exists():
        generator.add_training_summary(str(train_log_path))
    
    # 載入評估結果
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        generator.add_evaluation_results(metrics)
    
    # 載入權益曲線
    equity_path = run_dir / "test_equity_curve.csv"
    if equity_path.exists():
        generator.add_equity_curve(str(equity_path))
    
    # 生成報告
    return generator.generate(output_path)


def generate_comparison_report(
    run_dirs: List[str],
    output_path: str,
    title: str = "Experiment Comparison Report"
) -> str:
    """
    生成多個實驗的比較報告
    
    Args:
        run_dirs: 運行目錄列表
        output_path: 輸出路徑
        title: 報告標題
    
    Returns:
        報告文件路徑
    """
    generator = ReportGenerator(ReportConfig(title=title))
    
    # 收集所有實驗的指標
    all_metrics = {}
    
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        run_name = run_dir.name
        
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                all_metrics[run_name] = json.load(f)
    
    if all_metrics:
        # 建立比較表格
        comparison_df = pd.DataFrame(all_metrics).T
        generator.add_table("Experiment Comparison", comparison_df)
        
        # 建立比較圖表
        for metric_name in comparison_df.columns:
            if comparison_df[metric_name].dtype in [np.float64, np.int64]:
                fig = generator.chart_gen.create_comparison_bar(
                    comparison_df[metric_name].to_dict(),
                    title=f"{metric_name} Comparison",
                    ylabel=metric_name
                )
                generator.add_figure(fig)
    
    return generator.generate(output_path)


# =============================================================================
# 快速報告生成器
# =============================================================================

class QuickReportBuilder:
    """
    快速建立報告的輔助類
    使用鏈式調用方式
    """
    
    def __init__(self, title: str = "Quick Report"):
        self.generator = ReportGenerator(ReportConfig(title=title))
    
    def with_title(self, title: str) -> 'QuickReportBuilder':
        self.generator.config.title = title
        return self
    
    def with_metric(
        self,
        name: str,
        value: Union[float, int, str],
        unit: str = None
    ) -> 'QuickReportBuilder':
        self.generator.add_metric(name, value, unit)
        return self
    
    def with_equity_curve(
        self,
        data: Union[str, pd.DataFrame],
        title: str = "Equity Curve"
    ) -> 'QuickReportBuilder':
        if isinstance(data, str):
            data = pd.read_csv(data)
        fig = self.generator.chart_gen.create_equity_curve(data, title)
        self.generator.add_figure(fig)
        return self
    
    def with_rewards(
        self,
        rewards: List[float],
        title: str = "Training Rewards"
    ) -> 'QuickReportBuilder':
        fig = self.generator.chart_gen.create_reward_curve(rewards, title=title)
        self.generator.add_figure(fig)
        return self
    
    def with_comparison(
        self,
        data: Dict[str, float],
        title: str = "Comparison",
        ylabel: str = "Value"
    ) -> 'QuickReportBuilder':
        fig = self.generator.chart_gen.create_comparison_bar(data, title, ylabel)
        self.generator.add_figure(fig)
        return self
    
    def with_table(
        self,
        name: str,
        data: Union[Dict, pd.DataFrame]
    ) -> 'QuickReportBuilder':
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        self.generator.add_table(name, data)
        return self
    
    def with_section(
        self,
        title: str,
        content: str
    ) -> 'QuickReportBuilder':
        self.generator.add_section(title, content)
        return self
    
    def build(self, output_path: str, format: str = "html") -> str:
        return self.generator.generate(output_path, format)


# =============================================================================
# 主程式（測試用）
# =============================================================================

if __name__ == "__main__":
    import tempfile
    
    # 測試快速報告建立
    print("Testing QuickReportBuilder...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_report.html")
        
        report = (
            QuickReportBuilder("Test Report")
            .with_metric("Sharpe Ratio", 1.85)
            .with_metric("Max Drawdown", -0.15, unit="%")
            .with_metric("Total Return", 0.45, unit="%")
            .with_comparison(
                {"Strategy A": 100, "Strategy B": 85, "Strategy C": 120},
                title="Strategy Returns",
                ylabel="Return ($)"
            )
            .with_section("Summary", "This is a test report generated automatically.")
            .build(output_path)
        )
        
        print(f"Report generated: {report}")
        print(f"File size: {os.path.getsize(report)} bytes")
    
    print("\nReport generator ready!")
