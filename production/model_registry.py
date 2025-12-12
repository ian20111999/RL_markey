"""
Model Registry System for Production-Ready RL Market Making

This module provides a centralized registry for managing trained models,
their metadata, validation metrics, and versioning.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
from dataclasses import dataclass, asdict


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    mean_pnl: float
    std_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    mean_max_inventory: float
    profitability_score: float  # Composite score
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetrics':
        return cls(**data)


@dataclass
class ModelMetadata:
    """Comprehensive model metadata"""
    model_id: str
    version: str
    created_at: str
    algorithm: str
    config_path: str
    training_timesteps: int
    data_source: str
    metrics: ModelMetrics
    validation_status: str  # 'pending', 'passed', 'failed'
    production_ready: bool
    tags: List[str]
    notes: str
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['metrics'] = self.metrics.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        metrics = ModelMetrics.from_dict(data.pop('metrics'))
        return cls(metrics=metrics, **data)


class ModelRegistry:
    """
    Central registry for managing RL trading models in production
    
    Features:
    - Model versioning and metadata tracking
    - Performance metrics storage
    - Automated model validation
    - Best model selection
    - Model lifecycle management
    """
    
    def __init__(self, registry_dir: str = "models/registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_dir / "registry.json"
        self.models_dir = self.registry_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize registry
        self._load_registry()
    
    def _load_registry(self):
        """Load existing registry or create new"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                self.models = {k: ModelMetadata.from_dict(v) for k, v in data.items()}
        else:
            self.models = {}
            self._save_registry()
    
    def _save_registry(self):
        """Save registry to disk"""
        data = {k: v.to_dict() for k, v in self.models.items()}
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register_model(
        self,
        model_path: str,
        config_path: str,
        metrics: ModelMetrics,
        algorithm: str = "SAC",
        training_timesteps: int = 0,
        data_source: str = "",
        tags: List[str] = None,
        notes: str = ""
    ) -> str:
        """
        Register a new model in the registry
        
        Args:
            model_path: Path to the trained model file (.zip)
            config_path: Path to the configuration file
            metrics: Model performance metrics
            algorithm: Algorithm used (SAC, PPO, TD3)
            training_timesteps: Number of training timesteps
            data_source: Source of training data
            tags: Optional tags for categorization
            notes: Additional notes
            
        Returns:
            model_id: Unique identifier for the registered model
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"{algorithm.lower()}_{timestamp}"
        version = "v1.0.0"
        
        # Check if model passes production criteria
        production_ready = self._validate_production_criteria(metrics)
        validation_status = "passed" if production_ready else "failed"
        
        # Copy model and config to registry
        model_path = Path(model_path)
        config_path = Path(config_path)
        
        model_dir = self.models_dir / model_id
        model_dir.mkdir(exist_ok=True)
        
        shutil.copy(model_path, model_dir / "model.zip")
        shutil.copy(config_path, model_dir / "config.yaml")
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            version=version,
            created_at=datetime.now().isoformat(),
            algorithm=algorithm,
            config_path=str(config_path),
            training_timesteps=training_timesteps,
            data_source=data_source,
            metrics=metrics,
            validation_status=validation_status,
            production_ready=production_ready,
            tags=tags or [],
            notes=notes
        )
        
        self.models[model_id] = metadata
        self._save_registry()
        
        print(f"✅ Model registered: {model_id}")
        print(f"   Production Ready: {production_ready}")
        print(f"   Profitability Score: {metrics.profitability_score:.2f}")
        
        return model_id
    
    def _validate_production_criteria(self, metrics: ModelMetrics) -> bool:
        """
        Validate if model meets production criteria
        
        Criteria:
        - Mean PnL > 0 (profitable)
        - Win Rate > 50%
        - Sharpe Ratio > 1.0 (good risk-adjusted returns)
        - Max Drawdown < 20%
        - Profitability Score > 60
        """
        return (
            metrics.mean_pnl > 0 and
            metrics.win_rate > 0.5 and
            metrics.sharpe_ratio > 1.0 and
            metrics.max_drawdown < 0.2 and
            metrics.profitability_score > 60
        )
    
    def get_best_model(self, filter_production_ready: bool = True) -> Optional[str]:
        """
        Get the best performing model based on profitability score
        
        Args:
            filter_production_ready: Only consider production-ready models
            
        Returns:
            model_id of the best model, or None if no models available
        """
        candidates = self.models.values()
        
        if filter_production_ready:
            candidates = [m for m in candidates if m.production_ready]
        
        if not candidates:
            return None
        
        best = max(candidates, key=lambda m: m.metrics.profitability_score)
        return best.model_id
    
    def get_model_path(self, model_id: str) -> Optional[Path]:
        """Get the file path for a specific model"""
        if model_id not in self.models:
            return None
        return self.models_dir / model_id / "model.zip"
    
    def get_model_config(self, model_id: str) -> Optional[Path]:
        """Get the config path for a specific model"""
        if model_id not in self.models:
            return None
        return self.models_dir / model_id / "config.yaml"
    
    def list_models(
        self,
        production_ready_only: bool = False,
        min_profitability_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        List all registered models with filtering
        
        Args:
            production_ready_only: Filter to only production-ready models
            min_profitability_score: Minimum profitability score
            
        Returns:
            List of model metadata dictionaries
        """
        models = self.models.values()
        
        if production_ready_only:
            models = [m for m in models if m.production_ready]
        
        if min_profitability_score > 0:
            models = [m for m in models if m.metrics.profitability_score >= min_profitability_score]
        
        # Sort by profitability score (descending)
        models = sorted(models, key=lambda m: m.metrics.profitability_score, reverse=True)
        
        return [m.to_dict() for m in models]
    
    def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get metadata for a specific model"""
        return self.models.get(model_id)
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model from the registry"""
        if model_id not in self.models:
            return False
        
        # Remove from disk
        model_dir = self.models_dir / model_id
        if model_dir.exists():
            shutil.rmtree(model_dir)
        
        # Remove from registry
        del self.models[model_id]
        self._save_registry()
        
        print(f"🗑️  Model deleted: {model_id}")
        return True
    
    def export_leaderboard(self, output_path: str = "models/leaderboard.csv"):
        """Export model performance leaderboard to CSV"""
        models = self.list_models()
        
        if not models:
            print("No models in registry")
            return
        
        # Flatten for CSV
        rows = []
        for m in models:
            row = {
                'model_id': m['model_id'],
                'algorithm': m['algorithm'],
                'created_at': m['created_at'],
                'production_ready': m['production_ready'],
                'profitability_score': m['metrics']['profitability_score'],
                'mean_pnl': m['metrics']['mean_pnl'],
                'win_rate': m['metrics']['win_rate'],
                'sharpe_ratio': m['metrics']['sharpe_ratio'],
                'max_drawdown': m['metrics']['max_drawdown'],
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"📊 Leaderboard exported to: {output_path}")


def calculate_profitability_score(metrics: ModelMetrics) -> float:
    """
    Calculate a composite profitability score (0-100)
    
    Weights:
    - Mean PnL: 30%
    - Win Rate: 25%
    - Sharpe Ratio: 25%
    - Max Drawdown (inverted): 20%
    """
    # Normalize metrics to 0-100 scale
    pnl_score = min(100, max(0, metrics.mean_pnl / 100 * 100))  # Assume 100 is excellent
    win_rate_score = metrics.win_rate * 100
    sharpe_score = min(100, max(0, metrics.sharpe_ratio / 3.0 * 100))  # 3.0 is excellent
    drawdown_score = max(0, (1 - metrics.max_drawdown) * 100)  # Lower is better
    
    # Weighted average
    score = (
        pnl_score * 0.30 +
        win_rate_score * 0.25 +
        sharpe_score * 0.25 +
        drawdown_score * 0.20
    )
    
    return round(score, 2)
