"""
Production tests for Model Registry System
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from production.model_registry import (
    ModelRegistry,
    ModelMetrics,
    calculate_profitability_score
)


@pytest.fixture
def temp_registry():
    """Create a temporary registry for testing"""
    temp_dir = tempfile.mkdtemp()
    registry = ModelRegistry(registry_dir=temp_dir)
    yield registry
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_metrics():
    """Sample profitable model metrics"""
    return ModelMetrics(
        mean_pnl=150.0,
        std_pnl=50.0,
        sharpe_ratio=2.0,
        max_drawdown=0.15,
        win_rate=0.65,
        total_trades=100,
        mean_max_inventory=3.5,
        profitability_score=75.0
    )


def test_profitability_score_calculation():
    """Test profitability score calculation"""
    metrics = ModelMetrics(
        mean_pnl=100.0,
        std_pnl=50.0,
        sharpe_ratio=2.0,
        max_drawdown=0.10,
        win_rate=0.60,
        total_trades=100,
        mean_max_inventory=3.0,
        profitability_score=0
    )
    
    score = calculate_profitability_score(metrics)
    
    # Score should be between 0 and 100
    assert 0 <= score <= 100
    
    # Good metrics should give good score
    assert score > 60


def test_model_registration(temp_registry, sample_metrics, tmp_path):
    """Test model registration"""
    # Create dummy model and config files
    model_file = tmp_path / "model.zip"
    config_file = tmp_path / "config.yaml"
    model_file.write_text("dummy model")
    config_file.write_text("dummy config")
    
    # Register model
    model_id = temp_registry.register_model(
        model_path=str(model_file),
        config_path=str(config_file),
        metrics=sample_metrics,
        algorithm="SAC",
        training_timesteps=200000,
        data_source="btc_usdt_1m_2023.csv",
        tags=["test"],
        notes="Test model"
    )
    
    # Verify registration
    assert model_id in temp_registry.models
    metadata = temp_registry.get_model_metadata(model_id)
    assert metadata is not None
    assert metadata.algorithm == "SAC"
    assert metadata.production_ready is True  # Good metrics
    assert metadata.metrics.mean_pnl == 150.0


def test_production_criteria_validation(temp_registry):
    """Test production readiness validation"""
    # Good metrics
    good_metrics = ModelMetrics(
        mean_pnl=100.0,
        std_pnl=50.0,
        sharpe_ratio=2.0,
        max_drawdown=0.15,
        win_rate=0.65,
        total_trades=100,
        mean_max_inventory=3.0,
        profitability_score=75.0
    )
    assert temp_registry._validate_production_criteria(good_metrics) == True
    
    # Poor metrics - not profitable
    poor_metrics = ModelMetrics(
        mean_pnl=-50.0,
        std_pnl=50.0,
        sharpe_ratio=0.5,
        max_drawdown=0.30,
        win_rate=0.40,
        total_trades=100,
        mean_max_inventory=3.0,
        profitability_score=30.0
    )
    assert temp_registry._validate_production_criteria(poor_metrics) == False


def test_get_best_model(temp_registry, tmp_path):
    """Test best model selection"""
    # Register multiple models with different scores
    for i, score in enumerate([60, 80, 70]):
        metrics = ModelMetrics(
            mean_pnl=score,
            std_pnl=20.0,
            sharpe_ratio=2.0,
            max_drawdown=0.10,
            win_rate=0.60,
            total_trades=100,
            mean_max_inventory=3.0,
            profitability_score=score
        )
        
        model_file = tmp_path / f"model_{i}.zip"
        config_file = tmp_path / f"config_{i}.yaml"
        model_file.write_text(f"model {i}")
        config_file.write_text(f"config {i}")
        
        temp_registry.register_model(
            model_path=str(model_file),
            config_path=str(config_file),
            metrics=metrics,
            algorithm="SAC"
        )
    
    # Get best model
    best_id = temp_registry.get_best_model()
    assert best_id is not None
    
    best = temp_registry.get_model_metadata(best_id)
    assert best.metrics.profitability_score == 80.0


def test_list_models_filtering(temp_registry, tmp_path):
    """Test model listing with filters"""
    # Register models with varying scores
    scores = [45, 65, 85]  # One fails, two pass production criteria
    
    for i, score in enumerate(scores):
        metrics = ModelMetrics(
            mean_pnl=score,
            std_pnl=20.0,
            sharpe_ratio=2.0 if score > 50 else 0.5,
            max_drawdown=0.10,
            win_rate=0.60 if score > 50 else 0.40,
            total_trades=100,
            mean_max_inventory=3.0,
            profitability_score=score
        )
        
        model_file = tmp_path / f"model_{i}.zip"
        config_file = tmp_path / f"config_{i}.yaml"
        model_file.write_text(f"model {i}")
        config_file.write_text(f"config {i}")
        
        temp_registry.register_model(
            model_path=str(model_file),
            config_path=str(config_file),
            metrics=metrics,
            algorithm="SAC"
        )
    
    # List all models
    all_models = temp_registry.list_models()
    assert len(all_models) == 3
    
    # List production-ready only
    prod_models = temp_registry.list_models(production_ready_only=True)
    assert len(prod_models) == 2
    
    # Filter by score
    high_score = temp_registry.list_models(min_profitability_score=70)
    assert len(high_score) == 1


def test_model_deletion(temp_registry, sample_metrics, tmp_path):
    """Test model deletion"""
    model_file = tmp_path / "model.zip"
    config_file = tmp_path / "config.yaml"
    model_file.write_text("dummy model")
    config_file.write_text("dummy config")
    
    model_id = temp_registry.register_model(
        model_path=str(model_file),
        config_path=str(config_file),
        metrics=sample_metrics,
        algorithm="SAC"
    )
    
    # Verify exists
    assert model_id in temp_registry.models
    
    # Delete
    success = temp_registry.delete_model(model_id)
    assert success == True
    assert model_id not in temp_registry.models
    
    # Try to delete again
    success = temp_registry.delete_model(model_id)
    assert success == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
