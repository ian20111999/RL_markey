# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive project structure documentation (PROJECT_STRUCTURE.md)
- Development guide for contributors (DEVELOPMENT.md)
- Enhanced README with badges and better organization
- Docker deployment section in README

### Changed
- Consolidated all dependencies into single requirements.txt
- Updated all documentation to reference consolidated requirements
- Improved README structure with table of contents
- Merged develop-marketable-models branch into main

### Removed
- Redundant requirements-production.txt file

## [2.0.0] - 2025-12-13

### Added - Production-Ready Features
- **Model Registry System** (`production/model_registry.py`)
  - Automatic versioning and metadata tracking
  - Performance metrics storage (PnL, Sharpe, Win Rate, etc.)
  - 5-criteria production validation
  - Profitability scoring (0-100 scale)
  - Model lifecycle management

- **REST API** (`production/api.py`)
  - FastAPI-based production API
  - Health check endpoint
  - Model prediction endpoint
  - Model management endpoints
  - Automatic best model selection
  - Performance leaderboard
  - Interactive documentation at `/docs`

- **Production CLI** (`production/cli.py`)
  - Auto-train profitable models with retries
  - List registered models
  - Show best model info
  - Export leaderboard to CSV
  - Start API server

- **Web Dashboard** (`production/dashboard.py`)
  - Real-time system statistics
  - Best model highlights
  - Performance leaderboard
  - Beautiful responsive UI

- **Docker Support**
  - Production-ready Dockerfile
  - Docker Compose configuration
  - Volume mounting for persistence
  - Health checks and auto-restart

- **Documentation**
  - PRODUCTION_SUMMARY.md - Complete feature summary
  - docs/PRODUCTION_GUIDE.md - Deployment guide (English)
  - docs/USER_GUIDE_ZH.md - User guide (Chinese)
  - docs/QUICKSTART.md - 5-minute quickstart

- **Examples**
  - examples/api_usage.py - API usage examples
  - examples/complete_workflow.py - Complete workflow demo

- **Tests**
  - tests/test_production.py - Production feature tests

### Changed
- Enhanced README with production features section
- Updated project structure to include production components

## [1.0.0] - 2023-XX-XX

### Added - Core Framework
- **Multiple RL Algorithms**
  - SAC (Soft Actor-Critic) - default
  - PPO (Proximal Policy Optimization)
  - TD3 (Twin Delayed DDPG)

- **Advanced Trading Environment** (`envs/`)
  - V2 environment with Potential-based Reward Shaping
  - Domain Randomization support
  - Realistic fill model
  - Flexible reward modes: dense, sparse, shaped, hybrid
  - Extended observation space (17+ features)

- **Training Utilities** (`utils/`)
  - Algorithm factory (`algorithms.py`)
  - Risk-sensitive training (`risk_sensitive.py`)
  - Curriculum learning (`curriculum.py`)
  - Professional backtesting framework (`backtesting.py`)
  - Ensemble methods (`ensemble.py`)
  - Explainability analysis (`explainability.py`)
  - Online adaptation (`online_adaptation.py`)
  - Distributed training (`distributed_training.py`)
  - Report generator (`report_generator.py`)

- **Configuration System**
  - YAML-based configuration
  - Multiple environment configs (v2, v3, baseline)
  - Flexible parameter management

- **Training Scripts**
  - `scripts/train.py` - General training script
  - `scripts/evaluate.py` - Evaluation script
  - `scripts/visualize_episode.py` - Visualization tools

- **Documentation**
  - Comprehensive README
  - Development phases documentation

### Features
- Walk-Forward Analysis for backtesting
- Monte Carlo Simulation
- HTML/PDF report generation
- TensorBoard integration
- Multi-seed validation
- Hyperparameter optimization with Optuna

---

## Version History Summary

- **v2.0.0** (2025-12-13): Production-ready deployment features
- **v1.0.0** (2023-XX-XX): Core RL training framework

---

## Upgrade Guide

### From v1.0.0 to v2.0.0

#### Breaking Changes
None - All v1.0.0 features are preserved

#### New Requirements
Production features require additional dependencies (now consolidated in requirements.txt):
- fastapi>=0.104.0
- uvicorn[standard]>=0.24.0
- python-multipart>=0.0.6

#### Migration Steps

1. **Update dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Optional: Use production features**:
   ```bash
   # Train a model (as before)
   python scripts/train.py
   
   # NEW: Register and deploy
   python production/cli.py train --symbol btc
   python production/cli.py serve
   ```

3. **Optional: Docker deployment**:
   ```bash
   docker-compose up -d
   ```

---

## Roadmap

### v2.1.0 (Planned)
- [ ] Enhanced monitoring and alerting
- [ ] Model A/B testing framework
- [ ] Multi-exchange support
- [ ] Advanced risk metrics

### v2.2.0 (Planned)
- [ ] Web-based configuration editor
- [ ] Automated hyperparameter tuning UI
- [ ] Real-time performance tracking
- [ ] Integration with trading platforms

### v3.0.0 (Future)
- [ ] Multi-agent collaboration
- [ ] Advanced market regime detection
- [ ] Adaptive strategy selection
- [ ] Cloud deployment templates

---

## Contributing

See [DEVELOPMENT.md](DEVELOPMENT.md) for contribution guidelines.

---

[Unreleased]: https://github.com/ian20111999/RL_markey/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/ian20111999/RL_markey/releases/tag/v2.0.0
[1.0.0]: https://github.com/ian20111999/RL_markey/releases/tag/v1.0.0
