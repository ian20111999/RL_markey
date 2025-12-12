# 🚀 Quick Start Guide - Production RL Market Making

Get a profitable trading model up and running in **5 minutes**.

---

## Prerequisites

- Python 3.8+
- 4GB+ RAM
- Historical market data (or will download automatically)

---

## Step 1: Install (2 minutes)

```bash
# Clone repository
git clone https://github.com/ian20111999/RL_markey.git
cd RL_markey

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-production.txt
```

---

## Step 2: Download Data (1 minute)

If you don't have data yet:

```bash
# Download BTC/USDT 1-minute data for 2023
python scripts/fetch_binance_ohlcv.py --symbol BTC/USDT --timeframe 1m --days 365
```

Or use any existing CSV data file with columns: `timestamp, open, high, low, close, volume`

---

## Step 3: Train Your First Model (2 minutes)

The CLI will automatically:
- Train a model
- Validate profitability
- Retry if needed
- Register the best model

```bash
python production/cli.py train --symbol btc --attempts 3
```

**What it does:**
1. Trains a reinforcement learning model
2. Evaluates on validation data
3. Checks profitability criteria
4. Retries up to 3 times if not profitable
5. Registers the best model automatically

**Success Criteria:**
- ✅ Mean PnL > 0 (profitable)
- ✅ Win Rate > 50%
- ✅ Sharpe Ratio > 1.0
- ✅ Max Drawdown < 20%
- ✅ Profitability Score > 60

---

## Step 4: Deploy (Optional)

### Option A: Start API Server

```bash
python production/cli.py serve --port 8000
```

Visit `http://localhost:8000/docs` for interactive API documentation.

### Option B: Start Dashboard

```bash
python production/dashboard.py
```

Visit `http://localhost:8080` to see your models.

### Option C: Docker Deployment

```bash
docker-compose up -d
```

---

## Using Your Model

### View Registered Models

```bash
python production/cli.py list
```

### Get Best Model Info

```bash
python production/cli.py best
```

### Export Performance Leaderboard

```bash
python production/cli.py leaderboard
# Creates models/leaderboard.csv
```

### Make Predictions via API

```bash
# Get prediction from best model
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"observation": [0.5, 0.2, 0.1, ...], "deterministic": true}'
```

---

## Next Steps

1. **Read the Full Guide**: See [PRODUCTION_GUIDE.md](PRODUCTION_GUIDE.md) for comprehensive documentation

2. **Try Different Algorithms**:
   ```bash
   python production/cli.py train --algorithm PPO
   python production/cli.py train --algorithm TD3
   ```

3. **Increase Training**:
   ```bash
   python production/cli.py train --timesteps 500000
   ```

4. **Use Custom Config**:
   ```bash
   python production/cli.py train --config configs/default.yaml
   ```

5. **Run Examples**:
   ```bash
   python examples/complete_workflow.py
   python examples/api_usage.py
   ```

---

## Troubleshooting

### "No data file found"
Download data first:
```bash
python scripts/fetch_binance_ohlcv.py --symbol BTC/USDT --timeframe 1m --days 365
```

### "Training failed"
Try with more timesteps:
```bash
python production/cli.py train --timesteps 300000
```

### "No profitable model"
Increase attempts:
```bash
python production/cli.py train --attempts 5
```

### API won't start
Check if port is in use:
```bash
lsof -i :8000
```

---

## Command Reference

```bash
# Train a model
python production/cli.py train [OPTIONS]
  --symbol TEXT          Trading symbol (default: btc)
  --algorithm TEXT       SAC, PPO, or TD3 (default: SAC)
  --attempts INT         Max training attempts (default: 3)
  --timesteps INT        Training timesteps (default: 200000)
  --config TEXT          Config file path

# List models
python production/cli.py list [OPTIONS]
  --production-only      Show only production-ready models

# Get best model
python production/cli.py best

# Export leaderboard
python production/cli.py leaderboard

# Start API server
python production/cli.py serve [OPTIONS]
  --host TEXT           Host to bind (default: 0.0.0.0)
  --port INT            Port to bind (default: 8000)
```

---

## What Makes a Model "Production-Ready"?

A model is marked as production-ready if it meets ALL criteria:

1. **Profitable**: Mean PnL > 0
2. **Consistent**: Win Rate > 50%
3. **Risk-Adjusted**: Sharpe Ratio > 1.0
4. **Safe**: Max Drawdown < 20%
5. **High Score**: Profitability Score > 60/100

The system automatically validates and only deploys models that pass.

---

## Support

- **Documentation**: [PRODUCTION_GUIDE.md](PRODUCTION_GUIDE.md)
- **Examples**: `examples/` directory
- **API Docs**: http://localhost:8000/docs (when running)
- **Issues**: GitHub Issues

---

**Happy Trading! 📈**
