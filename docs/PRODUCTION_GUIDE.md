# 🚀 Production Deployment Guide

## Overview

This guide will help you deploy the RL Market Making system as a production-ready service. The system provides:

- **Automated Model Training**: Train profitable models with quality validation
- **REST API**: HTTP API for model inference and management
- **Model Registry**: Centralized tracking of all trained models
- **Docker Support**: Containerized deployment for easy scaling
- **Monitoring**: Health checks and performance tracking

---

## Quick Start (5 Minutes)

### 1. Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Production dependencies
# All dependencies are now consolidated in requirements.txt
```

### 2. Train Your First Profitable Model

```bash
# This will automatically train until you get a profitable model
python production/cli.py train --symbol btc --attempts 3
```

The CLI will:
- Train a model
- Validate its profitability
- Retry if not profitable (up to 3 attempts)
- Register the best model automatically

### 3. Start the API Server

```bash
# Start the production API
python production/cli.py serve --port 8000
```

Your API is now running at `http://localhost:8000`

Visit `http://localhost:8000/docs` for interactive API documentation.

---

## Production Deployment Options

### Option 1: Docker (Recommended)

**Single Command Deployment:**

```bash
# Build and start
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

**Manual Docker:**

```bash
# Build image
docker build -t rl-market-making:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  --name rl-market-making \
  rl-market-making:latest
```

### Option 2: Kubernetes

```bash
# Apply deployment (see k8s/ directory)
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Check status
kubectl get pods
kubectl get services

# Scale up
kubectl scale deployment rl-market-making --replicas=3
```

### Option 3: Direct Python

```bash
# Install as systemd service
sudo cp production/systemd/rl-market-making.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable rl-market-making
sudo systemctl start rl-market-making

# Check status
sudo systemctl status rl-market-making
```

---

## CLI Commands

### Train a Profitable Model

```bash
# Basic training
python production/cli.py train --symbol btc

# Advanced options
python production/cli.py train \
  --symbol btc \
  --algorithm SAC \
  --attempts 5 \
  --timesteps 300000 \
  --config configs/default.yaml
```

### List Models

```bash
# List all models
python production/cli.py list

# Production-ready models only
python production/cli.py list --production-only
```

### Get Best Model

```bash
python production/cli.py best
```

### Export Leaderboard

```bash
python production/cli.py leaderboard
# Creates models/leaderboard.csv
```

---

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "models_loaded": 1,
  "registry_size": 5,
  "production_models": 3
}
```

### Get Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "observation": [0.5, 0.2, 0.1, ...],
    "deterministic": true
  }'
```

Response:
```json
{
  "action": [0.3, 0.7],
  "model_id": "sac_20231215_143022",
  "version": "v1.0.0"
}
```

### List Models

```bash
curl http://localhost:8000/models
```

### Get Best Model

```bash
curl http://localhost:8000/models/best/current
```

### Model Management

```bash
# Load model into memory
curl -X POST http://localhost:8000/models/{model_id}/load

# Unload model
curl -X POST http://localhost:8000/models/{model_id}/unload

# Delete model
curl -X DELETE http://localhost:8000/models/{model_id}
```

---

## Model Registry

All trained models are automatically tracked in the registry at `models/registry/`.

### Registry Structure

```
models/registry/
├── registry.json          # Metadata database
└── models/
    ├── sac_20231215_143022/
    │   ├── model.zip      # Trained model
    │   └── config.yaml    # Configuration
    └── ...
```

### Quality Criteria

Models are marked as "production-ready" if they meet:

- ✅ Mean PnL > 0 (profitable)
- ✅ Win Rate > 50%
- ✅ Sharpe Ratio > 1.0
- ✅ Max Drawdown < 20%
- ✅ Profitability Score > 60

### Profitability Score

A composite score (0-100) calculated from:
- Mean PnL: 30%
- Win Rate: 25%
- Sharpe Ratio: 25%
- Max Drawdown: 20%

Higher is better. Models with score > 70 are considered excellent.

---

## Production Checklist

### Before Deployment

- [ ] Train and validate at least one profitable model
- [ ] Review model performance metrics
- [ ] Test API endpoints locally
- [ ] Configure monitoring and alerts
- [ ] Set up proper logging
- [ ] Review security settings

### Security Recommendations

1. **API Authentication**: Add API key authentication
   ```python
   # Add to production/api.py
   from fastapi.security import APIKeyHeader
   ```

2. **Rate Limiting**: Prevent abuse
   ```bash
   pip install slowapi
   ```

3. **HTTPS**: Use reverse proxy (nginx/traefik)
   ```nginx
   server {
       listen 443 ssl;
       location / {
           proxy_pass http://localhost:8000;
       }
   }
   ```

4. **Firewall**: Restrict access
   ```bash
   ufw allow 8000/tcp
   ufw enable
   ```

### Monitoring

1. **Health Endpoint**: `/health`
2. **Metrics Endpoint**: `/metrics/leaderboard`
3. **Logging**: Check `logs/` directory
4. **Alerts**: Set up for model performance degradation

---

## Troubleshooting

### API won't start

```bash
# Check if port is in use
lsof -i :8000

# Check logs
docker-compose logs -f
```

### No models available

```bash
# Train a model first
python production/cli.py train --symbol btc

# Verify registration
python production/cli.py list
```

### Poor model performance

```bash
# Try different algorithm
python production/cli.py train --algorithm TD3

# Increase training time
python production/cli.py train --timesteps 500000

# Adjust configuration
# Edit configs/default.yaml
```

### Memory issues

```bash
# Unload unused models
curl -X POST http://localhost:8000/models/{model_id}/unload

# Limit loaded models in production/api.py
MAX_LOADED_MODELS = 3
```

---

## Scaling

### Horizontal Scaling

```bash
# Docker Compose
docker-compose up -d --scale rl-market-making=3

# Kubernetes
kubectl scale deployment rl-market-making --replicas=5
```

### Load Balancing

Use nginx or traefik:

```nginx
upstream rl_backend {
    server localhost:8000;
    server localhost:8001;
    server localhost:8002;
}

server {
    location / {
        proxy_pass http://rl_backend;
    }
}
```

### Auto-Scaling

Kubernetes HPA:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rl-market-making-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rl-market-making
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## Best Practices

1. **Always validate models before production**
   ```bash
   python production/cli.py train --attempts 5
   ```

2. **Use the best model endpoint for inference**
   ```bash
   curl http://localhost:8000/models/best/current
   ```

3. **Monitor performance regularly**
   ```bash
   curl http://localhost:8000/metrics/leaderboard
   ```

4. **Retrain periodically with fresh data**
   ```bash
   # Weekly or monthly
   python production/cli.py train --symbol btc
   ```

5. **Keep backups of best models**
   ```bash
   cp -r models/registry models/registry_backup_$(date +%Y%m%d)
   ```

---

## Support

For issues or questions:
1. Check this documentation
2. Review API docs at `/docs`
3. Check logs in `logs/` directory
4. Open an issue on GitHub

---

## License

MIT License - See LICENSE file for details
