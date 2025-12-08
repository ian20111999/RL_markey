#!/bin/bash
cd /Users/ian/Desktop/Project/RL_markey
/Users/ian/Desktop/Project/RL_markey/.venv/bin/python scripts/run_v3_pipeline.py \
    --config configs/env_v3_robust.yaml \
    --algorithm SAC \
    --total_timesteps 200000 \
    --mode standard
