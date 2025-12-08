# v3 Nuclear Configuration - Successful Run

## Overview
This configuration achieved the first consistent positive PnL in OOS testing.

## Key Changes
1. **Reward Normalization**: `reward_scale=0.001` to stabilize critic loss.
2. **Nuclear Inventory Penalty**: `lambda_inventory=10.0` (1000x original) to force inventory management.
3. **Strict Dynamic Position Limits**: `volatility_threshold=0.0005` (0.05%) to activate limits on 1-min data.
4. **Simplified Observation**: Disabled trend features (`include_trend=False`) to reduce overfitting/speculation.

## Results (OOS)
- **Mean PnL**: +4,304 USDT
- **Win Rate**: 60% of episodes
- **Total PnL**: +86,098 USDT (20 episodes)
- **Max Inventory**: 5.0 (Still hitting limits, but managing them profitably)

## Reproduction
1. Use config: `configs/env_v3_normalized.yaml` (ensure nuclear settings are present).
2. Run training:
   ```bash
   python scripts/run_v3_pipeline.py --config configs/env_v3_normalized.yaml --algorithm SAC --total_timesteps 200000 --mode standard
   ```
3. Or continue from checkpoint:
   ```bash
   python scripts/continue_training.py
   ```

## Visualization
Use `scripts/visualize_episode.py` to plot specific episodes:
```bash
python scripts/visualize_episode.py --run_folder runs/v3_nuclear_continued_20251206_122315 --episode 18
```
