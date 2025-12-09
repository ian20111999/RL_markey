# Evaluation Summary - Run v3_fixed_20251209_155947

## Configuration
- **Model**: `runs/v3_fixed_20251209_155947/best_model/best_model.zip`
- **Config**: `configs/env_v3_stabilized.yaml`
- **Key Features**: 
    - `max_inventory`: 3.0
    - `vol_penalty`: Enabled (Threshold 0.2%, Coeff 200.0)
    - `lambda_inventory`: 10.0

## Results (30 Episodes OOS)
- **Mean PnL**: +2,051.53
- **Std Dev**: ± 5,264.98
- **Win Rate**: 70% (21/30)
- **Total PnL**: +61,545.91
- **Max Inventory Usage**: 3.0 (Full utilization)

## Analysis
- **Success**: The agent is profitable on average, a major turnaround from previous runs.
- **Risk**: High standard deviation and some large losses (e.g., -10k) indicate risk management is still a concern during extreme events.
- **Hypothesis**: The `vol_penalty` successfully filters out many bad trades, allowing the agent to profit from the spread in calmer markets.

## Next Steps
- Run a longer training session (1M steps) to allow the agent to converge further and potentially learn to handle the high-volatility edge cases better.
