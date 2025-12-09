# Evaluation Summary - Extended Run (1M Steps)

## Configuration
- **Run Folder**: `runs/v3_extended_1M_20251209_162838`
- **Base Model**: `runs/v3_fixed_20251209_155947` (300k steps)
- **Total Timesteps**: 1,000,000 (Extended)
- **Config**: `configs/env_v3_stabilized.yaml`

## Results (30 Episodes OOS)
- **Mean PnL**: +1,628.32
- **Std Dev**: ± 4,702.27
- **Win Rate**: 60% (18/30)
- **Total PnL**: +48,849.65
- **Max Inventory Usage**: 3.0

## Comparison with 300k Run
| Metric | 300k Run | 1M Run (Extended) | Change |
| :--- | :--- | :--- | :--- |
| Mean PnL | +2,051 | +1,628 | 🔻 -17% |
| Win Rate | 70% | 60% | 🔻 -10% |
| Std Dev | ± 5,265 | ± 4,702 | 🟢 Improved (Lower Risk) |

## Analysis
- The extended training slightly reduced the average profitability and win rate.
- However, the **Standard Deviation** decreased (from 5265 to 4702), indicating slightly more stable performance, although the "fat tails" (large losses) are still present.
- The model might have started to overfit to the training set or the high volatility penalty is making it too cautious in some OOS scenarios.

## Conclusion
- The 300k model might be the "sweet spot" for this specific configuration.
- The `vol_penalty` is effective but tuning its magnitude might be necessary to balance risk/reward further.
