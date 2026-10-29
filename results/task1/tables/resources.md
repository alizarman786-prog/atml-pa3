| Method | Trainable params | Train time (h) | s / update | Peak GPU mem allocated (GB) | Peak reserved (GB) | Saved trainable state | Final displacement ‖θ−θ0‖₂ | Method-specific overhead |
|---|---|---|---|---|---|---|---|---|
| Full FT | 596,049,920 | 2.53 | 7.10 | 12.56 | 15.31 | 1.20 GB (FP16 model) | 10.67 | — |
| Full FT + L2-SP | 596,049,920 | 2.46 | 6.92 | 13.75 | 15.27 | 1.20 GB (FP16 model) | 6.27 | θ0 reference: 1.19 GB (bf16), transfer 0.32s |
| LoRA r=4 | 2,523,136 | 2.75 | 7.86 | 3.44 | 6.36 | 10.2 MB adapter | — | — |
| LoRA r=16 | 10,092,544 | 2.80 | 7.97 | 3.53 | 6.48 | 40.5 MB adapter | — | — |
| OPLoRA r=16 | 10,092,544 | 3.66 | 10.49 | 3.91 | 6.85 | 40.5 MB adapter | — | bases: 323 MB, SVD 37.1s |

Recovery checkpoint (trainable state + optimizer + scheduler + scaler + RNG): Full FT/L2-SP 7.15 GB; LoRA-4 30.7 MB; LoRA-16/OPLoRA 121.6 MB. Full FT midpoint displacement: 10.03. Skipped (overflow) updates: Full FT 0, Full FT + L2-SP 0, LoRA r=4 0, LoRA r=16 0, OPLoRA r=16 0. Runs were trained two at a time on 2x Tesla T4 (one run per GPU).
