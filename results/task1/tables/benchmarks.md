| Condition | GSM8K | HellaSwag | BoolQ | ARC-Easy | HellaSwag acc | ARC-Easy acc | GSM8K trunc. % | GSM8K unextractable % |
|---|---|---|---|---|---|---|---|---|
| Baseline | 43.6 | 47.3 | 64.4 | 56.1 | 37.6 | 60.9 | 0.83 | 0.00 |
| Full FT (mid) | 57.0 (+13.4) | 45.7 (-1.6) | 60.1 (-4.3) | 61.7 (+5.6) | 36.9 | 65.5 | 0.99 | 0.00 |
| Full FT | 64.7 (+21.2) | 46.0 (-1.3) | 60.4 (-4.0) | 63.5 (+7.4) | 37.2 | 66.5 | 0.45 | 0.00 |
| Full FT + L2-SP | 65.0 (+21.5) | 46.6 (-0.8) | 64.7 (+0.3) | 62.3 (+6.3) | 37.5 | 65.5 | 0.83 | 0.00 |
| LoRA r=4 | 59.7 (+16.1) | 47.5 (+0.1) | 66.7 (+2.3) | 63.0 (+6.9) | 37.6 | 65.3 | 0.91 | 0.00 |
| LoRA r=16 | 61.8 (+18.2) | 47.6 (+0.3) | 72.6 (+8.2) | 63.9 (+7.9) | 37.9 | 66.7 | 0.83 | 0.08 |
| OPLoRA r=16 | 61.9 (+18.3) | 47.4 (+0.1) | 71.6 (+7.2) | 63.7 (+7.6) | 37.8 | 66.9 | 0.61 | 0.08 |

Primary metrics: GSM8K flexible-extract exact match; HellaSwag and ARC-Easy acc_norm; BoolQ acc. Parentheses: change from the shared baseline in percentage points. Strict-match GSM8K is 0.0 for all conditions (no response uses the "The answer is" pattern).
