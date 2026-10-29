| Condition | mean new tokens | truncated (/24) | JSON valid (/8, after fence strip) | template keys present (/8) | math markers (/24) | mean repeated-4gram frac |
|---|---|---|---|---|---|---|
| Baseline | 101 | 1 | 8 | 8 | 0 | 0.054 |
| Full FT (mid) | 202 | 5 | 2 | 8 | 11 | 0.324 |
| Full FT | 191 | 5 | 2 | 7 | 13 | 0.232 |
| Full FT + L2-SP | 188 | 5 | 2 | 8 | 11 | 0.171 |
| LoRA r=4 | 108 | 2 | 3 | 8 | 2 | 0.104 |
| LoRA r=16 | 155 | 3 | 2 | 8 | 8 | 0.219 |
| OPLoRA r=16 | 167 | 3 | 2 | 8 | 5 | 0.215 |

Automatic checks only (regex/parse based); they are not a substitute for the manual 42-response annotation.
