| Condition | Layer | ρ_k (k=128) | η | R math (mean) | R non-math (mean) | math/non-math |
|---|---|---|---|---|---|---|
| Full FT (mid) | 0 | 0.1337 | 0.0131 | 0.0514 | 0.0251 | 2.05 |
| Full FT (mid) | 13 | 0.1485 | 0.0154 | 0.0572 | 0.0302 | 1.89 |
| Full FT (mid) | 27 | 0.1160 | 0.0125 | 0.0194 | 0.0286 | 0.68 |
| Full FT | 0 | 0.1334 | 0.0136 | 0.0530 | 0.0257 | 2.06 |
| Full FT | 13 | 0.1485 | 0.0160 | 0.0578 | 0.0308 | 1.88 |
| Full FT | 27 | 0.1154 | 0.0130 | 0.0193 | 0.0290 | 0.67 |
| Full FT + L2-SP | 0 | 0.1332 | 0.0112 | 0.0456 | 0.0228 | 2.00 |
| Full FT + L2-SP | 13 | 0.1497 | 0.0137 | 0.0480 | 0.0276 | 1.74 |
| Full FT + L2-SP | 27 | 0.1068 | 0.0079 | 0.0181 | 0.0258 | 0.70 |
| LoRA r=4 | 0 | 0.1329 | 0.0093 | 0.0712 | 0.0147 | 4.83 |
| LoRA r=4 | 13 | 0.1389 | 0.0101 | 0.0975 | 0.0137 | 7.13 |
| LoRA r=4 | 27 | 0.1055 | 0.0068 | 0.0040 | 0.0051 | 0.78 |
| LoRA r=16 | 0 | 0.1452 | 0.0149 | 0.0450 | 0.0274 | 1.64 |
| LoRA r=16 | 13 | 0.1412 | 0.0183 | 0.0578 | 0.0262 | 2.20 |
| LoRA r=16 | 27 | 0.1159 | 0.0115 | 0.0098 | 0.0147 | 0.66 |
| OPLoRA r=16 | 0 | 0.0000 | 0.0147 | 0.0481 | 0.0278 | 1.73 |
| OPLoRA r=16 | 13 | 0.0000 | 0.0187 | 0.0593 | 0.0275 | 2.15 |
| OPLoRA r=16 | 27 | 0.0000 | 0.0094 | 0.0093 | 0.0136 | 0.68 |

Computed in FP32 on down_proj. For OPLoRA ρ_k is zero by construction (projected factors). z is the starting model's down_proj input at the last prompt token; GSM8K probe prompts all end in the same chat-template suffix, so math-prompt z vectors are highly similar.
