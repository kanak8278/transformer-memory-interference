# Query-Discriminating Heads — Extraction Report

FVQ - CVQ attention differences per (layer, head).  
**Positive** = head attends more to target round under FVQ than CVQ.  
**Negative** = head attends more under CVQ.  

## Qwen2.5-3B-Instruct__normal
- Model: `Qwen2.5-3B-Instruct` / mode: `normal`  
- K=2, N=30, trials=50

### generation token → primacy_round0
max |FVQ-CVQ| = 0.7519, mean |FVQ-CVQ| = 0.02003

**Top FVQ-dominant (FVQ > CVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 31 | 15 | +0.7519 |
| 2 | 31 | 7 | +0.5837 |
| 3 | 31 | 8 | +0.5621 |
| 4 | 30 | 11 | +0.5563 |
| 5 | 31 | 12 | +0.4596 |
| 6 | 29 | 1 | +0.3755 |
| 7 | 27 | 1 | +0.3436 |
| 8 | 26 | 12 | +0.3111 |
| 9 | 30 | 1 | +0.2934 |
| 10 | 29 | 5 | +0.2814 |
| 11 | 32 | 3 | +0.2810 |
| 12 | 31 | 6 | +0.2690 |
| 13 | 29 | 4 | +0.2548 |
| 14 | 32 | 7 | +0.2507 |
| 15 | 31 | 3 | +0.2038 |

**Top CVQ-dominant (CVQ > FVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|

### generation token → recency_last_round
max |FVQ-CVQ| = 0.1827, mean |FVQ-CVQ| = 0.00436

**Top FVQ-dominant (FVQ > CVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|

**Top CVQ-dominant (CVQ > FVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 31 | 15 | -0.1827 |
| 2 | 31 | 5 | -0.1533 |
| 3 | 30 | 11 | -0.1180 |
| 4 | 31 | 12 | -0.0963 |
| 5 | 31 | 8 | -0.0840 |
| 6 | 32 | 3 | -0.0728 |
| 7 | 30 | 1 | -0.0621 |
| 8 | 31 | 3 | -0.0610 |
| 9 | 32 | 7 | -0.0598 |
| 10 | 31 | 7 | -0.0494 |
| 11 | 29 | 2 | -0.0386 |
| 12 | 33 | 9 | -0.0373 |
| 13 | 31 | 6 | -0.0363 |
| 14 | 33 | 0 | -0.0361 |
| 15 | 27 | 10 | -0.0359 |

### query-cat token → primacy_round0
max |FVQ-CVQ| = 0.0540, mean |FVQ-CVQ| = 0.00209

**Top FVQ-dominant (FVQ > CVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 27 | 1 | +0.0540 |
| 2 | 27 | 0 | +0.0528 |
| 3 | 28 | 3 | +0.0344 |
| 4 | 17 | 10 | +0.0322 |
| 5 | 24 | 5 | +0.0219 |
| 6 | 31 | 8 | +0.0199 |
| 7 | 25 | 8 | +0.0198 |
| 8 | 13 | 6 | +0.0188 |
| 9 | 30 | 1 | +0.0186 |
| 10 | 32 | 3 | +0.0181 |
| 11 | 15 | 15 | +0.0156 |
| 12 | 31 | 10 | +0.0147 |
| 13 | 27 | 10 | +0.0144 |
| 14 | 20 | 5 | +0.0133 |
| 15 | 28 | 11 | +0.0132 |

**Top CVQ-dominant (CVQ > FVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 22 | 14 | -0.0275 |
| 2 | 25 | 2 | -0.0164 |
| 3 | 29 | 1 | -0.0132 |
| 4 | 5 | 15 | -0.0128 |
| 5 | 31 | 4 | -0.0115 |
| 6 | 31 | 2 | -0.0080 |
| 7 | 34 | 6 | -0.0072 |
| 8 | 24 | 11 | -0.0072 |

### query-cat token → recency_last_round
max |FVQ-CVQ| = 0.0270, mean |FVQ-CVQ| = 0.00064

**Top FVQ-dominant (FVQ > CVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 31 | 4 | +0.0018 |
| 2 | 28 | 3 | +0.0017 |
| 3 | 11 | 8 | +0.0016 |

**Top CVQ-dominant (CVQ > FVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 4 | 0 | -0.0270 |
| 2 | 31 | 3 | -0.0197 |
| 3 | 19 | 12 | -0.0115 |
| 4 | 31 | 7 | -0.0112 |
| 5 | 23 | 1 | -0.0110 |
| 6 | 18 | 13 | -0.0066 |
| 7 | 3 | 12 | -0.0056 |
| 8 | 6 | 1 | -0.0055 |
| 9 | 15 | 14 | -0.0050 |
| 10 | 8 | 4 | -0.0045 |
| 11 | 12 | 15 | -0.0040 |
| 12 | 34 | 12 | -0.0037 |
| 13 | 23 | 7 | -0.0033 |
| 14 | 2 | 14 | -0.0032 |
| 15 | 19 | 9 | -0.0030 |

## Qwen2.5-3B-Instruct__normal__K2N10
- Model: `Qwen2.5-3B-Instruct` / mode: `normal`  
- K=2, N=10, trials=50

### generation token → primacy_round0
max |FVQ-CVQ| = 0.7454, mean |FVQ-CVQ| = 0.02062

**Top FVQ-dominant (FVQ > CVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 31 | 15 | +0.7454 |
| 2 | 31 | 7 | +0.5760 |
| 3 | 30 | 11 | +0.5185 |
| 4 | 31 | 8 | +0.4942 |
| 5 | 31 | 12 | +0.4430 |
| 6 | 32 | 3 | +0.3716 |
| 7 | 27 | 1 | +0.3454 |
| 8 | 29 | 1 | +0.3350 |
| 9 | 32 | 7 | +0.3326 |
| 10 | 26 | 12 | +0.2746 |
| 11 | 30 | 1 | +0.2653 |
| 12 | 31 | 6 | +0.2503 |
| 13 | 29 | 5 | +0.2461 |
| 14 | 29 | 4 | +0.2094 |
| 15 | 31 | 3 | +0.2033 |

**Top CVQ-dominant (CVQ > FVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|

### generation token → recency_last_round
max |FVQ-CVQ| = 0.3837, mean |FVQ-CVQ| = 0.00938

**Top FVQ-dominant (FVQ > CVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|

**Top CVQ-dominant (CVQ > FVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 31 | 15 | -0.3837 |
| 2 | 31 | 12 | -0.2756 |
| 3 | 31 | 8 | -0.2264 |
| 4 | 30 | 11 | -0.2082 |
| 5 | 31 | 5 | -0.1818 |
| 6 | 32 | 3 | -0.1772 |
| 7 | 30 | 1 | -0.1737 |
| 8 | 31 | 7 | -0.1723 |
| 9 | 29 | 2 | -0.1256 |
| 10 | 29 | 5 | -0.1109 |
| 11 | 31 | 3 | -0.1108 |
| 12 | 29 | 4 | -0.1077 |
| 13 | 32 | 7 | -0.1072 |
| 14 | 31 | 6 | -0.1044 |
| 15 | 27 | 6 | -0.1002 |

### query-cat token → primacy_round0
max |FVQ-CVQ| = 0.0974, mean |FVQ-CVQ| = 0.00407

**Top FVQ-dominant (FVQ > CVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 28 | 3 | +0.0974 |
| 2 | 27 | 1 | +0.0819 |
| 3 | 31 | 8 | +0.0791 |
| 4 | 31 | 15 | +0.0727 |
| 5 | 31 | 7 | +0.0649 |
| 6 | 27 | 0 | +0.0580 |
| 7 | 30 | 1 | +0.0558 |
| 8 | 32 | 3 | +0.0468 |
| 9 | 31 | 12 | +0.0425 |
| 10 | 25 | 8 | +0.0395 |
| 11 | 29 | 1 | +0.0343 |
| 12 | 32 | 7 | +0.0343 |
| 13 | 24 | 5 | +0.0315 |
| 14 | 31 | 6 | +0.0312 |
| 15 | 17 | 10 | +0.0297 |

**Top CVQ-dominant (CVQ > FVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 22 | 14 | -0.0268 |
| 2 | 34 | 6 | -0.0157 |
| 3 | 6 | 9 | -0.0148 |

### query-cat token → recency_last_round
max |FVQ-CVQ| = 0.0324, mean |FVQ-CVQ| = 0.00116

**Top FVQ-dominant (FVQ > CVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 13 | 2 | +0.0179 |
| 2 | 10 | 2 | +0.0055 |
| 3 | 31 | 6 | +0.0051 |
| 4 | 11 | 8 | +0.0040 |

**Top CVQ-dominant (CVQ > FVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 4 | 0 | -0.0324 |
| 2 | 19 | 12 | -0.0173 |
| 3 | 23 | 1 | -0.0151 |
| 4 | 21 | 1 | -0.0143 |
| 5 | 34 | 12 | -0.0132 |
| 6 | 18 | 3 | -0.0117 |
| 7 | 18 | 13 | -0.0117 |
| 8 | 6 | 1 | -0.0106 |
| 9 | 21 | 10 | -0.0104 |
| 10 | 15 | 4 | -0.0097 |
| 11 | 24 | 5 | -0.0086 |
| 12 | 21 | 4 | -0.0078 |
| 13 | 21 | 11 | -0.0075 |
| 14 | 21 | 3 | -0.0071 |
| 15 | 22 | 7 | -0.0071 |

## Qwen2.5-3B-Instruct__normal__K5N20
- Model: `Qwen2.5-3B-Instruct` / mode: `normal`  
- K=5, N=20, trials=50

### generation token → primacy_round0
max |FVQ-CVQ| = 0.5431, mean |FVQ-CVQ| = 0.01566

**Top FVQ-dominant (FVQ > CVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 31 | 15 | +0.5431 |
| 2 | 27 | 12 | +0.3374 |
| 3 | 27 | 1 | +0.3321 |
| 4 | 31 | 8 | +0.3267 |
| 5 | 31 | 12 | +0.3132 |
| 6 | 31 | 7 | +0.3130 |
| 7 | 27 | 0 | +0.2827 |
| 8 | 30 | 11 | +0.2762 |
| 9 | 29 | 1 | +0.2726 |
| 10 | 32 | 3 | +0.2657 |
| 11 | 32 | 7 | +0.2382 |
| 12 | 27 | 6 | +0.2328 |
| 13 | 29 | 4 | +0.2099 |
| 14 | 29 | 5 | +0.1831 |
| 15 | 28 | 11 | +0.1746 |

**Top CVQ-dominant (CVQ > FVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|

### generation token → recency_last_round
max |FVQ-CVQ| = 0.1787, mean |FVQ-CVQ| = 0.00429

**Top FVQ-dominant (FVQ > CVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|

**Top CVQ-dominant (CVQ > FVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 31 | 15 | -0.1787 |
| 2 | 30 | 11 | -0.1271 |
| 3 | 31 | 12 | -0.1031 |
| 4 | 31 | 8 | -0.0981 |
| 5 | 32 | 3 | -0.0844 |
| 6 | 29 | 2 | -0.0766 |
| 7 | 32 | 7 | -0.0699 |
| 8 | 31 | 5 | -0.0663 |
| 9 | 31 | 7 | -0.0618 |
| 10 | 30 | 1 | -0.0506 |
| 11 | 32 | 10 | -0.0455 |
| 12 | 29 | 4 | -0.0434 |
| 13 | 29 | 5 | -0.0427 |
| 14 | 32 | 1 | -0.0426 |
| 15 | 31 | 6 | -0.0408 |

### query-cat token → primacy_round0
max |FVQ-CVQ| = 0.0716, mean |FVQ-CVQ| = 0.00146

**Top FVQ-dominant (FVQ > CVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 21 | 7 | +0.0716 |
| 2 | 27 | 1 | +0.0647 |
| 3 | 25 | 8 | +0.0273 |
| 4 | 17 | 10 | +0.0271 |
| 5 | 24 | 5 | +0.0229 |
| 6 | 23 | 5 | +0.0165 |
| 7 | 13 | 3 | +0.0160 |
| 8 | 31 | 10 | +0.0156 |
| 9 | 13 | 6 | +0.0135 |
| 10 | 12 | 8 | +0.0127 |
| 11 | 20 | 6 | +0.0127 |
| 12 | 31 | 7 | +0.0123 |
| 13 | 24 | 4 | +0.0114 |
| 14 | 27 | 0 | +0.0105 |
| 15 | 32 | 3 | +0.0094 |

**Top CVQ-dominant (CVQ > FVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 5 | 15 | -0.0203 |
| 2 | 25 | 2 | -0.0120 |
| 3 | 29 | 4 | -0.0073 |
| 4 | 30 | 11 | -0.0070 |
| 5 | 4 | 5 | -0.0051 |
| 6 | 29 | 5 | -0.0050 |
| 7 | 5 | 8 | -0.0045 |

### query-cat token → recency_last_round
max |FVQ-CVQ| = 0.0166, mean |FVQ-CVQ| = 0.00077

**Top FVQ-dominant (FVQ > CVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 10 | 0 | +0.0050 |

**Top CVQ-dominant (CVQ > FVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 19 | 12 | -0.0166 |
| 2 | 4 | 0 | -0.0161 |
| 3 | 23 | 1 | -0.0154 |
| 4 | 18 | 3 | -0.0143 |
| 5 | 21 | 10 | -0.0120 |
| 6 | 22 | 14 | -0.0085 |
| 7 | 23 | 0 | -0.0084 |
| 8 | 13 | 2 | -0.0078 |
| 9 | 22 | 10 | -0.0070 |
| 10 | 24 | 5 | -0.0067 |
| 11 | 21 | 1 | -0.0066 |
| 12 | 20 | 6 | -0.0057 |
| 13 | 20 | 4 | -0.0054 |
| 14 | 22 | 12 | -0.0052 |
| 15 | 23 | 5 | -0.0048 |

## Qwen2.5-3B-Instruct__reversal
- Model: `Qwen2.5-3B-Instruct` / mode: `reversal`  
- K=15, N=20, trials=50

### generation token → primacy_round0
max |FVQ-CVQ| = 0.2442, mean |FVQ-CVQ| = 0.00805

**Top FVQ-dominant (FVQ > CVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 31 | 15 | +0.2442 |
| 2 | 27 | 12 | +0.1942 |
| 3 | 27 | 0 | +0.1543 |
| 4 | 27 | 1 | +0.1529 |
| 5 | 28 | 11 | +0.1483 |
| 6 | 29 | 1 | +0.1469 |
| 7 | 32 | 3 | +0.1466 |
| 8 | 31 | 8 | +0.1442 |
| 9 | 31 | 12 | +0.1333 |
| 10 | 31 | 7 | +0.1296 |
| 11 | 32 | 7 | +0.1281 |
| 12 | 29 | 4 | +0.1253 |
| 13 | 28 | 10 | +0.1136 |
| 14 | 29 | 5 | +0.1018 |
| 15 | 30 | 11 | +0.1016 |

**Top CVQ-dominant (CVQ > FVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|

### generation token → recency_last_round
max |FVQ-CVQ| = 0.1743, mean |FVQ-CVQ| = 0.00377

**Top FVQ-dominant (FVQ > CVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|

**Top CVQ-dominant (CVQ > FVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 31 | 15 | -0.1743 |
| 2 | 30 | 11 | -0.1439 |
| 3 | 31 | 12 | -0.1258 |
| 4 | 32 | 3 | -0.1106 |
| 5 | 31 | 7 | -0.0800 |
| 6 | 31 | 8 | -0.0760 |
| 7 | 32 | 7 | -0.0743 |
| 8 | 30 | 1 | -0.0742 |
| 9 | 29 | 2 | -0.0595 |
| 10 | 27 | 1 | -0.0483 |
| 11 | 33 | 9 | -0.0463 |
| 12 | 29 | 4 | -0.0459 |
| 13 | 29 | 1 | -0.0457 |
| 14 | 29 | 5 | -0.0435 |
| 15 | 32 | 0 | -0.0391 |

### query-cat token → primacy_round0
max |FVQ-CVQ| = 0.0648, mean |FVQ-CVQ| = 0.00087

**Top FVQ-dominant (FVQ > CVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 21 | 7 | +0.0648 |
| 2 | 17 | 10 | +0.0371 |
| 3 | 13 | 3 | +0.0255 |
| 4 | 27 | 1 | +0.0136 |
| 5 | 12 | 3 | +0.0129 |
| 6 | 20 | 6 | +0.0124 |
| 7 | 19 | 8 | +0.0101 |
| 8 | 25 | 8 | +0.0101 |
| 9 | 14 | 4 | +0.0086 |
| 10 | 23 | 5 | +0.0084 |
| 11 | 5 | 14 | +0.0083 |
| 12 | 17 | 14 | +0.0079 |
| 13 | 21 | 2 | +0.0076 |
| 14 | 20 | 5 | +0.0066 |
| 15 | 24 | 5 | +0.0065 |

**Top CVQ-dominant (CVQ > FVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 25 | 2 | -0.0110 |
| 2 | 22 | 0 | -0.0041 |
| 3 | 30 | 11 | -0.0036 |
| 4 | 25 | 14 | -0.0033 |
| 5 | 22 | 7 | -0.0032 |
| 6 | 27 | 0 | -0.0032 |
| 7 | 31 | 7 | -0.0032 |
| 8 | 26 | 12 | -0.0029 |
| 9 | 21 | 14 | -0.0028 |
| 10 | 5 | 15 | -0.0027 |
| 11 | 28 | 3 | -0.0024 |
| 12 | 31 | 10 | -0.0022 |

### query-cat token → recency_last_round
max |FVQ-CVQ| = 0.0148, mean |FVQ-CVQ| = 0.00048

**Top FVQ-dominant (FVQ > CVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|

**Top CVQ-dominant (CVQ > FVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 17 | 11 | -0.0148 |
| 2 | 13 | 2 | -0.0145 |
| 3 | 20 | 4 | -0.0132 |
| 4 | 21 | 10 | -0.0122 |
| 5 | 23 | 1 | -0.0084 |
| 6 | 19 | 12 | -0.0061 |
| 7 | 22 | 14 | -0.0061 |
| 8 | 22 | 12 | -0.0058 |
| 9 | 4 | 0 | -0.0054 |
| 10 | 22 | 0 | -0.0054 |
| 11 | 22 | 10 | -0.0043 |
| 12 | 24 | 5 | -0.0043 |
| 13 | 22 | 13 | -0.0041 |
| 14 | 20 | 6 | -0.0040 |
| 15 | 18 | 3 | -0.0038 |

## gemma-3-4b-it__normal
- Model: `gemma-3-4b-it` / mode: `normal`  
- K=7, N=30, trials=50

### generation token → primacy_round0
max |FVQ-CVQ| = 0.7272, mean |FVQ-CVQ| = 0.01516

**Top FVQ-dominant (FVQ > CVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 23 | 3 | +0.7272 |
| 2 | 23 | 1 | +0.6153 |
| 3 | 23 | 0 | +0.5492 |
| 4 | 23 | 6 | +0.3994 |
| 5 | 23 | 7 | +0.2483 |
| 6 | 17 | 5 | +0.2095 |
| 7 | 17 | 0 | +0.2007 |
| 8 | 29 | 4 | +0.1904 |
| 9 | 17 | 2 | +0.1052 |
| 10 | 29 | 3 | +0.1012 |
| 11 | 29 | 5 | +0.0974 |
| 12 | 21 | 6 | +0.0726 |
| 13 | 29 | 2 | +0.0584 |
| 14 | 17 | 3 | +0.0543 |
| 15 | 23 | 2 | +0.0533 |

**Top CVQ-dominant (CVQ > FVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 17 | 4 | -0.0139 |
| 2 | 17 | 6 | -0.0094 |

### generation token → recency_last_round
max |FVQ-CVQ| = 0.1389, mean |FVQ-CVQ| = 0.00851

**Top FVQ-dominant (FVQ > CVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|

**Top CVQ-dominant (CVQ > FVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 23 | 3 | -0.1389 |
| 2 | 25 | 0 | -0.1057 |
| 3 | 22 | 6 | -0.0995 |
| 4 | 32 | 1 | -0.0991 |
| 5 | 29 | 4 | -0.0953 |
| 6 | 27 | 6 | -0.0897 |
| 7 | 23 | 1 | -0.0872 |
| 8 | 30 | 2 | -0.0871 |
| 9 | 28 | 1 | -0.0831 |
| 10 | 22 | 7 | -0.0787 |
| 11 | 23 | 0 | -0.0786 |
| 12 | 26 | 4 | -0.0764 |
| 13 | 22 | 5 | -0.0639 |
| 14 | 29 | 5 | -0.0620 |
| 15 | 26 | 0 | -0.0608 |

### query-cat token → primacy_round0
max |FVQ-CVQ| = 0.0828, mean |FVQ-CVQ| = 0.00126

**Top FVQ-dominant (FVQ > CVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 17 | 5 | +0.0828 |
| 2 | 17 | 0 | +0.0718 |
| 3 | 11 | 1 | +0.0326 |
| 4 | 23 | 3 | +0.0249 |
| 5 | 17 | 1 | +0.0176 |
| 6 | 17 | 2 | +0.0159 |
| 7 | 19 | 7 | +0.0094 |
| 8 | 23 | 1 | +0.0076 |
| 9 | 29 | 4 | +0.0069 |
| 10 | 11 | 5 | +0.0068 |
| 11 | 23 | 7 | +0.0067 |
| 12 | 23 | 6 | +0.0058 |
| 13 | 11 | 3 | +0.0043 |
| 14 | 29 | 5 | +0.0037 |
| 15 | 19 | 1 | +0.0024 |

**Top CVQ-dominant (CVQ > FVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 11 | 2 | -0.0056 |
| 2 | 5 | 1 | -0.0027 |
| 3 | 14 | 3 | -0.0015 |
| 4 | 5 | 6 | -0.0012 |
| 5 | 11 | 4 | -0.0012 |
| 6 | 12 | 7 | -0.0007 |
| 7 | 11 | 6 | -0.0006 |
| 8 | 15 | 4 | -0.0005 |
| 9 | 15 | 7 | -0.0005 |
| 10 | 10 | 3 | -0.0004 |

### query-cat token → recency_last_round
max |FVQ-CVQ| = 0.0099, mean |FVQ-CVQ| = 0.00034

**Top FVQ-dominant (FVQ > CVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 12 | 7 | +0.0011 |
| 2 | 7 | 2 | +0.0009 |
| 3 | 12 | 2 | +0.0009 |
| 4 | 13 | 3 | +0.0007 |
| 5 | 6 | 4 | +0.0006 |
| 6 | 22 | 5 | +0.0005 |

**Top CVQ-dominant (CVQ > FVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 12 | 1 | -0.0099 |
| 2 | 16 | 1 | -0.0038 |
| 3 | 4 | 3 | -0.0038 |
| 4 | 19 | 5 | -0.0035 |
| 5 | 18 | 2 | -0.0032 |
| 6 | 19 | 4 | -0.0029 |
| 7 | 15 | 5 | -0.0024 |
| 8 | 20 | 1 | -0.0019 |
| 9 | 16 | 3 | -0.0018 |
| 10 | 11 | 5 | -0.0018 |
| 11 | 12 | 6 | -0.0016 |
| 12 | 13 | 4 | -0.0013 |
| 13 | 4 | 2 | -0.0012 |
| 14 | 21 | 7 | -0.0012 |
| 15 | 17 | 0 | -0.0011 |

## gemma-3-4b-it__normal__K2N30
- Model: `gemma-3-4b-it` / mode: `normal`  
- K=2, N=30, trials=50

### generation token → primacy_round0
max |FVQ-CVQ| = 0.7961, mean |FVQ-CVQ| = 0.02931

**Top FVQ-dominant (FVQ > CVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 23 | 1 | +0.7961 |
| 2 | 23 | 3 | +0.6628 |
| 3 | 25 | 0 | +0.5300 |
| 4 | 23 | 6 | +0.4472 |
| 5 | 23 | 0 | +0.3480 |
| 6 | 21 | 6 | +0.2861 |
| 7 | 24 | 1 | +0.2518 |
| 8 | 22 | 7 | +0.2464 |
| 9 | 23 | 7 | +0.2077 |
| 10 | 29 | 4 | +0.2077 |
| 11 | 27 | 6 | +0.1944 |
| 12 | 24 | 4 | +0.1881 |
| 13 | 22 | 6 | +0.1851 |
| 14 | 30 | 2 | +0.1803 |
| 15 | 26 | 0 | +0.1574 |

**Top CVQ-dominant (CVQ > FVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 17 | 4 | -0.0393 |

### generation token → recency_last_round
max |FVQ-CVQ| = 0.3361, mean |FVQ-CVQ| = 0.01400

**Top FVQ-dominant (FVQ > CVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|

**Top CVQ-dominant (CVQ > FVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 25 | 0 | -0.3361 |
| 2 | 23 | 3 | -0.2929 |
| 3 | 27 | 6 | -0.1795 |
| 4 | 23 | 0 | -0.1342 |
| 5 | 23 | 6 | -0.1262 |
| 6 | 23 | 1 | -0.1254 |
| 7 | 30 | 2 | -0.1248 |
| 8 | 32 | 1 | -0.1212 |
| 9 | 21 | 6 | -0.1131 |
| 10 | 26 | 5 | -0.1112 |
| 11 | 29 | 4 | -0.1083 |
| 12 | 22 | 6 | -0.0996 |
| 13 | 22 | 7 | -0.0983 |
| 14 | 23 | 7 | -0.0927 |
| 15 | 26 | 4 | -0.0923 |

### query-cat token → primacy_round0
max |FVQ-CVQ| = 0.0393, mean |FVQ-CVQ| = 0.00145

**Top FVQ-dominant (FVQ > CVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 17 | 5 | +0.0393 |
| 2 | 28 | 0 | +0.0251 |
| 3 | 8 | 1 | +0.0174 |
| 4 | 12 | 7 | +0.0122 |
| 5 | 13 | 7 | +0.0105 |
| 6 | 11 | 5 | +0.0094 |
| 7 | 7 | 7 | +0.0061 |
| 8 | 17 | 0 | +0.0058 |
| 9 | 11 | 1 | +0.0056 |
| 10 | 22 | 7 | +0.0052 |
| 11 | 30 | 0 | +0.0049 |
| 12 | 6 | 2 | +0.0049 |
| 13 | 19 | 5 | +0.0047 |
| 14 | 22 | 6 | +0.0046 |
| 15 | 23 | 3 | +0.0040 |

**Top CVQ-dominant (CVQ > FVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 8 | 5 | -0.0278 |
| 2 | 11 | 2 | -0.0263 |
| 3 | 10 | 7 | -0.0211 |
| 4 | 14 | 3 | -0.0084 |
| 5 | 25 | 4 | -0.0052 |
| 6 | 22 | 5 | -0.0046 |
| 7 | 9 | 2 | -0.0046 |
| 8 | 23 | 1 | -0.0028 |
| 9 | 18 | 1 | -0.0023 |
| 10 | 11 | 4 | -0.0023 |
| 11 | 8 | 7 | -0.0020 |
| 12 | 17 | 6 | -0.0019 |
| 13 | 13 | 6 | -0.0018 |
| 14 | 30 | 2 | -0.0018 |
| 15 | 27 | 1 | -0.0018 |

### query-cat token → recency_last_round
max |FVQ-CVQ| = 0.0034, mean |FVQ-CVQ| = 0.00036

**Top FVQ-dominant (FVQ > CVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 6 | 0 | +0.0017 |
| 2 | 4 | 7 | +0.0008 |
| 3 | 8 | 3 | +0.0008 |
| 4 | 12 | 2 | +0.0008 |
| 5 | 7 | 6 | +0.0007 |
| 6 | 8 | 1 | +0.0007 |

**Top CVQ-dominant (CVQ > FVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 4 | 2 | -0.0034 |
| 2 | 15 | 5 | -0.0031 |
| 3 | 4 | 3 | -0.0030 |
| 4 | 33 | 6 | -0.0028 |
| 5 | 5 | 0 | -0.0027 |
| 6 | 11 | 5 | -0.0019 |
| 7 | 13 | 4 | -0.0016 |
| 8 | 16 | 2 | -0.0016 |
| 9 | 22 | 0 | -0.0015 |
| 10 | 12 | 1 | -0.0015 |
| 11 | 15 | 1 | -0.0014 |
| 12 | 33 | 4 | -0.0013 |
| 13 | 16 | 1 | -0.0012 |
| 14 | 25 | 5 | -0.0012 |
| 15 | 14 | 1 | -0.0012 |

## gemma-3-4b-it__normal__K5N15
- Model: `gemma-3-4b-it` / mode: `normal`  
- K=5, N=15, trials=50

### generation token → primacy_round0
max |FVQ-CVQ| = 0.7632, mean |FVQ-CVQ| = 0.02903

**Top FVQ-dominant (FVQ > CVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 23 | 3 | +0.7632 |
| 2 | 23 | 1 | +0.7545 |
| 3 | 25 | 0 | +0.5945 |
| 4 | 21 | 6 | +0.4832 |
| 5 | 23 | 0 | +0.4521 |
| 6 | 23 | 6 | +0.3833 |
| 7 | 22 | 7 | +0.3124 |
| 8 | 22 | 5 | +0.2163 |
| 9 | 29 | 4 | +0.1901 |
| 10 | 27 | 6 | +0.1869 |
| 11 | 24 | 1 | +0.1744 |
| 12 | 22 | 6 | +0.1729 |
| 13 | 23 | 7 | +0.1702 |
| 14 | 18 | 2 | +0.1693 |
| 15 | 26 | 4 | +0.1644 |

**Top CVQ-dominant (CVQ > FVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|

### generation token → recency_last_round
max |FVQ-CVQ| = 0.3953, mean |FVQ-CVQ| = 0.01529

**Top FVQ-dominant (FVQ > CVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|

**Top CVQ-dominant (CVQ > FVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 25 | 0 | -0.3953 |
| 2 | 23 | 3 | -0.3312 |
| 3 | 23 | 1 | -0.2090 |
| 4 | 23 | 0 | -0.2065 |
| 5 | 27 | 6 | -0.1720 |
| 6 | 21 | 6 | -0.1427 |
| 7 | 22 | 7 | -0.1371 |
| 8 | 22 | 6 | -0.1358 |
| 9 | 29 | 4 | -0.1353 |
| 10 | 30 | 2 | -0.1156 |
| 11 | 23 | 6 | -0.1145 |
| 12 | 26 | 4 | -0.1131 |
| 13 | 26 | 5 | -0.1014 |
| 14 | 26 | 0 | -0.1000 |
| 15 | 22 | 5 | -0.0974 |

### query-cat token → primacy_round0
max |FVQ-CVQ| = 0.0878, mean |FVQ-CVQ| = 0.00176

**Top FVQ-dominant (FVQ > CVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 17 | 5 | +0.0878 |
| 2 | 17 | 0 | +0.0610 |
| 3 | 19 | 7 | +0.0246 |
| 4 | 11 | 1 | +0.0209 |
| 5 | 17 | 1 | +0.0142 |
| 6 | 17 | 2 | +0.0121 |
| 7 | 19 | 5 | +0.0117 |
| 8 | 19 | 4 | +0.0092 |
| 9 | 23 | 3 | +0.0092 |
| 10 | 15 | 5 | +0.0085 |
| 11 | 22 | 7 | +0.0075 |
| 12 | 28 | 0 | +0.0074 |
| 13 | 11 | 5 | +0.0067 |
| 14 | 22 | 6 | +0.0065 |
| 15 | 13 | 7 | +0.0061 |

**Top CVQ-dominant (CVQ > FVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 8 | 5 | -0.0159 |
| 2 | 11 | 2 | -0.0127 |
| 3 | 14 | 3 | -0.0104 |
| 4 | 10 | 7 | -0.0037 |
| 5 | 16 | 5 | -0.0034 |
| 6 | 25 | 4 | -0.0028 |
| 7 | 18 | 1 | -0.0020 |
| 8 | 22 | 5 | -0.0020 |
| 9 | 30 | 2 | -0.0020 |
| 10 | 9 | 3 | -0.0020 |
| 11 | 16 | 1 | -0.0017 |

### query-cat token → recency_last_round
max |FVQ-CVQ| = 0.0074, mean |FVQ-CVQ| = 0.00046

**Top FVQ-dominant (FVQ > CVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 12 | 7 | +0.0010 |
| 2 | 6 | 0 | +0.0009 |
| 3 | 14 | 3 | +0.0008 |
| 4 | 8 | 1 | +0.0007 |

**Top CVQ-dominant (CVQ > FVQ at this target):**

| rank | layer | head | FVQ-CVQ |
|---:|---:|---:|---:|
| 1 | 11 | 5 | -0.0074 |
| 2 | 15 | 5 | -0.0057 |
| 3 | 13 | 4 | -0.0045 |
| 4 | 12 | 1 | -0.0042 |
| 5 | 16 | 3 | -0.0037 |
| 6 | 4 | 2 | -0.0035 |
| 7 | 16 | 1 | -0.0033 |
| 8 | 2 | 7 | -0.0030 |
| 9 | 18 | 2 | -0.0027 |
| 10 | 19 | 5 | -0.0023 |
| 11 | 17 | 2 | -0.0023 |
| 12 | 22 | 6 | -0.0020 |
| 13 | 19 | 4 | -0.0020 |
| 14 | 15 | 0 | -0.0018 |
| 15 | 17 | 0 | -0.0017 |


## Cross-Qwen-config consistency

Heads in the top-15 of MULTIPLE Qwen configs — robust query-discriminating heads.

### gen → primacy_round0  —  FVQ-dominant heads

| layer | head | appears in | values |
|---:|---:|---|---|
| 31 | 15 | normal, normal__K2N10, normal__K5N20, reversal | +0.752, +0.745, +0.543, +0.244 |
| 31 | 7 | normal, normal__K2N10, normal__K5N20, reversal | +0.584, +0.576, +0.313, +0.130 |
| 31 | 8 | normal, normal__K2N10, normal__K5N20, reversal | +0.562, +0.494, +0.327, +0.144 |
| 30 | 11 | normal, normal__K2N10, normal__K5N20, reversal | +0.556, +0.519, +0.276, +0.102 |
| 31 | 12 | normal, normal__K2N10, normal__K5N20, reversal | +0.460, +0.443, +0.313, +0.133 |
| 29 | 1 | normal, normal__K2N10, normal__K5N20, reversal | +0.376, +0.335, +0.273, +0.147 |
| 27 | 1 | normal, normal__K2N10, normal__K5N20, reversal | +0.344, +0.345, +0.332, +0.153 |
| 29 | 5 | normal, normal__K2N10, normal__K5N20, reversal | +0.281, +0.246, +0.183, +0.102 |
| 32 | 3 | normal, normal__K2N10, normal__K5N20, reversal | +0.281, +0.372, +0.266, +0.147 |
| 29 | 4 | normal, normal__K2N10, normal__K5N20, reversal | +0.255, +0.209, +0.210, +0.125 |
| 32 | 7 | normal, normal__K2N10, normal__K5N20, reversal | +0.251, +0.333, +0.238, +0.128 |
| 26 | 12 | normal, normal__K2N10 | +0.311, +0.275 |
| 30 | 1 | normal, normal__K2N10 | +0.293, +0.265 |
| 31 | 6 | normal, normal__K2N10 | +0.269, +0.250 |
| 31 | 3 | normal, normal__K2N10 | +0.204, +0.203 |
| 27 | 12 | normal__K5N20, reversal | +0.337, +0.194 |
| 27 | 0 | normal__K5N20, reversal | +0.283, +0.154 |
| 28 | 11 | normal__K5N20, reversal | +0.175, +0.148 |

### gen → recency_last_round  —  CVQ-dominant heads

| layer | head | appears in | values |
|---:|---:|---|---|
| 31 | 15 | normal, normal__K2N10, normal__K5N20, reversal | -0.183, -0.384, -0.179, -0.174 |
| 30 | 11 | normal, normal__K2N10, normal__K5N20, reversal | -0.118, -0.208, -0.127, -0.144 |
| 31 | 12 | normal, normal__K2N10, normal__K5N20, reversal | -0.096, -0.276, -0.103, -0.126 |
| 31 | 8 | normal, normal__K2N10, normal__K5N20, reversal | -0.084, -0.226, -0.098, -0.076 |
| 32 | 3 | normal, normal__K2N10, normal__K5N20, reversal | -0.073, -0.177, -0.084, -0.111 |
| 30 | 1 | normal, normal__K2N10, normal__K5N20, reversal | -0.062, -0.174, -0.051, -0.074 |
| 32 | 7 | normal, normal__K2N10, normal__K5N20, reversal | -0.060, -0.107, -0.070, -0.074 |
| 31 | 7 | normal, normal__K2N10, normal__K5N20, reversal | -0.049, -0.172, -0.062, -0.080 |
| 29 | 2 | normal, normal__K2N10, normal__K5N20, reversal | -0.039, -0.126, -0.077, -0.059 |
| 31 | 5 | normal, normal__K2N10, normal__K5N20 | -0.153, -0.182, -0.066 |
| 31 | 6 | normal, normal__K2N10, normal__K5N20 | -0.036, -0.104, -0.041 |
| 29 | 5 | normal__K2N10, normal__K5N20, reversal | -0.111, -0.043, -0.043 |
| 29 | 4 | normal__K2N10, normal__K5N20, reversal | -0.108, -0.043, -0.046 |
| 31 | 3 | normal, normal__K2N10 | -0.061, -0.111 |
| 33 | 9 | normal, reversal | -0.037, -0.046 |

### qcat → primacy_round0  —  FVQ-dominant heads

| layer | head | appears in | values |
|---:|---:|---|---|
| 27 | 1 | normal, normal__K2N10, normal__K5N20, reversal | +0.054, +0.082, +0.065, +0.014 |
| 17 | 10 | normal, normal__K2N10, normal__K5N20, reversal | +0.032, +0.030, +0.027, +0.037 |
| 24 | 5 | normal, normal__K2N10, normal__K5N20, reversal | +0.022, +0.031, +0.023, +0.006 |
| 25 | 8 | normal, normal__K2N10, normal__K5N20, reversal | +0.020, +0.039, +0.027, +0.010 |
| 27 | 0 | normal, normal__K2N10, normal__K5N20 | +0.053, +0.058, +0.010 |
| 32 | 3 | normal, normal__K2N10, normal__K5N20 | +0.018, +0.047, +0.009 |
| 28 | 3 | normal, normal__K2N10 | +0.034, +0.097 |
| 31 | 8 | normal, normal__K2N10 | +0.020, +0.079 |
| 13 | 6 | normal, normal__K5N20 | +0.019, +0.013 |
| 30 | 1 | normal, normal__K2N10 | +0.019, +0.056 |
| 31 | 10 | normal, normal__K5N20 | +0.015, +0.016 |
| 20 | 5 | normal, reversal | +0.013, +0.007 |
| 31 | 7 | normal__K2N10, normal__K5N20 | +0.065, +0.012 |
| 21 | 7 | normal__K5N20, reversal | +0.072, +0.065 |
| 23 | 5 | normal__K5N20, reversal | +0.017, +0.008 |
| 13 | 3 | normal__K5N20, reversal | +0.016, +0.026 |
| 20 | 6 | normal__K5N20, reversal | +0.013, +0.012 |

### qcat → primacy_round0  —  CVQ-dominant heads

| layer | head | appears in | values |
|---:|---:|---|---|
| 25 | 2 | normal, normal__K5N20, reversal | -0.016, -0.012, -0.011 |
| 5 | 15 | normal, normal__K5N20, reversal | -0.013, -0.020, -0.003 |
| 22 | 14 | normal, normal__K2N10 | -0.027, -0.027 |
| 34 | 6 | normal, normal__K2N10 | -0.007, -0.016 |
| 30 | 11 | normal__K5N20, reversal | -0.007, -0.004 |

### qcat → recency_last_round  —  FVQ-dominant heads

| layer | head | appears in | values |
|---:|---:|---|---|
| 11 | 8 | normal, normal__K2N10 | +0.002, +0.004 |

### qcat → recency_last_round  —  CVQ-dominant heads

| layer | head | appears in | values |
|---:|---:|---|---|
| 4 | 0 | normal, normal__K2N10, normal__K5N20, reversal | -0.027, -0.032, -0.016, -0.005 |
| 19 | 12 | normal, normal__K2N10, normal__K5N20, reversal | -0.011, -0.017, -0.017, -0.006 |
| 23 | 1 | normal, normal__K2N10, normal__K5N20, reversal | -0.011, -0.015, -0.015, -0.008 |
| 18 | 3 | normal__K2N10, normal__K5N20, reversal | -0.012, -0.014, -0.004 |
| 21 | 10 | normal__K2N10, normal__K5N20, reversal | -0.010, -0.012, -0.012 |
| 24 | 5 | normal__K2N10, normal__K5N20, reversal | -0.009, -0.007, -0.004 |
| 18 | 13 | normal, normal__K2N10 | -0.007, -0.012 |
| 6 | 1 | normal, normal__K2N10 | -0.005, -0.011 |
| 34 | 12 | normal, normal__K2N10 | -0.004, -0.013 |
| 21 | 1 | normal__K2N10, normal__K5N20 | -0.014, -0.007 |
| 22 | 14 | normal__K5N20, reversal | -0.009, -0.006 |
| 13 | 2 | normal__K5N20, reversal | -0.008, -0.015 |
| 22 | 10 | normal__K5N20, reversal | -0.007, -0.004 |
| 20 | 6 | normal__K5N20, reversal | -0.006, -0.004 |
| 20 | 4 | normal__K5N20, reversal | -0.005, -0.013 |
| 22 | 12 | normal__K5N20, reversal | -0.005, -0.006 |
