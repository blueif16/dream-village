# Final results

## LoCoMo (categories 1-4; 95% CI = cluster bootstrap over conversations)

| system | n | all-10 acc [95% CI] | held-out 6-10 acc [95% CI] | c1 acc (n) | c2 acc (n) | c3 acc (n) | c4 acc (n) | paper | note |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| full-context | 1540 | 0.846 [0.820, 0.871] | 0.841 [0.800, 0.871] | 0.780 (282) | 0.844 (321) | 0.615 (96) | 0.895 (841) | 0.825 |  |
| rag | 1540 | 0.657 [0.638, 0.677] | 0.654 [0.622, 0.683] | 0.479 (282) | 0.598 (321) | 0.510 (96) | 0.756 (841) | 0.633 |  |
| memora-p | 1537 | 0.826 [0.808, 0.844] | 0.835 [0.816, 0.852] | 0.745 (282) | 0.773 (321) | 0.573 (96) | 0.903 (838) | 0.863 |  |
| memora-s | 1540 | 0.816 [0.792, 0.838] | 0.819 [0.802, 0.835] | 0.727 (282) | 0.760 (321) | 0.583 (96) | 0.893 (841) | 0.849 |  |
| memora-noabs | 1540 | 0.836 [0.812, 0.861] | 0.838 [0.820, 0.852] | 0.755 (282) | 0.798 (321) | 0.615 (96) | 0.904 (841) | 0.653 | no-abstraction ablation: values embedded instead of abstraction index; same t |
| memora-j-8k | 1540 | 0.777 [0.750, 0.804] | 0.788 [0.774, 0.805] | 0.691 (282) | 0.779 (321) | 0.427 (96) | 0.845 (841) | - | Memora-J A4, 8k recall budget; Jev decision model; thresholds untuned |
| memora-j-2k (diagnostic) | 1540 | 0.137 [0.106, 0.175] | 0.127 [0.088, 0.172] | 0.156 (282) | 0.028 (321) | 0.323 (96) | 0.151 (841) | - | 2k budget: profile consumed budget; diagnostic only |
| memora-j-16k | 1540 | 0.782 [0.755, 0.809] | 0.793 [0.776, 0.817] | 0.699 (282) | 0.779 (321) | 0.438 (96) | 0.850 (841) | - | Memora-J A4, 16k recall budget |

### LoCoMo paired differences (system - reference) on shared questions

| reference | system | diff [95% paired cluster CI] | held-out 6-10 diff | n shared |
|---|---|---|---|---:|
| memora-p | full-context | +0.020 [+0.002, +0.038] | +0.006 [-0.020, +0.034] | 1537 |
| memora-p | rag | -0.169 [-0.191, -0.146] | -0.181 [-0.210, -0.148] | 1537 |
| memora-p | memora-s | -0.011 [-0.026, +0.005] | -0.017 [-0.040, +0.010] | 1537 |
| memora-p | memora-noabs | +0.010 [-0.007, +0.029] | +0.003 [-0.020, +0.028] | 1537 |
| memora-p | memora-j-8k | -0.049 [-0.067, -0.029] | -0.048 [-0.075, -0.018] | 1537 |
| memora-p | memora-j-2k (diagnostic) | -0.690 [-0.729, -0.645] | -0.708 [-0.759, -0.654] | 1537 |
| memora-p | memora-j-16k | -0.045 [-0.064, -0.023] | -0.043 [-0.075, -0.010] | 1537 |
| full-context | rag | -0.189 [-0.213, -0.167] | -0.186 [-0.229, -0.153] | 1540 |
| full-context | memora-p | -0.020 [-0.038, -0.002] | -0.006 [-0.034, +0.020] | 1537 |
| full-context | memora-s | -0.031 [-0.047, -0.010] | -0.022 [-0.046, +0.012] | 1540 |
| full-context | memora-noabs | -0.010 [-0.026, +0.015] | -0.003 [-0.027, +0.042] | 1540 |
| full-context | memora-j-8k | -0.069 [-0.089, -0.042] | -0.053 [-0.083, -0.009] | 1540 |
| full-context | memora-j-2k (diagnostic) | -0.709 [-0.751, -0.662] | -0.713 [-0.781, -0.640] | 1540 |
| full-context | memora-j-16k | -0.064 [-0.085, -0.038] | -0.048 [-0.082, -0.006] | 1540 |

## LongMemEval subset-24 (95% CI = bootstrap with each question its own cluster)

| system | n | accuracy [95% CI] | per question_type (correct/n) | note |
|---|---:|---|---|---|
| full-context | 24 | 0.917 [0.792, 1.000] | knowledge-update 4/4, multi-session 3/4, single-session-assistant 4/4, single-session-preference 3/4, single-session-user 4/4, temporal-reasoning 4/4 |  |
| full-context (all 500) | 500 | 0.878 [0.848, 0.906] | not restricted to subset |  |

### LongMemEval paired differences (system - reference) on shared subset questions

| reference | system | diff [95% paired CI] | n shared |
|---|---|---|---:|
