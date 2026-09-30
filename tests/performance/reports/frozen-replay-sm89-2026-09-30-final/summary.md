# Frozen replay performance qualification

API/kernel medians use the statistical protocol embedded in
`report.json`. Capture/replay times are five-call Nsight Systems kernel
aggregates.

Peak MiB is tracked application memory: persistent inputs, outputs, caller
workspace and transient LSM workspace; it excludes the CUDA driver context.

| Model | Replay | Execution | API median ms | Kernel median ms | Capture ms | Replay ms | Peak MiB | Regs | Stack B | Local B | Min occupancy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| black_scholes | frozen_exercise | mono | 0.748 | 0.555 | 0.474 | 0.040 | 0.17 | 130 | 512 | 512 | 0.250 |
| black_scholes | frozen_exercise | node_graph | 0.726 | 0.500 | 0.474 | 0.052 | 0.44 | 128 | 512 | 512 | 0.333 |
| black_scholes | frozen_policy | mono | 0.810 | 0.596 | 0.475 | 0.169 | 0.14 | 122 | 512 | 512 | 0.333 |
| black_scholes | frozen_policy | node_graph | 0.737 | 0.522 | 0.474 | 0.078 | 0.41 | 126 | 512 | 512 | 0.333 |
| heston | frozen_exercise | mono | 2.875 | 2.165 | 1.129 | 1.228 | 0.30 | 132 | 576 | 576 | 0.250 |
| heston | frozen_exercise | node_graph | 1.713 | 1.301 | 1.128 | 0.322 | 0.75 | 128 | 576 | 576 | 0.333 |
| heston | frozen_policy | mono | 3.854 | 3.057 | 1.129 | 2.253 | 0.27 | 150 | 576 | 576 | 0.250 |
| heston | frozen_policy | node_graph | 1.612 | 1.268 | 1.130 | 0.283 | 0.72 | 126 | 576 | 576 | 0.333 |
| bates | frozen_exercise | mono | 2.951 | 2.292 | 1.087 | 1.419 | 0.30 | 136 | 720 | 720 | 0.250 |
| bates | frozen_exercise | node_graph | 2.272 | 1.825 | 1.088 | 0.971 | 0.89 | 134 | 800 | 800 | 0.250 |
| bates | frozen_policy | mono | 3.514 | 2.777 | 1.082 | 1.984 | 0.27 | 172 | 624 | 624 | 0.167 |
| bates | frozen_policy | node_graph | 2.090 | 1.601 | 1.080 | 0.675 | 0.86 | 134 | 624 | 624 | 0.250 |
| cir | frozen_exercise | mono | 0.918 | 0.574 | 0.296 | 0.312 | 0.10 | 164 | 544 | 544 | 0.250 |
| cir | frozen_exercise | node_graph | 0.501 | 0.337 | 0.296 | 0.064 | 0.45 | 124 | 544 | 544 | 0.333 |
| cir | frozen_policy | mono | 1.069 | 0.675 | 0.296 | 0.441 | 0.08 | 158 | 544 | 544 | 0.250 |
| cir | frozen_policy | node_graph | 0.514 | 0.347 | 0.294 | 0.077 | 0.44 | 126 | 544 | 544 | 0.333 |
| g2 | frozen_exercise | mono | 0.906 | 0.617 | 0.445 | 0.236 | 0.22 | 246 | 576 | 576 | 0.167 |
| g2 | frozen_exercise | node_graph | 0.721 | 0.497 | 0.446 | 0.105 | 0.72 | 134 | 576 | 576 | 0.250 |
| g2 | frozen_policy | mono | 1.476 | 1.046 | 0.449 | 0.711 | 0.21 | 246 | 576 | 576 | 0.167 |
| g2 | frozen_policy | node_graph | 0.782 | 0.549 | 0.450 | 0.159 | 0.71 | 138 | 576 | 576 | 0.250 |
| g2_plus_plus_svensson | frozen_exercise | mono | 2.502 | 1.842 | 1.253 | 0.841 | 0.22 | 254 | 704 | 704 | 0.167 |
| g2_plus_plus_svensson | frozen_exercise | node_graph | 2.001 | 1.442 | 1.253 | 0.385 | 0.95 | 150 | 640 | 640 | 0.250 |
| g2_plus_plus_svensson | frozen_policy | mono | 3.710 | 2.927 | 1.222 | 2.147 | 0.21 | 254 | 704 | 704 | 0.167 |
| g2_plus_plus_svensson | frozen_policy | node_graph | 2.753 | 2.061 | 1.221 | 1.094 | 0.94 | 150 | 640 | 640 | 0.250 |

Ratios use frozen-exercise/mono as the within-model baseline.

| Model | Replay | Execution | API ratio | Memory ratio |
|---|---|---:|---:|---:|
| black_scholes | frozen_exercise | mono | 1.000 | 1.000 |
| black_scholes | frozen_exercise | node_graph | 0.971 | 2.538 |
| black_scholes | frozen_policy | mono | 1.082 | 0.823 |
| black_scholes | frozen_policy | node_graph | 0.985 | 2.361 |
| heston | frozen_exercise | mono | 1.000 | 1.000 |
| heston | frozen_exercise | node_graph | 0.596 | 2.522 |
| heston | frozen_policy | mono | 1.340 | 0.897 |
| heston | frozen_policy | node_graph | 0.561 | 2.419 |
| bates | frozen_exercise | mono | 1.000 | 1.000 |
| bates | frozen_exercise | node_graph | 0.770 | 2.992 |
| bates | frozen_policy | mono | 1.191 | 0.897 |
| bates | frozen_policy | node_graph | 0.708 | 2.889 |
| cir | frozen_exercise | mono | 1.000 | 1.000 |
| cir | frozen_exercise | node_graph | 0.546 | 4.750 |
| cir | frozen_policy | mono | 1.165 | 0.838 |
| cir | frozen_policy | node_graph | 0.560 | 4.588 |
| g2 | frozen_exercise | mono | 1.000 | 1.000 |
| g2 | frozen_exercise | node_graph | 0.796 | 3.272 |
| g2 | frozen_policy | mono | 1.629 | 0.930 |
| g2 | frozen_policy | node_graph | 0.864 | 3.202 |
| g2_plus_plus_svensson | frozen_exercise | mono | 1.000 | 1.000 |
| g2_plus_plus_svensson | frozen_exercise | node_graph | 0.799 | 4.265 |
| g2_plus_plus_svensson | frozen_policy | mono | 1.483 | 0.931 |
| g2_plus_plus_svensson | frozen_policy | node_graph | 1.100 | 4.196 |
