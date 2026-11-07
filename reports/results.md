# Measured experiment results

24 held-out families; seeds [11, 22, 33]; 192 search calls per method and run.
Offline data-generation costs are excluded from the online comparison. The baseline receives no learned prior.

| Metric | Multi-start | Conditional CVAE |
|---|---:|---:|
| Median objective (per-task median over seeds) | 0.1772 | 0.1624 |
| Feasible runs (search quadrature) | 100.0% | 100.0% |
| Feasible runs (dense independent audit) | 100.0% | 100.0% |
| Median within-task objective std across seeds | 0.0482 | 0.0432 |

Median paired task improvement: **11.39%**.
Task bootstrap 95% interval: -3.14% to 20.66% (5,000 resamples; fixed model seed).
Ratio-of-medians improvement: 8.38%. These are different estimands.
Agent wins on 62.5% of tasks, using task medians.

## All tasks

| Family | f (mm) | Half-FOV (°) | f/# | Multi-start median | CVAE median | Paired improvement |
|---|---:|---:|---:|---:|---:|---:|
| 17 | 43.63 | 2.13 | 6.75 | 0.1769 | 0.1528 | 13.6% |
| 22 | 61.47 | 10.00 | 8.60 | 0.1298 | 0.1415 | -9.0% |
| 25 | 45.08 | 6.73 | 7.22 | 0.1700 | 0.1687 | 0.8% |
| 41 | 42.26 | 4.83 | 8.16 | 0.1839 | 0.1408 | 23.4% |
| 44 | 37.53 | 7.97 | 7.69 | 0.1862 | 0.2180 | -17.1% |
| 57 | 47.89 | 5.13 | 4.73 | 0.2953 | 0.1780 | 39.7% |
| 64 | 74.76 | 8.86 | 5.81 | 0.2576 | 0.1740 | 32.4% |
| 67 | 44.70 | 6.21 | 7.21 | 0.1310 | 0.1403 | -7.1% |
| 68 | 65.15 | 2.78 | 7.20 | 0.1946 | 0.0954 | 51.0% |
| 89 | 73.98 | 9.49 | 5.56 | 0.2825 | 0.2204 | 22.0% |
| 100 | 69.78 | 3.20 | 6.42 | 0.1771 | 0.1286 | 27.4% |
| 101 | 32.41 | 7.83 | 7.66 | 0.1620 | 0.2067 | -27.6% |
| 117 | 55.51 | 2.75 | 7.88 | 0.1094 | 0.1128 | -3.1% |
| 120 | 61.28 | 4.99 | 7.62 | 0.1108 | 0.0921 | 16.8% |
| 144 | 77.99 | 7.11 | 4.73 | 0.2781 | 0.1790 | 35.6% |
| 146 | 31.25 | 4.27 | 8.58 | 0.1775 | 0.1557 | 12.2% |
| 149 | 39.70 | 6.84 | 4.21 | 0.3962 | 0.1876 | 52.6% |
| 171 | 43.43 | 5.61 | 7.97 | 0.1077 | 0.1060 | 1.6% |
| 196 | 74.78 | 8.04 | 7.70 | 0.1773 | 0.1796 | -1.3% |
| 200 | 67.59 | 9.71 | 5.61 | 0.2376 | 0.1951 | 17.9% |
| 229 | 73.79 | 8.72 | 6.40 | 0.1509 | 0.2124 | -40.8% |
| 230 | 25.84 | 3.84 | 6.38 | 0.1744 | 0.1561 | 10.5% |
| 237 | 62.34 | 5.97 | 7.21 | 0.1254 | 0.1416 | -12.9% |
| 247 | 29.20 | 4.80 | 4.15 | 0.2182 | 0.3454 | -58.3% |

## Interpretation

This is a synthetic, fixed-topology proof of concept. Family grouping prevents related prescription variants crossing splits, but does not establish generalization to new topology, glass catalogs, or real manufactured optics.
The three seeds vary online proposals/refinement only. There is one dataset seed and one training seed; the confidence interval does not include model-training uncertainty.
Transmission means the fraction of geometrically valid sampled rays, not optical energy throughput. Feasibility is sampled and is not a manufacturing certification.
Dense audit scores use 1,737 rays per design after optimization; these calls never feed back into the search. Both methods receive the same one-call audit.
No 27% target was imposed or copied from the project brief. The numbers above are computed from benchmark.json.
