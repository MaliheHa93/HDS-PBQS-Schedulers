# v0.7.0 Validation Protocol

## Hard gates

1. The July 30 manuscript equations map to an implementation function and a
   direct regression test.
2. Every raw matrix has the exact requested case grid, one row per key, and
   the paper-aligned weight/reuse schema.
3. Binary success curves use all 50 matched seeds.
4. Cost, delay, transfer, CPU, and RAM curves use only workflow/seed cases
   where all four configurations succeed globally.
5. A macro point requires at least five retained seeds in every workflow and
   weights the four workflow means equally.
6. Deployment runtime summaries include every row, including solver-limit
   outcomes, even though deployment runtime is reported in text rather than
   plotted.
7. Controlled-BoS runtime and acceptance use only the first submitted
   ready-set decision. The denominator must equal the requested BoS width.
8. Python and MATLAB generate the same four panel structures without
   interpolation or invented observations.
9. Every solver-limit event remains in the raw data and manuscript report.
10. All old v0.6.3 result claims and plots are removed from the release.

## Required matrices

| Study | Exact grid | Rows |
|---|---|---:|
| Deadline | 4 workflows × 30 deadline factors × 50 seeds × 4 configurations | 24,000 |
| Deployment | 10 profiles × 4 workflows × 5 seeds × 4 configurations | 800 |
| Controlled BoS | widths 2--20 × 5 seeds × 4 configurations | 380 |

## Figure gate

- Figure 3: Epigenomics and CyberShake deadline-success panels.
- Figure 4: workflow-weighted macro cost and delay.
- Figure 5: transferred data and combined CPU/RAM utilization.
- Figure 6: first-round BoS runtime and first-round accepted-SFC ratio.
