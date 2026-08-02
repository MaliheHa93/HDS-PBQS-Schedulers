# v0.7.0 Manuscript Result Claims

All values below are derived from the v0.7.0 raw CSV files.

## Abstract-safe quantitative sentence

At $\kappa=3$, after retaining within-workflow seeds where all four configurations meet the global deadline and averaging the four workflow means equally:

- For HDS, sharing reduces provisioning cost by 2.5\%, increases transferred-data volume by 0.9\%, increases end-to-end delay by 1.4\%, and changes CPU/RAM utilization by +0.19/+0.46 percentage points.
- For PBQS, sharing reduces provisioning cost by 26.4\%, reduces transferred-data volume by 20.6\%, increases end-to-end delay by 20.8\%, and changes CPU/RAM utilization by +5.12/+9.32 percentage points.

## Scalability wording

The evaluation produced complete records through 60 fog nodes, 1,000 VNFs, and a BoS width of 20. It recorded 0 deployment and 45 controlled-BoS rows with at least one MILP solver-limit status; therefore the paper must not claim that no solver limit was reached.

## Deadline-comparison caution

Use the full paired curve-level analysis rather than a selected point to claim HDS superiority. The JSON report records both the largest HDS advantage and disadvantage over the original $\kappa\leq3$ range.
