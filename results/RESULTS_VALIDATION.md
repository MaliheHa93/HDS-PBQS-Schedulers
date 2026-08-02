# v0.7.0 Paper-Aligned Results Validation

## Release status

**PASS, with measured stress warnings retained.**

The three requested raw matrices have their exact experimental grids, unique
case identities, complete HDS/PBQS pairs, paper-aligned scheduler settings,
and no scientific-integrity validation errors. The controlled-BoS study
contains real solver-limit and unfinished-workflow outcomes; these are warnings
to report, not observations to remove.

| Study | Exact design | Rows | Duplicate rows | Validation errors | Solver-limit rows | Unfinished rows |
|---|---|---:|---:|---:|---:|---:|
| Deadline | 4 families × 30 \(\kappa\) values × 50 seeds × 4 configurations | 24,000 | 0 | 0 | 0 | 7,353 |
| Deployment | 10 profiles × 4 families × 5 seeds × 4 configurations | 800 | 0 | 0 | 0 | 0 |
| Controlled BoS | widths 2--20 × 5 seeds × 4 configurations | 380 | 0 | 0 | 45 | 4 |

The deadline unfinished rows are the retained binary outcomes of deliberately
tight deadlines. Completion-dependent metrics remain undefined where
appropriate; they are not replaced by zero. The four unfinished BoS stress
rows are identified below.

## Raw-data integrity

| Raw matrix | SHA-256 |
|---|---|
| `deadline_curves_paper_aligned_50seeds.csv` | `3774a1a9ada5fb3c8bb6513c5c943feafd01eab546ced23de18debcefaf93244` |
| `deployment_scaling_paper_aligned.csv` | `7472d90d02c6aef6c079db0dbb599433c67b29f1f845d298968d40b48ca41423` |
| `bos_scaling_paper_aligned.csv` | `4df7b1072029388b94049f510965a9a6cbe9e40ce77d1a4d990d987545b66f15` |

The deadline and BoS master CSVs are cell-for-cell equal to their independently
flushed shards after identity sorting. The deployment master CSV is the
schema-normalized merge of ten independently validated 80-row profile shards.

Every raw row satisfies the fixed v0.7.0 method schema:

- HDS objective weights:
  \((\alpha,\beta,\gamma)=(0.7,0.2,0.1)\);
- PBQS weights: \((\omega_u,\omega_w)=(0.7,0.3)\);
- downstream-reserved TASTD;
- joint BoS optimization enabled;
- VM reuse enabled;
- idle-only reuse for HDS;
- queue-aware reuse for PBQS;
- non-sharable placement enforced within each submitted BoS.

The controlled-BoS first-round denominator equals the requested width in all
380 rows. The maximum numerical error in
`admitted / submitted == first_round_accepted_sfc_ratio` is
\(8.33\times10^{-17}\).

## D10 recovery provenance

A second worker was detected writing the original D10 file concurrently.
The mixed file was quarantined before any merge. Duplicate identities began
at D10 ordinal 36 (Epigenomics seed 4, PBQS-NonSharable) and continued through
ordinal 43 (Inspiral seed 1, PBQS-Sharable).

The recovery procedure:

1. preserved unique, one-writer ordinals 1--35 and 44--74;
2. discarded all 16 observations belonging to overlap ordinals 36--43;
3. regenerated those eight identities under one writer;
4. measured the six interrupted tail identities, ordinals 75--80;
5. validated the resulting 80-row D10 repair shard before merging.

`deployment_D10_prefix_recovery_manifest.json` records the quarantined input
hash, discarded interval, duplicate multiplicities, retained rows, and missing
tail. `deployment_D10_final_manifest.json` records the validated 80-row shard
hash and confirms that remerging D1--D10 is byte-identical to the authoritative
800-row master CSV. The quarantine and partial recovery files are audit
artifacts only; neither is used by the authoritative deployment CSV.

## Deadline robustness

Global-deadline success and all-SFC-subdeadline success are identical in all
24,000 deadline rows. Therefore, the manuscript must not claim that a gap
between these two criteria was observed in this result set.

The primary scheduler comparison integrates each seed's success curve over
\(\kappa\in[0.8,3]\), then performs paired inference with Holm correction:

| Workflow | Sharing | HDS--PBQS normalized-AUC difference (pp) | 95% CI (pp) | Holm-adjusted \(p\) |
|---|---|---:|---:|---:|
| CyberShake | Non-sharable | +1.98 | [1.32, 2.61] | \(8.30\times10^{-7}\) |
| CyberShake | Sharable | +20.41 | [18.95, 22.00] | \(1.26\times10^{-29}\) |
| Epigenomics | Non-sharable | +1.18 | [0.59, 1.82] | \(5.51\times10^{-4}\) |
| Epigenomics | Sharable | +4.59 | [3.82, 5.41] | \(2.72\times10^{-14}\) |
| Inspiral | Non-sharable | +4.93 | [3.86, 6.00] | \(2.20\times10^{-11}\) |
| Inspiral | Sharable | +12.39 | [11.16, 13.61] | \(2.67\times10^{-24}\) |
| Montage | Non-sharable | +2.25 | [1.68, 2.82] | \(8.98\times10^{-10}\) |
| Montage | Sharable | +25.48 | [23.77, 27.18] | \(1.74\times10^{-31}\) |

For the two Figure 3 workflows, the sustained 90% success thresholds are:

| Workflow | HDS-Sharable \(\kappa_{90}\) | PBQS-Sharable \(\kappa_{90}\) |
|---|---:|---:|
| Epigenomics | 1.15 | 1.30 |
| CyberShake | 1.50 | 2.25 |

At \(\kappa=3\), all four configurations achieve 100% global success in all
four workflow families.

## Paired-success efficiency at \(\kappa=3\)

Within each workflow and seed, only cases where all four configurations meet
the global deadline are retained. The four workflow means are then weighted
equally.

- HDS sharing reduces provisioning cost by 2.5%, increases transferred data
  by 0.9%, increases end-to-end delay by 1.4%, and changes CPU/RAM utilization
  by +0.19/+0.46 percentage points.
- PBQS sharing reduces provisioning cost by 26.4%, reduces transferred data
  by 20.6%, increases end-to-end delay by 20.8%, and changes CPU/RAM
  utilization by +5.12/+9.32 percentage points.

These values replace the earlier 27--28% cost, 56--57% transfer, and about 5%
delay claims. The effects cannot be reported as one common sharing range
because HDS and PBQS now differ materially, including the direction of the
HDS transferred-data effect.

## Deployment scaling

All 800 deployment cases complete and meet both deadline criteria. No
deployment row reaches the 15-second HDS solver limit.

At D10 (60 fog nodes and 1,000 VNFs), median total scheduler runtime and IQR
over four workflows × five seeds are:

| Configuration | Median (s) | IQR (s) | Maximum (s) |
|---|---:|---:|---:|
| HDS-Sharable | 9.54 | [6.26, 13.05] | 14.20 |
| HDS-NonSharable | 9.24 | [6.25, 11.92] | 16.43 |
| PBQS-Sharable | 143.93 | [107.67, 161.68] | 164.60 |
| PBQS-NonSharable | 142.11 | [109.58, 161.64] | 168.48 |

The result is a measured scalability reversal: at 1,000 VNFs, repeated PBQS
candidate evaluation across scheduling rounds dominates total scheduling
runtime. Deployment runtime is described in the evaluation text rather than
shown as a Figure 6 panel.

## Controlled-BoS stress results

All 45 solver-limit rows are HDS-Sharable stage-2 optimization outcomes.
They begin at width 8 and occur at every tested width from 8 through 20.
No HDS-NonSharable or PBQS row reaches a solver limit.

The four unfinished stress workflows are:

- width 18, seed 3, HDS-Sharable: 1 unfinished SFC;
- width 19, seed 3, HDS-Sharable: 1 unfinished SFC;
- width 20, seed 3, HDS-Sharable: 2 unfinished SFCs;
- width 20, seed 3, PBQS-Sharable: 1 unfinished SFC.

At width 20:

| Configuration | Median first-round runtime (s) | Mean first-round accepted-SFC ratio |
|---|---:|---:|
| HDS-Sharable | 15.2073 | 54% |
| HDS-NonSharable | 0.0217 | 25% |
| PBQS-Sharable | 0.00452 | 41% |
| PBQS-NonSharable | 0.00360 | 24% |

The higher HDS-Sharable admission ratio is accompanied by repeated
solver-limit outcomes. The paper must describe width 20 as a measured stress
boundary, not as evidence of limit-free or asymptotic scalability.

## Figure and test gates

| Figure | Pixels | DPI | SHA-256 |
|---|---:|---:|---|
| `Figure3_deadline_success.png` | 3043 × 1041 | 300 | `e63a3dc5e6d64d1adbf3a63364a7458c9f2b41ef74a4c36f65fde743aab02fd1` |
| `Figure4_macro_cost_delay.png` | 3223 × 1071 | 300 | `7641a1d0d49b28906a497cfa85500e5a1e21b0843e7b373638d807a89e2df617` |
| `Figure5_transfer_cpu_ram.png` | 3222 × 1071 | 300 | `117e287b12c6c6d5f3c2c3187a51fe8e121b39ab13157127ac1edc733201cabf` |
| `Figure6_runtime_bos_acceptance.png` | 3219 × 1044 | 300 | `e6bef18561ab6b5cda0ba44d6f86608748ceaa04d84c20268ee9bf45ebd3a152` |

Each manuscript-filename copy is byte-identical to its canonical PNG.
Visual inspection confirms that Figures 3--6 each contain two panels. Every
marker is measured; unsupported points are not interpolated.

The equation-, scheduler-, metric-, filtering-, claim-, and plotting-regression
suite passes all 55 tests. `VALIDATION_REPORT.json` records `"passed": true`, and
all required processed tables are present.
