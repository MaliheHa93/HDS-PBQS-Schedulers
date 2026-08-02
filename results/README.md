# HDS evaluation results

This directory contains the validated experiment matrices, processed
statistics, quantitative paper claims, and final publication figures for the
HDS study.

## Authoritative raw data

Only these three master CSV files are used for the reported results:

- `raw/deadline_curves_paper_aligned_50seeds.csv` — 24,000 deadline cases;
- `raw/deployment_scaling_paper_aligned.csv` — 800 deployment-scaling cases;
- `raw/bos_scaling_paper_aligned.csv` — 380 controlled BoS decisions.

`RESULTS_VALIDATION.md` documents the experiment grids, hashes, and validation
checks. `VALIDATION_REPORT.json` is the machine-readable release report.

The `*_shards*` and `deployment_recovery_artifacts/` folders preserve run and
merge provenance. They are included for auditability but are not figure inputs;
the three master CSV files above remain authoritative.

## Processed results

The `processed/` directory contains:

- normalized observations;
- paired-success comparisons;
- workflow-macro summaries;
- pointwise and curve-level deadline statistics;
- deployment and BoS scalability summaries.

`PAPER_CLAIMS_V0_7.json` and `PAPER_CLAIMS_V0_7.md` contain the quantitative
values derived from the raw matrices.

## Final figures

Use `figures_paper_aligned/` for the canonical 300-DPI PNG files and
`figures_paper_aligned_pdf/` for vector PDF files.

1. `Figure3_deadline_success` shows global-deadline and SFC-subdeadline success
   for Epigenomics and CyberShake over the complete `0.8--3.0` range.
2. `Figure4_macro_cost_delay` shows workflow-macro provisioning cost and
   end-to-end delay over `2.0--5.0`.
3. `Figure5_transfer_cpu_ram` shows workflow-macro transferred data and the
   separate CPU/RAM purchased-capacity utilization series over `2.0--5.0`.
4. `Figure6_runtime_bos_acceptance` shows BoS-width scheduling runtime and
   first-round accepted-SFC ratio for widths 2--20.

Deployment runtime is not plotted in Figure 6. The paper reports the deployment
study in text: all configurations completed through 60 fog nodes and 1,000 VNFs
without a deployment solver-limit event, and PBQS had the higher median runtime
at the largest deployment.

Every marker represents measured data. No missing point is interpolated,
imputed, or extrapolated.

## Interpretation rules

- Deadline success is calculated from all 50 binary repetitions at each
  displayed deadline factor.
- Cost, delay, transfer, CPU utilization, and RAM utilization use matched cases
  where all four principal configurations meet the global deadline.
- Each workflow contributes equal weight to the macro-average.
- `first_round_accepted_sfc_ratio` is
  `first_round_admitted_sfc_count / first_round_submitted_sfc_count`; it is not
  eventual workflow completion.
- Solver-limit rows in the controlled BoS study remain in the dataset and are
  explicitly reported.
