# July 30 Manuscript-to-Code Alignment

The July 30 eight-page manuscript is the source of truth for v0.7.0. The
previous v0.6.3 implementation and its results are retained only in Git
history; they must not be cited as evidence for the current algorithm.

| Manuscript requirement | v0.7.0 implementation | Direct regression evidence |
|---|---|---|
| Eq. (6): a linear SFC edge exists only when the parent has one child and the child has one parent | `src/hds/sfc_builder.py` ends chains at forks and starts a new SFC at joins while retaining every parent edge | `test_diamond_join_starts_new_sfc_and_retains_all_parents`; `test_join_never_merges_with_either_parent` |
| Eqs. (7)--(9): subdeadlines reserve all remaining downstream work and final delivery | `src/hds/tastd.py` calculates reference earliest finishes and longest remaining execution/transfer time, including terminal output | `test_tastd_follows_paper_equations_at_a_join`; `test_terminal_output_time_is_reserved_at_sink` |
| Persistent cross-SFC outputs contribute to storage | `src/hds/sfc_builder.py` uses maximum temporary VNF storage plus externally retained output | `test_sfc_storage_includes_retained_external_outputs` |
| Stage 1 maximizes admitted SFCs | `src/hds/milp.py` introduces admission variables and solves the cardinality problem first | `test_adaptive_bos_fallback_preserves_feasible_subset` |
| Stage 2 minimizes cost, communication, and completion at fixed admission | `src/hds/milp.py` uses \((\alpha,\beta,\gamma)=(0.7,0.2,0.1)\) after fixing the Stage-1 count | `test_scheduler_weight_constraints_match_paper`; MILP feasibility tests |
| Internal VNF deadlines, terminal delivery, startup, storage, and link capacity are constraints | `src/hds/milp.py`, `src/hds/timing.py`, and `src/hds/network_admission.py` model and check each constraint | `test_internal_vnf_deadline_is_enforced`; `test_terminal_output_deadline_is_enforced`; `test_storage_capacity_is_enforced`; `test_link_capacity_is_applied_to_hds_and_pbqs_paths` |
| HDS reuses only an idle paid VM | `src/hds/hds_scheduler.py` defaults to `idle_only` | `test_hds_idle_reuse_does_not_queue_second_ready_sfc` |
| Eq. (28): a NonSharable candidate receives at most one SFC from the current BoS; an idle paid VM remains reusable in a later round | `src/hds/milp.py` enforces the per-BoS sum and both schedulers prevent same-round duplicate reuse | `test_nonsharable_uses_distinct_candidates`; `test_nonsharable_reuses_idle_vm_but_not_twice_in_one_bos` |
| PBQS evaluates candidate-specific laxity and may queue on a reusable VM | `src/hds/pbqs_scheduler.py` evaluates each candidate with queue/startup, processing, internal deadline, transfer, and terminal-delivery effects | `test_hds_idle_and_pbqs_queue_aware_reuse_are_distinct`; `test_pbqs_defers_candidate_with_no_finite_laxity` |
| BoS acceptance means admitted from the submitted ready set | `src/hds/simulator.py` records first-round submitted, admitted, ratio, and runtime separately from eventual completion | `test_controlled_bos_records_first_round_admission` |

## Result invalidation

The changes alter SFC partitions, readiness, transfer paths, storage demand,
deadlines, admission, placement, VM reuse, and completion times. Therefore:

1. no v0.6.3 row may be copied into the v0.7.0 raw matrices;
2. Figures 3--6 must be regenerated from the v0.7.0 CSV files;
3. abstract percentages must be calculated again using the documented
   paired-success, workflow-first macro procedure; and
4. solver-limit statements must be based on the stored Stage-1 and Stage-2
   statuses, not inferred from whether a row was written.

## Figure definitions

- Figure 3: all 50 binary repetitions for Epigenomics and CyberShake through
  \(\kappa=3\).
- Figures 4--5: within each workflow and deadline factor, retain seeds where
  all four configurations succeed globally; require at least five retained
  seeds for every workflow; average the four workflow means with equal weight.
- Figure 6(a): median and IQR of first-round scheduler time by controlled BoS
  width.
- Figure 6(b): mean and 95% interval of
  `first_round_admitted_sfc_count / first_round_submitted_sfc_count`.
- Deployment runtime is summarized in the evaluation text rather than plotted:
  the study covers 5--60 fog nodes and 100--1,000 VNFs.
