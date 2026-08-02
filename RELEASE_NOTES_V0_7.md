# Release Notes — v0.7.0 July 30 Paper Alignment

Version 0.7.0 replaces v0.6.3 because the July 30 manuscript specifies a
different static phase, deadline rule, admission model, objective, constraint
set, and PBQS priority.

## Algorithm corrections

- Forks terminate chains and joins begin new SFCs while retaining every parent.
- VNF deadlines reserve the longest downstream execution and transfer path,
  including terminal delivery.
- Persistent inter-SFC outputs contribute to storage demand.
- HDS uses lexicographic two-stage optimization: maximize admitted SFCs, then
  minimize cost, communication, and completion time at that admission count.
- Internal VNF deadlines, final delivery, startup, storage, link capacity, and
  sharable sequencing are enforced during admission.
- PBQS uses candidate-specific laxity and defers infeasible ready SFCs.
- Controlled-BoS acceptance and runtime are recorded for the first scheduling
  decision rather than inferred from eventual completion.

## Evaluation replacement

The complete deadline, deployment, and controlled-BoS matrices are regenerated
from v0.7.0. No v0.6.3 observation is copied or relabeled. Figures 3--6 and the
MATLAB mirror consume only the regenerated CSV files.

The exact quantitative claims, retained paired-success sample counts, and
solver-limit totals are generated in `results/PAPER_CLAIMS_V0_7.json` and
`results/PAPER_CLAIMS_V0_7.md`; those files are the source for revising the
abstract and evaluation text.

## Compatibility

The legacy `eta` argument and `topology` deadline-mode alias remain accepted
for command compatibility, but both route to the downstream-reserved rule.
The output schema replaces the old two-weight ambiguity with explicit
`alpha`, `beta`, `gamma`, `omega_u`, and `omega_w` columns.
