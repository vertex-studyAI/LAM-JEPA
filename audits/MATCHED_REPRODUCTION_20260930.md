# New bounded matched-baseline reproduction (2026-09-30)

After the retained-row verifier recheck, a separately recorded reproduction
was executed to replace missing *accessible evidence*, not to reconstruct the
original byte-identified archive or seek a favorable result.

## Frozen inputs

Scientific source: `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`. All 259 retained
source/matched-runner blobs inspected match the frozen source tree. Five seeds
1–5, 20 epochs, batch 32, learning rate 0.0003, model steps 1, all 1,117 eligible
train and 295 validation rows, and parameter-match tolerance 0.01 were unchanged.
Train and validation parquet SHA-256 checks passed against the frozen protocol.
Only these two split paths were permitted by the loader wrapper; protected
ARC test access was forbidden. No test data was accessed.

The original job log supplied historical scientific package versions: Torch
2.13.0+cpu, NumPy 2.4.6, pandas 3.0.5, PyArrow 25.0.0 and tqdm 4.70.0. Those
versions were installed for this reproduction. Python 3.12 was used rather than
historical 3.11.15; some ancillary dependency versions also differ. This is not
a byte-identical historical environment recreation.

Before the full run, the exact command, environment and source/data identifiers
were frozen in a separate plan with SHA-256
`25a710d2040706ca3521c37efe2446b367f16b53d286b76e426bc0a670390678`.
The unchanged native runner was instrumented only to save returned model states;
serialization was checked not to advance the Torch RNG.

## Results and verification

A training-only one-epoch pilot passed. The full five-seed matched + LAM run
completed in 102.94 seconds using one CPU and 472.76 MiB peak sampled resident
memory. A first pilot attempt failed at a 2 GiB virtual-address-space limit;
that failure was preserved, and subsequent runs monitored actual resident
memory with a 2 GiB ceiling. No hyperparameter or scientific change followed.

| Model | Newly measured correct counts / 295, seeds 1–5 | Exact mean |
|---|---|---|
| LAM | 71, 78, 78, 71, 78 | 0.2549152542372881 |
| Matched supervised | 72, 83, 82, 80, 76 | 0.2664406779661017 |

Paired LAM-minus-matched mean: -0.011525423728813562. The native retained-row
verifier passes. Mean and sample-SD summaries reproduce the historical job-log
values within the existing 1e-6 tolerance (the exact fractions agree).
All ten newly saved inference checkpoints reload to exactly identical saved
per-example probabilities in this environment. Their optimizer states were
not retained by the original training functions; do not call them resumable
training checkpoints.

## Boundaries and remaining gaps

This new evidence is separately retained with its own predictions, checkpoints,
plan, hashes and execution records. It is not the missing original matched
archive and is not a new external scientific replication. No frozen paper
number or negative/inconclusive conclusion is changed. No protected test,
paid compute, workflow dispatch, submission, merge or release was performed.

The old matched archive and historical ARC checkpoints remain unavailable.
The external collapse probe's precise intervention/tolerances, historical
complete environment lock, final licensing/authorship decisions, and final
submission approval remain distinct gates. Private evidence archives are not
added to this public branch.
