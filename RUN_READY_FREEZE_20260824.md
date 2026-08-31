# LAM-JEPA Run-Ready Freeze — 24 August 2026

## Scientific status carried forward

The current ARC package is a reproducible internal negative/inconclusive result. This freeze does **not** attempt to rescue, retune, reinterpret, or unlock a protected evaluation set. The goal is to make the existing result maximally reproducible and externally reviewable.

## Morning target

LAM-JEPA is ready for release/reproduction work when the following are explicit and executable:

- exact source commit and environment identity;
- frozen ARC task/data manifest and checksums where redistribution permits;
- exact model/configuration used for the retained result;
- all baselines used in the retained evaluation;
- seed and budget policy;
- one bounded smoke command that validates the real loader/model/evaluator wiring without changing the result protocol;
- one exact reproduction command/runbook for the frozen experiment using only allowed data;
- raw artifact location/schema and checksum procedure;
- deterministic table/figure generation from those raw artifacts;
- claim ledger mapping manuscript statements to retained evidence;
- independent reproduction packet with expected hashes/outputs and discrepancy-report format;
- release metadata closure for authorship, citation, license and redistribution boundaries.

## Forbidden actions

- no metric switching after inspecting outcomes;
- no seed search or selective seed reporting;
- no locked-test rescue;
- no architecture or hyperparameter retune presented as the same frozen experiment;
- no deletion or replacement of negative/inconclusive evidence;
- no claim of external validation until a genuinely independent reproduction/review is returned.

## Recommended execution order

1. Run repository tests and bounded smoke checks only.
2. Verify the retained raw-result hashes and paper-asset provenance.
3. Rebuild tables/figures from immutable artifacts.
4. Check manuscript numbers against the claim ledger programmatically or line-by-line.
5. Produce an external reproduction command/runbook that requires no undocumented choices.
6. Freeze a discrepancy template covering environment, data hashes, command, stdout/stderr, raw metrics and deviations.
7. Close owner-controlled release metadata.
8. Hand the packet to an independent reproducer/skeptical reviewer.

## Success criterion

Success is **not** a positive ARC result. Success is that an outside researcher can take the packet, execute the documented allowed procedure, and either reproduce the retained negative/inconclusive result or return a precise, evidence-backed discrepancy. That is the scientifically valuable next state for the current version.
