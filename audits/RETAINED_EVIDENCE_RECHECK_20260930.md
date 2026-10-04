# Retained-evidence checks (2026-09-30)

This change hardens the ARC-v5 retained-row verifier. It does not alter models,
training, datasets, seeds, decision rules, or the protected-test boundary.
The frozen negative/inconclusive result is unchanged.

## Defect and narrow repair

The previous v5 verifier compared labels across saved conditions but did not
compare them to the labels in the checksum-locked validation data. It also did
not validate probability finiteness, normalization, class bounds, or consistency
between the saved prediction and probability argmax. A deliberately corrupted
probability vector passed the previous verifier. It is rejected after this fix.

The helper verifies canonical IDs/order and labels, unique canonical IDs,
nonempty row counts, four finite bounded probabilities summing to one, and
prediction/argmax agreement. Float32 normalization tolerance is 1e-5, matching
the existing v3 row checks. Class labels/predictions must be actual integers.
The empty-validation condition is made explicit rather than using chained
inequality. Historical raw evidence is not rewritten.

## Bounded checks performed

- 16 dependency-free unit tests, including malformed and nonfinite fixtures
- Existing v3 full-controls verifier on retained attempt 3: pass
- Existing v5 verifier on retained rerun: pass
- Hardened v5 verifier on the same retained rerun: pass, same scientific verdict
- Deliberately corrupted v5 probability fixture: old verifier accepts; hardened verifier rejects
- Independent raw-row checks against checksum-locked validation labels reproduce
  the previously reported v3 count vectors and v5 means/intervals

These are retained-evidence verification runs, not new training or independent
scientific replication. No protected test data was downloaded or opened.
Runs were restricted to one CPU with a 2 GiB address-space limit. Peak observed
resident memory was below 350 MiB. No experiment workflow was dispatched.

## Recheck commands

With the repository's declared benchmark dependencies installed:

```sh
python -m unittest discover -s tests -p 'test_retained_row_integrity.py' -v
PYTHONPATH=src python scripts/ci/verify_arc_v5_repaired_validation.py \
  --results /path/to/arc-v5-repaired-validation-reexecution.json \
  --protocol protocols/arc_challenge_v5_repaired_validation.json \
  --validation /path/to/arc-challenge-validation.parquet \
  --report /path/to/new-v5-verification.json
```

The confirmatory test split remains sealed. Use only the retained development-validation archive described above.
Do not replace a missing historical archive with a newly trained result under
the old identity. An archive digest is an identity record, not proof that a
remote copy is presently downloadable.

## Remaining evidence gates

The matched-supervised archive (historical artifact 9003785715, SHA-256
13268856e9be2d9da91addc7935a9cd7bdc4bc7b0a59e527905c6de6fa0f87cc)
was not recovered in this recheck. Its per-seed/per-example values must not be
inferred from aggregate means. Retained v3/v5 prediction packages do not contain
ARC model checkpoints or full historical dependency lockfiles. The external
collapse probe's exact intervention, tolerances and spread definition remain
separate documentation gates. Owner-controlled publication metadata/licensing
and final submission approval remain required.
