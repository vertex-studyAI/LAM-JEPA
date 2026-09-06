# ARC successor v1 — retained-evidence schema resolution

**Status:** PRE-OUTCOME CONTROL-PLANE CLOSURE ONLY  
**Scientific outcome generated:** no  
**Execution authorized:** no  
**Outcome access authorized:** no

This artifact closes one reproducibility/submission-preparation gap in the LAM successor protocol: the run-package and claim-ledger structure is now specified before any successor outcome access.

The machine-readable contract is `protocols/arc_successor_v1_evidence_schema.json`, with a blank fail-closed receipt template at `protocols/arc_successor_v1_run_receipt_template.json`. The verifier `tools/verify_arc_successor_evidence_schema.py` binds the contract to the already-frozen decision-rule, context/target, and encoder artifacts by exact SHA-256 and rejects drift in seeds, systems, authorization policy, retained-artifact roles, seed-retention rules, rescue prohibitions, or the old locked-test boundary.

## What the future retained package must contain

Every scientifically admissible system/seed run must retain a source SHA, the frozen protocol hashes, an independently approved authorization state, environment and host receipts, dataset/split/checkpoint identities, parameter and compute accounting, raw predictions, metric JSON, representation and optimizer diagnostics, stdout/stderr, machine-readable configuration, and cryptographic artifact identities.

Every reported number must have a claim-ledger entry pointing back to a retained artifact and JSON location. Failed seeds may not disappear from the package, secondary metrics may not rescue a failed primary gate, and a complete result package cannot be marked complete unless all authorization fields are true.

## Explicit non-resolution

This control-plane change does **not** resolve or weaken any remaining successor scientific blocker. In particular it does not supply an independent data-freshness review, approve a confirmatory dataset, create a dataset byte receipt, set collapse thresholds, set parameter-match tolerance, set a compute ratio, freeze implementation identities, or provide the final exact scientific reproduce command/environment lock.

The blank template intentionally keeps confirmatory dataset identity, future execution source SHA, environment lock, run artifacts, scientific claim ledger, and completion receipt unset. It also keeps `execution_authorized=false` and `outcome_access_authorized=false`.

No model is downloaded or trained by this control plane. No old ARC-v5 locked confirmatory test is opened. No successor accuracy, significance, mechanism, external-validation, or superiority claim is created.

## Submission use

Before results, the strongest supported statement is only that a cryptographically bound retained-evidence and claim-ledger schema was frozen pre-outcome. If the successor is later authorized and executed, the same contract should be used to fail closed on incomplete provenance before any manuscript number is treated as submission evidence.
