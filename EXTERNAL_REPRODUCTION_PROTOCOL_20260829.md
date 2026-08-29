# LAM-JEPA Frozen ARC External Reproduction Protocol

## Purpose

This document converts the remaining independent-review gate for the frozen ARC study into an executable reproduction protocol. It does **not** authorize retuning, rescue experiments, reinterpretation of the locked test set, or broader claims about JEPA-style methods.

The reproduction target is the exact evidence-bounded conclusion already frozen in `PAPER_FINALIZATION_20260822.md`: under the retained ARC-Challenge validation protocol, the tested LAM-JEPA configuration did not outperform the capacity-matched supervised baseline and did not satisfy the planner or EMA-target contribution criteria.

## Reproduction questions

An independent reviewer should answer only these questions:

1. Can the documented training/evaluation pipeline be executed from a clean environment using the repository's declared dependencies and instructions?
2. Do the retained five-seed validation aggregates reproduce within a predeclared numerical tolerance?
3. Does the capacity-matched supervised baseline remain at least as strong as the full frozen configuration under the same protocol?
4. Do the planner and EMA-target ablations remain unsupported by the retained validation evidence?
5. Does the repository contain enough provenance to trace reported manuscript claims to code, configuration, and retained outputs?

## Frozen reference values

The reference aggregates are:

| Variant | Mean | Reported spread |
|---|---:|---:|
| full | 0.2549152542 | 0.0129968064 |
| matched supervised | 0.2664406780 | 0.0154600058 |
| no_planner | 0.2501694915 | 0.0129968064 |
| no_target | 0.2616949153 | 0.0203954020 |

A reviewer must not modify hyperparameters, seeds, dataset partitioning, evaluation rules, or model-selection criteria in an attempt to improve these values. Any modified protocol is a new study and must be versioned separately.

## Locked-test rule

The confirmatory ARC test set remains unopened for this failed line. Independent reproduction should use the retained validation protocol only. The locked test must not be used as a rescue set, debugging set, model-selection set, or publication-improvement set.

## Clean-room procedure

### 1. Environment capture

Record before execution:

- operating system and version;
- CPU/GPU model;
- Python/runtime version;
- package-manager version;
- exact repository commit SHA;
- dependency lockfile or resolved dependency snapshot;
- hardware-specific deterministic settings, if any.

Do not silently upgrade dependencies after a failed run. If an environment repair is necessary, document the repair and distinguish infrastructure repair from scientific-protocol change.

### 2. Repository integrity check

Before running experiments, inspect at minimum:

- `CLAIM_LEDGER.md`;
- `MANUSCRIPT_PROVENANCE.md`;
- `METHOD_SOURCE_AUDIT_20260814.md`;
- `EVIDENCE_AUDIT_20260813.md`;
- `PAPER_FINALIZATION_20260822.md`;
- `MANUSCRIPT_DRAFT_NEGATIVE_ARC.md`.

Confirm that the implementation and manuscript refer to the same frozen study and that no local edits are present unless explicitly documented.

### 3. Protocol reconstruction

Extract and record the exact:

- dataset/version and split definitions;
- retained seed list;
- training budget;
- optimizer/scheduler configuration;
- architecture configuration for each variant;
- evaluation metric and aggregation rule;
- checkpoint-selection rule;
- stopping criteria.

If any required item cannot be reconstructed unambiguously, mark the reproduction **blocked by provenance** rather than guessing.

### 4. Execute variants

Run only the frozen variants required to test the paper's retained claims:

- full;
- capacity-matched supervised baseline;
- no_planner;
- no_target.

Do not add exploratory variants inside this reproduction attempt.

### 5. Evidence retention

For every run, retain:

- command or launcher invocation;
- resolved configuration;
- stdout/stderr log;
- seed;
- start/end timestamps;
- exit status;
- primary metric;
- artifact/checkpoint identifier where applicable;
- environment fingerprint.

Store raw per-seed metrics before computing aggregates.

## Acceptance criteria

A reproduction is **scientifically consistent** with the frozen report when all of the following hold:

1. The full configuration does not establish superiority over the capacity-matched supervised baseline under the frozen validation protocol.
2. The planner ablation does not establish a positive planner contribution.
3. The EMA-target ablation does not establish a positive EMA-target contribution.
4. No locked-test information is used.
5. Any numerical discrepancies are documented and are insufficient to reverse the qualitative verdict above.

A reproduction is **numerically close** when the independent aggregate for each reported variant falls within a tolerance declared *before* inspecting the reproduction result. Recommended default: absolute mean difference <= 0.01, unless the original metric definition implies a tighter deterministic expectation.

The qualitative scientific verdict takes precedence over cosmetic agreement in trailing decimals.

## Failure taxonomy

Use exactly one primary classification if reproduction does not complete:

- **INFRASTRUCTURE_FAILURE** — environment/build/runtime failure with no scientific result;
- **PROVENANCE_BLOCKED** — required frozen settings cannot be reconstructed without guessing;
- **NUMERICAL_DRIFT** — study executes but aggregate values differ materially while qualitative verdict is unchanged;
- **VERDICT_CONFLICT** — independent frozen-protocol result reverses a retained qualitative conclusion;
- **PROTOCOL_DEVIATION** — reviewer changed a frozen scientific setting;
- **REPRODUCED** — frozen qualitative verdict independently holds.

A `VERDICT_CONFLICT` should trigger investigation, not selective reruns or claim editing.

## Reviewer report template

Copy this block into the reproduction report:

```text
Commit SHA:
Environment fingerprint:
Protocol reconstructable without guessing: YES / NO
Locked test accessed: YES / NO

Variant results:
- full:
- matched supervised:
- no_planner:
- no_target:

Reference comparison completed: YES / NO
Primary classification:
Qualitative verdict matches frozen report: YES / NO
Numerical tolerance declared before inspection:
Numerical tolerance satisfied: YES / NO
Observed infrastructure/provenance issues:
Protocol deviations, if any:
Reviewer notes:
```

## Claim guard

Successful reproduction supports only the narrow statement that this specific frozen LAM-JEPA ARC configuration produced a reproducible negative/inconclusive result under the documented protocol. It does not support claims of general JEPA failure, universal architectural inferiority, planner uselessness outside this study, or broad reasoning conclusions.

## Completion condition

The external reproduction gate may be marked complete only when an independent reviewer provides the filled report, raw per-seed evidence, environment fingerprint, and a traceable commit SHA. Owner-approved authorship, license/release metadata, citation review, and venue formatting remain separate release gates.