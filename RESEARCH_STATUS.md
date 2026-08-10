# LAM-JEPA Research Status

**Evidence cutoff:** 10 August 2026  
**Source base:** `05c039fcc02c09c0aa1c1487596dcdd741ee6d51`  
**Classification:** `RESEARCH_ACTIVE / EXECUTION_REPRODUCIBLE / ARC SUPERIORITY HYPOTHESIS UNSUPPORTED / RESEARCH_COMPLETE_FALSE`

## Executive result

LAM-JEPA has a real, reproducible research pipeline with external ARC-Challenge benchmark integration, matched-capacity and pretrained comparison paths, multi-seed validation, controls, retained raw evidence, and independent verification. The scientific outcome on the current ARC line is negative/inconclusive rather than a superiority result.

The current evidence does **not** support claims that LAM-JEPA outperforms strong baselines on ARC, that the planner or target mechanism provides a validated ARC benefit, that the repaired quantizer provides a validated generalization/quantization advantage, or that LAM-JEPA is research-complete.

The locked ARC confirmatory test must not be used to rescue the failed validation hypothesis.

## Evidence ledger

### 1. External-benchmark protocol and eligibility

Protocol work moved from synthetic-only execution checks to ARC-Challenge train/validation with immutable preregistration and audit history. Protocol v3 corrected an observed pre-test choice-cardinality incompatibility using a feature-only eligibility rule before test access:

- train: 1,117 / 1,119 rows eligible;
- validation: 295 / 299 rows eligible;
- excluded rows are retained as evidence;
- source order is preserved;
- the locked ARC test was not used for the failed superiority claim.

### 2. Capacity-matched supervised baseline

The frozen matched-capacity comparison uses the ARC objective's actual gradient-active parameter count rather than total nominal parameters:

- LAM-JEPA gradient-active parameters: `86,372`;
- matched supervised gradient-active parameters: `86,644`;
- ratio: `1.0031491687`.

Under the frozen five-seed validation budget:

- LAM-JEPA accuracy: `0.2549152542 ± 0.0129968064`;
- matched supervised accuracy: `0.2664406780 ± 0.0154600058`;
- paired LAM minus matched: `-0.0115254237 ± 0.0140994131`.

**Verdict:** LAM-JEPA did not beat the capacity-matched supervised baseline on validation.

Evidence merge: `99a384f630fe469094ac5bb8cbff8e6a52191c4a`.

### 3. Required ARC controls and ablations

The frozen full-controls validation used five seeds, 20 epochs, batch size 32, learning rate `0.0003`, model steps 1, all 1,117 eligible train rows and all 295 eligible validation rows.

Verified aggregate results:

- full LAM-JEPA: `0.2549152542 ± 0.0129968064`;
- `no_planner`: `0.2501694915 ± 0.0129968064`;
- `no_target`: `0.2616949153 ± 0.0203954020`;
- deterministic shuffled-label control: `0.2630508475 ± 0.0145011862`.

Paired mechanism effects:

- full minus `no_planner`: `+0.0047457627`, 95% bootstrap CI `[0.0, 0.0142372881]`;
- full minus `no_target`: `-0.0067796610`, 95% bootstrap CI `[-0.0135593220, 0.0]`.

The shuffled-label result remained below the frozen `0.35` failure threshold, but neither required mechanism criterion was met.

**Verdict:** no planner or target-mechanism contribution is supported by the frozen ARC validation evidence.

Evidence merge: `db0de546e604b18def26499ff3f87bb95e632896`.

### 4. Strong pretrained comparator

The benchmark pipeline includes a pinned DeBERTa comparator:

- model: `microsoft/deberta-v3-xsmall`;
- immutable model revision: `14809e4f1fe1895fcba8b258271a940c6ca45ec4`;
- pretrained-baseline runtime is frozen before confirmatory access.

A bounded development comparison was adverse to LAM-JEPA:

- LAM-JEPA: `0.15625`;
- DeBERTa: `0.21875`;
- paired LAM minus DeBERTa: `-0.0625`.

This bounded comparison is characterization evidence, not a standalone final inferiority claim. It does reinforce the strict no-superiority boundary.

Relevant evidence merges include `3b6c22afa5a47ca2134c56834b9ef993753a6ec0` and runtime freeze `0eca0baf5dabf7e1b1dcf158e202471b2aaab5f3`.

### 5. ARC-v5 trainability repair

A train-only causal investigation localized a major failure to the quantized latent path. The narrow opt-in repair `arc-v5-stable-ema-residual-0.03125` restored the predeclared bounded trainability gate and was independently reproduced before repaired validation execution.

Repair merge: `df249086e9171febaa77333a4c62888f35265c40`.

This repair does **not** rescue the original hard-VQ mechanism claim.

### 6. ARC-v5 repaired validation

The repaired validation protocol was frozen before execution and merged as:

- protocol freeze: `168f6beb434610752da4cb2cb6161f15ee026663`;
- validation execution: `18bd608a05bc308056e6279b347ff3ddb2b751be`;
- verifier-only float32 tolerance fix: `05c039fcc02c09c0aa1c1487596dcdd741ee6d51`.

The independent recomputation verdict is:

`VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION`

The repaired validation evidence did not support the predeclared generalization gate or quantization-benefit gate. ARC test remained absent and `research_complete` remained false.

## Supported claims

The repository can defensibly state that:

1. the documented core training/checkpoint/evaluation pipeline executes reproducibly;
2. ARC-Challenge external-benchmark plumbing and evidence retention are implemented;
3. capacity-matched and strong-pretrained comparison paths are implemented;
4. frozen multi-seed ARC validation and required controls were executed;
5. adverse/negative results were retained rather than tuned away;
6. the bounded v5 repair improves trainability under its declared train-only gate;
7. repaired ARC validation remained negative/inconclusive.

## Claims not supported

Do not claim:

- LAM-JEPA superiority on ARC;
- planner benefit on ARC;
- target/EMA mechanism benefit on ARC;
- quantization benefit from repaired validation;
- externally validated educational effectiveness;
- general benchmark superiority;
- AGI or general intelligence capability;
- `RESEARCH_COMPLETE`.

## Scientific stop rule

Do not unlock or use the ARC confirmatory test to rescue the current failed superiority/mechanism hypothesis. Any future architectural repair, new benchmark, or new scientific hypothesis should be versioned separately and preregistered before observing its validation evidence.

Negative results in this repository are first-class research artifacts and should remain visible in manuscripts, technical reports, and portfolio summaries.

## Open release/package gate

Repository-level publication packaging remains separate from the scientific result. Issue #14 tracks owner-approved licensing, citation metadata, provenance, and release packaging. A license or author identity must not be invented.

## Canonical issue outcomes

- Issue #10: external ARC benchmark gate completed with unsupported superiority/mechanism hypothesis.
- Issue #38: repaired ARC-v5 validation completed with negative/inconclusive outcome; confirmatory test remains forbidden for that line.
- Issue #14: publication provenance/citation package remains open.
