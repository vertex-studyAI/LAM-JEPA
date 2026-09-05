# ARC successor data-freshness audit

**Audit date:** 2026-09-06  
**Repository evidence cutoff:** `dc1db5c0309705c3054687be7924e3255eba6240`  
**Successor protocol:** `protocols/arc_successor_v1_draft.md` / `protocols/arc_successor_v1_draft.json`  
**Status:** COMPLETE AS A REPOSITORY-EVIDENCE AUDIT; DOES NOT AUTHORIZE HELD-OUT EXECUTION

## 1. Purpose

This audit answers the successor protocol's freshness gate without reopening, relabeling, or rescuing the frozen ARC-v5 result. It records which ARC-Challenge surfaces are already contaminated by development or observed outcomes, which project-controlled surface remains locked, and what can and cannot be treated as fresh for the successor study.

The conservative rule used here is stronger than trying to reconstruct a perfect list of individual questions seen during every historical edit: **if a split was used for training, debugging, model selection, validation, ablation analysis, repair analysis, or reported aggregate outcomes, the entire split is treated as development-exposed and is ineligible for a new confirmatory claim.**

This document is about repository-controlled evidence. It cannot prove the absence of undocumented off-repository human access. Any such access discovered later must be added here and may only make the freshness classification stricter.

## 2. Canonical dataset identity

The repository manifest `data/manifests/arc_challenge.json` identifies AI2 ARC-Challenge and records:

| Split | Source rows | SHA-256 | Historical policy |
|---|---:|---|---|
| train | 1,119 | `e488c1587ffdcfc8443f916c53488a95cd471c5790e0746c6bfe4cecf20962cb` | training |
| validation | 299 | `395a5c88d1580d69855fbaee9450270578df1ad5af6259771cd0a42c20e99f05` | development/model selection |
| test | 1,172 | `62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9` | final confirmatory only |

The manifest contains source URLs and expected hashes for all three splits. **A manifest URL/hash is provenance metadata, not evidence that the corresponding rows or labels were opened.** Access classification below therefore relies on retained protocols, experiment metadata, result artifacts, and the claim ledger rather than on the existence of the manifest entry alone.

## 3. Historical access inventory

### 3.1 ARC-Challenge train

**Classification: ACCESSED / DEVELOPMENT-EXPOSED / NOT CONFIRMATORY-FRESH.**

Evidence:

- `protocols/arc_challenge_v1.json`, `v2.json`, and `v3.json` assign the train split to training.
- v3 freezes the exactly-four-choice eligibility rule and retains train examples after that feature-only filter.
- `protocols/arc_challenge_v5_repaired_validation.json` records 1,119 source rows, 1,117 eligible rows, and training on all 1,117 eligible rows.
- The retained full-controls, repair, ablation, and reproduction line therefore depends on ARC-Challenge train data.

Labels were necessarily used for supervised cross-entropy training and for the deterministic shuffled-label negative control. The split is fully contaminated for any new confirmatory claim.

**Successor use permitted:** development/training only, subject to a newly frozen internal train/dev construction created entirely within the already-contaminated development surface.

### 3.2 ARC-Challenge validation

**Classification: ACCESSED / OUTCOMES OBSERVED / NOT CONFIRMATORY-FRESH.**

Evidence:

- Historical protocols explicitly designate validation for development, debugging, and model selection.
- `protocols/arc_challenge_v5_repaired_validation.json` records 299 source rows and 295 eligible rows and declares all 295 eligible validation rows as the evaluation surface.
- `CLAIM_LEDGER.md` retains observed aggregate results, including the five-seed full-model mean validation accuracy, the matched-supervised comparison, planner and target ablation deltas, shuffled-label control performance, and repaired-v5 validation verdict.
- Paper/result artifacts retain per-seed and aggregate validation outcomes and externally reproduced bounded failure-mechanism analysis.

Because labels, per-example outputs, and aggregate outcomes have been inspected repeatedly, this split is contaminated more strongly than a normal development split.

**Successor use permitted:** descriptive/historical analysis only unless the successor protocol explicitly designates a development-only use. It must never be described as a fresh confirmatory surface.

### 3.3 ARC-Challenge test

**Project-controlled classification: LOCKED / NO RETAINED EVIDENCE OF DOWNLOAD OR EVALUATION / RESERVED FROM SUCCESSOR.**

Evidence:

- `protocols/arc_challenge_v3.json` states that no confirmatory test data had been downloaded or evaluated before the v3 pre-test correction.
- `protocols/arc_challenge_v5_repaired_validation.json` says the test split must not be downloaded, opened, evaluated, or used for selection.
- `CLAIM_LEDGER.md` C15/C16 records that the locked confirmatory test was not used to rescue the failed line and remains unopened for that failed hypothesis line.
- Multiple release/reproduction documents repeat the same locked-test boundary.

Important limitation: the manifest and older protocols contain the test split URL, hash, and planned role. That is not row/label access, but it also means the test is an historically reserved surface rather than a newly discovered benchmark.

**Successor use permitted:** none. `protocols/arc_successor_v1_draft.md` explicitly prohibits opening the old locked ARC test as part of the successor. This audit does not change that rule even though repository-controlled evidence says it remains unopened.

## 4. Development exposure relevant to the successor architecture

The following historical choices were made using ARC-Challenge train and/or validation evidence and therefore contaminate those surfaces for confirmatory use:

- four-choice ARC input/output plumbing and eligibility handling;
- supervised learning-rate/training-budget decisions in the frozen ARC line;
- matched-capacity baseline design and parameter accounting;
- planner, EMA-target, memory, and quantization ablation interpretation;
- repaired-v5 trainability work;
- shuffled-label negative-control analysis;
- choice-order robustness and calibration reporting;
- VQ-collapse diagnosis and the subsequent decision to keep VQ out of the successor's primary treatment;
- the successor motivation itself, which is explicitly informed by the negative ARC-v5 result and the external collapse diagnosis.

A complete historical list of every individual prompt/question displayed during every development action is not retained as a separate human-view ledger. The audit therefore applies the conservative whole-split rule: **all ARC-Challenge train and validation examples are considered seen for freshness purposes**, regardless of whether a particular row can be proven to have been manually inspected.

This conservative classification prevents missing prompt-level logs from being exploited as a loophole.

## 5. Successor development surface

The successor may use **ARC-Challenge train only as an already-contaminated development surface**. Before any B0/B1/T1/T2 comparison, create and freeze a new internal train/dev construction with:

1. exact source SHA (`e488...62cb`);
2. exact eligible ordered IDs or row indices;
3. deterministic split-construction code and seed;
4. train/dev index hashes;
5. overlap check proving disjoint internal partitions;
6. a rule that no ARC-Challenge validation or test rows enter the successor's internal development split;
7. one frozen development metric-selection policy shared by B0/B1/T1/T2.

This can support architecture engineering and development-only comparisons. It cannot by itself support a new confirmatory generalization claim.

## 6. Confirmatory surface

**Status: UNRESOLVED HARD BLOCKER.**

This audit does **not** select `CONFIRMATORY_DATASET`.

The following are explicitly ineligible:

- ARC-Challenge train — historically trained on;
- ARC-Challenge validation — outcomes repeatedly observed;
- ARC-Challenge test — remains locked and is explicitly excluded from successor access by the successor protocol.

A candidate confirmatory dataset must therefore be a different immutable evaluation surface and must pass a separate pre-outcome review showing:

- no prior treatment-family tuning or outcome inspection in this project;
- task compatibility with the frozen multiple-choice contract, or a separately frozen adapter with no label-informed design;
- license and redistribution status;
- immutable dataset revision and content hash;
- hidden labels from successor model/hyperparameter development;
- a one-shot evaluation rule;
- no item overlap with the successor development surface where overlap is semantically meaningful;
- exact selection rationale written before treatment outcomes are observed.

Absence of a dataset name from current repository search is not sufficient proof of freshness. Selection requires affirmative provenance evidence.

Until this blocker is resolved, the successor is **development-only**.

## 7. Freshness verdict

| Question | Verdict |
|---|---|
| Is ARC-Challenge train fresh for confirmation? | **No** |
| Is ARC-Challenge validation fresh for confirmation? | **No** |
| Does repository-controlled evidence show ARC-Challenge test evaluation? | **No retained evidence; project ledger says unopened** |
| May the successor use the old ARC test? | **No — explicitly prohibited** |
| Is there currently an authorized successor confirmatory dataset? | **No** |
| May successor engineering proceed on a frozen internal split of ARC train? | **Yes, development-only** |
| Is held-out treatment execution authorized by this audit? | **No** |

## 8. Blocker disposition

For the successor protocol, this document is sufficient to resolve the narrow **`DATA_FRESHNESS_AUDIT` documentation blocker** once independently reviewed, because it conservatively classifies every ARC-Challenge split and specifies the permitted development boundary.

It does **not** resolve `CONFIRMATORY_DATASET`, does not freeze the successor protocol, does not authorize held-out treatment evaluation, and does not alter any frozen ARC-v5 result.

Recommended independent-review question:

> Does the retained repository evidence justify treating all ARC-Challenge train/validation rows as development-exposed, the old ARC test as still locked but prohibited for successor use, and the successor as development-only until a separately evidenced confirmatory dataset is frozen?

Only after that review should the successor authorization manifest mark `DATA_FRESHNESS_AUDIT` as resolved with this file's SHA-256.
