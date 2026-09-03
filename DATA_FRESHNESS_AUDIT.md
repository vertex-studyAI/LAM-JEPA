# LAM successor v1 — data freshness audit

**Audit date:** 2026-09-03  
**Audit base:** `main` at `ff534a48553442c753249f89f5a6290c73c4add7`  
**Protocol:** `protocols/arc_successor_v1_draft.md`  
**Status:** **PARTIAL / FAIL-CLOSED — CONFIRMATORY SURFACE UNRESOLVED**

This audit exists to close the data-history question required by the LAM successor protocol. It does **not** authorize held-out treatment evaluation, reinterpret ARC-v5, unlock the old ARC test, or certify that the repository contains an exhaustive record of every human-visible example ever inspected outside version control.

## 1. Evidence sources reviewed

Repository evidence used for this audit:

- `RESULTS.md` — frozen ARC-v3 scientific result and split-use statement;
- `data/manifests/arc_challenge.json` — checksum-addressed ARC-Challenge train/validation/test identities and declared split policy;
- `protocols/arc_successor_v1_draft.md` — successor freshness and freeze requirements;
- retained ARC validation/reproducibility workflows and repository search results showing train/validation acquisition paths.

The audit is intentionally conservative: where repository evidence cannot prove historical non-exposure, the state remains unresolved rather than inferred clean.

## 2. ARC split exposure ledger

| Split | Repository-grounded exposure state | Labels / outcomes | Successor eligibility |
|---|---|---|---|
| ARC-Challenge **train** | **ACCESSED**. Frozen studies trained on checksum-addressed train data. | Training labels necessarily used by the frozen supervised objectives. | Development-only candidate; exact successor internal split indices/hashes still must be frozen. |
| ARC-Challenge **validation** | **ACCESSED AND OUTCOMES OBSERVED**. Frozen ARC-v3 uses validation for model comparison and reports aggregate/per-seed scientific outcomes. | Validation labels were used for evaluation; aggregate outcomes and mechanism comparisons are recorded in `RESULTS.md`. | **NOT CLEAN CONFIRMATORY DATA** for the successor. It may not be relabeled as a fresh held-out confirmatory surface. |
| ARC-Challenge **test** | Repository evidence says **LOCKED / NOT DOWNLOADED / NOT EVALUATED** for the failed ARC-v3/v5 hypothesis line. The public manifest records its URL, checksum, and row count. | No repository evidence reviewed here establishes model evaluation or label/outcome inspection. Public identity metadata is known. | Remains **forbidden to use as a rescue test for ARC-v5**. This audit does not automatically authorize it for the successor; a separate owner-approved protocol decision and proof of independence from treatment development would still be required. |

### Frozen ARC identities

From `data/manifests/arc_challenge.json`:

- train: 1,119 source rows, SHA-256 `e488c1587ffdcfc8443f916c53488a95cd471c5790e0746c6bfe4cecf20962cb`;
- validation: 299 source rows, SHA-256 `395a5c88d1580d69855fbaee9450270578df1ad5af6259771cd0a42c20e99f05`;
- test: 1,172 source rows, SHA-256 `62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9`.

The frozen v3 eligible-set ledger reports 1,117 train rows and 295 validation rows after the exactly-four-choice eligibility filter. This difference is preprocessing, not a new fresh data source.

## 3. Outcomes already known to treatment development

The successor must assume the following ARC-v3 validation information is contaminated for confirmatory purposes because it is already recorded in the repository:

- full LAM-JEPA mean validation accuracy: `0.2549152493`;
- capacity-matched supervised mean validation accuracy: `0.2664406780`;
- `no_planner` mean validation accuracy: `0.2501694888`;
- `no_target` mean validation accuracy: `0.2616949081`;
- shuffled-label control mean validation accuracy: `0.2630508393`;
- planner and target-path paired effects and bootstrap intervals;
- the negative/inconclusive scientific verdict;
- the later external-review finding that retained runs collapse to constant classifiers with the information-loss chain localized to the vector-quantizer path.

These observations can motivate a **new** hypothesis and diagnostics, but they cannot be treated as unseen confirmation of that new hypothesis.

## 4. Historical prompt/example exposure

**UNRESOLVED / NOT EXHAUSTIVELY PROVABLE FROM REPOSITORY STATE.**

The current repository proves train/validation split use at the experiment level, but it does not provide a complete immutable ledger of every ARC prompt/example that may have been manually inspected during prior architecture, debugging, manuscript, or hyperparameter work.

Therefore:

1. no ARC validation subset is eligible for clean confirmation;
2. a newly constructed split from previously used ARC train data can be used only as a **development surface**, not as evidence of dataset-level freshness;
3. any proposed external confirmatory dataset must be accompanied by a source/version/hash and an explicit declaration of prior project exposure before it can be frozen;
4. if historical exposure cannot be established conservatively, the successor must be labeled **development-only**.

## 5. Candidate successor development surface

The only repository-supported default at this audit stage is:

- ARC-Challenge train data only;
- exact deterministic internal train/dev construction to be created and checksum-frozen **before** B0/T1 outcome comparison;
- no ARC validation or ARC test data in successor hyperparameter selection;
- all mask/view construction derived deterministically from the frozen successor config and seed.

This audit does not choose the indices. A separate pre-outcome split artifact must record at minimum source SHA, eligible-row policy, row identifiers/indices, split algorithm, split seed, and resulting split hashes.

## 6. Confirmatory surface

`CONFIRMATORY_DATASET` remains **UNRESOLVED**.

A candidate is acceptable only if the successor protocol can prove all of the following before treatment results are inspected:

- exact dataset/version and immutable checksum;
- task compatibility with the frozen multiple-choice input/output contract or a separately frozen adapter;
- license/redistribution status;
- no prior use for LAM successor architecture/hyperparameter selection;
- an exposure declaration covering repository-visible and known manual project use;
- labels hidden from model/hyperparameter development;
- one-shot evaluation rule and release/claim boundary.

If no candidate clears those gates, v1 is **development-only** and no confirmatory claim is permitted.

## 7. ARC test boundary

The old ARC test remains locked for the failed ARC-v5 line. Its non-use is part of the retained scientific integrity record and must not be reversed to rescue that result.

For successor v1, this audit deliberately makes **no authorization decision** about ARC test use. Knowing a public split's identity/hash is not equivalent to having a scientifically clean confirmatory protocol. Any future proposal to use it must be separately preregistered before treatment results and must prove that doing so does not create a post-hoc rescue or leakage path.

## 8. Gate decision

### Closed by this audit

- ARC train historical access is explicit.
- ARC validation historical access and observed outcomes are explicit.
- ARC validation is permanently disqualified as a clean successor confirmatory surface.
- ARC test remains locked/unopened for the failed hypothesis line.
- uncertainty about manual example exposure is made explicit rather than silently treated as freshness.

### Still blocking protocol freeze

- exact successor development split indices/hashes;
- `CONFIRMATORY_DATASET` or explicit development-only downgrade;
- exhaustive-enough exposure declaration for any proposed external confirmatory candidate;
- encoder/checkpoint/tokenizer revision;
- context/target visibility construction;
- numerical success, uncertainty, collapse, parameter-match, and compute thresholds;
- exact environment and reproduce command.

## 9. Authorization state

**HELD-OUT TREATMENT EVALUATION: NOT AUTHORIZED.**

Allowed next work is pre-outcome protocol completion, deterministic data plumbing, B0 implementation on the development-safe surface, leakage tests, and representation-health instrumentation. No successor result claim is created by this audit.
