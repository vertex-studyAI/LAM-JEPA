# LAM-JEPA Release Readiness Gate

**Prepared:** 2026-08-17  
**Source revision reviewed:** `cf988f3275a25419995df60ade5931bc0270f9c0`  
**Scientific rule:** the frozen ARC result remains negative/inconclusive. This release gate must not be used to reopen tuning, access the locked test split, or convert the failed hypothesis into a positive claim.

## Current scientific state

The canonical current manuscript is `MANUSCRIPT_DRAFT_NEGATIVE_ARC.md`. `paper.tex` remains a pre-falsification architecture draft and is not an authorized statement of the current evidence.

Repository evidence currently supports a reproducible negative-result package on the repaired ARC-Challenge validation protocol. It does **not** establish LAM-JEPA superiority, planner benefit, target/EMA benefit, hard-VQ benefit, external generalization, confirmatory-test success, educational effectiveness, or `RESEARCH_COMPLETE` status.

## Release gates

| Gate | State | Evidence / required action |
|---|---|---|
| Canonical manuscript identified | PASS | `PAPER_SOURCE_STATUS.md` names `MANUSCRIPT_DRAFT_NEGATIVE_ARC.md` as canonical. |
| Frozen protocol and claim boundary recorded | PASS | `RELEASE_PROVENANCE.md` records the repaired-v5 validation protocol, hashes, row counts, decision rules, and prohibited claims. |
| Reproduction entry points documented | PASS | `RELEASE_PROVENANCE.md`, `REPRODUCE.md`, protocol files, scripts, and CI verification paths identify the executable surface. |
| Negative/inconclusive verdict preserved | PASS | Current repository release/provenance documents explicitly prohibit post-hoc rescue and locked-test access. |
| Owner-approved root license | BLOCKED_OWNER | Owner/maintainers must choose or approve a license and confirm compatibility with third-party code/data obligations. Do not infer a license from repository visibility. |
| Dataset redistribution/licensing review | BLOCKED_OWNER | Confirm what may be redistributed versus referenced by dataset identity/hash only. Existing provenance records dataset identity but grants no redistribution right. |
| Author list and order | BLOCKED_OWNER | No automatic author inference. Owner/research contributors must approve names, order, and contribution basis. |
| `CITATION.cff` | BLOCKED_OWNER | Create only after author list/order, release title/version, and identifiers are approved. |
| Immutable release revision/tag | BLOCKED_OWNER | After all owner-controlled metadata is approved, select the exact source revision, create the release tag, and bind all release artifacts to it. |
| Independent external reproduction/review | BLOCKED_EXTERNAL | A genuinely independent reviewer must reproduce or review the frozen package. Internal checks do not satisfy this gate. |
| Final related-work integration | BLOCKED_REVIEW | Reconcile the canonical manuscript against the verified related-work/originality audit before submission. This must not expand claims beyond evidence. |
| Public release/submission | BLOCKED | Release only when all preceding owner/external gates are closed and the immutable package is cross-linked. |

## Owner decision packet

To close the owner-controlled gates without changing the science, the minimum explicit decisions are:

1. approved root license, plus any exclusions or third-party notices;
2. approved author names and order;
3. approved release title and semantic/version identifier;
4. approved citation metadata and any persistent identifier strategy;
5. approval of the exact immutable source revision to tag;
6. confirmation of which datasets/artifacts may be redistributed versus referenced only;
7. confirmation that the public manuscript remains the negative/inconclusive manuscript and that no locked-test or post-hoc rescue run is authorized.

## External reviewer packet

An independent reviewer should receive an immutable revision/tag and be asked to verify, without retuning:

- environment/install succeeds from the declared package surface;
- frozen protocol JSON matches the documented hashes, row counts, eligibility rule, and decision criteria;
- reproduction commands run as documented;
- retained outputs match the package summaries within declared tolerances;
- no locked ARC test access or post-outcome hyperparameter search occurs;
- manuscript claims stay within the negative/inconclusive evidence boundary;
- any discrepancies are reported rather than repaired by changing the frozen protocol.

## Stop rules

Do not mark this project publication-ready merely because CI is green. Do not generate an author list, license, DOI, citation metadata, or ownership statement from guesswork. Do not run new major experiments to make the paper more attractive. Do not access the locked test split to rescue the failed hypothesis. Do not replace the negative/inconclusive conclusion with marketing language.

The strongest truthful next state is: **owner metadata decisions closed + immutable release candidate created + genuinely independent reproduction/review completed**.