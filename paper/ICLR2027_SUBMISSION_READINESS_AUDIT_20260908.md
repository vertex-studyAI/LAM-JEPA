# LAM-JEPA ICLR 2027 submission-readiness audit

**Audit date:** 2026-09-08  
**Base scientific source:** `main@41867c6801242c58f52703437655cc925cc7a56c`  
**Scope:** manuscript/evidence reconciliation and venue-readiness only  
**Scientific state:** **NEGATIVE / INCONCLUSIVE — UNCHANGED**

This audit does **not** run a model, change a seed, split, threshold, metric, retained result, or protocol, and does not authorize the separate successor study. The locked historical ARC confirmatory test remains unopened and prohibited as a rescue path. The separate OpenBookQA successor remains pre-outcome and must not be used to repair this paper after the fact.

## 1. Submission candidate boundary

The only scientifically supportable ICLR-facing paper on the current retained evidence is a bounded failure-mechanism / falsification-first report of the historical ARC-Challenge line. It must not be presented as architecture superiority, a general JEPA result, a general vector-quantization result, or a repaired positive successor.

The current manuscript already preserves the central retained conclusions:

- five frozen validation seeds do not satisfy the preregistered H1 superiority gate;
- planner and EMA-target contribution gates H2/H3 are not satisfied;
- the shuffled-label control is a validity control only;
- one genuinely external frozen-protocol rerun/review reproduced the retained headline metrics;
- the reviewed retained runs collapse to constant classifiers through a single-code VQ bottleneck;
- disabling quantization in the bounded external diagnostic restores input dependence but does **not** establish above-chance ARC performance;
- the historical locked confirmatory ARC test stays closed;
- the supported state remains negative/inconclusive.

Those boundaries are consistent with `CLAIM_LEDGER.md`, `paper/EXTERNAL_REVIEW_VQ_COLLAPSE_CORRECTION_20260831.md`, and the current `paper/main.tex`.

## 2. External-review correction reconciliation

### Reflected in current `paper/main.tex`

- [x] exact H1/H2/H3 preregistered thresholds are stated rather than summarized vaguely;
- [x] one external rerun/review is described as one bounded external rerun, not broad replication or peer review;
- [x] constant-classifier collapse is surfaced as part of the scientific result;
- [x] the measured chain `distinct pre-quantizer latents -> one VQ code -> constant post-quantizer representation -> constant predictions` is represented in bounded form;
- [x] the quantizer-off diagnostic is explicitly limited to restored input dependence and does not claim above-chance task performance;
- [x] the bounded pretrained comparison is not used as a broad superiority/inferiority claim;
- [x] architecture wording is constrained: the tested path is not described as a Transformer or canonical I-JEPA context-to-distinct-target task;
- [x] the locked confirmatory-test stop rule remains explicit.

### Evidence located in-repository during this audit

A follow-up source check found an exact implementation for the gradient-capacity accounting in `scripts/ci/measure_arc_gradient_capacity.py`. It defines the reported gradient-active capacity as:

> `sum numel(parameter) where parameter.grad is not None after exact _lam_arc_loss backward`

The script separately records total parameters, `requires_grad` parameters, gradient-active parameters, and gradient-inactive trainable parameters. This is materially narrower than saying “all trainable parameters” and, importantly, it does **not** mean every scalar had a nonzero numerical gradient on every batch. The manuscript should use this exact accounting definition and bind the published 86,372 / 86,644 values to retained generated reports before submission.

The retained external-review correction at `paper/EXTERNAL_REVIEW_VQ_COLLAPSE_CORRECTION_20260831.md` also contains the already-reviewed truncation observation: roughly 2% of retained examples exceed the 96-whitespace-token limit, and some validation rows lose the fourth-answer marker under truncation. That observation may be disclosed without rerunning the model, but the manuscript sentence must cite/bind to that retained review artifact rather than re-estimating from memory.

### Remaining paper-facing blockers

The unresolved work is now split into **missing retained numbers**, **evidence-located manuscript disclosures**, and **venue-format work** so that easy editorial fixes are not confused with new scientific analysis.

#### A. Missing retained quantitative evidence — hard blockers

- [ ] **Seed-level integer deltas / counts:** pair the five-seed bootstrap summaries with retained per-seed integer correct counts / differences. Because the reviewed conditions collapse to constant classifiers, these counts are important context for interpreting small mechanism deltas. Do not infer counts from rounded means.
- [ ] **Hash-collision quantification:** the manuscript states that text is hashed into a 256-ID vocabulary, while the external review explicitly asks for a collision statistic. Do not invent a rate. Either locate an already-retained deterministic audit or generate a separately reviewable deterministic diagnostic from the frozen eligible text surface without opening the locked test set.

#### B. Evidence located; manuscript-only disclosure still required

- [ ] **Five-seed uncertainty limitation:** `paper/main.tex` already says that five seeds quantify only a narrow protocol, but the bootstrap sentence should be explicit: uncertainty from five fixed seeds is a finite-seed summary and is not an asymptotic or population-level guarantee.
- [ ] **Truncation disclosure:** add the retained external-review observation described above and bind it to `paper/EXTERNAL_REVIEW_VQ_COLLAPSE_CORRECTION_20260831.md`.
- [ ] **ARC naming disambiguation:** at first dataset use, spell out **AI2 Reasoning Challenge (ARC-Challenge)** so it cannot be confused with the Abstraction and Reasoning Corpus.
- [ ] **Gradient-active parameter definition:** replace the manuscript’s shorthand with the exact accounting definition from `scripts/ci/measure_arc_gradient_capacity.py`; retain a provenance link/report for each published count.

These four items require no change to the frozen scientific protocol and no outcome access. They are editorial/provenance work only.

Until Sections A and B are closed from retained evidence, the paper is **scientifically coherent but not submission-ready**.

## 3. Current ICLR 2027 venue gate

Official ICLR 2027 requirements re-checked on 2026-09-08:

- abstract deadline: **2026-09-18 11:59 PM AoE**;
- full-paper deadline: **2026-09-25 11:59 PM AoE**;
- abstract must be genuine/informative; placeholder or duplicate abstracts are removed;
- no new authors may be added after the abstract deadline;
- all authors need current OpenReview profiles;
- submission is **double blind** and author identity in the paper/supplement can cause desk rejection;
- main text is **9 pages maximum** at initial submission; references are excluded from the page limit; appendices may follow the references;
- the official **ICLR 2027 LaTeX style** is required;
- an **AI use statement is mandatory** in the paper and does not count toward the page limit;
- a reproducibility statement is strongly recommended.

Primary sources:
- https://iclr.cc/Conferences/2027/CallForPapers
- https://iclr.cc/Conferences/2027/AuthorGuidelines
- https://iclr.cc/Conferences/2027/AIPolicyForAuthors

## 4. Venue-format blockers on the current manuscript

The authoritative `paper/main.tex` is presently an `11pt article` with `geometry`, not the ICLR 2027 style package. It also contains owner-placeholder author metadata rather than a final anonymous ICLR author block. Therefore:

- [ ] create a **separate ICLR 2027 submission source** from the evidence-bound manuscript; do not mutate scientific values while reformatting;
- [ ] use the official ICLR 2027 style and verify main-text page count <= 9;
- [ ] replace owner-placeholder metadata with the venue-correct anonymous submission form;
- [ ] add the required AI use statement describing actual tool use, with authors retaining responsibility for the final content;
- [ ] add/reconcile the recommended reproducibility statement;
- [ ] run a final anonymity scrub over main text, appendix, supplemental material, repository links, artifact names, acknowledgments, and metadata;
- [ ] freeze the final author list before the Sep 18 abstract deadline.

## 5. Safe work that can proceed without new experiments

The following work is explicitly safe under the current research freeze because it does not alter outcomes or access the locked confirmatory set:

1. Patch the four manuscript-only disclosures in Section 2B using the retained sources already identified.
2. Locate and checksum the retained generated gradient-capacity reports that substantiate the manuscript’s 86,372 / 86,644 figures; if a report is absent, treat the numerical values as unverified rather than regenerating silently.
3. Materialize the seed-level integer-count table only from retained result artifacts, with source paths / hashes adjacent to every row.
4. Add a deterministic hash-collision diagnostic only over the already-eligible frozen train/validation text surface; keep it as a separately reviewable artifact and do not touch the locked ARC test.
5. Create a venue-format-only ICLR source that imports/copies the evidence-bound text without changing scientific values.
6. Add the mandatory AI-use disclosure based on actual author/tool usage and run an anonymity scrub before any submission build.

None of these steps authorizes a new training run, threshold change, new seed, successor outcome, or historical test access.

## 6. Go / no-go rule

**GO for an ICLR abstract only if, before the abstract deadline, the submission source can honestly describe the existing frozen negative/collapse result without relying on an unfinished successor experiment.**

**NO-GO** if submission would require any of the following:

- opening the locked historical ARC confirmatory test;
- using OpenBookQA/successor outcomes to rescue the historical claim;
- changing H1/H2/H3 thresholds, seeds, split, metric, or matched-comparator definition;
- omitting the constant-classifier/VQ-collapse finding;
- describing one external rerun as broad independent replication;
- fabricating truncation, collision, seed-level, parameter-count, or other missing numbers to meet a deadline.

A missed venue is preferable to weakening the retained scientific boundary.

## 7. Strongest next gate

Before manuscript styling, close the **two missing-quantitative-evidence blockers in Section 2A** and then apply the four evidence-located disclosures in Section 2B. The highest-value single artifact remains a deterministic paper-facing table containing, for each frozen seed, retained predicted class/support and integer correct counts for `full`, `no_planner`, `no_target`, and matched supervised, with exact source paths/hashes next to the already-reviewed collapse/VQ evidence.

After that table and the collision statistic are independently reviewable, the remaining work is predominantly manuscript disclosure, venue formatting, author/AI-use compliance, and anonymity checking—not scientific rescue work.
