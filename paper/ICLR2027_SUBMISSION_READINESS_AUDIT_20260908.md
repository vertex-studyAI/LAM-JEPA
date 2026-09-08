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

### Still unresolved before submission

The external-review correction explicitly required several paper-facing disclosures that are not yet present in the current authoritative manuscript in a submission-ready form. These remain **hard manuscript blockers** until exact retained evidence is inserted:

- [ ] **Seed-level integer deltas / counts:** pair the five-seed bootstrap summaries with the retained per-seed integer differences/counts, because the collapsed classifiers can otherwise make small mechanism deltas look more informative than they are.
- [ ] **Five-seed uncertainty limitation:** explicitly state the narrow sampling interpretation of bootstrap intervals computed from only five frozen seeds.
- [ ] **Truncation disclosure:** insert the already-retained external-review observation that roughly 2% of retained examples exceed the 96-whitespace-token limit and that some validation rows lose the fourth-answer marker under truncation. Do not estimate new numbers from memory; bind the sentence to the retained audit/source artifact used for the correction.
- [ ] **Hash-collision quantification:** the manuscript currently notes hashing into 256 token IDs, but the external review requires an actual retained collision statistic rather than qualitative wording. Do not invent a rate; generate it deterministically from the frozen eligible text surface or cite an already-retained audit if one exists.
- [ ] **ARC naming disambiguation:** spell out **AI2 Reasoning Challenge** at first use so the dataset is not confused with the Abstraction and Reasoning Corpus.
- [ ] **Gradient-active parameter definition:** define precisely what the retained `86,372 / 86,644` count includes and avoid implying that every counted scalar received a nonzero gradient on every batch.

Until those six items are closed from retained evidence, the paper is **scientifically coherent but not submission-ready**.

## 3. Current ICLR 2027 venue gate

Official ICLR 2027 requirements checked on 2026-09-08:

- abstract deadline: **2026-09-18 11:59 PM AoE**;
- full-paper deadline: **2026-09-25 11:59 PM AoE**;
- abstract must be genuine/informative; placeholder abstracts are removed;
- no new authors may be added after the abstract deadline;
- all authors need current OpenReview profiles;
- submission is **double blind**;
- main text is **9 pages maximum** at initial submission; references are excluded from the page limit; appendices may follow the references;
- the official **ICLR 2027 LaTeX style** is required;
- an **AI use statement is required** and does not count toward the page limit;
- a reproducibility statement is strongly recommended.

Primary sources:
- https://iclr.cc/Conferences/2027/CallForPapers
- https://iclr.cc/Conferences/2027/AuthorGuidelines

## 4. Venue-format blockers on the current manuscript

The authoritative `paper/main.tex` is presently an `11pt article` with `geometry`, not the ICLR 2027 style package. It also contains owner-placeholder author metadata rather than a final anonymous ICLR author block. Therefore:

- [ ] create a **separate ICLR 2027 submission source** from the evidence-bound manuscript; do not mutate scientific values while reformatting;
- [ ] use the official ICLR 2027 style and verify main-text page count <= 9;
- [ ] replace owner-placeholder metadata with the venue-correct anonymous submission form;
- [ ] add the required AI use statement;
- [ ] add/reconcile the recommended reproducibility statement;
- [ ] run a final anonymity scrub over main text, appendix, supplemental material, repository links, artifact names, acknowledgments, and metadata;
- [ ] freeze the final author list before the Sep 18 abstract deadline.

## 5. Go / no-go rule

**GO for an ICLR abstract only if, before the abstract deadline, the submission source can honestly describe the existing frozen negative/collapse result without relying on an unfinished successor experiment.**

**NO-GO** if submission would require any of the following:

- opening the locked historical ARC confirmatory test;
- using OpenBookQA/successor outcomes to rescue the historical claim;
- changing H1/H2/H3 thresholds, seeds, split, metric, or matched-comparator definition;
- omitting the constant-classifier/VQ-collapse finding;
- describing one external rerun as broad independent replication;
- fabricating truncation, collision, seed-level, or other missing numbers to meet a deadline.

A missed venue is preferable to weakening the retained scientific boundary.

## 6. Strongest next gate

Before more manuscript styling, close the **six evidence-disclosure blockers in Section 2** from retained artifacts. The highest-value single item is to materialize a deterministic paper-facing diagnostic table containing, for each frozen seed, the retained predicted class/support and integer correct counts for `full`, `no_planner`, `no_target`, and matched supervised, plus the exact collapse/VQ evidence already externally reviewed. That table would directly answer the strongest remaining methodological critique without changing the scientific conclusion.
