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

Those boundaries remain consistent with `CLAIM_LEDGER.md`, `paper/EXTERNAL_REVIEW_VQ_COLLAPSE_CORRECTION_20260831.md`, and `paper/main.tex`.

## 2. External-review correction reconciliation

### Already reflected in `paper/main.tex`

- [x] exact H1/H2/H3 preregistered thresholds are stated rather than summarized vaguely;
- [x] one external rerun/review is described as one bounded external rerun, not broad replication or peer review;
- [x] constant-classifier collapse is surfaced as part of the scientific result;
- [x] the measured chain `distinct pre-quantizer latents -> one VQ code -> constant post-quantizer representation -> constant predictions` is represented in bounded form;
- [x] the quantizer-off diagnostic is explicitly limited to restored input dependence and does not claim above-chance task performance;
- [x] the bounded pretrained comparison is not used as a broad superiority/inferiority claim;
- [x] architecture wording is constrained: the tested path is not described as a Transformer or canonical I-JEPA context-to-distinct-target task;
- [x] the locked confirmatory-test stop rule remains explicit.

### Quantitative evidence blockers closed on 2026-09-08

A retained paper-facing diagnostic now exists at:

`paper/ICLR2027_RETAINED_DIAGNOSTICS_20260908.md`

It closes the two missing-quantitative-evidence blockers without a new training run or test access.

- [x] **Seed-level integer counts / deltas.** Artifact `9162165932` was downloaded again and its archive SHA-256 re-matched the retained provenance digest `caa898f1ff046a337db9b5ddbffe1b332943a732868e2fd809abeda8ee89c30b`. Counts were recomputed directly from the raw `predictions` arrays in `arc-protocol-v3-full-controls-validation.json` (file SHA-256 `76aad8b1327e21470aeed137bac341b75b4fcf1f37e5394047642d395e8070f8`). The retained counts are:
  - `full`: `71, 78, 78, 71, 78` correct of 295;
  - `no_planner`: `71, 78, 71, 71, 78`;
  - `no_target`: `71, 78, 83, 71, 83`;
  - shuffled: `78, 71, 83, 78, 78`;
  - `full-no_planner` item deltas: `[0,0,+7,0,0]`;
  - `full-no_target` item deltas: `[0,0,-5,0,-5]`;
  - every listed retained run has prediction support 1.
- [x] **Hash-collision quantification.** The original retained external-review note reports `7,030` distinct word types mapped into `256` hash buckets, mean `27.5` distinct word types per bucket, maximum `47`. This is retained as an external-review diagnostic, not promoted to a newly generated model outcome and not substituted for the measured VQ bottleneck.

### Additional retained disclosure evidence closed

- [x] **Truncation counts independently rechecked.** The non-expired `ARC Canonical Input Visibility Audit` artifact `9884278396` was downloaded and its archive digest re-matched `eaf1b43a781d52b74bcf14b23e36acec27e8f4870dd182b57c23badf5d9df8c3`. Its retained row records give 23/1117 eligible train prompts and 6/295 validation prompts over the 96-whitespace-token limit; 3/295 validation rows place the fourth-answer marker `[3]` beyond the retained surface; `test_split_accessed=false`.
- [x] **Gradient-active accounting source located.** `scripts/ci/measure_arc_gradient_capacity.py` defines the reported accounting as the sum of parameter tensor sizes for parameters whose `.grad` is not `None` after the exact `_lam_arc_loss` backward. This is not a claim that every scalar had a nonzero numerical gradient.
- [x] **ARC naming correction identified.** First dataset use must read **AI2 Reasoning Challenge (ARC-Challenge)**.
- [x] **Five-seed uncertainty wording determined.** The manuscript must state that the retained bootstrap over five fixed seeds is a finite-seed protocol summary, not an asymptotic or population-level uncertainty guarantee.

These gates are now **evidence-closed but not yet manuscript-closed**: `paper/main.tex` still needs the exact disclosures inserted before submission.

## 3. Current ICLR 2027 venue gate

Requirements re-checked on 2026-09-08 from the official ICLR 2027 Call for Papers, Author Guidelines, and AI Policy:

- abstract deadline: **2026-09-18 11:59 PM AoE**;
- full-paper deadline: **2026-09-25 11:59 PM AoE**;
- abstract must be genuine/informative; placeholder or duplicate abstracts are removed;
- no new authors may be added after the abstract deadline;
- all authors need current OpenReview profiles;
- submission is **double blind**;
- main text is **9 pages maximum** at initial submission, excluding references;
- official **ICLR 2027 LaTeX style** is required;
- an **AI use statement is mandatory** and does not count toward the page limit;
- a reproducibility statement is strongly recommended.

Primary sources remain:
- https://iclr.cc/Conferences/2027/CallForPapers
- https://iclr.cc/Conferences/2027/AuthorGuidelines
- https://iclr.cc/Conferences/2027/AIPolicyForAuthors

## 4. Remaining submission blockers

### Manuscript disclosure

- [ ] insert the exact seed-level integer count table from `ICLR2027_RETAINED_DIAGNOSTICS_20260908.md`;
- [ ] explicitly describe the five-seed bootstrap as a finite-seed summary;
- [ ] add the 23/1117 and 6/295 truncation counts and the 3/295 fourth-marker cutoff;
- [ ] add the external-review collision statistic while keeping VQ as the measured primary bottleneck;
- [ ] spell out AI2 Reasoning Challenge at first use;
- [ ] define gradient-active parameter accounting exactly and avoid implying every scalar has nonzero gradient.

### Venue format and metadata

- [ ] create a separate ICLR 2027 submission source from the evidence-bound manuscript;
- [ ] use official ICLR 2027 style and verify main-text page count <= 9;
- [ ] replace owner-placeholder metadata with venue-correct anonymous submission form;
- [ ] add the mandatory AI-use statement based on actual author/tool use;
- [ ] add/reconcile the recommended reproducibility statement;
- [ ] run an anonymity scrub across manuscript, supplement, repository links, artifact names, acknowledgments, and PDF metadata;
- [ ] freeze final authors before the Sep 18 abstract deadline.

## 5. Archival reproducibility risk

The historical matched-supervised raw artifact `9003785715` is now expired in GitHub Actions. Its digest and raw-result provenance remain recorded in `MANUSCRIPT_PROVENANCE.md`, and the external rerun reproduced the retained matched headline value, but the old ZIP can no longer be freshly downloaded from GitHub. Do not silently replace its identity.

If a fresh clean-room rerun of the **unchanged frozen validation protocol** is performed for archival retention, it must be labeled as a new reproduction artifact, preserve seeds/split/budget/metric/locked-test boundaries exactly, and be compared against the retained historical digest/values rather than treated as a new scientific trial.

## 6. Go / no-go rule

**GO for an ICLR abstract only if the submission source can honestly describe the existing frozen negative/collapse result without relying on an unfinished successor experiment.**

**NO-GO** if submission would require:

- opening the locked historical ARC confirmatory test;
- using OpenBookQA/successor outcomes to rescue the historical claim;
- changing H1/H2/H3 thresholds, seeds, split, metric, or comparator definition;
- omitting the constant-classifier/VQ-collapse finding;
- describing one external rerun as broad independent replication;
- fabricating missing values or rewriting the adverse result to meet the deadline.

A missed venue is preferable to weakening the retained scientific boundary.

## 7. Strongest next gate

The two missing quantitative evidence blockers are now closed. The strongest next submission gate is to **patch `paper/main.tex` from the retained diagnostics and then create the venue-format-only ICLR source**, without changing any scientific value.

A second independent frozen-protocol reproduction would strengthen the scientific package further, but it is not permission to alter or rescue the existing negative result.