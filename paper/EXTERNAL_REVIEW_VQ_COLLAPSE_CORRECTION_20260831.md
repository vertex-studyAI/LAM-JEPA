# LAM-JEPA ARC manuscript — external-review mechanism correction

**Date:** 2026-08-31  
**Status:** REQUIRED BEFORE NEXT PAPER RELEASE  
**Scientific state:** frozen negative/inconclusive result preserved

## Why this correction exists

A genuinely external reviewer independently reran the frozen ARC protocol and reproduced the retained headline numbers, checksums, eligibility counts, per-seed counts, capacity-match ratio, method/source correspondence and references. The review also found a material mechanism-level fact that changes how the paper should describe the experiment: **all retained full/control runs collapse to constant classifiers, and the measured information-loss bottleneck is the vector quantizer.**

This document does not create a positive result, change a seed, open the locked test set, or authorize a broader claim. It records the external review and freezes the minimum manuscript corrections required before any next submission/release.

## 1. Independently reproduced frozen results

The external rerun reported:

| Condition | retained paper value | external rerun |
|---|---:|---:|
| full | 0.2549152493 | 0.2549152493 |
| no planner | 0.2501694888 | 0.2501694888 |
| no target | 0.2616949081 | 0.2616949081 |
| shuffled | 0.2630508393 | 0.2630508393 |
| matched supervised | 0.2664406780 | 0.2664406806 |

The reviewer also reported identical per-seed counts to retained artifact `9149336081`, matching ARC hashes/eligibility counts, reproduction of the `86,372 / 86,644` parameter-count comparison, source-method agreement, and valid references.

**Interpretation:** the numerical artifact chain survives outside rerun. The scientific interpretation still requires correction below.

## 2. Material finding: the retained runs are constant classifiers

Across every retained condition and seed reviewed, predictions have support 1 over all 295 validation rows. Softmax outputs are effectively input-invariant (~`6e-8` spread).

Validation class counts are `63, 71, 78, 83` out of 295. The reported per-seed correct counts are drawn from those class base rates, consistent with each run selecting one constant class.

The existing collapse criterion also rejects these runs: predicted support is 1 and the largest predicted-class share is 100%.

### Required claim correction

Do **not** describe the primary table merely as two functioning classifiers where one failed to outperform the other. The accurate bounded statement is:

> Under the frozen validation protocol, both the tested LAM-JEPA configurations and the relevant retained outputs exhibited prediction collapse; LAM-JEPA did not outperform the matched supervised comparator, and the retained mechanism comparisons do not establish planner or EMA-target benefit.

The constant-output behavior is part of the result, not an implementation footnote.

## 3. Material finding: information loss localizes to the VQ bottleneck

The external reviewer traced the retained representation path:

| Stage | distinct values over 295 validation inputs | observed spread |
|---|---:|---:|
| pre-quantizer latent `z` | 295 / 295 | per-dimension std roughly 3.2–5.2 |
| VQ assignment | **1 of 32 codes** | — |
| post-quantizer `z_q` | constant | ~`4e-8` |
| output softmax | constant | ~`6e-8` |

The encoder therefore remains input-dependent immediately before quantization. In the reviewer's causal check, removing quantization restores input-dependent predictions, while removing the alignment term does not remove collapse. The quantizer-off run does **not** establish above-chance performance; it only localizes the collapse mechanism.

### Required failure-analysis correction

Current discussion of 256-token hashing, mean pooling and the constant numeric branch may remain as upstream representation limitations, but they must not be presented as the measured primary bottleneck. The failure-analysis chain should instead lead with:

`distinct encoder latents -> one VQ code -> constant z_q -> constant predictions`

Then separately report upstream limitations as plausible contributors that remain unisolated with respect to task accuracy.

## 4. Mechanism claims must be narrowed

Because the full/no-planner/no-target runs are constant functions, the H2/H3 deltas are largely changes in which base-rate class a run collapses onto. Seed-level integer differences should be shown beside any bootstrap summary.

The manuscript must state the actual predeclared thresholds rather than relying only on the phrase `preregistered criteria`:

- H1: effect threshold and confidence requirement from the frozen verifier;
- H2/H3: mean delta >= 0.01 with lower interval bound > 0;
- shuffled-label validity ceiling: accuracy < 0.35.

No positive-control result currently establishes that this exact frozen setup would have detected the intended planner/target mechanism if present. Therefore prefer `unsupported under the frozen protocol` / `negative or inconclusive` over a broad mechanistic `falsified` claim.

## 5. Other corrections required before release

- **Bounded DeBERTa comparison:** state its tiny development budget/sample explicitly or remove it from the compact paper. It is characterization only.
- **Gradient-active parameters:** define exactly how the count is computed and avoid implying every counted scalar received nonzero gradient.
- **Bootstrap at five seeds:** retain if desired, but pair it with seed-level integer deltas and state the narrow sampling limitation.
- **Truncation:** disclose that roughly 2% of retained examples exceed the 96-whitespace-token limit; a few validation rows lose the fourth-answer marker under truncation.
- **Hash collisions:** quantify rather than describing only a 256-ID vocabulary.
- **ARC naming:** spell out `AI2 Reasoning Challenge` early to prevent confusion with the Abstraction and Reasoning Corpus.
- **External-review state:** remove statements that genuine outside review/reproduction is absent. The current state is one external frozen-protocol rerun with material methodological critique; a second independent reproduction remains a stronger next gate.

## 6. Authorized claim after this review

A conservative research claim now supported by the retained evidence plus external rerun is:

> An independent rerun reproduced the frozen negative/inconclusive ARC-Challenge results. Follow-up inspection found that the tested quantized path collapsed 295 distinct pre-quantizer latents to a single VQ code per run, yielding constant downstream predictions. Removing the quantizer restored input dependence in the reviewer's bounded check but did not establish above-chance performance. The experiment therefore supports a reproducible failure-mechanism report, not an architecture-superiority or general JEPA conclusion.

## 7. Claims still prohibited

- LAM-JEPA improves ARC performance.
- Planner or EMA-target alignment provides a validated benefit.
- Vector quantization is generally harmful.
- JEPA methods fail on reasoning tasks.
- Quantizer removal solves the ARC task.
- The matched-supervised comparison alone establishes meaningful model superiority/inferiority beyond this bounded collapsed setup.
- The external review constitutes peer-reviewed publication or broad independent replication.

## 8. Next release gate

Before the next paper/preprint release:

- [ ] revise `paper/main.tex` abstract, results, failure analysis, limitations, discussion and conclusion to expose constant prediction collapse;
- [ ] revise `paper/icdm_teen_2026.tex` if retained as an archival compact source;
- [ ] state the measured VQ-collapse chain;
- [ ] update the external-review/reproducibility status;
- [ ] preserve the exact frozen numeric tables and locked confirmatory-test state;
- [ ] rerun paper/source claim gates on the revised exact head;
- [ ] obtain a second independent rerun/reviewer if promotion beyond the current evidence level is desired.

## Evidence provenance

The full external notes were received by email as `REVIEW_FOR_ISSUE_102.md` and summarized into GitHub issue #102 on 2026-08-31. The reviewer stated that the notes were intended for that review gate and that they made no repository changes.

This correction is intentionally additive until the manuscript text is revised and recertified; it prevents the earlier internal audit from being mistaken for the final interpretation after external review.
