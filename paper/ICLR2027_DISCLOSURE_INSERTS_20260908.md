# ICLR 2027 retained disclosure inserts

**Date:** 2026-09-08  
**Scientific state:** **NEGATIVE / INCONCLUSIVE — UNCHANGED**  
**Purpose:** submission packaging from already retained evidence only.

This file contains manuscript-ready disclosure material derived from the retained artifacts already reconciled in `ICLR2027_RETAINED_DIAGNOSTICS_20260908.md`. It is not a new experiment, does not change any seed/split/threshold/metric, does not authorize the successor study, and does not open the locked historical ARC confirmatory test.

## 1. Seed-level frozen-control table

Source: GitHub artifact `9162165932`, run `31203337502`, archive SHA-256 `caa898f1ff046a337db9b5ddbffe1b332943a732868e2fd809abeda8ee89c30b`; raw result `arc-protocol-v3-full-controls-validation.json`, SHA-256 `76aad8b1327e21470aeed137bac341b75b4fcf1f37e5394047642d395e8070f8`.

```latex
\begin{table}[t]
\centering
\caption{Seed-level frozen ARC-Challenge validation counts. Each entry is correct predictions out of 295 retained validation examples. Every listed run has prediction support 1, i.e. it emits a single class across the retained validation set.}
\begin{tabular}{lrrrrr}
\toprule
Configuration & Seed 1 & Seed 2 & Seed 3 & Seed 4 & Seed 5 \\
\midrule
Full LAM-JEPA & 71 & 78 & 78 & 71 & 78 \\
\texttt{no\_planner} & 71 & 78 & 71 & 71 & 78 \\
\texttt{no\_target} & 71 & 78 & 83 & 71 & 83 \\
Shuffled-label control & 78 & 71 & 83 & 78 & 78 \\
\bottomrule
\end{tabular}
\end{table}
```

The paired item-count differences are `full - no_planner = [0, 0, +7, 0, 0]` and `full - no_target = [0, 0, -5, 0, -5]`. These counts are descriptive retained evidence and do not replace the preregistered H2/H3 decision rules.

## 2. Finite-seed uncertainty wording

Manuscript-safe wording:

> The retained paired bootstrap intervals summarize variation over the five fixed preregistered seeds in this protocol. With only five seeds, they should be read as finite-seed protocol summaries rather than asymptotic or population-level uncertainty guarantees.

This wording does not alter the already frozen bootstrap values or decision gates.

## 3. Input-visibility / truncation disclosure

Source: `ARC Canonical Input Visibility Audit` artifact `9884278396`, archive SHA-256 `eaf1b43a781d52b74bcf14b23e36acec27e8f4870dd182b57c23badf5d9df8c3`; retained `input-visibility-audit.json` SHA-256 `1d894781e4476505adb1c6464cea0cf232f1dfa13cb4e69f0e0b2d1c313c5f7a`.

Manuscript-safe wording:

> The frozen whitespace-token surface truncates at 96 positions. A retained input-visibility audit found 23/1,117 eligible training prompts and 6/295 validation prompts above that limit; in 3/295 validation rows, the fourth-answer marker `[3]` lies beyond the retained 96-token surface. The locked test split was not accessed. These are upstream representation limitations and are not used to rescue the failed hypothesis.

## 4. Hash-collision disclosure

Source: original retained external-review diagnostic.

Manuscript-safe wording:

> The hashed-token representation maps 7,030 distinct observed word types into 256 buckets in the retained external-review diagnostic, with a mean of 27.5 distinct word types per bucket and a maximum of 47. This is a representation limitation, not a substitute for the directly measured downstream VQ-collapse diagnosis.

Do not reinterpret this statistic as proving that hash collisions caused the scientific failure.

## 5. Gradient-active parameter accounting

Source: `scripts/ci/measure_arc_gradient_capacity.py`.

Manuscript-safe wording:

> Gradient-active parameter count is defined as the sum of parameter tensor sizes for parameters whose `.grad` is not `None` after backpropagating the exact retained `_lam_arc_loss`. This accounting establishes which tensors participate in the retained backward graph; it does not claim that every scalar gradient is numerically nonzero.

The existing reported counts remain 86,372 for LAM-JEPA and 86,644 for the matched supervised comparator; this file does not recompute or alter them.

## 6. Dataset naming

At first use, spell the benchmark as **AI2 Reasoning Challenge (ARC-Challenge)**. Later uses may use `ARC-Challenge`.

## 7. Claim boundary to retain beside these inserts

The inserts above do not change the supported conclusion:

- H1 superiority failed under the frozen five-seed validation protocol;
- H2 planner contribution and H3 target-path contribution were not supported;
- the reviewed retained runs behave as constant classifiers through a single-code quantized bottleneck;
- the bounded quantizer-off diagnostic restored input dependence but did not establish above-chance ARC performance;
- one external rerun/review is one bounded external reproduction, not broad replication;
- no general JEPA or vector-quantization conclusion is supported;
- the historical locked ARC confirmatory test remains closed;
- the OpenBookQA/successor line remains separate, pre-outcome, and cannot rescue this manuscript.

## 8. Venue packaging note

Current official ICLR 2027 requirements were rechecked on 2026-09-08: genuine abstract by **2026-09-18 11:59 PM AoE**, full paper by **2026-09-25 11:59 PM AoE**, no new authors after the abstract deadline, double-blind submission, official ICLR 2027 style, at most 9 pages of main text at initial submission, and a mandatory AI-use statement outside the page limit. The final venue-formatted source still needs to be built and scrubbed for anonymity.

Official sources:
- https://iclr.cc/Conferences/2027/CallForPapers
- https://iclr.cc/Conferences/2027/AuthorGuidelines
- https://iclr.cc/Conferences/2027/AIPolicyForAuthors
