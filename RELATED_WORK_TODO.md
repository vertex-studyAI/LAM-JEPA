# LAM-JEPA Related-Work Verification Ledger

**Purpose:** close the manuscript bibliography without inventing citations or inflating novelty. This ledger is deliberately a TODO/evidence map, not a completed literature review.

## A. Joint-embedding predictive architectures

Required checks:

- identify the canonical JEPA/I-JEPA/V-JEPA papers actually relevant to latent prediction;
- record exact mechanism overlap with LAM-JEPA;
- distinguish predictive representation learning from the planner/target/quantized mechanisms tested here;
- do not imply that latent-space prediction or stop-gradient target networks are novel to this project.

Status: **TODO — source verification required before manuscript insertion**.

## B. Discrete/quantized latent representation learning

Required checks:

- VQ-VAE and subsequent vector-quantized representation-learning work;
- codebook collapse / dead-code / optimization literature where directly relevant;
- EMA codebook updates and residual/near-continuous alternatives where relevant to the repaired-v5 line.

Claim boundary: the repository's trainability repair is an engineering result under a frozen gate, not evidence that vector quantization is new or generally superior.

Status: **TODO — source verification required**.

## C. ARC and multiple-choice reasoning baselines

Required checks:

- canonical AI2 ARC dataset paper/source;
- baseline/model families used on ARC-Challenge that are comparable to the scope of this small-model experiment;
- exact provenance for the pinned `microsoft/deberta-v3-xsmall` comparator;
- avoid comparing against incomparable test-set/SOTA numbers when this project deliberately keeps the ARC test locked.

Status: **PARTIAL — dataset/model identifiers frozen; final paper citations still need source verification**.

## D. Ablation, capacity matching, and negative controls

Required checks:

- neural architecture ablation methodology;
- matched-capacity or matched-compute comparison principles;
- shuffled-label/randomization controls and sanity-check literature where directly applicable.

The manuscript should explain why gradient-active parameter matching was used instead of nominal total parameters, but it must not claim this matching rule is novel.

Status: **TODO — source verification required**.

## E. Reproducibility, preregistration, and negative results in ML

Required checks:

- reproducibility/checklist literature or venue guidance appropriate to the eventual target;
- preregistration/frozen-protocol methodology where relevant;
- responsible reporting of failed hypotheses and adverse baselines.

The paper's methodological contribution is evidence discipline, not a claim that preregistration or negative-result reporting was invented here.

Status: **TODO — venue-dependent source verification required**.

## F. Determinism and floating-point reproducibility

Required checks:

- authoritative PyTorch determinism/reproducibility documentation or primary technical references;
- literature on non-associative floating-point execution only if necessary for the final explanation.

Required wording: independent reruns reproduce exact aggregate conclusions and verifier outputs, while low-level probabilities/checkpoint bytes are not guaranteed identical.

Status: **PARTIAL — empirical repository evidence complete; citation source still required**.

## Citation acceptance gate

A reference enters the manuscript only when all fields are recorded:

- exact title;
- authors;
- year;
- venue/source;
- persistent identifier/official URL;
- exact claim it supports;
- whether it threatens any novelty wording.

No generated or memory-only citation is permitted into the final bibliography.
