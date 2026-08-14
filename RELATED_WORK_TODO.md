# LAM-JEPA Related-Work Verification Ledger

**Updated:** 2026-08-14  
**Purpose:** close the manuscript bibliography without inventing citations or inflating novelty. A reference is marked verified only when title/authors/source identifier were checked against a primary source. This remains a literature ledger, not evidence that the current LAM-JEPA mechanism is novel.

## A. Joint-embedding predictive architectures

### Verified anchors

1. **Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, Nicolas Ballas.**  
   *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.*  
   arXiv:2301.08243, 2023.  
   **Supports:** JEPA prediction in representation space; learned target encoder; EMA target update; collapse considerations.  
   **Novelty threat:** latent-space prediction and EMA target encoders are established, not LAM-JEPA inventions.

2. **Mido Assran et al.**  
   *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.*  
   arXiv:2506.09985, 2025.  
   **Supports:** modern JEPA representations extended to action-conditioned latent world modeling and planning.  
   **Novelty threat:** combining JEPA-style latent prediction with action-conditioned planning is now clearly an established research direction; LAM-JEPA must rely on its exact reasoning-domain question/evidence, not the generic combination.

### Additional current landscape requiring manuscript-specific triage

- Garrido et al., *Learning Latent Action World Models In The Wild*, arXiv:2601.05230.
- Yan et al., *Is Forward Prediction Enough? Physical State Grounding for JEPA World Models*, arXiv:2608.06799.
- Lin et al., *JEPA-WAM: Learning Vision-Language-Action Policies with Joint-Embedding World Modeling*, arXiv:2608.09381.

These are relevant because they further narrow any novelty claim based only on latent actions, world modeling, prediction, grounding, or planning. They do **not** invalidate the current negative ARC result; they affect framing.

**Status:** **CORE ANCHORS VERIFIED; broader citation sweep still required before submission.**

## B. Discrete/quantized latent representation learning

### Verified anchor

**Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu.**  
*Neural Discrete Representation Learning.*  
arXiv:1711.00937, 2017/2018.  
**Supports:** vector-quantized discrete latent representation learning and learned codebook use.  
**Novelty threat:** discrete vector quantization is established. The repository's quantized bottleneck is an implementation choice, not novel by naming.

### Still required if used in final discussion

- codebook-collapse/dead-code literature directly relevant to the observed trainability failure;
- EMA codebook-update literature and residual/near-continuous alternatives if the repaired-v5 discussion needs them.

**Claim boundary:** the repository's trainability repair is an engineering result under a frozen gate, not evidence that vector quantization is new or generally superior.

**Status:** **FOUNDATIONAL SOURCE VERIFIED; failure-mechanism literature still TODO.**

## C. ARC and multiple-choice reasoning baselines

### Verified anchors

1. **Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, Oyvind Tafjord.**  
   *Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge.*  
   arXiv:1803.05457, 2018.  
   **Supports:** provenance and original framing of ARC / ARC-Challenge.

2. **Pengcheng He, Jianfeng Gao, Weizhu Chen.**  
   *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing.*  
   arXiv:2111.09543; ICLR 2023.  
   **Supports:** architecture family for the pinned bounded `microsoft/deberta-v3-xsmall` comparator.  
   **Boundary:** the repository's exact model checkpoint/revision provenance comes from the frozen comparator protocol, not from the paper citation alone.

### Required manuscript discipline

- do not compare the locked-validation study rhetorically with incomparable ARC test-set leaderboard numbers;
- keep the pinned DeBERTa result labeled **bounded development characterization**, not broad inferiority proof;
- state that the current ARC confirmatory test remains untouched for the failed hypothesis line.

**Status:** **DATASET + MODEL-FAMILY SOURCES VERIFIED; comparator checkpoint provenance already frozen in repository evidence.**

## D. Ablation, capacity matching, and negative controls

Required checks before making a methodological novelty claim:

- neural architecture ablation methodology;
- matched-capacity or matched-compute comparison principles;
- shuffled-label/randomization sanity checks where directly applicable.

The manuscript may explain why gradient-active parameter matching was used, but it must **not** claim the matching rule is novel unless a serious methodology review establishes that.

**Status:** **TODO — source verification still required only if these principles are explicitly cited/claimed.**

## E. Reproducibility, preregistration, and negative results in ML

Required checks:

- venue-specific reproducibility/checklist guidance;
- preregistration/frozen-protocol methodology where directly relevant;
- responsible adverse-result and negative-result reporting guidance.

The paper's methodological value is evidence discipline in this case study, not a claim that preregistration or negative-result reporting was invented here.

**Status:** **TODO — venue-dependent.**

## F. Determinism and floating-point reproducibility

The repository itself establishes the important empirical boundary: aggregate conclusions and verifier outputs reproduce, while low-level probability-bearing arrays can drift across independent runners.

For the final explanatory note, use authoritative framework documentation/technical sources only if needed. Required wording remains conservative: independent reruns reproduce the scientific conclusion and declared aggregates; byte-identical probabilities/checkpoints across platforms/runners are not claimed.

**Status:** **EMPIRICAL REPOSITORY EVIDENCE COMPLETE; optional framework citation remains TODO.**

## Current novelty conclusion

The literature checked so far weakens any claim that the component stack is a new JEPA mechanism. The strongest defensible paper contribution is instead:

- a frozen, reproducible ARC evaluation of this particular reasoning-oriented LAM-JEPA configuration;
- a matched supervised comparator and mechanism ablations;
- preservation of adverse pretrained characterization and shuffled-label evidence;
- a documented engineering trainability repair that did not rescue the scientific hypothesis;
- explicit non-use of the locked confirmatory test after validation failure.

This is best framed as a **negative-result / falsification-first / reproducibility case study**, unless a later, separately versioned research question establishes new mechanism evidence.

## Citation acceptance gate

A reference enters the final manuscript only when all fields are recorded:

- exact title;
- authors (or complete bibliography-generated author list for long collaborations);
- year;
- venue/source;
- persistent identifier;
- exact claim it supports;
- whether it threatens novelty wording.

No memory-only or generated citation is permitted into the final bibliography.