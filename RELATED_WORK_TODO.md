# LAM-JEPA Related-Work Verification Ledger

**Last verified:** 2026-08-14  
**Purpose:** close the ARC negative-result bibliography without inventing citations or inflating novelty.  
**Scientific source boundary:** ARC architecture statements refer to frozen commit `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb` unless explicitly labeled as later repair/evidence tooling.

## Accepted primary sources

### RW-01 — I-JEPA
- **Title:** Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture
- **Authors:** Mahmoud Assran; Quentin Duval; Ishan Misra; Piotr Bojanowski; Pascal Vincent; Michael Rabbat; Yann LeCun; Nicolas Ballas
- **Year/source:** 2023, arXiv
- **Identifier:** arXiv:2301.08243
- **Supports:** canonical JEPA representation-space prediction; context-to-distinct-target embedding prediction; learned target encoder updated by EMA.
- **Novelty effect:** prevents representation-space targets or EMA target encoders from being claimed as novel; also provides a contrast showing that the frozen ARC path is not a canonical I-JEPA context/target task.
- **Status:** VERIFIED.

### RW-02 — VQ-VAE
- **Title:** Neural Discrete Representation Learning
- **Authors:** Aaron van den Oord; Oriol Vinyals; Koray Kavukcuoglu
- **Year/source:** 2017 (revised 2018), arXiv
- **Identifier:** arXiv:1711.00937
- **Supports:** learned vector-quantized discrete latent representations as established prior art.
- **Novelty effect:** prevents a generic vector-quantization/discrete-code novelty claim.
- **Status:** VERIFIED.

### RW-03 — ARC
- **Title:** Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge
- **Authors:** Peter Clark; Isaac Cowhey; Oren Etzioni; Tushar Khot; Ashish Sabharwal; Carissa Schoenick; Oyvind Tafjord
- **Year/source:** 2018, arXiv
- **Identifier:** arXiv:1803.05457
- **Supports:** provenance and intended scope of the AI2 Reasoning Challenge benchmark family.
- **Novelty effect:** benchmark use is not a contribution by itself.
- **Status:** VERIFIED.

### RW-04 — DeBERTaV3
- **Title:** DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing
- **Authors:** Pengcheng He; Jianfeng Gao; Weizhu Chen
- **Year/source:** 2021 arXiv; conference paper at ICLR 2023
- **Identifier:** arXiv:2111.09543
- **Supports:** provenance of the DeBERTaV3 family used for the bounded pretrained comparison.
- **Novelty effect:** comparator only; no SOTA inference from the project's development slice.
- **Status:** VERIFIED.

### RW-05 — FF-JEPA
- **Title:** FF-JEPA: Long-Horizon Planning in World Models with Latent Planners
- **Authors:** Sergi Masip; Jonathan Swinnen; Yutong Hu; Renaud Detry; Tinne Tuytelaars
- **Year/source:** 2026, arXiv
- **Identifier:** arXiv:2606.09311
- **Supports:** explicit latent planners in a JEPA/world-model setting are public prior art by June 2026.
- **Novelty effect:** prevents a generic claim that adding a latent planner to a JEPA setting is novel.
- **Status:** VERIFIED.

### RW-06 — reproducibility programme
- **Title:** Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)
- **Authors:** Joelle Pineau; Philippe Vincent-Lamarre; Koustuv Sinha; Vincent Lariviere; Alina Beygelzimer; Florence d'Alche-Buc; Emily Fox; Hugo Larochelle
- **Year/source:** 2021, Journal of Machine Learning Research 22(164):1–20
- **Persistent source:** JMLR paper `20-303`
- **Supports:** reproducibility as a reliability practice and the context of reproducibility checklists/programmes in ML research.
- **Novelty effect:** the LAM evidence pipeline may be useful engineering, but reproducibility practice is not invented here.
- **Status:** VERIFIED.

### Optional library-context source — VICReg
- **Title:** VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning
- **Authors:** Adrien Bardes; Jean Ponce; Yann LeCun
- **Year/source:** 2021, arXiv
- **Identifier:** arXiv:2105.04906
- **Use only if needed:** the repository's generic non-ARC loss library contains variance/covariance terms, but the frozen ARC benchmark `_lam_arc_loss` does **not** use those terms. Do not cite VICReg as if it were part of the reported ARC objective.
- **Status:** VERIFIED / NOT REQUIRED FOR CURRENT ARC RESULTS.

## A. JEPA / target-encoder positioning

**Status: VERIFIED.**

RW-01 provides the canonical comparison. I-JEPA predicts representations of distinct masked target blocks from a context block. By contrast, the frozen LAM ARC forward computes `target_z` by running the EMA target encoder/projector on the **same `tokens` and `numeric_x`** supplied to the online encoder. The ARC loss then aligns `z_q` to this same-input EMA representation.

Required manuscript wording: describe the tested model as the project-named **LAM-JEPA** or as a **same-input EMA target-alignment architecture**. Do not imply that the frozen ARC experiment instantiated canonical context-to-future/masked-target JEPA prediction.

## B. Discrete/quantized latent representation learning

**Status: VERIFIED for generic prior-art boundary.**

RW-02 establishes vector-quantized discrete latent learning as prior art. The later trainability repair is an engineering result under a frozen train-only gate, not evidence that vector quantization is new or generally superior.

A specialized codebook-collapse citation is optional only if the final discussion makes a literature-level claim beyond the repository's own diagnostic observation.

## C. ARC and pretrained comparison

**Status: VERIFIED.**

Use RW-03 for ARC provenance and RW-04 for the DeBERTaV3 family. The repository's pinned model revision remains the source of truth for the exact `microsoft/deberta-v3-xsmall` artifact used.

Do not compare the frozen validation result to unrelated public test-set/SOTA numbers. The locked ARC confirmatory test remains untouched for the failed hypothesis.

## D. Planner / world-model novelty boundary

**Status: VERIFIED enough to block broad novelty language.**

RW-05 shows explicit latent planning in a JEPA/world-model setting by June 2026. The current paper therefore cannot claim that “latent planner + JEPA” is generically new. The LAM ARC planner is a short discrete latent-action rollout evaluated only through the bounded ARC setup.

## E. Ablation, capacity matching, and negative controls

**Status: SOURCE-LOCAL CLAIM ONLY; no novelty citation required.**

The paper can report exactly what it did: gradient-active parameter matching, `no_planner`, `no_target`, and deterministic shuffled-label controls. It must not claim the matching rule or ablation methodology is novel.

## F. Reproducibility and adverse-result retention

**Status: PARTIAL / SUFFICIENT FOR CURRENT DRAFT.**

RW-06 supports the general value of reproducibility. The project's stronger statements are local and artifact-backed: protocol/configs, adverse results, hashes, replay attempts, and the stop rule are retained. Do not claim preregistration or negative-result reporting was invented here.

## G. Determinism and floating-point reproducibility

**Status: EMPIRICAL REPOSITORY EVIDENCE COMPLETE; OPTIONAL FRAMEWORK DOC CITATION OPEN.**

Required wording:

> independent reruns reproduce the aggregate scientific conclusion and verifier outputs; low-level probabilities/checkpoint bytes are not claimed bitwise identical across all platforms/releases.

If the final manuscript discusses framework-level determinism, verify the then-current official PyTorch reproducibility documentation before citation.

## Frozen ARC source warnings

These points must survive editorial cleanup:

1. `text_to_tokens` lowercases/splits whitespace tokens and maps each token deterministically with BLAKE2b into a 256-entry vocabulary. The ARC adapter uses a 96-token maximum.
2. ARC `numeric_x` is zero for every example, so the numeric branch supplies no example-varying information in this benchmark.
3. `TokenEncoder.encoder = nn.Identity()`; after embedding + learned positional vectors, the encoder applies `LayerNorm` and mean pooling. “Transformer encoder” is forbidden for the frozen ARC configuration.
4. The ARC-specific objective is `_lam_arc_loss = CE + 0.5*alignment + 0.25*quantization + 0.25*trajectory`. The larger repository `total_loss` is not the reported ARC training objective.
5. The alignment target is the EMA target encoder/projector on the same input, not a separate future/masked target view.

## Citation acceptance gate

Any additional reference enters the manuscript only when all fields are recorded:
- exact title;
- authors;
- year;
- venue/source;
- persistent identifier or official primary-source location;
- exact claim it supports;
- whether it threatens novelty wording.

No generated, memory-only, or secondary-summary citation is permitted into the final bibliography.
