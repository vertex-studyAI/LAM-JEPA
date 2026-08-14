# LAM-JEPA Related-Work Verification Ledger

**Last verified:** 2026-08-14  
**Purpose:** close the manuscript bibliography without inventing citations or inflating novelty.  
**Scientific source boundary:** architecture statements refer to frozen commit `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb` unless explicitly labeled as later repair/evidence tooling.

## Accepted primary sources

A source below is eligible for manuscript insertion because its title/authors/year/source/identifier and supported claim were checked against a primary paper page.

### RW-01 — I-JEPA

- **Title:** Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture
- **Authors:** Mahmoud Assran; Quentin Duval; Ishan Misra; Piotr Bojanowski; Pascal Vincent; Michael Rabbat; Yann LeCun; Nicolas Ballas
- **Year/source:** 2023, arXiv
- **Identifier:** arXiv:2301.08243
- **Supports:** JEPA/representation-space prediction; context-to-target embedding prediction; learned target encoder updated by EMA; collapse concerns in joint-embedding learning.
- **Novelty effect:** prevents LAM-JEPA from claiming representation-space prediction or EMA targets as novel.
- **Status:** VERIFIED.

### RW-02 — VQ-VAE

- **Title:** Neural Discrete Representation Learning
- **Authors:** Aaron van den Oord; Oriol Vinyals; Koray Kavukcuoglu
- **Year/source:** 2017 (revised 2018), arXiv
- **Identifier:** arXiv:1711.00937
- **Supports:** learned vector-quantized discrete latent representations as established prior art.
- **Novelty effect:** prevents a generic vector-quantization/discrete-code novelty claim.
- **Status:** VERIFIED.

### RW-03 — VICReg

- **Title:** VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning
- **Authors:** Adrien Bardes; Jean Ponce; Yann LeCun
- **Year/source:** 2021, arXiv
- **Identifier:** arXiv:2105.04906
- **Supports:** explicit variance and covariance regularization to stabilize non-contrastive representation learning.
- **Novelty effect:** LAM-JEPA's variance/covariance loss terms are established categories, not mechanism novelty.
- **Status:** VERIFIED.

### RW-04 — ARC

- **Title:** Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge
- **Authors:** Peter Clark; Isaac Cowhey; Oren Etzioni; Tushar Khot; Ashish Sabharwal; Carissa Schoenick; Oyvind Tafjord
- **Year/source:** 2018, arXiv
- **Identifier:** arXiv:1803.05457
- **Supports:** provenance and intended difficulty/scope of the AI2 Reasoning Challenge benchmark family.
- **Novelty effect:** benchmark use is not a contribution by itself.
- **Status:** VERIFIED.

### RW-05 — DeBERTaV3

- **Title:** DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing
- **Authors:** Pengcheng He; Jianfeng Gao; Weizhu Chen
- **Year/source:** 2021 arXiv; conference paper at ICLR 2023
- **Identifier:** arXiv:2111.09543
- **Supports:** provenance of the DeBERTaV3 model family used for the pinned bounded pretrained comparison.
- **Novelty effect:** comparator only; no SOTA inference from the project's development slice.
- **Status:** VERIFIED.

### RW-06 — FF-JEPA

- **Title:** FF-JEPA: Long-Horizon Planning in World Models with Latent Planners
- **Authors:** Sergi Masip; Jonathan Swinnen; Yutong Hu; Renaud Detry; Tinne Tuytelaars
- **Year/source:** 2026, arXiv
- **Identifier:** arXiv:2606.09311
- **Supports:** explicit latent planners in a JEPA/world-model setting are public prior art by June 2026.
- **Novelty effect:** prevents a generic claim that adding a latent planner to a JEPA setting is novel.
- **Status:** VERIFIED.

### RW-07 — reproducibility programme

- **Title:** Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)
- **Authors:** Joelle Pineau; Philippe Vincent-Lamarre; Koustuv Sinha; Vincent Lariviere; Alina Beygelzimer; Florence d'Alche-Buc; Emily Fox; Hugo Larochelle
- **Year/source:** 2021, Journal of Machine Learning Research 22(164):1–20
- **Persistent source:** JMLR paper `20-303`
- **Supports:** reproducibility as a reliability practice and the context of reproducibility checklists/programmes in ML research.
- **Novelty effect:** the LAM evidence pipeline may be useful engineering, but reproducibility practice is not invented here.
- **Status:** VERIFIED.

## A. Joint-embedding predictive architectures

**Status: VERIFIED for the claims needed by the current negative-result manuscript.**

The manuscript may cite RW-01 for the core JEPA idea and RW-06 when discussing why latent-planner novelty language is unsafe in 2026. The paper should describe LAM-JEPA as a **hybrid JEPA-style objective**, because the frozen implementation includes supervised cross-entropy and several auxiliary/regularization terms rather than the pure I-JEPA masked-image objective.

Do not imply that latent-space prediction, EMA targets, stop-gradient target representations, or planning in JEPA/world-model settings are new to this project.

## B. Discrete/quantized latent representation learning

**Status: VERIFIED for generic prior-art boundary; specialized codebook-collapse bibliography remains OPTIONAL.**

RW-02 establishes vector-quantized discrete latent learning as prior art. The repository's trainability repair is an engineering result under a frozen train-only gate, not evidence that vector quantization is new or generally superior.

A specialized citation on codebook collapse/dead codes may be added only if the final discussion makes a literature-level claim about that phenomenon. It is not required merely to report the repository's observed trainability failure.

## C. ARC and multiple-choice reasoning baselines

**Status: VERIFIED for dataset and pinned model family.**

Use RW-04 for ARC provenance and RW-05 for the DeBERTaV3 family. The pinned repository comparator revision remains the source of truth for the exact `microsoft/deberta-v3-xsmall` artifact used by the experiment.

Do not compare this frozen validation result to unrelated public test-set/SOTA numbers. The project's locked ARC confirmatory test remains untouched for the failed hypothesis.

## D. Ablation, capacity matching, and negative controls

**Status: SOURCE-LOCAL CLAIM ONLY; no novelty citation required.**

The paper can report exactly what it did: gradient-active parameter matching, `no_planner`, `no_target`, and deterministic shuffled-label controls. It must not claim the matching rule or ablation methodology is novel. If a venue later requires a methodological citation, add one only after primary-source verification.

## E. Reproducibility, preregistration, and negative results in ML

**Status: PARTIAL / SUFFICIENT FOR CURRENT DRAFT.**

RW-07 supports the importance of reproducibility practice. The project's stronger claim is local and artifact-backed: the protocol, adverse results, hashes, replay attempts, and stop rule are retained. Do not claim that preregistration or negative-result reporting was invented here.

Venue-specific checklist/preregistration references can be added after a target venue is selected.

## F. Determinism and floating-point reproducibility

**Status: EMPIRICAL REPOSITORY EVIDENCE COMPLETE; OPTIONAL DOC CITATION STILL OPEN.**

Required wording remains:

> independent reruns reproduce the aggregate scientific conclusion and verifier outputs; low-level probabilities/checkpoint bytes are not claimed bitwise identical across all platforms/releases.

If the final manuscript explicitly discusses framework-level determinism, verify and cite the then-current official PyTorch reproducibility documentation. Do not use framework documentation to imply bitwise portability that it does not guarantee.

## Source-verified architecture warning

The frozen ARC scientific source does **not** contain a Transformer block in `TokenEncoder`: after embedding plus learned positional vectors, `self.encoder = nn.Identity()`, followed by layer normalization and mean pooling. Therefore manuscript wording such as “Transformer encoder” is forbidden for the frozen tested configuration unless referring to a distinct later implementation.

Likewise, the frozen `total_loss` combines classification cross-entropy with target alignment, variance/covariance/uniformity/geodesic regularization, confidence/verifier/rubric terms, trajectory consistency, and quantization. The paper should not characterize the evaluated objective as pure self-supervised JEPA training.

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
