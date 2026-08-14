# LAM-JEPA Related-Work Verification Ledger

**Last verified:** 2026-08-14  
**Purpose:** close the ARC negative-result bibliography using verified primary sources while keeping novelty claims conservative.  
**Scientific source boundary:** architecture statements refer to frozen commit `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb` unless a later repair is explicitly identified.

## Accepted primary sources

### RW-01 — I-JEPA
- **Title:** Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture
- **Authors:** Mahmoud Assran; Quentin Duval; Ishan Misra; Piotr Bojanowski; Pascal Vincent; Michael Rabbat; Yann LeCun; Nicolas Ballas
- **Source/year:** arXiv:2301.08243, 2023
- **Supports:** canonical JEPA context-to-distinct-target representation prediction; EMA target encoder.
- **Novelty effect:** latent representation targets and EMA targets are established; also highlights that the frozen LAM ARC path is not the canonical I-JEPA prediction task because its target receives the same serialized input.
- **Status:** **VERIFIED**.

### RW-02 — VQ-VAE
- **Title:** Neural Discrete Representation Learning
- **Authors:** Aaron van den Oord; Oriol Vinyals; Koray Kavukcuoglu
- **Source/year:** arXiv:1711.00937, 2017; revised 2018
- **Supports:** learned vector-quantized discrete latent representations.
- **Novelty effect:** generic vector quantization/discrete-code learning is prior art.
- **Status:** **VERIFIED**.

### RW-03 — Latent Action Pretraining (LAPA)
- **Title:** Latent Action Pretraining from Videos
- **Authors:** Seonghyeon Ye; Joel Jang; Byeongguk Jeon; Sejune Joo; Jianwei Yang; Baolin Peng; Ajay Mandlekar; Reuben Tan; Yu-Wei Chao; Bill Yuchen Lin; Lars Liden; Kimin Lee; Jianfeng Gao; Luke Zettlemoyer; Dieter Fox; Minjoon Seo
- **Source/year:** arXiv:2410.11758, 2024
- **Supports:** learning discrete latent actions from video with a VQ-VAE-based objective before downstream action-model pretraining.
- **Novelty effect:** discrete latent actions are not unique to LAM-JEPA.
- **Status:** **VERIFIED**.

### RW-04 — Latent action world models
- **Title:** Learning Latent Action World Models In The Wild
- **Authors:** Quentin Garrido; Tushar Nagarajan; Basile Terver; Nicolas Ballas; Yann LeCun; Michael Rabbat
- **Source/year:** arXiv:2601.05230, 2026
- **Supports:** latent action/world-model learning and planning from action-free video using learned constrained latent action spaces.
- **Novelty effect:** broad latent-action world-model language cannot be treated as novel here.
- **Status:** **VERIFIED**.

### RW-05 — FF-JEPA
- **Title:** FF-JEPA: Long-Horizon Planning in World Models with Latent Planners
- **Authors:** Sergi Masip; Jonathan Swinnen; Yutong Hu; Renaud Detry; Tinne Tuytelaars
- **Source/year:** arXiv:2606.09311, 2026
- **Supports:** an explicit latent planner in a JEPA/world-model setting.
- **Novelty effect:** prevents generic “latent planner + JEPA” novelty language.
- **Status:** **VERIFIED**.

### RW-06 — ARC
- **Title:** Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge
- **Authors:** Peter Clark; Isaac Cowhey; Oren Etzioni; Tushar Khot; Ashish Sabharwal; Carissa Schoenick; Oyvind Tafjord
- **Source/year:** arXiv:1803.05457, 2018
- **Supports:** provenance and intended scope of ARC.
- **Novelty effect:** benchmark use is not a contribution.
- **Status:** **VERIFIED**.

### RW-07 — DeBERTaV3
- **Title:** DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing
- **Authors:** Pengcheng He; Jianfeng Gao; Weizhu Chen
- **Source/year:** arXiv:2111.09543, 2021; ICLR 2023
- **Supports:** provenance of the DeBERTaV3 family used for the bounded pretrained characterization.
- **Novelty effect:** comparator only; it does not establish a broad SOTA/inferiority claim.
- **Status:** **VERIFIED**.

### RW-08 — ML reproducibility programme
- **Title:** Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)
- **Authors:** Joelle Pineau; Philippe Vincent-Lamarre; Koustuv Sinha; Vincent Lariviere; Alina Beygelzimer; Florence d'Alche-Buc; Emily Fox; Hugo Larochelle
- **Source/year:** Journal of Machine Learning Research 22(164):1–20, 2021; JMLR paper 20-303
- **Supports:** reproducibility/checklist practice as established ML methodology.
- **Novelty effect:** LAM-JEPA may contribute a strong local evidence package, but reproducibility practice is not invented here.
- **Status:** **VERIFIED**.

### Optional context — VICReg
- **Title:** VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning
- **Authors:** Adrien Bardes; Jean Ponce; Yann LeCun
- **Source/year:** arXiv:2105.04906, 2021
- **Use only if needed:** the generic repository loss library contains variance/covariance terms, but the frozen ARC benchmark `_lam_arc_loss` does **not** use those terms. Do not cite VICReg as if those terms were part of the reported ARC result.
- **Status:** **VERIFIED / NOT REQUIRED FOR CURRENT ARC RESULT**.

## A. JEPA / target-encoder positioning

**Status: CLOSED FOR CURRENT MANUSCRIPT.**

I-JEPA provides the canonical comparison. It predicts representations of distinct masked target blocks from a context block. In the frozen LAM ARC forward, `target_z` is computed by sending the **same `tokens` and `numeric_x`** through the EMA target encoder/projector. The ARC-specific objective aligns `z_q` to that same-input target.

Required wording: use the project identifier **LAM-JEPA** while describing the tested mechanism as **same-input EMA target alignment**. Do not imply that the ARC experiment instantiated canonical context-to-future/masked-target JEPA prediction.

## B. Discrete / latent-action positioning

**Status: CLOSED FOR CURRENT MANUSCRIPT.**

VQ-VAE establishes vector-quantized latent learning. LAPA and later latent-action world-model work establish learned latent actions as an active prior direction. FF-JEPA further blocks generic novelty claims around latent planning in a JEPA/world-model setting.

The repository's quantizer repair remains an engineering/trainability result under its own frozen gate, not evidence that vector quantization or latent actions are new or generally superior.

## C. ARC and pretrained comparator

**Status: CLOSED FOR CITATION; RAW ARTIFACT PROVENANCE REMAINS PARTIAL.**

Use RW-06 for ARC provenance and RW-07 for the DeBERTaV3 family. The exact pinned comparator revision remains `microsoft/deberta-v3-xsmall@14809e4f1fe1895fcba8b258271a940c6ca45ec4` from repository evidence.

Do not compare this validation result to unrelated public test-set/SOTA numbers. The locked ARC confirmatory test remains untouched for the failed hypothesis.

## D. Ablation and capacity matching

**Status: SOURCE-LOCAL CLAIM; NO NOVELTY CLAIM.**

The manuscript may state exactly what the experiment did: gradient-active parameter matching, `no_planner`, `no_target`, and shuffled-label controls. It must not claim this matching/ablation methodology as novel.

A key semantic warning must remain: `no_target` replaces the EMA target with `z.detach()` while the alignment term remains present. It is therefore an EMA-target-path ablation, not “target prediction versus no target objective.”

## E. Reproducibility / negative-result practice

**Status: SUFFICIENT FOR CURRENT DRAFT.**

RW-08 supplies general reproducibility context. The project's stronger statements are local and artifact-backed: frozen protocols, retained adverse evidence, hashes, repeat reruns, bug-before/fix-after lineage, and a stop rule. The manuscript must not claim preregistration or negative-result reporting as an invention.

Venue-specific reproducibility/checklist citations may be added after a target venue is chosen.

## F. Determinism and floating-point reproducibility

**Status: EMPIRICAL CLAIM CLOSED; OPTIONAL FRAMEWORK DOCUMENTATION IF DISCUSSED.**

Required wording:

> Independent reruns reproduce the aggregate scientific conclusion and verifier outputs; low-level probabilities/checkpoint bytes are not claimed bitwise identical across all independent runners.

If the final manuscript makes framework-level claims about determinism, verify and cite the then-current official PyTorch reproducibility documentation. Do not infer hardware-portable bitwise determinism.

## Frozen ARC source warnings

These must survive editorial cleanup:

1. `text_to_tokens` lowercases/splits whitespace and maps tokens deterministically with BLAKE2b into a 256-entry vocabulary; ARC uses maximum length 96.
2. ARC `numeric_x` is zero for every example, so the numeric branch supplies no example-varying ARC information.
3. `TokenEncoder.encoder = nn.Identity()`; the tested token path is embedding + learned positions + LayerNorm + mean pooling, not a Transformer encoder.
4. The ARC-specific objective is `CE + 0.5*alignment + 0.25*quantization + 0.25*trajectory`; the larger generic repository `total_loss` is not the reported ARC training objective.
5. The alignment target is the EMA target encoder/projector on the same input, not a separate future/masked target view.
6. The planner trajectory loss pulls rollout states toward `z_q.detach()`; it is not direct held-out-future supervision.

## Citation acceptance gate

Any additional reference enters the manuscript only when all fields are recorded:
- exact title;
- authors;
- year;
- venue/source;
- persistent identifier or official primary-source location;
- exact claim it supports;
- whether it weakens or changes novelty wording.

No generated, memory-only, or secondary-summary citation is permitted into the final bibliography.
