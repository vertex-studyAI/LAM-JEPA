# MANUSCRIPT CITATION MAP — NEGATIVE ARC DRAFT

**Date:** 2026-08-14  
**Target:** `MANUSCRIPT_DRAFT_NEGATIVE_ARC.md`  
**Rule:** use these sources only for the claims listed here; do not turn related-work citations into evidence for LAM-JEPA performance.

## Verified primary sources

| Key | Source | Verified scope | Safe use in manuscript |
|---|---|---|---|
| `assran2023ijepa` | Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*, arXiv:2301.08243 | I-JEPA predicts target-block representations from context-block representations instead of reconstructing pixels; demonstrates JEPA representation learning in vision | Establish JEPA prior art and the core target/context latent-prediction motif |
| `assran2025vjepa2` | Assran et al., *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*, arXiv:2506.09985 | V-JEPA 2 uses predictive representations; V-JEPA 2-AC is action-conditioned and is used for planning in robotic tasks | Establish that action-conditioned JEPA-style latent planning predates this LAM-JEPA manuscript |
| `huang2025llmjepa` | Huang, LeCun, Balestriero, *LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures*, arXiv:2509.14252 / NeurIPS 2025 UniReps | JEPA objectives applied to LLM pretraining/finetuning and evaluated on language/reasoning datasets including GSM8K | Establish that JEPA for language/reasoning is prior art; do not claim first JEPA reasoning model |
| `bui2025textjepa` | Bui et al., *Speaking in Words, Thinking in Logic: A Dual-Process Framework in QA Systems*, arXiv:2507.20491 | Describes Text-JEPA-style natural-language-to-logic representation pipeline for QA | Establish related JEPA-inspired reasoning/application work |
| `schmidt2024lapo` | Schmidt & Jiang, *Learning to Act without Actions*, ICLR 2024 | Introduces Latent Action Policies (LAPO), recovering latent action information and policies/world models from observations without action labels | Establish latent-action learning as prior art |
| `ye2024lapa` | Ye et al., *Latent Action Pretraining from Videos*, arXiv:2410.11758 | Learns discrete latent actions with a VQ-style objective and uses them for downstream VLA pretraining | Establish discrete latent action pretraining as prior art |
| `rybkin2019clasp` | Rybkin et al., *Learning what you can do before doing anything*, ICLR 2019/OpenReview | Learns composable latent action representations from observations for prediction/planning | Establish earlier learned latent action-space literature |
| `power2022grokking` | Power et al., *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*, arXiv:2201.02177 | Establishes delayed generalization after overfitting on small algorithmic datasets | Establish grokking as prior phenomenon, not LAM-JEPA novelty |
| `oord2017vqvae` | van den Oord et al., *Neural Discrete Representation Learning*, arXiv:1711.00937 | Establishes vector-quantized discrete latent representations/codebooks | Establish quantization/codebooks as standard prior art |

## Related-work section replacement outline

### Paragraph 1 — JEPA lineage

State that JEPA methods predict representations rather than reconstructing raw observations, with I-JEPA as a foundational example. Then note that the family expanded to video/predictive planning and, by 2025, language/reasoning applications. Cite `assran2023ijepa`, `assran2025vjepa2`, and `huang2025llmjepa`; optionally use `bui2025textjepa` for domain-adjacent QA reasoning.

**Do not say:** “JEPA has not been applied to language/reasoning.” That statement is false by the current literature.

### Paragraph 2 — latent actions and planning

State that learning or using latent action representations is an established line, including CLASP, LAPO, and LAPA. V-JEPA 2-AC further demonstrates action-conditioned predictive latent planning. LAM-JEPA’s distinction is the use of such motifs in a frozen educational/reasoning stack, not the invention of latent actions themselves.

Cite `rybkin2019clasp`, `schmidt2024lapo`, `ye2024lapa`, and `assran2025vjepa2`.

### Paragraph 3 — discrete bottlenecks and grokking

State that vector quantization and discrete latent codes are established by VQ-VAE-style work, while grokking is a known delayed-generalization phenomenon on algorithmic tasks. LAM-JEPA combines these ideas, but the current ARC experiment does not establish that either mechanism causes improved generalization.

Cite `oord2017vqvae` and `power2022grokking`.

### Paragraph 4 — contribution boundary

Explicitly distinguish this manuscript from positive architecture papers:

> The contribution evaluated here is not a claim that JEPA, latent actions, quantization, or planning are new. It is a falsification-first evaluation of one frozen combination under matched controls, with retained mechanism ablations and a locked confirmatory test.

This paragraph needs no novelty flourish. Its strength is the evidence discipline.

## Claims requiring additional source verification before submission

The following topics still need a focused primary-source audit before final prose:

- matched-capacity architecture comparison methodology;
- preregistration/frozen-evaluation practices in machine learning;
- publication/reporting of negative results in ML;
- ARC-Challenge benchmark provenance and the exact dataset citation/licensing record;
- DeBERTa-v3 primary model citation and license/revision metadata;
- any claim about educational tutoring, automated grading, rubric modeling or student-state modeling.

Until those sources are checked, keep `[CITATION TODO]` markers rather than filling references from memory.

## Bibliographic URLs checked

- https://arxiv.org/abs/2301.08243
- https://arxiv.org/abs/2506.09985
- https://arxiv.org/abs/2509.14252
- https://arxiv.org/abs/2507.20491
- https://openreview.net/forum?id=rvUq3cxpDF
- https://arxiv.org/abs/2410.11758
- https://openreview.net/forum?id=SylPMnR9Ym
- https://arxiv.org/abs/2201.02177
- https://arxiv.org/abs/1711.00937

## Result

The Related Work TODO is now reduced from “unknown literature” to a bounded insertion task. The safest novelty framing remains **reproducible negative empirical observation + disciplined provenance**, not mechanism firstness.
