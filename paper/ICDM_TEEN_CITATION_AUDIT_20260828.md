# ICDM 2026 Teen — Citation and Related-Work Audit

**Scope:** `paper/icdm_teen_2026.tex` only.  
**Date:** 2026-08-28.  
**Scientific boundary:** bibliographic verification only. This file does not change any experiment, metric, threshold, seed, split, or scientific verdict.

## Audit rule

Every citation used by the compact ICDM manuscript must (a) resolve to a real primary publication/source, (b) support the exact nearby sentence, and (c) not be used to imply that the frozen LAM-JEPA ARC implementation is equivalent to the cited method.

| Key | Primary source verified | Nearby manuscript use | Verdict |
|---|---|---|---|
| `assran2023ijepa` | Assran et al., *Self-Supervised Learning From Images With a Joint-Embedding Predictive Architecture*, CVPR 2023, pp. 15619–15629. https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html | I-JEPA predicts representations of distinct target blocks from a context block; used only to distinguish canonical I-JEPA from this repo's same-input target-alignment path. | **VERIFIED / bounded** |
| `vandenOord2017vqvae` | van den Oord, Vinyals, Kavukcuoglu, *Neural Discrete Representation Learning*, NeurIPS 2017. https://papers.nips.cc/paper/7210-neural-discrete-representation-learning | Establishes that vector-quantized latent representation learning predates this project. | **VERIFIED** |
| `ye2024lapa` | Ye et al., *Latent Action Pretraining from Videos*, arXiv:2410.11758 (2024). https://arxiv.org/abs/2410.11758 | Establishes a modern latent-action learning precedent. It is not claimed to be method-equivalent to this ARC implementation. | **VERIFIED / bounded** |
| `clark2018arc` | Clark et al., *Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge*, arXiv:1803.05457 (2018). https://arxiv.org/abs/1803.05457 | Identifies ARC/ARC-Challenge and its science QA setting. | **VERIFIED** |
| `he2023debertav3` | He, Gao, Chen, *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing*, ICLR 2023. https://openreview.net/forum?id=sE7-XhLxHA | Identifies the model family of the pinned development comparator. The manuscript does not claim a compute-matched or broad superiority comparison against DeBERTaV3. | **VERIFIED / bounded** |
| `pineau2021reproducibility` | Pineau et al., *Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)*, JMLR 22(164):1–20, 2021. https://jmlr.org/papers/v22/20-303.html | Supports the general statement that formal reproducibility programs/practices are established in ML. | **VERIFIED** |

## Related-work boundary

The compact manuscript intentionally does **not** claim novelty for JEPA, vector quantization, latent actions, ARC, pretrained language models, or reproducibility practice. Its defensible contribution is the evidence discipline around a frozen negative result: matched comparison, mechanism controls, retained adverse evidence, source audit, provenance chain, and a stop rule that keeps the confirmatory test locked after validation failure.

Two bibliography entries (`garrido2026latentaction`, `masip2026ffjepa`) are retained in the shared `.bib` for the longer manuscript but are not cited by the compact ICDM source. Their mere presence must not be interpreted as a claim or as related-work coverage in the submitted paper.

## Release gate

Before upload, verify mechanically that every `\\cite{...}` key used in `paper/icdm_teen_2026.tex` exists in `paper/references.bib`; verify that no citation key is silently dropped during bibliography generation; and inspect the compiled PDF to ensure all references resolve (no `?` citation markers). This audit does not prove successful compilation or the 5-page limit.