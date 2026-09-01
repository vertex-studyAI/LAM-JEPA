# LAM-JEPA bibliography audit

Audit date: 2026-09-01

This ledger verifies the identity and retained use of every entry in `references.bib` against a primary publication record. It does not claim that any cited method was reproduced, budget-matched, or executed as an experimental comparator in this repository. All eight citations are contextual or methodological background only.

| BibTeX key | Primary record | Identity reconciliation | Retained manuscript use | Experimental status |
|---|---|---|---|---|
| `assran2023ijepa` | [CVPR 2023 Open Access](https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html) | Title, eight authors, venue, year, and pages 15619–15629 agree. | Establishes canonical I-JEPA context-to-distinct-target prediction and EMA-target background; the manuscript explicitly distinguishes its same-input alignment. | Context only; not reproduced or run as a comparator. |
| `vandenOord2017vqvae` | [NeurIPS 2017 proceedings](https://proceedings.neurips.cc/paper/2017/hash/7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html) | Title, three authors, venue volume 30, and year agree. | Establishes that vector-quantized discrete latent learning predates LAM-JEPA. | Context only; not reproduced or run as a comparator. |
| `ye2024lapa` | [arXiv:2410.11758](https://arxiv.org/abs/2410.11758) | Title, 17-author list, identifier, and 2024 first-submission year agree. | Supports the statement that latent-action pretraining learns discrete latent actions from videos. | Context only; no LAPA experiment was executed. |
| `garrido2026latentaction` | [arXiv:2601.05230](https://arxiv.org/abs/2601.05230) | Title, six authors, identifier, and 2026 first-submission year agree. | Context for constrained latent actions and planning from action-free in-the-wild video. | Context only; no matched comparison was executed. |
| `masip2026ffjepa` | [arXiv:2606.09311](https://arxiv.org/abs/2606.09311) | Title, five authors, identifier, and 2026 first-submission year agree. | Context for an action-free latent planner paired with forward dynamics for long-horizon planning. | Context only; no FF-JEPA experiment was executed. |
| `clark2018arc` | [arXiv:1803.05457](https://arxiv.org/abs/1803.05457) | Title, seven authors, identifier, and 2018 first-submission year agree. | Establishes the identity and stated purpose of the AI2 Reasoning Challenge benchmark. | Dataset context only; the confirmatory ARC test remains locked and unopened. |
| `he2023debertav3` | [ICLR 2023 OpenReview](https://openreview.net/forum?id=sE7-XhLxHA) | Title, three authors, venue decision, and 2023 publication year agree. | Identifies the model family of the pinned DeBERTa-v3-xsmall bounded comparison. | The cited family is contextual; only the repository-pinned xsmall checkpoint comparison was executed. |
| `pineau2021reproducibility` | [JMLR 22(164)](https://jmlr.org/papers/v22/20-303.html) | Title, eight authors, volume 22, article 164, pages 1–20, and 2021 year agree. | Establishes prior machine-learning reproducibility programs and checklists. | Methodological context only. |

## Claim boundary

- The ledger verifies bibliographic identity, not novelty, priority, correctness, or reproducibility of the cited works.
- No cited result is treated as a strong matched baseline unless it was actually executed under the frozen LAM-JEPA protocol; none of these eight cited papers meets that condition.
- The only executed pretrained comparison remains the bounded repository-pinned DeBERTa-v3-xsmall diagnostic already described in the manuscript; it is not evidence of broad model-family inferiority.
- The ledger supplies no positive LAM-JEPA evidence and does not alter `VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION`.
- The matched supervised comparator was not outperformed, mechanism benefits remain unsupported, and the confirmatory ARC test remains locked and unopened.
