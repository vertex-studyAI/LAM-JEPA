# Shared bibliography: primary-source metadata audit

Review date: 26 September 2026. Scope: the eight records in `paper/references.bib` at source commit `7f3de951975fa5ea1d2fbb3a3ae7f87178c4bb83`.

**Outcome: bibliographic identity and existing publication metadata verified for 8/8 entries.** This is not a full-paper citation-support audit, a successful TeX build, a scientific replication, or submission approval.

## Relationship to the earlier audit

`paper/ICDM_TEEN_CITATION_AUDIT_20260828.md` verified six entries used by the compact ICDM source. It explicitly excluded the Garrido and FF-JEPA entries from that manuscript's coverage. This record covers the shared bibliography's eight entries, including those two, without superseding the older sentence-level review or its scope.

## Verified primary records

| Citation key | Metadata checked | Primary source |
|---|---|---|
| `assran2023ijepa` | Title, eight authors in order, CVPR 2023, pages 15619-15629. | https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html |
| `vandenOord2017vqvae` | Title, three authors in order, NeurIPS 2017, volume 30. | https://proceedings.neurips.cc/paper/2017/hash/7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html |
| `ye2024lapa` | Title, sixteen authors in order, arXiv:2410.11758, initial 2024 publication. | https://arxiv.org/abs/2410.11758 |
| `garrido2026latentaction` | Title, six authors in order, arXiv:2601.05230, 2026. Latest version displayed at review: v2, 20 January 2026. | https://arxiv.org/abs/2601.05230 |
| `masip2026ffjepa` | Title, five authors in order, arXiv:2606.09311, 2026. Version displayed at review: v1, 8 June 2026. | https://arxiv.org/abs/2606.09311 |
| `clark2018arc` | Title, seven authors in order, arXiv:1803.05457, 2018. This is AI2 science question answering, not ARC-AGI. | https://arxiv.org/abs/1803.05457 |
| `he2023debertav3` | Title, three authors in order; the authors' arXiv record explicitly reports publication at ICLR 2023. The 2021 preprint date does not invalidate the existing 2023 conference citation. | https://arxiv.org/abs/2111.09543 |
| `pineau2021reproducibility` | Title, eight authors in order, JMLR 22(164):1-20, 2021. | https://jmlr.org/papers/v22/20-303.html |

LAPA also has an official ICLR 2025 proceedings record: https://proceedings.iclr.cc/paper_files/paper/2025/hash/45d74e190008c7bff2845ffc8e3facd3-Abstract-Conference.html . The existing 2024 preprint citation is valid and is intentionally retained. Switching cited versions would be an editorial choice, not correction of a nonexistent reference. No acceptance status is inferred for the two 2026 preprints.

## Patch and local checks

The accompanying change only adds a primary-source `url` field to each bibliography record. Citation keys, entry types, titles, authors, years, venues, volume/issue/page fields and ordering are unchanged. No new reference or scientific claim is added.

The reconstructed original was checked against its Git blob SHA-1, `12da73fca87e40d29563b4a9e4acdd09a0fd8c91` (2756 bytes), before editing. Local checks passed: eight entries and eight added URL fields; original key sequence preserved; balanced braces; removing exactly the new URL fields restores the original bytes. Updated bibliography SHA-256: `3bc7be36e1b0a173fd8caeb4ceee9c8266b11435dc8c91ce736b020c3a720a00`.

These are structural checks, not a BibTeX compiler invocation. No model, dataset, experiment, held-out split, training seed or result generator was executed. CI is intentionally not represented as green; this documentation-only change remains a draft pending the ordinary authorized build/review path.

## What remains open

The current venue manuscript still needs sentence-by-sentence source support review, all citation keys checked against the exact chosen TeX revision, an authorized clean bibliography/PDF build, and visual inspection for link wrapping and unresolved citations. This record does not approve authorship, author order, licensing, redistribution, venue selection, release or submission. It does not supply a submission receipt.

Preserve the frozen AI2 ARC-Challenge negative/inconclusive result, its local collapse interpretation, all adverse evidence and the locked confirmatory-test boundary. Bibliographic completion is not scientific or publication completion.
