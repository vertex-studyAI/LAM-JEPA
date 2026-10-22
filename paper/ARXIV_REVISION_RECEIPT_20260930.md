# LAM-JEPA manuscript revision: bounded execution receipt

Date: 2026-09-30. Scope: close the supplied manuscript's feasible editorial, method-specification, reported-count analysis and build defects. The original ARC-v3 negative/inconclusive result and separately versioned ARC-v5 follow-up remain unchanged.

## Disposition

**VERIFIED_MANUSCRIPT_REVISION / RELEASE_BLOCKED / NOT_SUBMITTED**

This branch records the closeout and an executable synthetic source check. The complete revised manuscript, four generated figure PDFs, source archive and 21-test analysis bundle were delivered as conversation files; they are not implicitly committed by this receipt. It does not replace `paper/main.tex`, approve authorship/licensing, merge, submit, publish a release, or authorize protected evaluation.

Repository head inspected: `49982fd2196e7045e9b54c2413917b3b73c71218`. Frozen scientific source: `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`. Original full-controls workflow checkout observed in job 94291056903: `ed81a16c5b2e3379eb37c4d94a79941d0cd0ff10`.

## Delivered artifact identities

- `LAM_JEPA_revised_manuscript.pdf`: 13 pages; SHA-256 `56de0370d8c666c4137e3cf3010c86e39da0b09491c6b710b4df9a6e9ca9d558`.
- `LAM_JEPA_revised_source_candidate.zip`: eight actual build inputs; SHA-256 `3ca90275e33f5abb3b704accba833474b7b7ab90c6f939d1c577dcfe7eb2c8e7`.
- Revised `main.tex`: SHA-256 `4392cec038ba35ce7dd479bd25b25076b5ab7b3e6bcf03dc64e0b3b142b27744`.

These names identify the conversation deliverables, not repository download URLs. The source archive contains main.tex, matching main.bbl, references.bib, generated macros and four quantitative figure PDFs. The larger downloadable bundle retains analysis scripts, reported evidence, tests, receipts and the original-to-revised manuscript patch.

## Actual revisions and verification

Completed the quantizer loss/initialization/EMA specification, corrected the trajectory MSE reduction and teacher input diagram, documented memory and matched-baseline details, disclosed clipping asymmetry and tensor-gradient-count limitations, separated planned from retained protocol evidence, labeled the validation-label oracle correctly, and preserved adverse matched/shuffled/repair outcomes.

Four figures were generated from explicit reported counts/summaries, not invented measurements. Original component intervals enumerate all 3,125 paired bootstrap resamples. Twenty-one local unit tests pass. Nine citations resolve in both BIB and BBL. Repeated PDFLaTeX compilation and strict source/log checks pass without missing dependencies, unresolved references/citations or overfull boxes. All 13 final pages were visually inspected. A fresh-directory build from the exact source ZIP passes and matches the inspected PDF's text and rendered pixels on all 13 pages. The precompiled BBL was supplied; BibTeX itself was not newly run.

No ARC models were trained, no protected evaluation was opened, no dataset was downloaded, and no new independent scientific replication was performed. Existing summaries are not relabeled as newly reproduced per-example outputs.

## Additional bounded source finding

The inspected EMAQuantizer initializes counts to zero and EMA weights independently of the codebook. For a code receiving no assignments in its first training forward pass, smoothing followed by the epsilon clamp yields an overwrite of `0.99 / 1e-5 = 99,000` times its initial EMA-weight vector. The accompanying exact extracted class and synthetic check verify this before any optimizer step, using PyTorch 2.10.0+cpu. This is an implementation-scale identity, not proof of the cause of the retained ARC collapse or a claim that a repair improves ARC.

Run the bounded check with an already installed PyTorch environment:

```sh
python paper/arxiv_revision_20260930/tools/check_quantizer_identity.py
```

It loads no ARC data and uses only an eight-vector synthetic fixture. Output is written under `paper/arxiv_revision_20260930/verification/`.

## Evidence recovery outcome

Direct download of original full-controls artifact 9162165932 returned HTTP 404. The original full-controls run 31203337502 returned an empty current artifact listing, but job 94291056903's log was recovered. Direct download of matched artifact 9003785715 also returned HTTP 404. These observations do not establish that every copy is lost, or determine the cause of the 404s. Historical digests remain identity records, not proof of present accessibility.

The original external reviewer report was recovered and read. It corroborates the four LAM condition count vectors and bounded collapse observations; it is not a new review of the revised manuscript. Missing matched and repaired per-seed outputs remain missing. No private email metadata or full private attachment is added here.

## Remaining gates and next bounded action

Restore checkable full-controls, matched and repaired per-example evidence/checkpoints; recover unavailable paired comparisons and per-seed diagnostics; specify the collapse-probe tolerance/spread operator and exact quantizer-off intervention; obtain owner approval for final metadata and licensing. Restore old packages under their recorded identities and rerun existing verifiers without accessing the locked test. Any later authorized replay must have its own execution identity rather than impersonating recovery of an old byte-identified archive.

The negative/inconclusive scientific conclusion is preserved. Passing these local checks does not establish novelty, model superiority, causal identification, external replication, acceptance or release readiness.
