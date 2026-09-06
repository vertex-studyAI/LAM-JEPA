# Submission-format paper source

This directory is a typesetting layer over the frozen evidence-backed manuscript. It does **not** change the scientific result, seeds, protocol, architecture, metrics, or locked-test state.

## Build

With a standard TeX Live installation:

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Expected output: `paper/main.pdf`.

## Source of truth

Scientific prose and claim boundaries originate from:

- `../MANUSCRIPT_DRAFT_NEGATIVE_ARC.md`
- `../PAPER_FINALIZATION_20260822.md`
- `../MANUSCRIPT_PROVENANCE.md`
- `../REPRODUCE.md`
- `EXTERNAL_REVIEW_VQ_COLLAPSE_CORRECTION_20260831.md`
- `../CLAIM_LEDGER.md`

If any numerical or scientific discrepancy is found, those evidence/provenance files and the frozen artifacts take precedence over this typesetting layer.

## Remaining release gates

The TeX source intentionally leaves authorship unresolved. Before any public submission or tagged release, the owner must approve:

- author names and order;
- release/license metadata and third-party compatibility;
- final citation metadata;
- venue-specific formatting/required declarations.

One genuinely external frozen-protocol rerun/review has already reproduced the retained headline metrics and supplied the bounded VQ-collapse diagnosis recorded in `EXTERNAL_REVIEW_VQ_COLLAPSE_CORRECTION_20260831.md`. That evidence must be described only at one-reviewer/frozen-protocol scope; it is not broad multi-site replication or peer review. A second independent rerun/reviewer remains a stronger promotion gate if the project is to claim broader external reproducibility.

## Scientific boundary

The supported conclusion is a reproducible negative/inconclusive result for the tested frozen ARC configuration, sharpened by one bounded external failure-mechanism review. Do not claim ARC superiority, planner benefit, EMA-target benefit, general JEPA failure, general vector-quantization failure, Transformer reasoning capability, quantizer-off task success, or successful use of the locked confirmatory test.
