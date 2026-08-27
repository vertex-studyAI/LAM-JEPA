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
- `../EXTERNAL_VALIDATION_PACKET_20260814.md`

If any numerical or scientific discrepancy is found, those evidence/provenance files and the frozen artifacts take precedence over this typesetting layer.

## Remaining release gates

The TeX source intentionally leaves authorship unresolved. Before any public submission or tagged release, the owner must approve:

- author names and order;
- release/license metadata and third-party compatibility;
- final citation metadata;
- venue-specific formatting/required declarations.

Independent external reproduction is also still pending and must not be claimed as complete until performed by a genuinely independent party.

## Scientific boundary

The supported conclusion is a reproducible negative/inconclusive result for the tested frozen ARC configuration. Do not claim ARC superiority, planner benefit, EMA-target benefit, general JEPA failure, Transformer reasoning capability, quantization/generalization benefit, or successful use of the locked confirmatory test.
