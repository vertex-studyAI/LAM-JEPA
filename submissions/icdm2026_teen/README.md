# ICDM 2026 Teen Research Track — LAM-JEPA submission workspace

This directory is the only venue-format workspace authorized to represent the frozen negative/inconclusive ARC line.

## Canonical scientific source

- `../../MANUSCRIPT_DRAFT_NEGATIVE_ARC.md`
- scientific source SHA: `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`
- current release branch base: `16a0c8ba675acc7366af51a15f58bf26a92a39b9`

The root `paper.tex` and `paper.pdf` are pre-falsification architecture artifacts. They are **not** submission sources and must not be copied forward without full claim reconciliation.

## Venue facts re-verified 2026-08-23

Official ICDM Teen Research Track call: submission deadline August 30, 2026; IEEE Computer Society Proceedings format; maximum five pages including figures, tables, and references; first author must be an enrolled high-school student and primary contributor; first-author affiliation must explicitly contain `High School Student`; review is single-blind.

## Working source

`lam_jepa_negative.tex` is an evidence-bounded IEEE-format draft derived from the canonical negative manuscript. It deliberately retains an author-metadata placeholder. That placeholder is a release gate, not missing technical prose.

## Automated checks

`.github/workflows/icdm-teen-paper-ci.yml`:

1. checks required negative-result/locked-test language;
2. rejects several known unsupported positive-claim phrases;
3. compiles the IEEE source twice with `pdflatex`;
4. reads the generated PDF page count;
5. fails if the result exceeds five pages;
6. fails on hard LaTeX errors.

## Owner-controlled blockers before submission

Do not guess or auto-fill these:

- final author list and order;
- confirmation that first author is an enrolled high-school student and primary contributor;
- first-author affiliation containing `High School Student`;
- contact email and any contribution statement;
- repository/license choice and release metadata;
- final citation metadata.

## External/submission blockers

- independent skeptical review/reproduction status must be recorded truthfully;
- submission account must be accessible and Teen Research Track selected;
- final PDF metadata and portal author metadata must agree;
- submission receipt/paper ID must be retained after upload.

## Scientific stop rule

Do not rerun, retune, cherry-pick, or open the locked ARC confirmatory test to seek a positive result for this frozen line. Any scientifically changed architecture or protocol is a new versioned study.
