# papers

This directory contains paper artifacts and generated publication evidence.

## Canonical current scientific manuscript

The current evidence-backed ARC paper is the root document:

- `../MANUSCRIPT_DRAFT_NEGATIVE_ARC.md` — **canonical working manuscript** for the frozen ARC superiority/mechanism hypothesis line.
- `MANUSCRIPT_RESULTS_20260813.md` — conservative manuscript-ready results text.
- `../ORIGINALITY_AUDIT.md` — closest-work and novelty-boundary audit.
- `../REVIEWER_ATTACK_20260814.md` — three-reviewer skeptical attack.
- `../FIGURE_PROVENANCE.md` — publication figure/table provenance contract.

The scientific result is negative/inconclusive: LAM-JEPA does not beat the retained gradient-active-parameter-matched supervised baseline in mean frozen ARC validation accuracy, and the planner/target mechanism gates are unsupported. The locked ARC test remains unused for this failed hypothesis line.

## Legacy paper warning

The repository also contains historical root artifacts `../paper.tex` and `../paper.pdf` describing a broader positive architectural vision for adaptive educational reasoning, planning, verification, grokking-style generalization, and related mechanisms. **Those files are superseded as scientific claim sources for the current ARC line.** They predate the final frozen negative evidence and contain aspirational or unvalidated mechanism/product language that must not be cited as empirical findings.

Preserve those artifacts as provenance; do not delete them. Before any public research release, either archive them explicitly as historical design documents or regenerate the public-facing paper artifact from the canonical negative manuscript. A stale positive `paper.pdf` must not be presented as the current evidence-backed paper.

## Generation policy

Generated reports, previews, tables and figures belong under this directory when the relevant scripts are run. Every evidence-bearing paper figure/table must trace to raw artifacts and include source/generator hashes as specified in `../FIGURE_PROVENANCE.md`.
