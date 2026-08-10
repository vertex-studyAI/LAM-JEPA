# LAM-JEPA release provenance and claim boundary

This document records repository-verifiable provenance for the current research package. It intentionally does **not** supply a license, author list, citation metadata, or ownership statement that has not been explicitly approved by the repository owner.

## 1. Repository implementation surface

The installable package is defined by `pyproject.toml` and loads Python modules from `src/lam_jepa/`. Reproducibility and research execution are implemented through repository-local scripts under `scripts/`, frozen protocols under `protocols/`, tests under `tests/`, and exact-head GitHub Actions workflows under `.github/workflows/`.

This identifies where the implementation lives. It does not independently establish original authorship or legal ownership of every historical line of code.

## 2. Current external validation dataset

The frozen ARC-v5 repaired-validation protocol declares:

- dataset: `allenai/ai2_arc`, `ARC-Challenge`;
- train split SHA-256: `e488c1587ffdcfc8443f916c53488a95cd471c5790e0746c6bfe4cecf20962cb`;
- validation split SHA-256: `395a5c88d1580d69855fbaee9450270578df1ad5af6259771cd0a42c20e99f05`;
- train source rows: `1119`;
- train eligible rows: `1117`;
- validation source rows: `299`;
- validation eligible rows: `295`;
- eligibility rule: exactly four answer choices using the frozen protocol implementation;
- confirmatory test policy: the test split must not be downloaded, opened, evaluated, or used for selection.

Canonical protocol: `protocols/arc_challenge_v5_repaired_validation.json`.

The protocol is explicitly development-validation evidence, not confirmatory test evidence.

## 3. Frozen scientific question and conditions

The repaired-v5 validation asks whether the independently reproduced ARC-v5 trainability repair generalizes to the frozen ARC-Challenge validation split under supervised cross-entropy training without returning to input-insensitive one-class collapse.

The frozen comparison conditions are:

- legacy quantized CE classifier;
- repaired-v5 quantized CE classifier;
- no-quantizer CE classifier with memory/planner retained;
- repaired-v5 shuffled-label negative control.

The predeclared decision rules and claim boundaries live in the protocol file and must not be loosened after observing validation results.

## 4. Current scientific verdict boundary

Repository verification preserves the ARC-v5 validation outcome as negative or inconclusive rather than converting it into a positive result by post-hoc tuning.

The frozen protocol explicitly leaves these claims unauthorized:

- planner-mechanism claim;
- target-mechanism claim;
- original hard-VQ mechanism claim;
- external-generalization claim;
- confirmatory-test claim;
- `RESEARCH_COMPLETE` status.

Independent result reproduction is required, further v5 hyperparameter tuning on validation is prohibited, and confirmatory test access requires a separate explicit authorization boundary.

## 5. Runtime/package dependencies

The current installable project declares these core Python dependencies in `pyproject.toml`:

- PyTorch (`torch>=2.2`);
- NumPy (`numpy>=1.26`);
- tqdm (`tqdm>=4.66`);
- SymPy (`sympy>=1.12`);
- scikit-learn (`scikit-learn>=1.4`);
- Matplotlib (`matplotlib>=3.8`);
- pandas (`pandas>=2.2`).

Optional dependency groups expose `pyarrow` for external benchmarks and pinned `transformers` / `sentencepiece` packages for pretrained-baseline work. Their presence in package metadata does not by itself prove that a particular frozen experiment used a pretrained checkpoint; experiment-specific provenance must come from the corresponding protocol/artifact.

## 6. Reproduction entry points

General repository execution is documented in `README.md`. The ARC-v5 validation-specific executable/verification paths include:

- `scripts/benchmark/run_arc_v5_repaired_validation.py`;
- `scripts/ci/verify_arc_v5_repaired_validation.py`;
- `scripts/ci/verify_arc_v5_validation_protocol.py`;
- `protocols/arc_challenge_v5_repaired_validation.json`;
- `.github/workflows/arc-v5-repaired-validation.yml`;
- `.github/workflows/arc-v5-validation-protocol-freeze.yml`.

Any release reproduction command should be pinned to an immutable release commit/tag once the owner approves the release package and the remaining release gates are closed.

## 7. Legal and bibliographic blockers that remain intentionally unresolved

The following items are **not** filled in automatically:

1. **Root license** — the owner must select/approve the license and confirm compatibility with third-party code/data obligations.
2. **`CITATION.cff` author list and release metadata** — names, authorship order, release title/version, identifiers, and publication metadata require explicit owner approval.
3. **Historical code-origin attestation** — if a formal release requires per-component authorship or third-party code provenance beyond what Git history/repository metadata establishes, that attestation must be supplied by the owner/maintainers.
4. **Dataset licensing/redistribution review** — this document records the dataset identity and hashes used by the protocol; it does not grant redistribution rights.

These blockers must not be bypassed merely to make the repository look release-ready.

## 8. Release rule

Passing CI proves executable repository paths and reproducibility checks. It does not prove benchmark validity, educational effectiveness, novelty, model superiority, legal packageability, or research completion.

A release should link an immutable source revision, the exact frozen protocol, retained raw/summary evidence, independent verification, approved legal metadata, and the scientific claim boundary together as one package.
