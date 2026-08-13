# Claim-boundary CI recheck — 13 August 2026

PR #72 initially ran the `Research claim boundary` workflow against branch commit `a835d27d9a767c3b41d963d0dde2d88fc48b3c86` and failed while verifying `RESEARCH_STATUS.md`.

The unit tests passed, but the repository-level guard reported two missing exact safety fragments:

- `ARC SUPERIORITY HYPOTHESIS UNSUPPORTED`
- `The locked ARC confirmatory test must not be used to rescue the failed validation hypothesis.`

This was a documentation/claim-boundary failure, not a scientific execution failure. No ARC data, model, seed, metric, threshold, ablation, verifier, result artifact, or locked-test policy was changed in response.

The current branch head restores both guard fragments verbatim in `RESEARCH_STATUS.md` while retaining the negative/inconclusive scientific conclusion and the unused locked ARC test. This file preserves the failed CI event as evidence rather than erasing it.

The next PR CI run is expected to re-evaluate the claim boundary on the current exact head. A passing claim-boundary check is required before this evidence-only PR should be merged.
