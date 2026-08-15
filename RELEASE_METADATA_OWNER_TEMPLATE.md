# LAM-JEPA owner release-metadata handoff

**Purpose:** close the remaining owner-controlled legal/bibliographic fields without inventing authorship, licensing, publication, or external validation.

## Immutable technical package

- Repository: `vertex-studyAI/LAM-JEPA`
- Current source revision reviewed for this handoff: `bf8311e1a4d240e2891e51af38eaf7754944e300`
- Internally verified paper/reproducibility package: `725ae2fb17de9c988938d4b03bd8a6be456b8e8b`
- Immutable external reproduction/review packet commit: `218ea1bea686cdf8c281520b2b636897bc8b8dd2`
- Scientific framing: **reproducible negative result** under the frozen ARC development-validation protocol; no superiority claim and no locked ARC test access.
- External-validation state: **PENDING / NOT YET PERFORMED BY AN INDEPENDENT PARTY**.

The technical reproduction instructions, retained artifact hashes, expected numerical recomputation, deterministic paper-asset command, limitations, non-claims and locked-test boundary are already recorded in `EXTERNAL_VALIDATION_PACKET_20260814.md`. Runtime/package dependency provenance and the remaining legal boundary are recorded in `RELEASE_PROVENANCE.md`.

## Owner decisions required before a public tagged release

Fill these fields only after explicit approval. Leaving them unresolved is preferable to guessing.

| Field | Owner-approved value |
|---|---|
| Release title | `[OWNER APPROVAL REQUIRED]` |
| Release/version identifier | `[OWNER APPROVAL REQUIRED]` |
| Author names | `[OWNER APPROVAL REQUIRED]` |
| Author order | `[OWNER APPROVAL REQUIRED]` |
| ORCID identifiers, if any | `[OWNER APPROVAL REQUIRED / OPTIONAL]` |
| Root software/content license | `[OWNER APPROVAL REQUIRED]` |
| Third-party compatibility review complete | `[YES/NO — OWNER OR MAINTAINER REVIEW REQUIRED]` |
| Dataset redistribution boundary reviewed | `[YES/NO — OWNER OR MAINTAINER REVIEW REQUIRED]` |
| Release date | `[OWNER APPROVAL REQUIRED]` |
| DOI/archive identifier, if later created | `[NONE YET / OWNER TO UPDATE]` |

## Citation metadata template

Do **not** commit a final `CITATION.cff` by substituting repository usernames or commit authors for approved authorship. Once the table above is approved, use this template as the basis for `CITATION.cff`:

```yaml
cff-version: 1.2.0
message: "If you use this reproducibility package, please cite it using the metadata below."
title: "[OWNER-APPROVED RELEASE TITLE]"
type: software
authors:
  - family-names: "[OWNER-APPROVED]"
    given-names: "[OWNER-APPROVED]"
version: "[OWNER-APPROVED VERSION]"
date-released: "[YYYY-MM-DD]"
repository-code: "https://github.com/vertex-studyAI/LAM-JEPA"
```

Add identifiers only after they actually exist. Do not claim a DOI, venue, publication status, peer review, or independent reproduction merely because this metadata file is complete.

## Public claim boundary

A truthful public description at the current evidence state is:

> This repository contains a frozen, internally reproduced negative ARC development-validation result for the evaluated LAM-JEPA configuration, together with retained raw-artifact provenance and an immutable packet for independent reproduction and skeptical review. The result does not establish that JEPA, planning, latent actions, Transformers, or educational systems fail or succeed as broader families. Independent external validation has not yet occurred.

## Release gate

A public tagged release remains **BLOCKED** until the owner approves license/compatibility and authorship/citation metadata. Independent external reproduction is a separate evidence gate: it may remain pending after a source release, but must never be described as completed until a genuinely independent validator returns evidence against the immutable packet.
