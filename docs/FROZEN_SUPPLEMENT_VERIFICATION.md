# Verify the frozen supplement against its source

The internal `BUNDLE_MANIFEST.json` detects inconsistent file changes. It cannot
establish the source of a coherently rewritten archive: changed scientific code
or expected results can be accompanied by new checksums. Use the additional
maintainer-side source check before relying on a supplement as a copy of the
retained frozen package.

```bash
python scripts/paper/verify_frozen_supplement.py /path/to/supplement.zip \
  --report /path/to/fresh-source-verification.jsonl
```

The report path must be new and its parent directory must exist. The JSONL
receipt records `STARTED` before verification and then `FROZEN_SUPPLEMENT_SOURCE_VERIFIED`
or `FROZEN_SUPPLEMENT_REJECTED`. A started receipt without a terminal record is
an incomplete attempt. Existing receipts and the input archive are never
overwritten. Keep this maintainer receipt outside the anonymous package: it
contains repository revision identities and may contain local path information.

## Trusted inputs and exactly what passes

The scientific source revision is
`760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`. The fixed package path list,
generated README, expected-result literals, and declared scientific revision
come from `scripts/paper/build_aistats2027_anonymous_bundle.py` at producer
revision `328d33f5d2e49a72ebdaf042cec318798c606563` (the separate supplement
hardening PR #200). The verifier parses these literals without executing the
producer. The archive and CLI cannot substitute different reference revisions.

The verifier checks raw Git object hashes from each pinned commit through every
necessary tree and blob. It does not trust a loose object's filename or Git's
reported object ID alone. Replacement objects, automatic missing-object fetches,
and network protocols are disabled for Git reads. Use a trusted checkout with
the two revisions and their required objects already present. A shallow or
partial checkout missing those objects fails explicitly; obtain the required
objects separately through the repository's normal Git workflow before retrying.

Every package member is compared byte for byte against the external reference,
including the canonical generated documents and checksum manifest. Missing,
extra, duplicate, aliased, encrypted, nonregular, corrupt, oversized, and
unsupported-compression members are rejected. Stored and DEFLATE compression
are supported. Container timestamps, compression choices, and member order do
not change this source-identity result; the separate package check owns its
deterministic packaging requirements. The receipt hashes the exact bounded ZIP
snapshot inspected and includes each member's SHA-256, length, and scientific
Git blob identity where applicable.

## Reproduce the engineering evidence

With the required objects present, run:

```bash
python -m pytest -q -W error \
  tests/test_frozen_supplement_source.py \
  tests/test_release_artifact_boundary.py \
  tests/test_research_claim_boundary.py

python research/verification/frozen_supplement_source_20261010/reproduce_artifacts.py \
  --output /path/to/fresh-source-audit
```

The second command performs two canonical builds from the pinned producer and
compares their ZIP hashes. It also creates two explicitly invalid engineering
witnesses: one changes the model's planner default, and one changes its declared
expected accuracy to 0.99. Both recompute the internal checksum manifest. The
older package verifier accepts these internally consistent archives; the new
source verifier must reject both and identify the altered member. No model is
imported, trained, or evaluated, and no dataset is downloaded. **The files named
`INVALID-coherent-*.zip` are corruption witnesses and must not be submitted.**

The test suite separately checks corruption of a Git source blob and an
intermediate Git tree, model/working-tree changes, replacement references,
complete package coverage, archive corruption, and receipt preservation.

## Remaining release and scientific boundaries

Source identity is one release check. Run the anonymous-bundle checker from
PR #200 separately; it checks anonymity and deterministic package structure.
The source verifier does not certify anonymity, licensing, authorship, venue
eligibility, manuscript correctness, scientific validity, or submission approval.

The LAM study remains a closed negative / failure-mechanism study within the
frozen tested configuration. Its ARC confirmatory test remains unopened. No
old result, threshold, manuscript number, or scientific claim is updated by this
maintenance work. A new scientific mechanism requires a successor study.
