# Frozen supplement source verification — prospective engineering contract

Recorded 2026-10-10 before implementation in response to the renewed project
pipeline instruction. Parent: `d40a4c4a3a9490a252265b1c54229506dd6a7b3a`.
This is a new bounded maintenance session, not an extension of the previous
statistical-report session or a new scientific evaluation.

## Observation and falsifier

The supplement checker verifies files against a manifest inside the same
archive. A changed model, protocol, or expected-result file can remain internally
consistent after its checksum is recomputed. PR #200 already repairs archive
structure and anonymity checks; those changes remain separately reviewable.

Add a maintainer-side verifier that obtains its reference from independently
pinned Git objects. Its falsifier is a coherently rewritten supplement that
passes the internal-manifest check but also passes the new source comparison.
The new verifier must reject that artifact and identify the mismatched member.

## Exact implementation boundary

Scientific source is fixed at
`760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`. The package recipe's literal
`FIXED_PATHS`, `README`, `EXPECTED`, and `SCIENTIFIC_REVISION` are read with
`ast.literal_eval` from `scripts/paper/build_aistats2027_anonymous_bundle.py`
at reviewed producer revision
`328d33f5d2e49a72ebdaf042cec318798c606563` (PR #200). The producer is not
executed by verification. Neither reference is selected by the archive or by a
CLI override. Git replacement objects and automatic missing-object downloads
must be disabled. Missing reference objects fail explicitly.

The expected package consists of the fixed paths and the frozen package's
Python source files, the two generated literal documents, and the canonical
checksum manifest. Compare every uncompressed member byte against that
external reference. Require exact member coverage and regular unambiguous
members; reject duplicates and unsupported encodings/compression. Bound reads
using reference lengths. ZIP container compression bytes are not scientific
content and need not be identical. Record their hash to identify the checked
artifact. Output the pinned revisions, source blob identities, per-member hashes
and lengths, and the verifier's source hash. Optional reports use exclusive
creation and preserve failed verification attempts. Never extract or execute
archive contents.

Verification proves byte identity to these pinned package inputs. It does not
prove that the retained scientific result is valid, that all publication or
anonymity requirements are satisfied, or that authorship and venue gates have
been approved. Run PR #200's anonymity/structure check separately. No change to
the frozen scientific source, package recipe, paper, expected results, ARC
splits, previous evidence, or study disposition is authorized by this contract.

## Budget, evidence, and stopping rule

- At most 30 distinct crafted archive/reference regression cases.
- At most two real canonical archive builds, with bounded source-only checks
  and coherent-rewrite witnesses; no model execution or data acquisition.
- At most 10 CPU minutes of engineering verification; zero training runs,
  protected-outcome accesses, protected runs, provider calls, or paid compute.
- Retain commands, raw test output, actual counts, source hashes, and the
  coherent-rewrite witness. Do not count engineering checks as efficacy.
- Stop after the falsifier is rejected, genuine archives verify, the relevant
  existing release/claim-boundary tests pass, and independent source review has
  resolved concrete findings. Do not reopen the closed negative study.

## Historical boundary and next gate

`FINAL_STATUS_2026-09-30.md`, `REPRODUCE.md`, the paper's release boundaries,
and the previous canonical state remain authoritative. The ARC confirmatory
test is unopened. The next deliverable is a separately reviewable additive
release-verification change, not a research-complete or submission-approved
claim. A new scientific hypothesis requires a separately versioned successor.
