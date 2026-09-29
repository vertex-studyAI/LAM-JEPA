# AISTATS 2027 reproducibility checklist status

Evidence-bounded checklist draft prepared from the current frozen ARC manuscript and retained provenance.

## Current answers

- 1a model/algorithm description: Yes.
- 1b formal complexity analysis: No.
- 1c anonymized source bundle: No; pending.
- 2a–2c theoretical claims/proofs: Not Applicable; this is an empirical falsification study.
- 3a code/data/instructions in anonymous review package: No; internal exact commands and artifact lineage exist, but the anonymous bundle is pending.
- 3b training details: Yes.
- 3c measures/statistics/error bars: Yes.
- 3d computing infrastructure: No; CPU execution is retained, but exact physical CPU model metadata was not retained.
- 4a citations for existing assets: Yes.
- 4b asset license metadata: No; pending integration.
- 4c new/released assets in supplement or URL: No; anonymous release artifact pending.
- 4d consent: Not Applicable.
- 4e sensitive content: Not Applicable.
- 5a–5c crowdsourcing/human subjects: Not Applicable.

## Verified evidence used

- `MANUSCRIPT_PROVENANCE.md`: complete raw provenance for the five-seed full controls, capacity-matched baseline, bounded pretrained characterization, source-level method claims, and deterministic paper-asset generator.
- `REPRODUCE.md`: exact frozen five-seed command, verifier command, CPU execution boundary, scientific source revision, rerun artifacts/digests, and locked-test policy.
- `paper/aistats2027_submission.tex`: method, experimental setup, statistics, limitations, AI Use Statement, and anonymous manuscript surface.

## Remaining packaging blockers

1. Integrate the checklist into the submission PDF after references.
2. Verify the exact checklist question wording against the official AISTATS 2027 sample source before upload.
3. Prepare an anonymous reproduction bundle if 3a/1c are to be upgraded to Yes.
4. Add verified asset-license metadata if 4b is to be upgraded to Yes.
5. Do not reconstruct missing CPU model metadata; leave 3d No unless evidence is found.
