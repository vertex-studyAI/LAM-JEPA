# Retained-evidence presentation draft — 7 October 2026

[One-page poster](poster.pdf) and [250-word abstract](abstract.txt), prepared as a nonarchival presentation candidate. Author review and venue-specific format/overlap checks remain open. This package was not submitted, and does not mark the project complete.

The renderer verifies every pinned source hash, recomputes the displayed retained arithmetic, and emits [claim data](claim_data.json) and [checks](checks.json). No model training, checkpoint replay, new outcome sweep or protected evaluation is performed. The complete final poster page was rendered and visually inspected after embedding DejaVu fonts; the initial font fallback defect was corrected.

## Rebuild

Run from this package directory with Python 3.11+, ReportLab and pypdf installed:

```sh
python -B build_materials.py --source-root ../.. --output .
```

The default font directory is `/usr/share/fonts/truetype/dejavu`. Set `CONFERENCE_FONT_DIR` to a directory containing `DejaVuSans.ttf` and `DejaVuSans-Bold.ttf` on other systems. The supplied PDF embeds its fonts and does not require them to view. Building regenerates presentation outputs only; pinned scientific sources are read-only.

## Evidence and boundary

- [audits/MATCHED_REPRODUCTION_20260930.md](https://github.com/vertex-studyAI/LAM-JEPA/blob/c6ee9d1dcfdf6bc9efbde61877cc300528964e73/audits/MATCHED_REPRODUCTION_20260930.md)
- [audits/independent_artifact_audit_2026-08-13.json](https://github.com/vertex-studyAI/LAM-JEPA/blob/c6ee9d1dcfdf6bc9efbde61877cc300528964e73/audits/independent_artifact_audit_2026-08-13.json)
- [paper/aistats2027_submission.tex](https://github.com/vertex-studyAI/LAM-JEPA/blob/c6ee9d1dcfdf6bc9efbde61877cc300528964e73/paper/aistats2027_submission.tex)

Input commit: `c6ee9d1dcfdf6bc9efbde61877cc300528964e73`. [Source manifest](source_manifest.json) records SHA-256 identities. The poster's empirical statements remain attributed to those retained sources; report arithmetic is not original model replay.

The failed confirmatory hypothesis stays closed; historical raw archive recovery and author/venue decisions remain separate.

The unbranded poster is a draft, not a claim of workshop selection or acceptance. The final venue call, authorship/presenter details and any archival restrictions must be reconciled before submission. Ordinary drafting and review work can continue within the already authorized scope.
