# LAM-JEPA independent sentence-level claim audit

Audit scope: `paper/main.tex`, `paper/icdm_teen_2026.tex`, the frozen evidence summary in
`experiments/reproducibility-wave-20260812.json`, the release manifest, and the retained
provenance/claim ledgers. This audit does not rerun training, open the ARC test split, or
authorize a successor experiment.

## Verdict

**PASS — evidence-bounded negative/inconclusive manuscript.** Every retained quantitative
claim is traceable to frozen project evidence at the precision printed in the manuscript.
No sentence supports superiority, statistical significance, a validated planner/EMA-target/
repaired-quantization mechanism, external reproduction, novelty, or release readiness.

## Independent arithmetic checks

The canonical summary values independently imply:

- full minus matched supervised: `0.2549152542 - 0.2664406780 = -0.0115254238`, agreeing
  with the retained paired estimate `-0.0115254237` to the published precision permitted
  by rounded arm means;
- full minus no-planner: `0.2549152542 - 0.2501694915 = 0.0047457627`;
- full minus no-target: `0.2549152542 - 0.2616949153 = -0.0067796611`, agreeing with the
  retained paired estimate `-0.0067796610` to the published precision permitted by rounded
  arm means;
- bounded pretrained characterization: `0.15625 - 0.21875 = -0.0625`;
- gradient-active parameter ratio: `86644 / 86372 = 1.0031491687` to ten decimal places.

The paired estimates and bootstrap intervals remain authoritative because arm means printed
to ten decimals cannot reconstruct unrounded per-seed pairs exactly. No p-value or
significance claim is present or inferred.

## Claim-to-evidence reconciliation

| Claim class | Manuscript statement | Direct evidence | Audit disposition |
|---|---|---|---|
| System identity | Small recurrent latent-action controller; not a Transformer or canonical distinct-target I-JEPA | `MANUSCRIPT_PROVENANCE.md`; source SHA `760aa7f...` | Retain narrow architecture description |
| EMA path | Same-input EMA target with momentum `0.996` | Frozen source/protocol provenance | Do not generalize to canonical JEPA |
| Frozen protocol | ARC train/validation only; seeds 1–5; 20 epochs; batch 32; LR `3e-4`; one model step | Retained full-controls artifact lineage and provenance ledger | Protocol unchanged |
| Data eligibility | 1,117/1,119 train and 295/299 validation rows eligible | Frozen artifact/provenance reports | Descriptive only |
| Full system | `0.2549152542 ± 0.0129968064`, n=5 | `canonical_metrics.full_lam_jepa_accuracy` | Reconciled exactly |
| Matched comparator | `0.2664406780 ± 0.0154600058`, n=5; 86,372 versus 86,644 active parameters | Canonical summary; matched-baseline provenance | Comparator is adverse evidence, not omitted |
| Primary paired result | `-0.0115254237 ± 0.0140994131`, n=5 | `canonical_metrics.paired_lam_minus_matched` | No superiority or significance |
| Planner ablation | Full minus no-planner `+0.0047457627`, CI `[0, 0.0142372881]` | Frozen paired bootstrap evidence | Planner benefit unsupported |
| EMA-target ablation | Full minus no-target `-0.0067796610`, CI `[-0.0135593220, 0]` | Frozen paired bootstrap evidence | EMA-target benefit unsupported |
| Shuffled-label control | `0.2630508475 ± 0.0145011862` | Canonical frozen summary | Validity concern retained; never favorable evidence |
| Bounded pretrained arm | LAM `0.15625`, DeBERTa `0.21875`, paired `-0.0625`, n=2 | Bounded pinned-pretrained artifact lineage | Characterization only; not matched or confirmatory |
| Repair studies | Quantization/repair criteria failed | Retained repair verdicts and claim ledger | No repaired-mechanism rescue |
| Reproducibility | Aggregate conclusions/verifier outputs reproduced; checkpoint bytes not universally identical | Independent rerun digests and documented floating-point drift | Semantic/project-controlled only; external reproduction pending |
| Literature | Eight primary records verified | `paper/BIBLIOGRAPHY_AUDIT.md` | Context/methodology only; cited methods were not executed comparators |
| Locked test | ARC test locked and unopened | Protocol and claim-boundary checks | No access or rescue permitted |
| Readiness | Internal package complete for frozen evidence; public release blocked | Release manifest and owner-approval placeholders | `preprint_ready=false`; authorship, licensing, permanent archive unresolved |

## Fail-closed boundaries

- Scientific status remains `VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION`.
- The matched supervised comparator was not outperformed.
- Planner, EMA-target, and repaired-quantization benefits remain unsupported.
- The confirmatory ARC test remains `locked/unopened`.
- Rescue tuning, seed expansion, threshold movement, and successor execution remain unauthorized.
- Independent external reproduction, broad generalization, novelty, significance, and
  submission/preprint readiness are not established.
- The historical ICDM PDF is provenance evidence for an expired target, not an authorized
  new submission.

No manuscript sentence, scientific value, protocol field, seed, threshold, or conclusion was
changed by this audit.
