# LAM-JEPA Manuscript Provenance Matrix

**As of:** 2026-08-14  
**Scope:** frozen ARC negative/inconclusive paper line only. This file does not authorize test access, hyperparameter rescue, publication, or a superiority claim.

## Provenance rule

Every quantitative manuscript statement must resolve as:

`claim -> manuscript table/figure -> processed metric -> raw artifact -> frozen protocol/config -> scientific code revision`

A missing edge is a paper-package blocker.

## Current quantitative claims

| Claim / display | Processed metric | Raw / retained artifact | Protocol / execution | Code / provenance | Status |
|---|---|---|---|---|---|
| Full LAM-JEPA validation accuracy `0.2549152542 ± 0.0129968064`, `n=5` | matched-comparison aggregate retained in `EVIDENCE_AUDIT_20260813.md` / `RESULTS.md` | full-controls artifact `9162165932`; digest `sha256:caa898f1ff046a337db9b5ddbffe1b332943a732868e2fd809abeda8ee89c30b`; matched-baseline lineage retained separately | `lam-jepa-arc-challenge-v3`; seeds `1..5`; 20 epochs; batch 32; LR `0.0003`; model steps 1; train 1117 eligible rows; validation 295 eligible rows; CPU | frozen scientific SHA `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`; current audit lineage through `6c6f5c10...` | **TRACEABLE** |
| Capacity-matched supervised `0.2664406780 ± 0.0154600058`; paired LAM-minus-matched `-0.0115254237 ± 0.0140994131` | matched-baseline aggregate | retained matched-baseline artifact lineage referenced by evidence audit | same frozen validation data/seed budget; gradient-active parameter counts LAM `86,372`, supervised `86,644` | frozen baseline path + scientific lineage in evidence audit | **TRACEABLE; artifact ID should be copied into final release manifest if available** |
| `no_planner` `0.2501694915 ± 0.0129968064`; full-minus-no_planner `+0.0047457627`; bootstrap 95% interval `[0, 0.0142372881]` | paired seed aggregate + bootstrap interval | full-controls artifact `9162165932` | same frozen v3 full-controls workflow | scientific SHA `760aa7...` | **TRACEABLE** |
| `no_target` `0.2616949153 ± 0.0203954020`; full-minus-no_target `-0.0067796610`; bootstrap 95% interval `[-0.0135593220, 0]` | paired seed aggregate + bootstrap interval | full-controls artifact `9162165932` | same frozen v3 full-controls workflow | scientific SHA `760aa7...` | **TRACEABLE** |
| Shuffled-label control `0.2630508475 ± 0.0145011862` | five-seed aggregate | full-controls artifact `9162165932` | deterministic shuffled-label arm in v3 full-controls workflow | scientific SHA `760aa7...` | **TRACEABLE** |
| Bounded DeBERTa-v3-xsmall characterization: LAM `0.15625`, DeBERTa `0.21875`, delta `-0.0625` | bounded development comparison | retained pretrained-comparator lineage | pinned `microsoft/deberta-v3-xsmall` revision `14809e4f1fe1895fcba8b258271a940c6ca45ec4` | comparator path retained in repository | **TRACEABLE AS CHARACTERIZATION ONLY; not a final inferiority claim** |
| Repaired ARC-v5 verdict `VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION` | independent verifier verdict | versioned repaired-v5 validation artifacts | `protocols/arc_challenge_v5_repaired_validation.json`; train/validation only | repair and verifier lineage recorded in `EVIDENCE_AUDIT_20260813.md` | **TRACEABLE; cannot retroactively validate original hard-VQ mechanism** |
| Independent rerun conclusion reproduction | aggregate/verifier equality across attempts | artifacts `9149336081` and `9162165932` | GitHub-hosted Ubuntu, Python 3.11, CPU; same seeds/budget | scientific SHA `760aa7...` | **TRACEABLE** |
| Low-level nondeterministic drift boundary | 35,526 numeric leaves differ; max drift ~`5.9186e-4`; no non-numeric differences | artifacts `9149336081` vs `9162165932` | independent runners | same scientific SHA | **TRACEABLE; byte-identical probabilities/checkpoints are not claimed** |

## Figure provenance plan

### Figure 1 — Primary baseline/control comparison

- **Source data:** retained full-controls and matched-baseline aggregates only.
- **Allowed content:** full LAM-JEPA, matched supervised, `no_planner`, `no_target`, shuffled-label control, mean ± sample SD, `n=5`.
- **Forbidden interpretation:** a visual implying LAM beats the matched baseline.
- **Generation gate:** plot-generation script must consume a retained machine-readable summary, not hard-coded manuscript numbers.

### Figure 2 — Paired mechanism effects

- **Source data:** paired per-seed full-minus-`no_planner` and full-minus-`no_target` results from the frozen artifact.
- **Allowed content:** point estimate, retained bootstrap 95% interval, zero-effect reference.
- **Interpretation:** neither mechanism satisfies its frozen positive-contribution criterion.

### Figure 3 — Reproducibility lineage

- **Source data:** Git/Actions provenance only.
- **Sequence:** pre-fix seed-order defect -> minimal seed-before-model-construction repair -> same-seed replay -> independent full scientific reruns -> unchanged negative/inconclusive conclusion.
- **Interpretation:** engineering reproducibility repair did not rescue the scientific hypothesis.

### Table 1 — Frozen experimental protocol

Must state: ARC-Challenge train/validation only, seeds `1..5`, 20 epochs, batch size 32, LR `0.0003`, model steps 1, 1117 eligible training rows, 295 eligible validation rows, CPU execution, test split not downloaded/evaluated.

## Environment boundary

The full-controls workflow itself proves `ubuntu-latest`, Python `3.11`, and an explicit CPU PyTorch environment (`--device cpu`). The exact physical CPU model of the GitHub-hosted runner is **not retained here and must not be invented**.

## Architecture/source boundary

The current source defines:

1. a token/numeric multi-view encoder and linear projector;
2. an optional EMA-updated vector quantizer;
3. optional sparse-memory retrieval with gated correction;
4. a latent-action policy and residual stochastic transition used for rollout when the planner is enabled;
5. an EMA target encoder/projector, falling back to detached online representation when the target path is disabled;
6. output, value, confidence, verifier, rubric, uncertainty and latent-summary heads.

The general composite loss in `src/lam_jepa/losses.py` combines supervised cross-entropy with latent alignment, variance/covariance/uniformity/geodesic regularizers, confidence/verifier calibration, rollout consistency, rubric loss and quantization loss using repository-defined fixed weights. The paper must distinguish this general source implementation from any benchmark-specific loss simplifications or overrides in the frozen ARC runner.

## Remaining internal blockers

- Final manuscript-specific source check for any benchmark-specific model/loss overrides before locking Method text.
- Figure-generation commands/artifacts still need to be tied to source data; do not hand-enter a publication figure.
- Final release manifest should include the exact retained artifact identifier for the matched-supervised lineage if it can be recovered from existing evidence.

## External/owner blockers

- root license decision and compatibility review;
- approved author list/order and `CITATION.cff` metadata;
- dataset redistribution/legal review as applicable;
- independent outside reproduction/reviewer report.

These remain separate from internal scientific reproducibility.