# LAM-JEPA conference-ready claim gates — 2026-09-05

## Purpose

This note supplies manuscript-ready replacement text for the abstract, hypothesis/decision-gate paragraph, and failure-mechanism discussion using only evidence already frozen or merged on `main`. It does **not** open the locked ARC test, modify a protocol, rerun an experiment, change a threshold, or upgrade a negative/inconclusive result into a positive claim.

The relevant frozen scientific protocol is `protocols/arc_challenge_v3.json`. The current paper is `paper/main.tex`. The current public claim boundary is `CLAIM_LEDGER.md`, including the externally reproduced VQ-collapse correction merged through PR #152. The external reproduction is one genuinely external frozen-protocol rerun/review; it is not broad multi-site replication or peer-reviewed publication.

## Frozen decision gates

The paper should state the actual preregistered gates rather than only saying that an effect had to be “sufficiently” large.

| Claim | Frozen gate | Observed retained evidence | Verdict |
|---|---|---|---|
| H1 — superiority over a trained baseline | mean LAM-minus-baseline accuracy gain at least **+0.02**, and paired seed-level 95% bootstrap CI excludes zero | full LAM-JEPA `0.2549152542 ± 0.0129968064`; matched supervised `0.2664406780 ± 0.0154600058`; paired mean `-0.0115254237` | **FAIL**; the observed mean is negative and therefore cannot satisfy the +0.02 gate |
| H2 — planner contribution | paired full-minus-`no_planner` mean accuracy at least **+0.01**, and paired 95% bootstrap CI excludes zero | `+0.0047457627`, CI `[0.0, 0.0142372881]` | **FAIL** |
| H3 — EMA target-path contribution | paired full-minus-`no_target` mean accuracy at least **+0.01**, and paired 95% bootstrap CI excludes zero | `-0.0067796610`, CI `[-0.0135593220, 0.0]` | **FAIL** |
| Shuffled-label validity control | validation accuracy must remain at or below the frozen **0.35** ceiling | `0.2630508475 ± 0.0145011862` | **PASS as a validity control only** |

Passing the shuffled-label control is not favorable evidence for H1–H3. Failure of H1–H3 does not license opening the locked ARC confirmatory test to rescue the line.

## Manuscript-ready abstract replacement

We evaluate the project-named LAM-JEPA system on ARC-Challenge under a frozen falsification-first protocol with a gradient-active-parameter-matched supervised comparator, mechanism ablations, a shuffled-label validity control, and a bounded pinned pretrained comparison. Source inspection constrains the architecture claim: the evaluated ARC path is a small hashed-token, mean-pooled embedding model with vector quantization, learned sparse memory, a one-step latent-action rollout, and same-input exponential-moving-average target alignment; it is neither a Transformer encoder nor the canonical I-JEPA context-to-distinct-target prediction task. The preregistered superiority gate required a mean accuracy gain of at least 0.02 with a paired 95% bootstrap confidence interval excluding zero, while component-attribution gates required a paired full-minus-ablation gain of at least 0.01 with the paired interval excluding zero. Across five frozen validation seeds, full LAM-JEPA achieved `0.2549152542 ± 0.0129968064` accuracy versus `0.2664406780 ± 0.0154600058` for the matched supervised model, with a paired mean difference of `-0.0115254237`; the superiority gate therefore failed. Full-minus-`no_planner` was `+0.0047457627` with 95% bootstrap CI `[0.0, 0.0142372881]`, and full-minus-`no_target` was `-0.0067796610` with CI `[-0.0135593220, 0.0]`, so neither mechanism gate was met. A later trainability repair remained negative/inconclusive on repaired validation. One genuinely external frozen-protocol rerun/review reproduced the retained headline metrics and further found single-code vector-quantizer collapse with constant downstream predictions in the reviewed runs; disabling quantization restored input-dependent predictions but did not establish above-chance ARC performance. We therefore report a bounded, reproducible failure-mechanism case study rather than architecture superiority, a general JEPA conclusion, or a general claim about vector quantization, and we keep the confirmatory ARC test locked for this failed hypothesis line.

## Manuscript-ready hypothesis and decision-gate replacement

Let `A_full` denote validation accuracy of the frozen full configuration, `A_match` the gradient-active-parameter-matched supervised comparator, `A_np` the `no_planner` ablation, and `A_nt` the `no_target` ablation. The frozen protocol defines three primary scientific gates. **H1 (superiority)** is supported only if the mean paired gain `A_full - A_baseline` is at least `+0.02` and the paired seed-level 95% bootstrap confidence interval excludes zero; any headline superiority statement must also hold against the strongest trained non-JEPA baseline. **H2 (planner contribution)** is supported only if the paired mean `A_full - A_np` is at least `+0.01` and its paired 95% bootstrap interval excludes zero. **H3 (target-path contribution)** uses the same `+0.01` and interval-excludes-zero rule for `A_full - A_nt`. The shuffled-label run is a validity control with a frozen `0.35` accuracy ceiling and cannot establish H1–H3. If the frozen validation gates fail, the confirmatory ARC test remains locked and successor mechanisms require a new versioned hypothesis rather than post-outcome threshold changes.

## Manuscript-ready results interpretation

The full model did not meet the frozen superiority gate. Its five-seed validation mean was `0.2549152542`, compared with `0.2664406780` for the matched supervised comparator, and the paired LAM-minus-matched mean was `-0.0115254237`. Because the preregistered H1 gate requires a positive mean gain of at least `+0.02`, the observed negative mean is already incompatible with H1; no favorable wording should be inferred from uncertainty alone. The planner ablation also failed its preregistered attribution rule: full-minus-`no_planner` was `+0.0047457627`, below the `+0.01` practical threshold, with retained 95% bootstrap CI `[0.0, 0.0142372881]` that does not exclude zero. The target-path comparison failed more strongly: full-minus-`no_target` was `-0.0067796610`, with CI `[-0.0135593220, 0.0]`. The shuffled-label control stayed below its frozen `0.35` ceiling, but this establishes only that the declared validity control did not trigger its stop rule.

## Manuscript-ready external failure-mechanism paragraph

The external reproduction sharpens the negative result without changing its scope. In the independently reviewed retained runs, each evaluated condition/seed behaved as a constant classifier across the 295-row retained evaluation set, while the examined quantized representation selected a single code from the 32-code vocabulary within each reviewed run. In a bounded external diagnostic, disabling quantization restored input-dependent predictions, localizing the constant-output collapse to the tested quantized path in this implementation. That intervention did **not** establish above-chance ARC accuracy. The supported conclusion is therefore mechanistic and narrow: the reviewed retained configuration suffered a reproducible quantized-path collapse that helps explain its adverse result. It does not establish that vector quantization is generally harmful, that removing quantization solves ARC, or that JEPA-style representation learning fails broadly.

## Conference-facing claim boundary

Allowed, with scope:

- the frozen five-seed ARC validation is negative/inconclusive for the tested superiority and mechanism claims;
- project-controlled reruns reproduce the aggregate adverse conclusion and verifier verdict despite low-order floating-point drift;
- one genuinely external frozen-protocol rerun/review reproduced the retained headline metrics and bounded collapse diagnosis;
- the reviewed quantized path collapsed to one VQ code per run and constant predictions, while quantizer removal restored input dependence without establishing above-chance performance;
- the locked confirmatory test remains unused for this failed line.

Not supported:

- LAM-JEPA superiority on ARC;
- a validated planner or EMA-target contribution;
- quantization benefit;
- a general failure claim about JEPA, vector quantization, latent planning, Transformers, or representation learning;
- broad independent replication, multi-site validation, peer-reviewed publication, or research completeness.

## Evidence map

1. `protocols/arc_challenge_v3.json` — frozen thresholds, uncertainty rule, ablation rule, shuffled-label stop rule, locked-test policy.
2. `paper/main.tex` — current manuscript values and bounded interpretation.
3. `CLAIM_LEDGER.md` — current public claim boundary after PR #152.
4. `paper/EXTERNAL_REVIEW_VQ_COLLAPSE_CORRECTION_20260831.md` — bounded external reproduction and collapse diagnosis.
5. `EXTERNAL_VALIDATION_PACKET_20260814.md` — immutable retained artifact identities and independent recomputation task.

## Remaining owner-controlled release gates

This text does not resolve authorship/order, repository/publication license, final citation metadata, venue formatting, or any additional independent-reproduction claim. Those remain separate release decisions. Scientific thresholds, metrics, and the adverse outcome should not be changed while resolving them.
