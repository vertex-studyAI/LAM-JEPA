# LAM-JEPA — RESULTS

**Audited source commit:** `2f59b4297e5978d4ce769ebe95adb363e1e75d7a`  
**Evidence class:** canonical frozen multi-seed ARC evidence + current-head GitHub Actions artifact verified/downloaded in the 2026-08-12 reproducibility wave. The wave did not launch a new full ARC training run.

## Hypothesis
Under a frozen ARC-Challenge protocol, LAM-JEPA should outperform a capacity-matched supervised baseline and show positive contributions from its planner/target mechanisms.

## Dataset / task
ARC-Challenge with feature-only eligibility fixed before confirmatory test access: 1,117/1,119 train rows and 295/299 validation rows eligible. Confirmatory test remains locked for this failed hypothesis line.

## Baselines
Capacity-matched supervised baseline (86,644 gradient-active parameters vs 86,372 for LAM-JEPA), planner/target ablations, deterministic shuffled-label negative control, and a pinned DeBERTa-v3-xsmall development comparator.

## Protocol / seeds
Frozen five-seed validation: 20 epochs, batch 32, learning rate 0.0003, model steps 1, all eligible training and validation rows. Current-head CI artifacts use tiny smoke budgets and must not be interpreted as the scientific benchmark.

## Result
LAM-JEPA validation accuracy `0.2549152542 ± 0.0129968064`; capacity-matched supervised `0.2664406780 ± 0.0154600058`; paired LAM-minus-matched `-0.0115254237 ± 0.0140994131`.

Full minus no-planner: `+0.0047457627`, bootstrap 95% CI `[0, 0.0142372881]`. Full minus no-target: `-0.0067796610`, bootstrap 95% CI `[-0.0135593220, 0]`. Neither mechanism criterion was met. A bounded DeBERTa development comparison was also adverse (`0.15625` vs `0.21875`).

## Current-head execution evidence
GitHub Actions run `31610608912`, attempt 2, produced artifact `9149221882` at the audited head with digest `sha256:5546b87ac6170d6b5a5c58404c57fec7c79fc481b351ce619c62292859102a94`. The CI-sized training smoke is only one step and the tiny synthetic evaluation is 0.0 accuracy. A 16-example ARC smoke scored 12.5%/18.75% for LAM-JEPA across seeds 1/2 versus 18.75%/25% for a non-parameter-matched hash baseline. This is plumbing evidence only.

## Verdict
**NEGATIVE / INCONCLUSIVE.** ARC superiority, planner benefit, target-mechanism benefit, and quantization benefit are unsupported. The locked confirmatory test must not be used to rescue the result.
