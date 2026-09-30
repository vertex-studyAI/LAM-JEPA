# LAM-JEPA — Final Research Status (2026-09-30)

## Disposition
**CLOSED AS A REPRODUCIBLE NEGATIVE / FAILURE-MECHANISM STUDY.**

The frozen ARC-Challenge evidence does not support architecture superiority or the preregistered planner/target mechanism claims. The locked ARC confirmatory test remains unopened and must not be used to rescue this failed line.

## Supported headline
Under the frozen ARC-Challenge validation protocol:
- LAM-JEPA mean validation accuracy: 0.2549152542 ± 0.0129968064;
- capacity-matched supervised baseline: 0.2664406780 ± 0.0154600058;
- paired LAM − matched difference: -0.0115254237 ± 0.0140994131;
- full − no_planner: +0.0047457627, bootstrap 95% CI [0.0, 0.0142372881];
- full − no_target: -0.0067796610, bootstrap 95% CI [-0.0135593220, 0.0].

The planner and target-path criteria were not met.

## Failure mechanism
The externally reviewed retained runs support a bounded diagnosis: the tested quantized path collapsed distinct pre-quantizer latents to a single VQ code per run and produced constant downstream predictions. This is a diagnosis of the tested configuration, not a general claim about vector quantization or JEPA.

## Reproducibility
Independent project-controlled reruns reproduce the aggregate negative conclusion and verifier verdict. Low-order floating-point drift exists in some raw artifacts, so byte-identical raw replay is not claimed.

## Claim boundary
Do not claim:
- ARC superiority;
- validated planner or target-path benefit;
- general JEPA failure;
- general vector-quantization failure;
- peer-reviewed publication;
- broad multi-site independent replication.

## Research state
The current hypothesis line is scientifically finished. Any architecture repair, new target construction, new benchmark, or stronger capability claim is a separately versioned successor hypothesis with a new frozen protocol.

## Final rule
Preserve the negative result. Do not open the locked ARC confirmatory test or rerun the line merely to obtain a more favorable outcome.
