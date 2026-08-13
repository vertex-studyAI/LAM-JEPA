# Independent Retained-Artifact Audit — 13 August 2026

This audit is additive to the canonical evidence audit already on `main`. It records an independent download/hash/parse check performed against the two retained frozen ARC-v3 full-controls GitHub Actions artifacts. No scientific code, protocol, data split, seeds, thresholds, architecture, or locked-test policy was changed.

## Audit target

- canonical main at audit reconciliation: `9f71ae89c79ecba55925ec60cdde22153915b7ef`
- frozen scientific revision: `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`
- seed-order repair: `b72a97a99769b278eb8ec75bc5eab62dc9599f29`
- audit environment: Python 3.13.5, Linux 6.18.35 x86_64
- fresh repository clone/training in the audit sandbox: not performed because outbound DNS to `github.com` was unavailable; this is an audit-environment limitation, not a scientific failure.

## Authenticated artifact download and digest verification

| Attempt | Run/attempt | Job | Artifact | Recorded digest | Independently downloaded digest |
|---|---|---|---|---|---|
| 2 | `31203337502` / `2` | `94178988063` | `9149336081` | `c45710b5dae6a767ccb6bab7f6e3d8e9578752d8cf9b79fd82a65ae824dded1b` | exact match |
| 3 | `31203337502` / `3` | `94291056903` | `9162165932` | `caa898f1ff046a337db9b5ddbffe1b332943a732868e2fd809abeda8ee89c30b` | exact match |

Each archive contains the same 10 retained paths. Eight files are byte-identical. Only the raw result JSON and its verifier normalized-input copy differ bytewise.

Independent recursive comparison found:

- numeric leaf differences: `35,526`;
- non-numeric leaf differences: `0`;
- maximum absolute numeric drift: `0.0005918592214584351`;
- one maximum-drift location: `/negative_control/records/1/predictions/97/probabilities/0`.

The normalized-input copies show the same difference count and maximum drift. Aggregate scientific conclusions and verifier outputs are unchanged.

## Independent protocol/result parsing

Retained verifier:

- verdict: `PROTOCOL_V3_FULL_CONTROLS_VALIDATION_VERIFIED`;
- train rows used: `1117` of `1117` eligible;
- validation rows used: `295` of `295` eligible;
- seeds: `[1,2,3,4,5]`;
- epochs: `20`;
- batch size: `32`;
- locked test evaluated: `false`;
- final five-seed/20-epoch protocol executed: `true`;
- mechanism claim authorized: `false`;
- research complete: `false`;
- maximum observed aggregate drift after allowed normalization: `1.075914349280005e-08` under tolerance `1e-6`.

Independent aggregate parse:

| Condition | Mean validation accuracy | Sample SD | n |
|---|---:|---:|---:|
| Full LAM-JEPA | `0.2549152542372881` | `0.01299680644927512` | 5 |
| `no_planner` | `0.2501694915254237` | `0.01299680644927512` | 5 |
| `no_target` | `0.26169491525423727` | `0.02039540197490349` | 5 |
| Shuffled-label control | `0.2630508474576271` | `0.014501186194038939` | 5 |

Paired effects:

- full − `no_planner`: mean `+0.004745762711864404`, sample SD `0.010611848028812555`, bootstrap 95% CI `[0.0, 0.014237288135593213]`; frozen criterion not met.
- full − `no_target`: mean `-0.006779661016949157`, sample SD `0.009283433178053668`, bootstrap 95% CI `[-0.013559322033898313, 0.0]`; frozen criterion not met.

## Verdict

`REPRODUCED_NEGATIVE_OR_INCONCLUSIVE`.

The independent artifact audit supports the canonical claim that the frozen aggregate scientific conclusion and verifier decision reproduce despite low-order per-example probability drift. It does not support ARC superiority, planner benefit, target/EMA benefit, research completion, or use of the locked confirmatory test.

This is independent verification of retained authenticated artifacts, not a seventh scientific training sample.
