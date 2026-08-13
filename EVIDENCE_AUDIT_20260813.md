# LAM-JEPA Evidence Audit — 13 August 2026

**Audit target:** current `main` lineage through `6c6f5c10e8610239ce6c72a4fa7f549659662014`  
**Frozen full scientific run:** `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`  
**Seed-order software repair:** `b72a97a99769b278eb8ec75bc5eab62dc9599f29`  
**Scientific verdict:** negative / inconclusive for ARC superiority and planner/target/quantization mechanism claims  
**Locked ARC test:** not accessed for this failed hypothesis line.

## Reviewer traceability matrix

| Headline item | Command / workflow | Source / fix commit | Environment | Seeds | Raw evidence | Aggregate / verifier | Conclusion |
|---|---|---|---|---|---|---|---|
| Full ARC controls | `.github/workflows/arc-protocol-v3-full-controls-validation.yml` / `scripts/benchmark/run_arc_protocol_v3_controls.py` | scientific SHA `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb` | GitHub Ubuntu, Python 3.11, CPU | 1–5 | artifacts `9149336081`, `9162165932` | verifier exact across reruns | no superiority or mechanism support |
| Capacity-matched baseline | frozen matched-baseline path retained in repo | frozen baseline lineage cited in `RESULTS.md` | matched ARC validation environment | 5 paired seeds | retained matched-baseline artifact lineage | LAM `0.2549152542 ± 0.0129968064`; matched `0.2664406780 ± 0.0154600058` | LAM mean lower |
| Planner ablation | same full-controls runner | `760aa7...` | same as full model | 1–5 | full-controls artifacts | full−no_planner `+0.0047457606`, CI `[0, 0.0142372817]` | criterion not met |
| Target-path ablation | same full-controls runner | `760aa7...` | same as full model | 1–5 | full-controls artifacts | full−no_target `−0.0067796588`, CI `[−0.0135593176, 0]` | criterion not met |
| Shuffled-label negative control | same full-controls runner | `760aa7...` | same as full model | 1–5 | full-controls artifacts | `0.2630508393 ± 0.0145011803`, below frozen 0.35 ceiling | control gate passed; no rescue of main claim |
| Exact aggregate rerun | Actions run `31203337502`, attempts 2 and 3 | `760aa7...` | independent GitHub-hosted runners | 1–5 | 10 files per artifact | aggregate summaries and strict verifier exact | scientific conclusion reproduced |
| Low-level numeric drift | artifact comparison between attempts 2 and 3 | `760aa7...` | independent runners | same | raw result JSONs | 35,526 numeric leaves differ; max drift ~`5.9186e-4`; no non-numeric differences | byte-exact raw probability identity not claimed |
| Pre-fix seed defect | deterministic training smoke before repair | pre-fix lineage | CPU | requested same seed | preserved discrepant losses `10.853294...` vs `10.348778...` | mismatch | reproducibility defect confirmed |
| Seed-order repair | seed before `LAMJEPA(cfg)` construction | `b72a97a...`, PR #61 | CPU replay | fixed same-seed replay | six retained replay attempts | final loss `11.704492568969727` and final accuracy `0.0` exact across attempts; secondary floats may drift | software reproducibility repaired; scientific ARC verdict unchanged |
| Repaired ARC-v5 line | frozen repaired-v5 validation lineage | protocol `168f6be...`, validation `18bd608...`, verifier fix `05c039f...` | retained CPU evidence | frozen repaired protocol | versioned validation artifacts | `VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION` | trainability repair did not establish generalization/quantization benefit |

## Canonical numbers checked

Full frozen ARC controls:

- full LAM-JEPA accuracy: `0.2549152493 ± 0.0129968006`, `n=5`;
- `no_planner`: `0.2501694888 ± 0.0129968006`, `n=5`;
- `no_target`: `0.2616949081 ± 0.0203953938`, `n=5`;
- shuffled-label control: `0.2630508393 ± 0.0145011803`, `n=5`;
- full − `no_planner`: `+0.0047457606`, sample SD `0.0106118432`, bootstrap 95% CI `[0.0, 0.0142372817]`;
- full − `no_target`: `−0.0067796588`, sample SD `0.0092834301`, bootstrap 95% CI `[−0.0135593176, 0.0]`.

Separately retained matched supervised comparison:

- LAM-JEPA: `0.2549152542 ± 0.0129968064`;
- capacity-matched supervised: `0.2664406780 ± 0.0154600058`;
- paired LAM − matched: `−0.0115254237 ± 0.0140994131`.

Bounded pretrained characterization:

- LAM-JEPA: `0.15625`;
- pinned DeBERTa: `0.21875`;
- paired delta: `−0.0625`.

The minor final-decimal differences between full-controls summaries and the separately retained matched-comparison summaries come from distinct retained result lineages and should not be silently rounded into one artifact identity.

## Artifact provenance

### Full scientific rerun attempt 2

- workflow run: `31203337502` attempt 2;
- job: `94178988063`;
- artifact: `9149336081`;
- digest: `sha256:c45710b5dae6a767ccb6bab7f6e3d8e9578752d8cf9b79fd82a65ae824dded1b`;
- status: success.

### Full scientific rerun attempt 3

- workflow run: `31203337502` attempt 3;
- job: `94291056903`;
- artifact: `9162165932`;
- digest: `sha256:caa898f1ff046a337db9b5ddbffe1b332943a732868e2fd809abeda8ee89c30b`;
- status: success.

Both artifacts retain the same 10-file scientific package. Eight files are byte-identical. The two raw probability-bearing JSON payloads are semantically but not byte-identically reproduced across independent runners.

## Bug-before / fix-after lineage

The pre-fix deterministic training path seeded after model construction, so requesting the same seed did not fully determine initial parameters. The defect was demonstrated with preserved conflicting one-step losses. PR #61 made the smallest software repair by applying the seed before model initialization while preserving trainer-side seeding for subsequent streams.

This fix is classified as an execution reproducibility repair. It did **not** change the frozen ARC dataset, scientific seed set, model architecture, metric, hypothesis threshold, or locked-test policy, and it must not be presented as rescuing the scientific result.

## Locked-test audit

The frozen reproduction instructions explicitly download `train` and `validation` only and assert that the ARC test parquet does not exist. The current negative/inconclusive hypothesis line remains closed to confirmatory-test rescue. Any future hypothesis must be separately versioned and preregistered before new validation or test access.

## Reporting-metadata defect

The frozen full-controls raw payload contains a stale sentence describing the run as not being the final five-seed/20-epoch protocol. The actual workflow arguments and independent verifier demonstrate that the final five-seed/20-epoch budget executed. The raw artifact is preserved unchanged; the stale sentence is documented as a non-invalidating reporting-metadata defect.

## Statistical audit

- all headline ARC means report `n=5` where based on the frozen five-seed validation;
- sample SD is reported rather than population SD;
- paired mechanism deltas preserve seed pairing;
- bootstrap intervals are reported only where retained;
- no significance claim is made from error-bar overlap or interval endpoints;
- adverse and inconclusive results are retained;
- no favorable-seed selection is used.

## Figure / table readiness

The strongest manuscript-ready evidence displays are:

1. **Primary ARC comparison table:** full LAM-JEPA, capacity-matched supervised, `no_planner`, `no_target`, shuffled-label control, mean ± SD, `n=5`.
2. **Paired mechanism-effect interval plot:** full−`no_planner` and full−`no_target` with retained bootstrap 95% intervals and a zero-effect reference line.
3. **Reproducibility lineage diagram:** pre-fix seed defect → PR #61 software repair → six same-seed replay attempts → independent full scientific reruns attempts 2/3 → unchanged negative conclusion.
4. **Artifact reproducibility table:** exact aggregate/verifier equality versus low-order raw probability drift.

Do not create a “LAM beats baseline” figure because the evidence does not support that statement.

## Promotion verdict

### GREEN

- frozen protocol traceability;
- deterministic seed policy documented;
- exact commands documented;
- scientific source and repair commits identified;
- independent full reruns retained;
- artifact digests retained;
- baseline and ablation tables retained;
- negative/inconclusive scientific conclusion preserved;
- bug-before/fix-after history retained;
- locked-test non-access documented.

### NOT GREEN / not claimed

- LAM-JEPA superiority on ARC;
- validated planner contribution;
- validated target/EMA contribution;
- validated quantization benefit;
- general benchmark superiority;
- externally validated educational effectiveness;
- research completion;
- publication/release provenance until owner-approved licensing/citation requirements are closed.

## Skeptical-reviewer verdict

A reviewer can now trace the current headline ARC result from command → frozen source revision → environment class → fixed seed set → retained artifacts/digests → aggregate and independent verifier → negative/inconclusive conclusion. The remaining material scientific limitation is not reproducibility of the current result; it is that the result does not support the proposed superiority or mechanism claims.
