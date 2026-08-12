# LAM-JEPA Results Ledger

**Reproducibility wave:** 2026-08-12  
**Frozen scientific source revision:** `2f59b4297e5978d4ce769ebe95adb363e1e75d7a`  
**Reproducibility-repair revision:** `b72a97a99769b278eb8ec75bc5eab62dc9599f29`  
**Scientific status:** negative / inconclusive on the frozen ARC-Challenge superiority and mechanism hypotheses  
**Confirmatory test status:** LOCKED; do not use it to rescue the failed validation hypothesis.

## Research question

Under a frozen ARC-Challenge train/validation protocol, does LAM-JEPA improve validation accuracy over a gradient-active-parameter-matched supervised baseline, and do the planner, target/EMA path, or repaired quantized latent mechanism contribute a reproducible validation benefit?

## Hypothesis and falsification

The superiority hypothesis requires a reproducible positive paired validation advantage over a meaningful matched supervised baseline. Mechanism claims require positive paired effects when the relevant component is present versus removed. The current evidence falsifies or fails to support those stronger claims; negative results are retained as first-class outcomes.

## Dataset and task

ARC-Challenge multiple-choice reasoning, using checksum-addressed train and validation data under the repository's frozen protocol. The current protocol retains excluded rows and preserves source order. The locked ARC test is excluded from this failed hypothesis line.

## Baselines and controls

- **Simple/reference controls:** matched label-distribution references and a deterministic shuffled-label control.
- **Standard architecture baseline:** capacity-matched supervised model using gradient-active parameter matching.
- **Strong pretrained comparator:** pinned `microsoft/deberta-v3-xsmall` at revision `14809e4f1fe1895fcba8b258271a940c6ca45ec4` for bounded development characterization.
- **Ablations:** `no_planner` and `no_target` paired against the full model.

## Frozen validation protocol

For the full-controls ARC validation: five paired seeds, 20 epochs, batch size 32, learning rate `0.0003`, model steps 1, 1,117 eligible training rows, and 295 eligible validation rows. The same evaluation semantics are used for the paired comparisons.

## Canonical retained results

| System / control | Validation accuracy, mean ± sample SD | n | Interpretation |
|---|---:|---:|---|
| Full LAM-JEPA | 0.2549152542 ± 0.0129968064 | 5 seeds | Proposed model |
| Capacity-matched supervised | 0.2664406780 ± 0.0154600058 | 5 seeds | Strong matched baseline |
| LAM − matched | -0.0115254237 ± 0.0140994131 | 5 paired seeds | No superiority evidence |
| `no_planner` | 0.2501694915 ± 0.0129968064 | 5 seeds | Planner ablation |
| `no_target` | 0.2616949153 ± 0.0203954020 | 5 seeds | Target-path ablation |
| Shuffled-label control | 0.2630508475 ± 0.0145011862 | 5 seeds | Control; below frozen 0.35 failure threshold |

Paired component effects:

- full − `no_planner`: `+0.0047457627`, 95% bootstrap CI `[0.0, 0.0142372881]`;
- full − `no_target`: `-0.0067796610`, 95% bootstrap CI `[-0.0135593220, 0.0]`.

Neither required mechanism criterion was met. No statistical significance claim is made from these intervals alone.

A bounded development comparison against the pinned pretrained comparator was adverse to LAM-JEPA (`0.15625` vs `0.21875`; paired delta `-0.0625`). This is characterization evidence, not a final inferiority test.

## Pre-fix exact-head reproduction

The repository's `Reproducibility CI` was rerun on `2f59b4297e5978d4ce769ebe95adb363e1e75d7a` after prior results were observed. Run `31610608912`, attempt 2, job `94178401933` completed successfully on GitHub-hosted Ubuntu / Python 3.11 / CPU in about 101 seconds.

That workflow success did **not** establish same-seed checkpoint reproducibility. A subsequent exact rerun showed that `train_single.py` sampled initial weights before the requested seed was applied. Under nominally identical SHA / CLI / seed / CPU execution, the one-step loss changed from `10.853294372558594` to `10.34877872467041`. This pre-fix nondeterminism is retained as invalidated reproducibility evidence rather than overwritten.

## Deterministic seed-order repair and replay

PR #61 applied the smallest repair: seed before `LAMJEPA(cfg)` construction while retaining trainer-side seeding for the subsequent data/training stream. No ARC split, seed set, threshold, metric, architecture, gate, or locked-test policy changed.

The repaired PR head `ced95ee10021d09419816aade3f5906a3d99663c` passed Reproducibility CI `31618228743`, deterministic replay `31618227708`, ARC Protocol V2 QA `31618228252`, and Research claim boundary `31618228424`. Its replay artifact was ID `9150159954`, SHA-256 `6ebd9a6e2d55b6cb2b06a65dc267cd354088ed314b0c41469fd5e76ddbd49c6c`. PR #61 merged as `b72a97a99769b278eb8ec75bc5eab62dc9599f29`.

The independent replay lineage now contains four verified attempts. The latest retained replay is workflow run `31631032761` on head `6aceefa1f4afb0e01869eda2734744753965c976`, artifact ID `9156974552`, SHA-256 `91f4aae1f5f02b7b9fae24909e34f31b2067d7325957c4d78b66dfcbe1751a49`. Within each attempt, same-seed model state, final metrics, and RNG state were exact. Across attempts, final loss and final accuracy were exact, while some floating-point submetrics drifted at roughly `1e-6` to `1e-7` and serialized PyTorch checkpoint bytes were not identical. The cross-attempt verifier JSON SHA-256 remained `1080efccc40d7a931451ec3fa5094113e877d54b4c16739cfe1861e22292f4af`.

Therefore the defensible claim is **semantic same-seed reproducibility under the verified CI path**, not byte-for-byte checkpoint identity across independent GitHub runners.

## Repaired ARC-v5 line

A train-only causal investigation found a failure in the quantized latent path. The opt-in repair `arc-v5-stable-ema-residual-0.03125` restored its bounded trainability gate. Repaired validation was then frozen and run separately. Its independent verdict remained `VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION`; neither the predeclared generalization gate nor the quantization-benefit gate was supported.

## Result

The defensible conclusion is **not** that LAM-JEPA beats ARC baselines. The current pipeline executes, the seed-order defect is preserved and minimally repaired, four independent same-seed replay attempts support semantic reproducibility under the documented CI path, adverse ARC evidence is retained, and the frozen ARC superiority/mechanism hypotheses remain unsupported. The repaired training path does not rescue the original hard-VQ claim.

## Uncertainty and limitations

- Five validation seeds do not justify broad benchmark-general significance claims.
- ARC-Challenge is one benchmark family and the test split remains intentionally locked for this failed hypothesis line.
- The pretrained comparator result is bounded development characterization, not a full matched confirmatory trial.
- CI smoke settings establish execution/reproducibility plumbing, not scientific effect size.
- Independent runners exhibit low-order floating-point drift and non-identical serialized checkpoint bytes; byte-exact cross-run identity is not claimed.
- The deterministic replay is not a substitute for an independent full five-seed ARC scientific rerun.
- No claim of educational effectiveness, general benchmark superiority, AGI, or general intelligence is supported.

## Stop rule

Do not tune the current architecture or thresholds against the locked ARC test. Any new repair, benchmark, or scientific hypothesis must receive a new versioned protocol before its validation evidence is observed.
