# LAM-JEPA Results Ledger

**Reproducibility wave:** 2026-08-12  
**Frozen source revision:** `2f59b4297e5978d4ce769ebe95adb363e1e75d7a`  
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

## 2026-08-12 exact-head reproduction

The repository's `Reproducibility CI` job was rerun without changing the source revision after observing prior results.

- workflow run: `31610608912`
- rerun attempt: `2`
- job: `94178401933` (`deterministic-training-smoke`)
- head SHA: `2f59b4297e5978d4ce769ebe95adb363e1e75d7a`
- runner: GitHub-hosted `ubuntu-latest`
- Python: 3.11 as pinned by workflow
- compute boundary: CPU-only; workflow asserts CUDA unavailable
- started: `2026-08-12T16:06:02Z`
- completed: `2026-08-12T16:07:43Z`
- wall-clock job duration: about 101 s
- conclusion: **SUCCESS**

The rerun successfully re-exercised protocol verification, checksum-addressed ARC train/validation download, a bounded external ARC smoke, a paired two-seed benchmark interface, deterministic training/checkpoint verification, all-task evaluation, matched reference baselines, exact-row comparison, a paper-results package, paired component ablations, and evidence upload.

**Important distinction:** this CI rerun verifies the executable evidence pipeline at the exact current revision. It is not a new five-seed 20-epoch confirmatory statistical sample and must not be reported as one. The canonical scientific estimates above remain the retained frozen validation evidence.

## Repaired ARC-v5 line

A train-only causal investigation found a failure in the quantized latent path. The opt-in repair `arc-v5-stable-ema-residual-0.03125` restored its bounded trainability gate. Repaired validation was then frozen and run separately. Its independent verdict remained `VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION`; neither the predeclared generalization gate nor the quantization-benefit gate was supported.

## Result

The defensible conclusion is **not** that LAM-JEPA beats ARC baselines. The reproducible result is that the current pipeline executes, adverse evidence is retained, and the frozen ARC superiority/mechanism hypotheses are unsupported. The repaired training path does not rescue the original hard-VQ claim.

## Uncertainty and limitations

- Five validation seeds are enough to expose instability but not to justify broad significance or benchmark-wide generalization claims.
- ARC-Challenge is one benchmark family and the test split remains intentionally locked for this failed hypothesis line.
- The pretrained comparator result is bounded development characterization, not a full matched confirmatory trial.
- CI smoke settings are deliberately tiny and establish reproducibility plumbing, not scientific effect size.
- No claim of educational effectiveness, general benchmark superiority, AGI, or general intelligence is supported.

## Stop rule

Do not tune the current architecture or thresholds against the locked ARC test. Any new repair, benchmark, or scientific hypothesis must receive a new versioned protocol before its validation evidence is observed.
