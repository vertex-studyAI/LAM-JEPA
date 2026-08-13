# LAM-JEPA paper result tables and figure provenance

All values below are retained evidence. No locked ARC test data are used.

## Table 1 — frozen ARC validation comparison

| System | Validation accuracy | Evidence class |
|---|---:|---|
| LAM-JEPA | `0.2549152493 ± 0.0129968006` | frozen 5-seed full-controls validation |
| Capacity-matched supervised | `0.2664406780 ± 0.0154600058` | retained matched-baseline validation |
| Paired LAM − matched | `-0.0115254237 ± 0.0140994131` | retained paired comparison |

Interpretation: the frozen evidence does not support LAM-JEPA superiority.

## Table 2 — mechanism controls

| Condition | Mean validation accuracy | Sample SD | n |
|---|---:|---:|---:|
| Full LAM-JEPA | `0.2549152493` | `0.0129968006` | 5 |
| `no_planner` | `0.2501694888` | `0.0129968006` | 5 |
| `no_target` | `0.2616949081` | `0.0203953938` | 5 |
| Shuffled-label control | `0.2630508393` | `0.0145011803` | 5 |

Paired effects:

| Effect | Mean | Bootstrap 95% CI | Frozen criterion |
|---|---:|---:|---|
| Full − `no_planner` | `+0.0047457606` | `[0.0, 0.0142372817]` | NOT MET |
| Full − `no_target` | `-0.0067796588` | `[-0.0135593176, 0.0]` | NOT MET |

## Figure files

- `paper/figures/arc_validation_accuracy.svg` — visual comparison including the capacity-matched baseline.
- `paper/figures/arc_mechanism_effects.svg` — paired planner/target effects with retained 95% bootstrap intervals.

### Figure evidence sources

Primary machine-readable source:

`experiments/repro_wave_2026_08_12/experiment_metadata.json`

The capacity-matched baseline value used in `arc_validation_accuracy.svg` is independently retained in `RESEARCH_STATUS.md`, `RESULTS.md`, and the matched-baseline evidence lineage. It is intentionally not inferred from a post-hoc fit.

The figures are descriptive visualizations only. Their source values, not graphical geometry, are the scientific evidence.

## Independent-rerun qualifier

The aggregate full/no-planner/no-target scores, paired effects, shuffled-label summary, verifier summary, and strict verifier verdict reproduce exactly across independent full scientific rerun attempts 2 and 3. Low-level per-example probabilities are not byte-identical across runners; the maximum observed numeric drift is approximately `5.9186e-4`.
