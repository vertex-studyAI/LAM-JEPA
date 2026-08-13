# LAM-JEPA — Canonical Frozen ARC Result Tables

**Scientific conclusion:** `ARC_SUPERIORITY_AND_MECHANISM_HYPOTHESES_UNSUPPORTED`  
**Locked ARC test:** not evaluated for this failed hypothesis line.

These tables are manuscript-ready views of already-retained evidence. They introduce no new experiment, tuning, seed selection, or statistical analysis.

## Table 1 — full-controls validation

| Condition | Mean validation accuracy | Sample SD | n |
|---|---:|---:|---:|
| Full LAM-JEPA | 0.2549152493 | 0.0129968006 | 5 |
| `no_planner` | 0.2501694888 | 0.0129968006 | 5 |
| `no_target` | 0.2616949081 | 0.0203953938 | 5 |
| Shuffled-label control | 0.2630508393 | 0.0145011803 | 5 |

Source lineage: frozen full-controls artifacts and `EVIDENCE_AUDIT_20260813.md`.

## Table 2 — preregistered mechanism effects

| Paired effect | Mean | Sample SD | Bootstrap 95% CI | Criterion |
|---|---:|---:|---:|---|
| Full − `no_planner` | +0.0047457606 | 0.0106118432 | [0.0000000000, 0.0142372817] | NOT MET |
| Full − `no_target` | −0.0067796588 | 0.0092834301 | [−0.0135593176, 0.0000000000] | NOT MET |

Interpretation: neither tested mechanism satisfies its frozen benefit criterion.

## Table 3 — separately retained capacity-matched comparison

| System / effect | Mean accuracy/effect | Sample SD |
|---|---:|---:|
| LAM-JEPA | 0.2549152542 | 0.0129968064 |
| Capacity-matched supervised | 0.2664406780 | 0.0154600058 |
| Paired LAM − matched | −0.0115254237 | 0.0140994131 |

The final-decimal difference between the LAM value in this table and Table 1 comes from distinct retained result lineages. Do not silently rewrite them as one byte-identical artifact.

## Table 4 — bounded pretrained characterization

| System | Accuracy |
|---|---:|
| LAM-JEPA | 0.15625 |
| Pinned DeBERTa development comparator | 0.21875 |
| Paired delta | −0.06250 |

This is a bounded development characterization, not a broad inferiority theorem.

## Reproducibility wording

Independent full scientific reruns reproduce the aggregate full/no-planner/no-target scores, paired mechanism effects, shuffled-label summary, verifier summary, and strict verifier verdict. Low-level per-example probability values are not byte-identical across independent runners; maximum observed numeric drift is approximately `5.9186e-4` with no non-numeric leaf changes.
