# LAM-JEPA external validation report index

**Updated:** 2026-09-03  
**Scientific boundary:** frozen negative/inconclusive ARC result; locked confirmatory test unchanged.

This index counts **people/reports**, not CI reruns. Internal GitHub Actions reproductions remain reproducibility evidence but are not external validators.

## Retained external reports

| ID | Date | Role | Source | Status | Core conclusion | Count toward 3-reproducer target? |
|---|---|---|---|---|---|---|
| `EXT-001` | 2026-08-31 | independent reproducer + methods reviewer | `paper/EXTERNAL_REVIEW_VQ_COLLAPSE_CORRECTION_20260831.md`; underlying notes named `REVIEW_FOR_ISSUE_102.md`, summarized in issue #102 | `REPRODUCED_WITH_MATERIAL_METHOD_CRITIQUE` | Frozen headline metrics reproduced; retained runs are constant classifiers; measured information-loss bottleneck localizes to the VQ path; no above-chance or superiority result created | **Yes: 1/3** |

## EXT-001 evidence summary

### Numerics reproduced

| Condition | retained | external rerun |
|---|---:|---:|
| full | 0.2549152493 | 0.2549152493 |
| no planner | 0.2501694888 | 0.2501694888 |
| no target | 0.2616949081 | 0.2616949081 |
| shuffled | 0.2630508393 | 0.2630508393 |
| matched supervised | 0.2664406780 | 0.2664406806 |

The retained review summary also records matching ARC hashes/eligibility counts, per-seed counts, the `86,372 / 86,644` parameter-count comparison, source-method correspondence, and valid references.

### Material discrepancy / new diagnosis

The external review found that all reviewed retained conditions/seeds predict a single class across 295 validation rows. The measured representation path is:

`295/295 distinct pre-quantizer latents -> 1/32 VQ codes -> constant post-quantizer latent -> constant output probabilities`

Removing quantization restored input-dependent predictions in the reviewer's bounded causal check, but did **not** establish above-chance task performance. Therefore the supported contribution is a reproducible failure-mechanism report, not evidence that a repaired LAM-JEPA beats a baseline.

### Claim impact

Allowed bounded wording:

- one independent rerun reproduced the frozen negative/inconclusive result;
- external inspection localized the retained collapse to the quantized path in this bounded setup;
- planner/EMA-target benefit remains unsupported;
- the experiment is informative as a reproducible failure/mechanism study.

Still prohibited:

- LAM-JEPA improves ARC performance;
- vector quantization is generally harmful;
- JEPA methods generally fail on reasoning;
- quantizer removal solves ARC;
- one report equals broad or multi-site external replication.

## Evidence completeness

The repository summary states that the full external notes were received as `REVIEW_FOR_ISSUE_102.md`. The following fields must remain `NOT_RETAINED_IN_THIS_INDEX` unless recovered from the original report with permission:

- reviewer identity if not already permissioned for public retention;
- exact OS / accelerator / Python version;
- exact command transcript;
- full deviation log;
- raw returned artifacts beyond the summarized evidence.

Missing fields are not fabricated. The report remains useful as the first auditable external rerun/review, but a second and third independently retained reproduction are still required by the campaign target.

## Internal reproducibility evidence — not counted as external reports

`RESULTS.md` retains multiple frozen GitHub Actions reruns that reproduce the aggregate scientific conclusion. These are important reproducibility evidence but do **not** increase the external-validator count.

## Next report slots

- `EXT-002`: open — second independent frozen-protocol rerun/review.
- `EXT-003`: open — third independent frozen-protocol rerun/review.
- `VENUE-001`: open — independent venue-fit critique focused on negative/reproducibility value.

Every failed, partial, or blocked external attempt should be indexed rather than silently dropped.
