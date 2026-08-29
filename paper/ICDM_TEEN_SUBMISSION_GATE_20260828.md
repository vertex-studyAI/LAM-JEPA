# ICDM 2026 Teen Research Track submission gate — 2026-08-28

Canonical target: IEEE ICDM 2026 Teen Research Track / High School Student Research Symposium.

## Official venue requirements reverified 2026-08-28
- Deadline: 2026-08-30 AoE.
- IEEE Computer Society Proceedings manuscript format.
- Maximum 5 pages total, including figures, tables, and references.
- Single-blind review: author identities are visible.
- High-school student must be primary contributor.
- First-author affiliation must clearly include `High School Student`.
- Electronic submission through the ICDM 2026 submission system under Teen Research Track.
- Accepted papers require at least one author registration; accepted work must be presented in person to appear in proceedings.

Primary venue page used for this gate: https://icdm2026.neu.edu.cn/CallforTeen_en/

## Scientific GO boundary
The submission is a narrow negative/inconclusive falsification-first ARC-Challenge study. It MUST NOT be reframed as architecture superiority or a successful JEPA result.

Supported central statements:
1. Frozen LAM-JEPA validation accuracy: 0.2549152542 +/- 0.0129968064.
2. Capacity-matched supervised validation accuracy: 0.2664406780 +/- 0.0154600058.
3. Paired LAM-minus-matched difference: -0.0115254237 +/- 0.0140994131.
4. Planner and target-path ablations do not satisfy their frozen contribution criteria.
5. The bounded trainability repair did not convert validation to a positive result.
6. The ARC confirmatory test remains locked for this failed line.

Forbidden submission claims:
- LAM-JEPA outperforms the matched supervised baseline.
- planner benefit is established.
- EMA-target benefit is established.
- quantization benefit is established.
- the evaluated ARC encoder is a Transformer.
- the evaluated target objective is canonical I-JEPA context-to-distinct-target prediction.
- the negative result establishes general JEPA failure.
- independent external reproduction has occurred.

## Closed gates in this wave
- [x] target venue and deadline reverified against current official venue page
- [x] single-blind status reverified
- [x] five-page inclusive limit reverified
- [x] first-author High School Student affiliation requirement reverified
- [x] venue-specific IEEE source created at `paper/icdm_teen_2026.tex`
- [x] abstract tightened around the actual negative result
- [x] contribution boundary narrowed to evidence/reproducibility rather than superiority
- [x] architecture wording corrected: no Transformer claim
- [x] I-JEPA relationship wording constrained to same-input target alignment
- [x] frozen dataset retained-row counts stated
- [x] frozen seeds stated as 1,2,3,4,5
- [x] frozen core optimization settings stated
- [x] matched active-parameter counts stated
- [x] primary results table included
- [x] mechanism/validity controls table included
- [x] uncertainty intervals retained for mechanism effects
- [x] bounded pretrained comparison explicitly limited in scope
- [x] trainability-repair evidence separated from original hypothesis
- [x] locked-confirmatory-test stop rule retained
- [x] limitations retained in the short venue manuscript
- [x] claim -> metric -> artifact -> protocol -> source-SHA provenance chain retained

## Hard gates that remain before READY
- [ ] OWNER: truthful author name/order approved.
- [ ] OWNER: truthful school/city/country/email metadata approved; first author visibly marked `High School Student`.
- [ ] BUILD: compile `paper/icdm_teen_2026.tex` with the official IEEE-compatible toolchain.
- [ ] PAGE COUNT: inspect the actual PDF and prove <=5 pages INCLUDING references.
- [ ] CITATIONS: verify every bibliography entry and every novelty/comparison sentence against its primary paper/source.
- [ ] PROVENANCE: run the repository paper/provenance verifiers on the exact submission-source SHA.
- [ ] NUMBERS: regenerate or machine-check every quantitative value in the submission source against retained artifacts.
- [ ] PDF INSPECTION: verify tables do not overflow, references are readable, fonts embedded, no placeholders except owner-controlled metadata before final fill.
- [ ] SOURCE/PDF CONSISTENCY: freeze exact source SHA and checksum of the final uploaded PDF.
- [ ] OWNER: upload through official Teen Research Track submission system and retain confirmation/receipt.

## Hard decision rule
GO only if every hard gate above is closed without modifying frozen scientific thresholds, seeds, splits, locked-test state, or negative/mixed results. Otherwise NO-GO for upload even though the scientific paper itself is defensible.
