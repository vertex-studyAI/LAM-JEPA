# LAM-JEPA publication next actions — 2026-09-15

## Publication decision

Treat the current ARC line as a bounded reproducible negative-result and failure-mechanism paper. Do not rescue it into a positive architecture paper, open the locked ARC confirmatory test, retune against the frozen outcome, or mix the successor OpenBookQA study into the current manuscript.

## Evidence already strong enough to write around

- Frozen ARC full-controls validation: five seeds, 20 epochs, all eligible train/validation rows, locked test split.
- Full LAM-JEPA mean frozen validation accuracy is approximately 0.2549.
- The capacity-matched supervised baseline performs better under the frozen protocol.
- Planner ablation: full minus no-planner = +0.0047457606 with 95% bootstrap CI [0.0, 0.0142372817]; the frozen mechanism criterion is not met.
- Target-path ablation: full minus no-target = -0.0067796588 with 95% bootstrap CI [-0.0135593176, 0.0]; the frozen mechanism criterion is not met.
- Deterministic shuffled-label control remains below the frozen 0.35 ceiling.
- Project-controlled reruns reproduce the aggregate negative verdict despite low-order floating-point drift.
- One genuinely external frozen-protocol rerun/review reproduces the headline result and localizes collapse in the reviewed quantized path to a single VQ code per run with constant downstream predictions.

## Experiment gap

No rescue experiment is required for the current bounded paper. The current scientific story is already falsifiable and complete enough to report as a negative result. Any successor confirmatory study must remain a separate protocol and paper line.

## Baseline gap

The capacity-matched supervised baseline is the primary comparator and should remain primary. The shuffled-label control is a sanity control, not a performance baseline. Do not add weaker baselines only to improve relative appearance.

## Ablation gap

The planner and target-path ablations already answer the principal component-contribution claims. The external VQ collapse diagnostic supports a bounded mechanism diagnosis only. It does not support a general claim against vector quantization or JEPA architectures. No additional ablation is required unless a reviewer identifies a specific confound before submission.

## Reproducibility gap

Scientific reproducibility is comparatively strong. The remaining work is packaging reproducibility:

1. Freeze one publication-candidate Git SHA.
2. Generate the current negative-result paper only from the canonical negative manuscript source.
3. Bind the PDF, figures, tables, bibliography, claim ledger, environment metadata, and retained result artifacts to that SHA with checksums.
4. Run a clean rerender and verify deterministic or explicitly explained output differences.
5. Keep the legacy positive `paper.tex` and tracked legacy `paper.pdf` quarantined from the submission package.

## Manuscript gap

`MANUSCRIPT_DRAFT_NEGATIVE_ARC.md` is the canonical current source. The legacy architecture manuscript is not scientifically current. Convert the canonical negative source into a venue-neutral paper whose headline is the reproducible failure result, not architecture superiority. Every quantitative sentence should map to `CLAIM_LEDGER.md` and retained artifacts.

The manuscript should center four contributions:

1. a frozen controlled evaluation that falsifies the original superiority hypothesis;
2. matched-baseline and mechanism-ablation evidence;
3. reproducibility analysis including the bounded floating-point drift result;
4. the externally reproduced VQ-collapse failure mechanism, stated only at the tested scope.

## Submission gap

The ICDM 2026 Teen Research Track deadline has passed, so that route is closed unless an official late-submission path is independently documented. The next live venue must be selected separately. Before any submission, close truthful author order and affiliations, contribution metadata, licensing/release permissions, related-work integration, final claim audit, venue formatting, and submission-receipt retention.

## Single next move

Freeze one publication-candidate SHA and build a clean venue-neutral PDF plus evidence manifest from `MANUSCRIPT_DRAFT_NEGATIVE_ARC.md`. Do not run a result-rescue experiment first.

## Stop rules

- Do not open the locked ARC confirmatory test to improve this paper.
- Do not retune a frozen result after seeing outcomes.
- Do not describe the externally reviewed VQ collapse as a universal JEPA or vector-quantization result.
- Do not submit or cite the legacy positive paper as the current scientific record.
