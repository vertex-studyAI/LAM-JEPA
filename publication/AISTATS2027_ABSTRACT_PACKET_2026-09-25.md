# AISTATS 2027 abstract-registration packet — 25 September 2026

## Purpose

This packet retargets the existing bounded LAM-JEPA negative-result manuscript from the missed ICLR 2027 registration lane to AISTATS 2027 without changing the frozen science.

AISTATS 2027 deadlines:
- abstract registration: **29 September 2026, 23:59 AoE**;
- full paper + all supplementary material: **6 October 2026, 23:59 AoE**.

The author list is frozen at abstract registration. Major title/abstract changes after registration can trigger desk rejection.

## Proposed title

**LAM-JEPA on ARC-Challenge: A Reproducible Falsification-First Evaluation**

## Proposed abstract

We evaluate the project-named LAM-JEPA system on the AI2 Reasoning Challenge (ARC-Challenge) under a frozen falsification-first protocol with a gradient-active-parameter-matched supervised comparator, mechanism ablations, a shuffled-label validity control, and a bounded pinned pretrained comparison. Source inspection constrains the architecture claim: the evaluated ARC path is a small hashed-token, mean-pooled embedding model with vector quantization, learned sparse memory, a one-step latent-action rollout, and same-input exponential-moving-average target alignment; it is neither a Transformer encoder nor the canonical I-JEPA context-to-distinct-target prediction task. The preregistered superiority gate required a mean paired accuracy gain of at least +0.02 with a paired seed-level 95% bootstrap confidence interval excluding zero; planner and target-path attribution each required a paired full-minus-ablation gain of at least +0.01 with the paired interval excluding zero. Across five frozen validation seeds, full LAM-JEPA achieved 0.2549152542 ± 0.0129968064 accuracy versus 0.2664406780 ± 0.0154600058 for the matched supervised model, with paired mean difference -0.0115254237; the superiority gate therefore failed. Full-minus-no_planner was +0.0047457627 with 95% bootstrap CI [0.0, 0.0142372881], and full-minus-no_target was -0.0067796610 with CI [-0.0135593220, 0.0], so neither mechanism gate was met. One independent frozen-protocol external rerun/review reproduced the retained headline metrics and found single-code vector-quantizer collapse with constant downstream predictions in the reviewed runs; disabling quantization restored input-dependent predictions but did not establish above-chance ARC performance. We therefore report a bounded, reproducible failure-mechanism case study rather than architecture superiority, a general JEPA conclusion, or a general claim about vector quantization, and we keep the confirmatory ARC test locked for this failed hypothesis line.

## Scientific freeze

Registration must preserve these boundaries:
- negative/inconclusive frozen ARC result;
- no ARC superiority claim;
- no validated planner or EMA-target benefit claim;
- no general JEPA or vector-quantization failure claim;
- locked confirmatory ARC test remains unopened;
- no rescue tuning, seed replacement, threshold changes, or outcome-bearing reruns to improve the submission narrative.

## Portal gates before abstract registration

Human owner action is required for:
1. **Final author set.** Every author must be listed at abstract registration; no author may be added or removed afterward.
2. **OpenReview profiles.** Every author must have an accurate, current OpenReview profile.
3. **Submission quota check.** Confirm every author is within the AISTATS 2027 submission quota.
4. **Reciprocal reviewer.** Nominate one eligible author as the AISTATS reviewer, or make the venue's exceptional exemption request if no author qualifies. Do not invent reviewer qualifications.
5. **Title/abstract approval.** Treat the title and abstract above as the frozen registration text unless authors explicitly approve a correction before the abstract deadline.
6. **Presentation intent.** Confirm that at least one author intends to register and present if accepted.
7. **Receipt retention.** Save the OpenReview submission/forum ID and timestamp immediately after registration.

## Full-paper gates for 6 October

- use the official AISTATS 2027 style;
- maximum 8 pages of main text for the initial submission;
- double-blind anonymization;
- mandatory **AI Use Statement** immediately before references;
- reproducibility checklist;
- all supplementary material submitted by the paper deadline;
- retain exact source SHA and built PDF digest;
- reconcile the seed-level mechanism evidence without strengthening the negative result.

## Existing source state

The current venue-source lane already contains:
- an anonymous conference-form manuscript;
- the five-seed validation counts;
- bounded external collapse diagnosis;
- the locked-test stop rule;
- a detailed AI-use disclosure;
- green paper/reproducibility/claim-boundary CI on the relevant publication heads.

The next source change should be formatting/packaging for AISTATS, not scientific rescue.
