# AISTATS 2027 abstract packet

Status: **prepared only — not submitted**.

Source basis: `paper/iclr2027_submission.tex` at `0c428bcb57346b4929f24e837f00ee1f5f880d9b`.

## Title

**LAM-JEPA on ARC-Challenge: A Reproducible Falsification-First Evaluation**

## Abstract

We evaluate the project-named LAM-JEPA system on the AI2 Reasoning Challenge (ARC-Challenge) under a frozen falsification-first protocol with a gradient-active-parameter-matched supervised comparator, mechanism ablations, a shuffled-label validity control, and a bounded pinned pretrained comparison. Source inspection constrains the architecture claim: the evaluated ARC path is a small hashed-token, mean-pooled embedding model with vector quantization, learned sparse memory, a one-step latent-action rollout, and same-input exponential-moving-average target alignment; it is neither a Transformer encoder nor the canonical I-JEPA context-to-distinct-target prediction task. The preregistered superiority gate required a mean paired accuracy gain of at least +0.02 with a paired seed-level 95% bootstrap confidence interval excluding zero; planner and target-path attribution each required a paired full-minus-ablation gain of at least +0.01 with the paired interval excluding zero. Across five frozen validation seeds, full LAM-JEPA achieved 0.2549152542 ± 0.0129968064 accuracy versus 0.2664406780 ± 0.0154600058 for the matched supervised model, with paired mean difference -0.0115254237; the superiority gate therefore failed. Full-minus-no_planner was +0.0047457627 with 95% bootstrap CI [0.0, 0.0142372881], and full-minus-no_target was -0.0067796610 with CI [-0.0135593220, 0.0], so neither mechanism gate was met. One independent frozen-protocol external rerun/review reproduced the retained headline metrics and found single-code vector-quantizer collapse with constant downstream predictions in the reviewed runs; disabling quantization restored input-dependent predictions but did not establish above-chance ARC performance. We therefore report a bounded, reproducible failure-mechanism case study rather than architecture superiority, a general JEPA conclusion, or a general claim about vector quantization, and we keep the confirmatory ARC test locked for this failed hypothesis line.

## Freeze rule

If this abstract is registered, treat the title, abstract, and complete human author set as frozen except for corrections permitted by the venue. Do not strengthen the claim, open the locked confirmatory ARC test, add rescue results, or change thresholds/seeds/splits to improve the venue story.
