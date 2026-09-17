# ICLR 2027 abstract / portal packet — 2026-09-17

## Purpose

This is a submission-operations packet for the bounded negative/inconclusive LAM-JEPA ARC-Challenge paper. It is derived from the exact ICLR source at commit `0c428bcb57346b4929f24e837f00ee1f5f880d9b` and does not change any scientific result, frozen protocol, seed, threshold, claim boundary, or locked-test state.

## Official deadline checkpoint

Verified against the official ICLR 2027 Call for Papers and Author Guidelines on 2026-09-17:

- Abstract deadline: **2026-09-18 11:59 PM Anywhere on Earth (UTC-12)**.
- Full paper deadline: **2026-09-25 11:59 PM Anywhere on Earth (UTC-12)**.
- The abstract must be genuine and informative; placeholder or duplicate abstracts may be removed.
- **No authors can be added or removed after the abstract deadline.** Author order may still be changed up to the full-paper deadline.
- Every author should have an up-to-date OpenReview profile before abstract submission.
- Submission is double blind; the review PDF and supplementary material must not reveal author identity.

Official sources:

- https://www.iclr.cc/Conferences/2027/CallForPapers
- https://iclr.cc/Conferences/2027/AuthorGuidelines
- https://openreview.net/group?id=ICLR.cc/2027/Conference

## Portal title — copy exactly unless the authors deliberately change it before submission

**LAM-JEPA on ARC-Challenge: A Reproducible Falsification-First Evaluation**

## Portal abstract — copy from the frozen ICLR source

We evaluate the project-named LAM-JEPA system on the AI2 Reasoning Challenge (ARC-Challenge) under a frozen falsification-first protocol with a gradient-active-parameter-matched supervised comparator, mechanism ablations, a shuffled-label validity control, and a bounded pinned pretrained comparison. Source inspection constrains the architecture claim: the evaluated ARC path is a small hashed-token, mean-pooled embedding model with vector quantization, learned sparse memory, a one-step latent-action rollout, and same-input exponential-moving-average target alignment; it is neither a Transformer encoder nor the canonical I-JEPA context-to-distinct-target prediction task. The preregistered superiority gate required a mean paired accuracy gain of at least +0.02 with a paired seed-level 95% bootstrap confidence interval excluding zero; planner and target-path attribution each required a paired full-minus-ablation gain of at least +0.01 with the paired interval excluding zero. Across five frozen validation seeds, full LAM-JEPA achieved 0.2549152542 ± 0.0129968064 accuracy versus 0.2664406780 ± 0.0154600058 for the matched supervised model, with paired mean difference -0.0115254237; the superiority gate therefore failed. Full-minus-no_planner was +0.0047457627 with 95% bootstrap CI [0.0, 0.0142372881], and full-minus-no_target was -0.0067796610 with CI [-0.0135593220, 0.0], so neither mechanism gate was met. One independent frozen-protocol external rerun/review reproduced the retained headline metrics and found single-code vector-quantizer collapse with constant downstream predictions in the reviewed runs; disabling quantization restored input-dependent predictions but did not establish above-chance ARC performance. We therefore report a bounded, reproducible failure-mechanism case study rather than architecture superiority, a general JEPA conclusion, or a general claim about vector quantization, and we keep the confirmatory ARC test locked for this failed hypothesis line.

## Scientific claim boundary that must remain intact

The submission may claim only the bounded evidence represented in the frozen source and retained artifacts. In particular:

- the frozen LAM-JEPA configuration does **not** satisfy the superiority gate;
- planner contribution is **not supported** under the preregistered criterion;
- target-path contribution is **not supported** under the preregistered criterion;
- the external rerun/review supports a **bounded collapse diagnosis in the reviewed quantized path**, not a general claim about JEPA or vector quantization;
- disabling quantization restored input-dependent predictions in the bounded diagnostic but did **not** establish above-chance ARC performance;
- the historical confirmatory ARC test remains locked and must not be opened to rescue this hypothesis line;
- the successor study is separate and cannot be used to retroactively rescue this paper.

## Exact-source / evidence checkpoint

Submission source branch at the time this packet was created:

- PR #175 head: `0c428bcb57346b4929f24e837f00ee1f5f880d9b`
- source: `paper/iclr2027_submission.tex`
- style blob: `paper/iclr2027_conference.sty`
- official style blob SHA: `f61ad7efce0855557694078c0945e6c33feb8236`
- retained anonymous submission artifact: `10319170333`
- retained archive digest: `sha256:6f8b06efa8aad774236a945f15cc23174672e9b6f73863a9f930bce8b51098e8`

The exact-head technical workflows recorded on PR #175 were terminal green for that head, but canonical `main` subsequently advanced. Therefore those checks establish the integrity of that exact submission source; they are not evidence of a fresh integration against current `main`.

## Author-set freeze checklist — HUMAN AUTHORITY REQUIRED

Before the abstract is submitted, freeze the complete author set. For **every** intended author, verify all of the following:

- [ ] legal/preferred author name is correct;
- [ ] author has explicitly agreed to authorship;
- [ ] contribution meets the team’s authorship standard;
- [ ] OpenReview profile exists and can be selected in the portal;
- [ ] preferred email in OpenReview is current;
- [ ] affiliation/profile information is current;
- [ ] no intended author is missing;
- [ ] no contributor who should not be an author is included;
- [ ] everyone understands that authors cannot be added or removed after the abstract deadline.

Do not submit the abstract while this checklist is incomplete.

## Abstract-submission portal checklist

- [ ] Open the ICLR 2027 Conference submission page on OpenReview.
- [ ] Create the submission using the final author set.
- [ ] Paste the title above.
- [ ] Paste the genuine abstract above.
- [ ] Verify every author resolves to the correct OpenReview profile.
- [ ] Check author order. It can still be reordered before the full-paper deadline, but the set is frozen after the abstract deadline.
- [ ] Confirm there is no author-identifying material in the anonymous PDF if one is uploaded at this stage.
- [ ] Review subject areas/keywords for truthful fit to representation learning, evaluation/reproducibility, reasoning/QA, and failure analysis as supported by the portal’s available categories.
- [ ] Save the OpenReview submission URL / forum ID.
- [ ] Save a timestamped receipt or screenshot after successful abstract submission.
- [ ] Record the exact portal title, abstract, author set, order, and OpenReview IDs in the project submission ledger.

## Full-paper follow-through after abstract lock

Before 2026-09-25 11:59 PM AoE:

- visually inspect the exact anonymous PDF page by page;
- verify bibliography/citations and references are resolved;
- ensure title/abstract in OpenReview still match the intended manuscript;
- confirm author order and profile metadata;
- preserve double-blind anonymity in main and supplementary material;
- retain the AI-use disclosure required by the ICLR 2027 source/policy workflow;
- rerun the claim/evidence audit if any quantitative or interpretation wording changes;
- retain exact final PDF hash, source SHA, build log, and portal submission receipt.

## Do-not-cross lines

Do not:

- open the locked ARC confirmatory test;
- add a positive/superiority framing unsupported by the frozen evidence;
- generalize the collapse diagnosis to JEPA or vector quantization broadly;
- mix successor-study outcomes into this paper as rescue evidence;
- add or remove authors after the abstract deadline;
- submit a placeholder abstract;
- expose author identity in the anonymous review package.

## Current stop condition

Technical packaging is substantially prepared. The remaining abstract-deadline gate is principally **human**: finalize the complete author set, confirm each OpenReview profile, review the exact title/abstract, and submit through OpenReview before the official deadline.