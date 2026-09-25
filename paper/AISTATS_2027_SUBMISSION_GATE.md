# LAM-JEPA — AISTATS 2027 submission gate

Last verified against the official AISTATS 2027 CFP / FAQ: **2026-09-25**.

This packet prepares an optional venue route. It does not authorize submission and does not alter the frozen scientific result.

## Official deadlines

- Abstract registration: **2026-09-29 23:59 AoE**
- Full paper and all supplementary material: **2026-10-06 23:59 AoE**
- Conference: **2027-05-03 through 2027-05-06, Montréal**

An abstract must be registered by the abstract deadline for the paper to proceed.

## Submission constraints to enforce

- [ ] OpenReview abstract record exists with a retained timestamped receipt.
- [ ] Complete human author list is final at abstract submission.
- [ ] Every author has an up-to-date OpenReview profile and required conflict/profile metadata.
- [ ] Submission-quota eligibility is checked for every author.
- [ ] One eligible author is nominated as reciprocal reviewer, or a justified venue exemption is requested.
- [ ] Paper remains double blind: no author names, affiliations, acknowledgements, or identity-revealing links.
- [ ] No concurrent archival-conference submission exists during AISTATS review.
- [ ] Main text is at most 8 pages under the **official AISTATS 2027 paper pack**.
- [ ] References, AI Use Statement, reproducibility checklist, and appendices are handled under the venue's exclusions/order.
- [ ] Mandatory **AI Use Statement** appears before references and is reconciled to actual tool usage.
- [ ] Reproducibility checklist is complete.
- [ ] Supplementary material is consistent with the main submission and uploaded by the full-paper deadline.
- [ ] Any public preprint remains venue-neutral and is never advertised as an AISTATS submission during review.

## Scientific boundary

The retained conclusion remains negative/inconclusive under the frozen ARC-Challenge validation protocol:

- no superiority over the matched supervised comparator;
- neither preregistered planner nor target-path attribution gate was met;
- the reviewed VQ-collapse diagnosis is local to the evaluated path;
- disabling quantization restored input dependence but did not establish above-chance ARC performance;
- no general JEPA or vector-quantization failure claim;
- locked ARC confirmatory test remains closed for this failed hypothesis line.

No new experiment is required to prepare the venue package. Any successor scientific experiment must be separately preregistered and cannot be inserted post-outcome into this submission line.

## Source and formatting provenance

Current scientific/manuscript source basis:
`0c428bcb57346b4929f24e837f00ee1f5f880d9b`

The official AISTATS 2027 paper pack is linked by the conference CFP. Do **not** substitute an unofficial or locally modified style file. Before creating the AISTATS TeX candidate:

1. obtain the official paper pack from the conference link;
2. record the downloaded archive digest and style-file blob/digest;
3. create a separate AISTATS source file rather than mutating the historical ICLR source;
4. compile from a clean environment;
5. verify eight-page main-text limit, anonymity, citations, figures/tables, AI-use statement, and reproducibility checklist;
6. retain the PDF/source artifact bound to the exact source SHA.

## Human decision checkpoint

By 2026-09-26, record exactly one state in issue #184:

- `SUBMIT_AISTATS`: proceed with official-style adaptation and abstract registration; or
- `DO_NOT_SUBMIT_AISTATS`: close the conference chase and release as a technical report/preprint/reproducibility package without changing the science.
