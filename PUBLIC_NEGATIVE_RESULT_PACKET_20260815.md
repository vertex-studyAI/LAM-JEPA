# LAM-JEPA — Public Negative-Result Release Packet

**State:** PARTIAL — internally reproducible negative result; owner metadata and genuinely independent external validation remain pending.

This packet packages the strongest truthful public boundary available without changing the frozen scientific result or touching the locked ARC test split.

## Immutable scientific identity

- Repository: `vertex-studyAI/LAM-JEPA`
- Numeric/provenance basis: `bf8311e1a4d240e2891e51af38eaf7754944e300`
- External reproduction/review task definition: `EXTERNAL_VALIDATION_PACKET_20260814.md`
- Frozen ARC-v3 scientific source identified by retained protocol/evidence: `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`
- Locked ARC test: **not used; must remain unused for this failed hypothesis line**.

## Result framing

Under the frozen exactly-four-choice ARC-Challenge development-validation protocol, this specific small LAM-JEPA configuration did not outperform its gradient-active-parameter-matched supervised comparator. The evaluated one-step latent-action-rollout and EMA target-path ablations did not meet the preregistered contribution criteria.

This is a **negative result about the evaluated configuration and protocol**, not a family-level claim that JEPA, planning, Transformers, or educational systems fail in general.

## Reproduction command

After independently downloading and hash-checking the three raw GitHub Actions artifacts listed in `EXTERNAL_VALIDATION_PACKET_20260814.md`, regenerate the deterministic paper assets with:

```bash
python scripts/analysis/generate_arc_negative_paper_assets.py \
  --full-controls /ABS/PATH/arc-protocol-v3-full-controls-validation.json \
  --matched /ABS/PATH/matched-v3-full-validation.json \
  --pretrained /ABS/PATH/arc-pretrained-v2-deberta.json \
  --out-dir /ABS/PATH/lam-paper-assets
```

The external-validation packet records the expected ZIP/raw hashes and regenerated asset hashes. Any mismatch must be reported rather than normalized away.

## Environment

Repository dependency declaration: `requirements.txt` at the immutable numeric/provenance basis. It currently declares:

- `torch>=2.2`
- `numpy>=1.26`
- `tqdm>=4.66`
- `sympy>=1.12`
- `scikit-learn>=1.4`
- `matplotlib>=3.8`
- `pandas>=2.2`

For a public immutable release, the validator/releaser must record the exact Python version, resolved dependency versions, OS, hardware, artifact hashes and repository revision used. The broad lower-bound dependency file is not by itself an exact environment lock.

## Limitations

- Development-validation evidence only; locked ARC test remains untouched.
- Five-seed primary comparison is bounded and should not be generalized beyond the evaluated setup.
- The bounded pinned-DeBERTa comparison is characterization only, not a final compute-matched superiority baseline.
- The retained planner comparison is a one-step latent-action-rollout ablation, not a broad test of search/planning.
- The evaluated target path does not establish a general benefit of target networks.
- No evidence here establishes educational/tutoring effectiveness.
- No genuinely independent outside reproduction/review has yet been returned.

## Owner-controlled release metadata

These fields are intentionally **not inferred from repository history**:

- License / redistribution compatibility: **[OWNER APPROVAL REQUIRED]**
- Author list and order: **[OWNER APPROVAL REQUIRED]**
- Release version/tag: **[OWNER APPROVAL REQUIRED]**
- Dataset/code redistribution boundary: **[OWNER APPROVAL REQUIRED]**

Do not publish an authoritative license, author order, or release identifier until the owner approves it.

## Citation metadata

Until owner approval, use only a non-authoritative working record:

- Working title: **LAM-JEPA ARC Negative-Result Package**
- Repository: `vertex-studyAI/LAM-JEPA`
- Scientific/provenance revision: `bf8311e1a4d240e2891e51af38eaf7754944e300`
- Authors: **[OWNER APPROVAL REQUIRED]**
- Version/tag: **[OWNER APPROVAL REQUIRED]**
- Persistent identifier / DOI: **none claimed**
- Publication status: **not claimed**

A final `CITATION.cff` should be created only after author/title/version/identifier approval.

## External-validation state

**BLOCKED / PENDING EXTERNAL PARTY.**

The repository contains an immutable independent reproduction/review task with exact artifact identities, expected numerical values, deterministic asset generation and skeptical source/method questions. That preparation is not external validation. The state may be upgraded only after a genuinely independent party returns a report based on the immutable packet.

## Public claim allowed

> A reproducible development-validation study found that this evaluated LAM-JEPA configuration did not outperform its matched supervised comparator and that the tested planner/target-path contributions failed their frozen criteria; the locked ARC test remained unused.

## Public claims forbidden

Do not claim:

- LAM-JEPA superiority;
- JEPA family failure or superiority;
- general planning/search failure;
- educational effectiveness;
- peer review, publication, acceptance, citation or adoption;
- independent reproduction or external validation before an outside report exists.
