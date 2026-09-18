# LAM successor v1 — contextual encoder resolution

**Status:** frozen pre-outcome identity only; **not execution authorization**.  
**Date:** 2026-09-06.  
**Machine-readable source of truth:** `protocols/arc_successor_v1_encoder.json`.

## What this resolves

The successor hypothesis requires a genuine contextual encoder shared by the matched supervised baseline and JEPA treatment. Before any successor B0/T1 held-out outcome access, `ENCODER_FAMILY_AND_REVISION` is frozen to:

- Hugging Face repository: `distilbert/distilroberta-base`
- immutable revision: `fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b`
- checkpoint: `model.safetensors`
- checkpoint SHA-256 from Hub metadata: `2b11ca9cf3d2cbb44cc1a93ad96aedc2894231ae6e33e2d2268c3d7b1ff97663`
- architecture contract: RoBERTa-family base encoder loaded through `AutoModel`; 6 hidden layers, 768 hidden dimensions, 12 attention heads
- tokenizer: same repository and immutable revision through `AutoTokenizer(..., use_fast=True)`
- maximum experiment sequence length: 512 tokens with right truncation and right padding
- additional vocabulary: none
- `[WITHHELD_SPAN]`: ordinary native-tokenizer text, not a new special token or trainable vocabulary row
- representation: final hidden states pooled by attention-mask-weighted arithmetic mean over non-padding positions, including native special tokens; output dimension 768

The exact checked-in freeze artifact is SHA-256 `8d9ea79aa7ee9777d3aad9044b63f1db5813b6e214d2bc7d559539700f09b516`.

## Matching contract

B0 and T1 must use the same pinned base encoder and tokenizer for the supervised path. T1's target encoder initializes as an exact copy of the same pinned base weights. A post-outcome checkpoint/tokenizer swap, treatment-specific added token, or treatment-specific maximum sequence length is prohibited.

This resolution does **not** yet freeze encoder trainability, classifier-head implementation, predictor, EMA rule, optimizer, parameter-match tolerance, compute ratio, or representation-collapse thresholds. Those must be frozen in their own evidence-backed contracts before execution.

## Why this checkpoint

The predecessor's frozen negative/inconclusive ARC study used hashed whitespace-token representations. The successor asks a different question in which a contextual encoder predicts a genuinely withheld target view. DistilRoBERTa provides a compact contextual English encoder while allowing B0 and T1 to start from the exact same immutable pretrained weights. This is a design choice made before successor outcomes, not a result.

## Important contamination and byte-identity limits

The checkpoint was pretrained on OpenWebText. Nothing in this freeze proves that ARC/OpenBookQA or semantically overlapping material was absent from the model's pretraining corpus. That uncertainty remains part of the independent freshness/contamination review and must be disclosed in any eventual paper.

The Hub revision and published checkpoint hash are pinned, but the project has **not** yet retained local download receipts for every model/tokenizer/config byte. Before an admissible run, the downloaded files must be hashed locally and bound to the exact environment/run manifest.

## Remaining successor hard blockers

Resolving the encoder leaves six hard blockers in `protocols/arc_successor_v1_draft.json`:

1. independent review of `DATA_FRESHNESS_AUDIT`;
2. independently approved `CONFIRMATORY_DATASET` with exact local-byte receipt;
3. `COLLAPSE_THRESHOLDS`;
4. `PARAMETER_MATCH_TOLERANCE`;
5. `MAX_COMPUTE_RATIO`;
6. `EXACT_REPRODUCE_COMMAND` plus environment/source bindings.

`execution_authorized=false` remains unchanged. The prior ARC-v5 negative/inconclusive result and old locked confirmatory-test boundary remain unchanged.
