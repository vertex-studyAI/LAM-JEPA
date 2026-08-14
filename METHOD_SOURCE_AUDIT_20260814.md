# LAM-JEPA ARC Method Source Audit — 2026-08-14

**Purpose:** map every method statement in the negative ARC manuscript to executable source or the frozen protocol. This audit describes the configuration actually exercised by the frozen ARC line; it does not promote disconnected auxiliary modules into scientific claims.

## Source identity

Audit performed against repository head `b213ffa7c96e93ffc703565e3402d485643be18b`.

Primary implementation sources:

- `src/lam_jepa/model.py`
- `src/lam_jepa/benchmarking/arc_challenge.py`
- `scripts/benchmark/run_arc_protocol_v3_controls.py`
- `scripts/benchmark/run_arc_matched_baseline.py`
- `scripts/benchmark/run_arc_matched_baseline_v3.py`
- `protocols/arc_challenge_v3.json`
- `REPRODUCE.md`

The scientific full-controls artifact remains tied to scientific revision `760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb`; later documentation/reproducibility commits do not rewrite that outcome.

## 1. Frozen default configuration used by the ARC controls

`run_arc_protocol_v3_controls.py` constructs `LAMJEPAConfig()` and changes only the component switches needed for the named ablations. Therefore the full ARC configuration inherits these source defaults:

| Parameter | Value |
|---|---:|
| `input_dim` | 32 |
| `vocab_size` | 256 |
| `embed_dim` | 32 |
| `hidden_dim` | 64 |
| `proj_dim` | 32 |
| `pred_dim` | 16 |
| `num_codes` | 32 |
| `num_actions` | 8 |
| `num_rubric` | 4 |
| `num_layers` | 1 |
| `num_heads` | 4 |
| `dropout` | 0.1 |
| `momentum` | 0.996 |
| `temperature` | 0.07 |
| `max_steps` | 3 |
| `memory_size` | 64 |
| `use_quantizer` | true |
| `use_memory` | true |
| `use_planner` | true |
| `use_target` | true |
| `latent_noise_std` | 0.10 |
| `rollout_samples` | 1 |
| `use_uncertainty` | true |
| `use_counterfactuals` | true |

For the final ARC controls invocation, `model_steps=1`, so the full model performs exactly one planner transition during training/evaluation. `no_planner` changes only `use_planner=False`; `no_target` changes only `use_target=False`.

## 2. ARC input representation

Each ARC row is represented as:

`Question: <question> Choices: [0] <choice0> [1] <choice1> ...`

The formatted prompt is deterministically converted to token IDs with maximum length 96. The ARC numeric input is a single zero scalar, then padded by `MultiViewEncoder` to its configured 32-dimensional numeric projection input. Thus the ARC scientific result is driven by the tokenized question/choice text; the numeric branch receives no item-specific information.

The token encoder uses:

1. a learned token embedding of size `256 × 32`;
2. a learned positional parameter with capacity 512 positions;
3. identity sequence `encoder` in the current implementation;
4. layer normalization;
5. mean pooling over sequence positions.

The numeric branch is a linear projection. Token and numeric vectors are concatenated, passed through an MLP, then layer-normalized to produce the multi-view encoder representation.

### Important non-claim

Although `num_heads` and `num_layers` are fields in `LAMJEPAConfig`, the current `TokenEncoder.encoder` is `nn.Identity()`. The frozen ARC implementation is therefore **not** accurately described as a Transformer encoder solely because those config fields exist. The manuscript must describe the executable forward path, not inferred architectural intent.

## 3. Online/target latent path

The online path is:

`MultiViewEncoder -> Linear(embed_dim, proj_dim) -> z`.

The model also owns a structurally matched target encoder/projector. At construction the target is synchronized from the online encoder/projector. After each ARC optimizer step, `update_target()` applies EMA updates with `momentum=0.996`:

`theta_target <- tau * theta_target + (1 - tau) * theta_online`.

When `use_target=True`, the target latent is computed under `torch.no_grad()`. When `use_target=False`, the target is replaced by `z.detach()`.

## 4. Quantization

With `use_quantizer=True`, the projected latent enters `EMAQuantizer` with 32 codes of dimension 32. The nearest code is selected by squared Euclidean distance. The quantization objective is:

`L_quant = MSE(z_q.detach(), z) + MSE(z_q, z.detach())`.

The returned quantized latent uses a straight-through estimator:

`z_q <- z + (z_q - z).detach()`.

During training, code statistics are updated through EMA buffers and the codebook parameter data is replaced by the normalized EMA weights.

## 5. Sparse memory

With `use_memory=True`, `SparseMemory.retrieve(z_q)` is called and the returned memory read is fused by `SparseMemory.gated_correction(z_q, r)` to obtain `z_mem`. The ARC loss does not include a separately weighted memory-specific loss; memory influences the objective only through the downstream latent trajectory and answer head.

## 6. Latent-action planner

With planner enabled and `model_steps=1`, `LatentActionModel.rollout` performs one transition from `z_mem`.

For each step:

1. an action policy maps the current 32-dimensional latent to 8 action logits;
2. training samples an action from the temperature-scaled distribution; deterministic evaluation takes `argmax`;
3. the selected action embedding is concatenated with the current latent, memory read, and a zero uncertainty/intervention input;
4. separate MLPs predict transition mean and log-variance;
5. during stochastic training, Gaussian transition noise is sampled; evaluation is deterministic;
6. the transition is residual (`z_next = LayerNorm(z + delta)`).

The full-controls runner explicitly verifies that the full and `no_target` variants execute the requested planner step and that `no_planner` executes zero planner steps.

## 7. ARC answer head and training objective

`LAMARCClassifier` applies a dedicated four-choice linear head to `outputs["latent_summary"]`, where `latent_summary` is an MLP projection of the final rollout state.

For labels `y`, choice logits `c`, online quantized latent `z_q`, target latent `z_t`, and rollout states `s_1,...,s_K`, the executable ARC loss is:

`L_ARC = CE(c, y) + 0.5 L_align + 0.25 L_quant + 0.25 L_traj`,

where:

- `L_align = 1 - cosine_similarity(z_q, z_t)` through `cosine_alignment`;
- `L_quant` is the quantizer loss above;
- `L_traj = mean_k MSE(s_k, stopgrad(z_q))` over rollout states after the initial state, and is zero if no transition state exists.

For the full model at `model_steps=1`, `L_traj` contains one rollout-state MSE term. For `no_planner`, the trajectory has no post-initial state and `L_traj=0`.

Training uses AdamW at learning rate `3e-4`, zeroes gradients each minibatch, backpropagates `L_ARC`, clips global gradient norm to 1.0, takes one optimizer step, and then updates the EMA target.

### Auxiliary heads that are *not* independently supervised by the ARC objective

The backbone computes decoder, value, confidence, verifier, rubric, uncertainty and latent-summary outputs. Under `_lam_arc_loss`, only the dedicated four-choice answer head, the latent alignment path, quantizer path, and rollout trajectory receive direct loss terms. Auxiliary heads disconnected from this ARC objective must not be described as empirically validated by the ARC result.

## 8. Frozen training/evaluation budget

The final full-controls command is fixed at:

- seeds `[1,2,3,4,5]`;
- 20 epochs;
- batch size 32;
- learning rate `0.0003`;
- one model/planner step;
- all 1,117 protocol-eligible training rows;
- all 295 protocol-eligible validation rows;
- CPU execution;
- locked test not downloaded/evaluated.

Eligibility is feature-only: retain exactly four-choice rows, preserve source order, and retain exclusions/digests as evidence.

## 9. Required controls

### `no_planner`
Same default configuration except `use_planner=False`. The final state becomes the memory-corrected quantized latent with no latent-action transition, and the trajectory term is zero.

### `no_target`
Same default configuration except `use_target=False`. The alignment target becomes `z.detach()` rather than the EMA target path.

### Shuffled-label negative control
The full configuration is retrained with a deterministic permutation of training labels using seed `20260807`. The label multiset is preserved and a changed label digest is required. The frozen failure ceiling is validation accuracy `>0.35`; exceeding it would stop the protocol for leakage/shortcut investigation. Passing the ceiling is only a diagnostic and does not validate the representation mechanism.

### Choice-order robustness
Validation choices are deterministically reversed with exact answer-label remapping. Per-example predictions are retained, and the runner checks identical row identity/order.

## 10. Capacity-matched supervised baseline

The matched baseline is intentionally non-JEPA. Its forward path is:

`MultiViewEncoder -> Linear projector -> N residual supervised MLP blocks -> LayerNorm -> four-choice classifier`.

It excludes EMA target encoding, latent alignment, latent-action planning, sparse memory and vector quantization.

Capacity matching is based on **gradient-active** LAM-JEPA parameters under the exact ARC loss, not total nominal model parameters. A probe executes `_lam_arc_loss.backward()` and counts trainable parameters with non-`None` gradients. The supervised architecture search chooses depth/hidden size to fall within the frozen 0.99–1.01 parameter ratio, and then requires every matched-baseline trainable parameter to receive a gradient under supervised cross-entropy.

The matched baseline is trained on the same eligible rows, seeds, 20 epochs, batch size, AdamW family and `3e-4` learning rate. Its loss is plain four-choice cross-entropy.

The retained final evidence records 86,372 gradient-active LAM-JEPA parameters and 86,644 matched-baseline parameters (ratio approximately 1.00315). Those numbers are evidence values, not values hard-coded into the architecture-selection routine.

## 11. Statistical and claim gates

Frozen protocol-v3 defines:

- primary metric: multiple-choice accuracy;
- practical superiority threshold: absolute mean gain at least 0.02;
- uncertainty: paired seed-level 95% bootstrap CI for LAM-JEPA minus each trained baseline;
- superiority additionally requires the paired CI to exclude zero;
- mechanism attribution requires full-minus-ablation mean at least 0.01 and paired 95% CI excluding zero;
- negative-control ceiling: 0.35 validation accuracy;
- choice-order robustness maximum allowed LAM accuracy drop: 0.05.

Failure of a gate requires retaining the negative/inconclusive result; it does not authorize changing thresholds or unlocking test data.

## 12. Manuscript-safe description

A source-accurate compact description is:

> The frozen ARC model encodes the concatenated question and four choices using learned token and positional embeddings followed by normalization and mean pooling, fuses this representation with a zero-valued numeric branch, projects it to a 32-dimensional latent, optionally vector-quantizes and memory-corrects the latent, and—when enabled—takes one stochastic latent-action transition during training (deterministic at evaluation). A four-choice classifier operates on an MLP summary of the final latent state. Training minimizes answer cross-entropy plus weighted latent alignment, quantization, and trajectory-consistency terms. An EMA target encoder supplies the alignment target in the full model. Required ablations remove the planner or EMA target path one at a time. The supervised comparator removes JEPA/target/planner/memory/quantization machinery and is matched to the number of LAM-JEPA parameters that are gradient-active under the exact ARC objective.

## 13. Explicit non-claims from source audit

Do not claim from this ARC implementation that:

- the token encoder is a Transformer;
- `pred_dim`, `num_heads`, or `num_layers` establish an exercised mechanism merely because they exist in config;
- value/confidence/verifier/rubric/decoder/counterfactual/uncertainty heads are separately trained or validated by the ARC objective;
- latent actions have semantic tutoring meanings in ARC;
- sparse memory has an independently demonstrated contribution (it was not a required frozen ablation here);
- quantization has an independently demonstrated generalization benefit;
- target/planner contribution is positive—the frozen ablations do not support those claims.

This audit closes the source-to-method ambiguity without changing the frozen scientific result.
