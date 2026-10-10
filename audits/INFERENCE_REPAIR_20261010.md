# Prospective inference API repair — 2026-10-10

This change repairs public inference utilities. It does not revise the submitted
negative/inconclusive ARC study, the frozen model, any protocol, any retained
checkpoint or result, or the status of unopened confirmatory data. It does not
introduce a new scientific comparison.

## Repaired behavior

`lam_jepa.inference.predict.load_model` now loads the actual architecture saved
by the canonical trainer under `extra.config`. The former loader read only a
root-level `config` and silently built defaults when that key was absent; valid
non-default canonical checkpoints could therefore fail strict weight loading.
Legacy checkpoints with an explicit root-level configuration remain supported.
Missing, malformed, or conflicting configuration metadata is rejected.

The inference loader deserializes a trusted checkpoint once, strictly loads its
weights, and leaves the caller's Python, NumPy and CPU Torch random streams
unchanged. It no longer calls the training resume utility, which restores saved
training RNG state. This does not change the existing trusted-checkpoint input
boundary: canonical checkpoint payloads include pickled training metadata.

The public `predict` and `rollout` helpers temporarily enter evaluation mode and
request deterministic latent transitions. `torch.no_grad()` alone did not do
this: when handed a model still in training mode, the former helpers could apply
dropout, sample transitions, and mutate the quantizer's codebook and EMA buffers.
The repaired helpers restore each module's previous mode even after an exception,
including intentionally mixed target/predictor modes. The model implementation
and its training behavior remain byte-for-byte unchanged. These helpers are for
ordinary sequential inference; they do not synchronize concurrent training on a
shared model object.

## Usable API

```python
import torch

from lam_jepa.inference.predict import load_model, predict
from lam_jepa.inference.rollout import rollout

model = load_model("outputs/final.pt", device="cpu")
tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
result = predict(model, tokens, steps=2)
trajectory, actions, logits = rollout(model, tokens, steps=2)
```

Token IDs must use the checkpoint's vocabulary. Both helpers accept optional
`numeric_x`; `steps` must be a non-negative integer. A caller opting into these
repaired helpers gets state-preserving deterministic inference; direct calls to
the frozen model methods retain their existing behavior.

## Validation and scientific boundary

`tests/unit/test_inference_contracts.py` uses a tiny non-default real LAM-JEPA
model and synthetic inputs. It covers canonical checkpoint round trips, legacy
metadata, strict incompatible-weight rejection, single deserialization, RNG
preservation, repeated deterministic outputs, unchanged model parameters and
buffers, mixed-mode restoration on success/failure, and invalid step admission.
No training dataset or benchmark rows are required.

The new `Inference runtime contracts` PR workflow runs these regressions and the
existing model unit tests on CPU with one numerical thread. Existing workflows,
source/claim gates, training code and publication artifacts are unchanged. Local
compilation and focused lint are available in this workspace; tensor execution
is validated by the linked hosted job in the pull request because local Torch
installation was unavailable. A passing synthetic unit job establishes these
runtime contracts, not scientific superiority or completion of a new study.

## Explanation-path follow-up

Base: `2323d33e2be76c3121e0c7ca68c8bf69619b8624` (PR #201).

The public `explain()` helper still bypassed the inference guard applied to
`predict()` and `rollout()`. A caller supplying a training-mode model could get
different predictions from consecutive explanation requests, consume randomness
and update quantizer state. It also allowed invalid step counts to reach model
execution.

`explain()` now uses the same step validation, temporary evaluation mode,
deterministic transition request and per-module mode restoration as the other
helpers. Its output fields and tensor shapes are preserved. Explanation outputs
are explicitly compared with the corresponding prediction and rollout outputs.
This changes the prospective inference helper only; the frozen model, trainer,
checkpoint utilities, protocol files and retained negative ARC results are
unchanged.

The new explanation cases first produced seven failures alongside 40 passing
cases on the parent implementation. The repaired suite, including a new
cross-helper agreement case, passes 48 tests locally with CPU PyTorch 2.14.1.
All examples use the existing tiny constructed model and synthetic tensors.
No dataset, protected outcome, benchmark result or scientific training run is
accessed. The dedicated inference workflow now checks out and asserts the exact
PR head before running its compile, lint and runtime checks, and includes
`explain.py` in lint.

## Second development pass: checkpoint tensor admission

The inspected parent was `0cf0b7a18572fcf9e07c573092b53fe2e2d5dcc9`, including the separately authored explanation-path repair. `load_state_dict(strict=True)` checked names and shapes but accepted nonfinite parameter/buffer values and silently converted float64, integer or complex weight tensors to the reconstructed model dtype. The complex case discarded imaginary values. Nine new regression cases reproduced those failures; the preceding 48 selected tests passed.

The inference loader now checks that checkpoint state entries are tensors, match the reconstructed architecture's expected dtype, and contain only finite values before strict weight loading. Name/shape checks remain delegated to strict loading, and metadata/weights still come from one deserialization. Validation occurs inside the existing RNG guard. Explicitly incompatible dtypes now fail rather than silently changing checkpoint values; the canonical model's expected dtype is the supported inference serialization contract.

The selected suite passes **57 local tests** under CPU PyTorch 2.14.1, including the nine corruption/coercion cases, canonical checkpoint round trips, deterministic prediction/explanation/rollout, mixed modes, exception restoration and RNG preservation. This loader still accepts only trusted serialized checkpoints; the change is a numerical/format contract and does not change pickle trust.

The frozen model, trainer, resume loader, protocols, submitted negative/inconclusive ARC results and retained artifacts remain unchanged. No scientific or submission workflow was dispatched. Hosted verification is bound to the source revision in the PR.
