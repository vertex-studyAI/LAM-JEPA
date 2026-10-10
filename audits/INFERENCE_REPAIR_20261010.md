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
