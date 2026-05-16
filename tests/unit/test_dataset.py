from __future__ import annotations

from lam_jepa.data import sample_batch


def test_dataset_shapes_and_labels():
    for task in ("parity", "modadd", "algebra", "chain"):
        batch = sample_batch(task, batch=8, vocab_size=256)
        assert batch.tokens.shape[0] == 8
        assert batch.labels.shape[0] == 8
        assert batch.rubric.shape[0] == 8
