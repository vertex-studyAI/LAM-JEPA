"""Actual current-package integration on generated tensors; no ARC files."""
import copy

import pytest
import torch

from lam_jepa import LAMJEPA, LAMJEPAConfig
from lam_jepa.eval.classification_v2 import evaluate_classification_v2


def real_model_and_batches():
    torch.manual_seed(211)
    cfg = LAMJEPAConfig(vocab_size=16, embed_dim=8, hidden_dim=16,
                       proj_dim=8, pred_dim=4, num_codes=4, num_actions=3,
                       num_heads=2, max_steps=2, memory_size=8)
    model = LAMJEPA(cfg)
    model.train()
    model.encoder.token.eval()  # Deliberately mixed module modes.
    batches = [
        {'tokens': torch.arange(12).reshape(3, 4) % 16,
         'numeric_x': torch.tensor([[0.1], [0.2], [0.3]]),
         'labels': torch.tensor([0, 4, 7])},
        {'tokens': torch.arange(8).reshape(2, 4) % 16,
         'numeric_x': torch.tensor([[0.4], [0.5]]),
         'labels': torch.tensor([1, 3])},
    ]
    return model, batches


def snapshot(model):
    return ([module.training for module in model.modules()],
            {key: value.detach().clone() for key, value in model.state_dict().items()},
            torch.random.get_rng_state().clone())


def assert_unchanged(model, before):
    modes, tensors, rng = before
    assert [module.training for module in model.modules()] == modes
    assert set(model.state_dict()) == set(tensors)
    assert all(torch.equal(model.state_dict()[key], value) for key, value in tensors.items())
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_current_model_matches_direct_sample_weighted_reference_without_mutation():
    model, batches = real_model_and_batches()
    reference = copy.deepcopy(model).eval()
    with torch.no_grad():
        logits = torch.cat([
            reference(batch['tokens'], numeric_x=batch['numeric_x'],
                      sample_rollout=False, noise_std=0.0)['logits'].double()
            for batch in batches
        ])
    labels = torch.cat([batch['labels'] for batch in batches])
    before = snapshot(model)
    result = evaluate_classification_v2(model, batches, n_bins=4)
    assert result['examples'] == 5
    assert result['classes'] == 16
    assert result['accuracy'] == (logits.argmax(dim=1) == labels).double().mean().item()
    assert result['negative_log_likelihood'] == pytest.approx(
        torch.nn.functional.cross_entropy(logits, labels).item(), abs=2e-14)
    probabilities = logits.softmax(dim=1)
    target = torch.nn.functional.one_hot(labels, 16).double()
    assert result['brier_multiclass'] == pytest.approx(
        ((probabilities - target) ** 2).sum(dim=1).mean().item(), abs=2e-14)
    assert result['scientific_clearance'] is False
    assert_unchanged(model, before)


def test_current_model_rejects_broadcast_labels_before_any_forward():
    model, batches = real_model_and_batches()
    before = snapshot(model)
    calls = []
    handle = model.register_forward_pre_hook(lambda *_: calls.append(True))
    bad = dict(batches[0], labels=batches[0]['labels'].reshape(-1, 1))
    try:
        with pytest.raises(ValueError, match='broadcasting is forbidden'):
            evaluate_classification_v2(model, [bad])
    finally:
        handle.remove()
    assert calls == []
    assert_unchanged(model, before)


def test_current_model_loader_failure_preserves_modes_and_no_partial_return():
    model, batches = real_model_and_batches()
    before = snapshot(model)

    def interrupted_loader():
        yield batches[0]
        raise RuntimeError('constructed loader interruption')

    with pytest.raises(RuntimeError, match='constructed loader interruption'):
        evaluate_classification_v2(model, interrupted_loader())
    assert_unchanged(model, before)
