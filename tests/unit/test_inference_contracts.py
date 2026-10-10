"""Prospective inference regressions using tiny synthetic checkpoints only."""

import random
from dataclasses import asdict

import numpy as np
import pytest
import torch

from lam_jepa.callbacks.checkpointing.save import save_checkpoint
from lam_jepa.inference.predict import load_model, predict
from lam_jepa.inference.rollout import rollout
from lam_jepa.model import LAMJEPA, LAMJEPAConfig


@pytest.fixture
def model():
    cfg = LAMJEPAConfig(
        input_dim=4,
        vocab_size=13,
        embed_dim=8,
        hidden_dim=12,
        proj_dim=6,
        pred_dim=4,
        num_codes=5,
        num_actions=3,
        num_rubric=2,
        num_heads=2,
        dropout=0.25,
        memory_size=6,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29)
        return LAMJEPA(cfg)


def assert_tree_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert not actual.requires_grad
        assert torch.isfinite(actual).all()
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_tree_equal(actual[key], expected[key])
    else:
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_tree_equal(left, right)


def snapshot_rng():
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()


def assert_rng_equal(actual, expected):
    assert actual[0] == expected[0]
    assert actual[1][0] == expected[1][0]
    np.testing.assert_array_equal(actual[1][1], expected[1][1])
    assert actual[1][2:] == expected[1][2:]
    assert torch.equal(actual[2], expected[2])


def test_canonical_checkpoint_round_trip_nondefault_architecture(tmp_path, model):
    path = save_checkpoint(tmp_path / "canonical.pt", model, extra={"config": asdict(model.cfg)})
    loaded = load_model(path)
    assert loaded.cfg == model.cfg
    assert all(not module.training for module in loaded.modules())
    assert loaded.state_dict().keys() == model.state_dict().keys()
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key]), key
    tokens = torch.tensor([[1, 2, 3], [3, 2, 1]])
    assert_tree_equal(predict(loaded, tokens, steps=2), predict(model, tokens, steps=2))


@pytest.mark.parametrize("duplicate_canonical", [False, True])
def test_legacy_root_configuration_is_supported(tmp_path, model, duplicate_canonical):
    payload = {"model": model.state_dict(), "config": asdict(model.cfg)}
    if duplicate_canonical:
        payload["extra"] = {"config": asdict(model.cfg)}
    path = tmp_path / "legacy.pt"
    torch.save(payload, path)
    assert load_model(path).cfg == model.cfg


def test_inference_load_preserves_callers_rng_instead_of_restoring_training_rng(tmp_path, model):
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    path = save_checkpoint(tmp_path / "canonical.pt", model, extra={"config": asdict(model.cfg)})
    random.seed(47)
    np.random.seed(47)
    torch.manual_seed(47)
    before = snapshot_rng()
    load_model(path)
    assert_rng_equal(snapshot_rng(), before)


def test_checkpoint_is_deserialized_once(tmp_path, model, monkeypatch):
    path = save_checkpoint(tmp_path / "canonical.pt", model, extra={"config": asdict(model.cfg)})
    original = torch.load
    calls = []

    def counted(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", counted)
    load_model(path)
    assert len(calls) == 1
    assert calls[0][1]["map_location"] == "cpu"


@pytest.mark.parametrize(
    "payload, message",
    [
        ([], "model state dictionary"),
        ({}, "model state dictionary"),
        ({"model": []}, "model state dictionary"),
        ({"model": {}}, "model configuration"),
        ({"model": {}, "extra": None}, "extra metadata"),
        ({"model": {}, "extra": {"config": []}}, "model configuration"),
        ({"model": {}, "config": {}}, "model configuration"),
        ({"model": {}, "config": {"unknown_model_option": 1}}, "configuration is invalid"),
        (
            {"model": {}, "config": {"vocab_size": 13}, "extra": {"config": {"vocab_size": 17}}},
            "conflicting model configurations",
        ),
    ],
)
def test_malformed_checkpoint_metadata_fails_closed(tmp_path, payload, message):
    path = tmp_path / "invalid.pt"
    torch.save(payload, path)
    before = snapshot_rng()
    with pytest.raises(ValueError, match=message):
        load_model(path)
    assert_rng_equal(snapshot_rng(), before)


def test_incompatible_weights_fail_strictly_without_consuming_rng(tmp_path, model):
    state = model.state_dict()
    state["quantizer.codebook"] = torch.zeros(1, model.cfg.proj_dim)
    path = tmp_path / "incompatible.pt"
    torch.save({"model": state, "extra": {"config": asdict(model.cfg)}}, path)
    before = snapshot_rng()
    with pytest.raises(RuntimeError, match="size mismatch for quantizer.codebook"):
        load_model(path)
    assert_rng_equal(snapshot_rng(), before)


@pytest.mark.parametrize("infer", [predict, rollout], ids=["predict", "rollout"])
@pytest.mark.parametrize("training", [False, True], ids=["eval", "train"])
@pytest.mark.parametrize("steps", [0, 2])
def test_inference_is_deterministic_and_preserves_state_and_mixed_modes(model, infer, training, steps):
    model.train(training)
    # A caller may intentionally freeze only the target encoder during training.
    model.target_encoder.eval()
    modes = [(module, module.training) for module in model.modules()]
    before = {key: value.clone() for key, value in model.state_dict().items()}
    rng = snapshot_rng()
    tokens = torch.tensor([[1, 2, 3], [3, 2, 1]])
    numeric = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    first = infer(model, tokens, numeric_x=numeric, steps=steps)
    second = infer(model, tokens, numeric_x=numeric, steps=steps)
    assert_tree_equal(first, second)
    assert_rng_equal(snapshot_rng(), rng)
    assert all(module.training == was_training for module, was_training in modes)
    for key, value in before.items():
        assert torch.equal(value, model.state_dict()[key]), key


@pytest.mark.parametrize("infer", [predict, rollout], ids=["predict", "rollout"])
def test_inference_restores_mixed_modes_after_forward_exception(model, infer, monkeypatch):
    model.train()
    model.target_encoder.eval()
    modes = [(module, module.training) for module in model.modules()]

    def fail(*args, **kwargs):
        assert all(not module.training for module in model.modules())
        assert kwargs["sample_rollout"] is False
        raise RuntimeError("synthetic forward failure")

    monkeypatch.setattr(model, "forward", fail)
    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        infer(model, torch.ones(1, 3, dtype=torch.long), steps=2)
    assert all(module.training == was_training for module, was_training in modes)


@pytest.mark.parametrize("infer", [predict, rollout], ids=["predict", "rollout"])
@pytest.mark.parametrize("steps", [-1, True, 1.5, float("nan")])
def test_invalid_steps_fail_before_model_execution(model, infer, steps, monkeypatch):
    calls = []

    def record(*args, **kwargs):
        calls.append(True)
        raise AssertionError("invalid steps reached the model")

    monkeypatch.setattr(model, "forward", record)
    with pytest.raises(ValueError, match="non-negative integer"):
        infer(model, torch.ones(1, 3, dtype=torch.long), steps=steps)
    assert not calls


@pytest.mark.parametrize("infer", [predict, rollout], ids=["predict", "rollout"])
def test_numpy_integer_steps_are_accepted(model, infer):
    infer(model, torch.ones(1, 3, dtype=torch.long), steps=np.int64(1))
