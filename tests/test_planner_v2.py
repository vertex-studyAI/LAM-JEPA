import pytest
import torch
from torch import nn

from lam_jepa.model import LatentActionModel, ValueHead, ConfidenceHead
from lam_jepa.planner import beam_plan
from lam_jepa.planner_v2 import beam_plan_v2


class TreeDynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_embed = nn.Embedding(2, 2)

    def step(self, z, *, action_override, **kwargs):
        node = z[:, 0].long()
        nxt = torch.where(node == 0, 1 + action_override, torch.where(node == 1, 3, 4))
        states = torch.stack((nxt.to(z.dtype), z[:, 1]), dim=1)
        return states, action_override, None, None, None, None


class TreeValue(nn.Module):
    def forward(self, z):
        values = z.new_tensor([[0, 2, 1, 0, 10], [0, 1, 2, 10, 0]])
        return values[z[:, 1].long(), z[:, 0].long()].unsqueeze(-1)


class ZeroHead(nn.Module):
    def forward(self, z):
        return z.new_zeros(len(z), 1)


def test_legacy_api_failure_remains_reproducible_and_v2_uses_six_outputs():
    torch.manual_seed(5)
    model = LatentActionModel(4, 3, 8, .1)
    value, verifier = ValueHead(4, 8, .1), ConfidenceHead(4, 8, .1)
    z = torch.randn(2, 4)
    with pytest.raises(ValueError, match="too many values to unpack"):
        beam_plan(model, value, verifier, z, steps=1)
    result = beam_plan_v2(model, value, verifier, z, steps=2, return_all=True)
    assert result.final_state.shape == z.shape
    assert len(result.trajectory) == 3 and len(result.actions) == 2
    assert torch.isfinite(result.score).all()


def test_width_two_recovers_better_non_greedy_path_and_ancestry():
    model, value, verifier = TreeDynamics(), TreeValue(), ZeroHead()
    z = torch.tensor([[0., 0.]])
    greedy = beam_plan_v2(model, value, verifier, z, steps=2, beam_width=1, return_all=True)
    search = beam_plan_v2(model, value, verifier, z, steps=2, beam_width=2, return_all=True)
    assert greedy.final_state[0, 0] == 3
    assert search.final_state[0, 0] == 4
    assert search.score > greedy.score
    assert [int(action[0]) for action in search.actions] == [1, 0]
    assert [int(state[0, 0]) for state in search.trajectory] == [0, 2, 4]


def test_each_batch_member_has_independent_beams():
    model, value, verifier = TreeDynamics(), TreeValue(), ZeroHead()
    z = torch.tensor([[0., 0.], [0., 1.]])
    together = beam_plan_v2(model, value, verifier, z, steps=2, beam_width=2, return_all=True)
    assert together.final_state[:, 0].tolist() == [4, 3]
    for i in range(2):
        alone = beam_plan_v2(model, value, verifier, z[i:i+1], steps=2, beam_width=2, return_all=True)
        torch.testing.assert_close(together.final_state[i:i+1], alone.final_state)
        torch.testing.assert_close(together.score[i:i+1], alone.score)
        for combined, single in zip(together.actions, alone.actions):
            torch.testing.assert_close(combined[i:i+1], single)


def test_eval_search_preserves_parameters_rng_and_mixed_training_modes():
    torch.manual_seed(88)
    model = LatentActionModel(4, 3, 8, .5)
    value, verifier = ValueHead(4, 8, .5), ConfidenceHead(4, 8, .5)
    model.transition_mu.eval()
    models = (model, value, verifier)
    flags = {module: module.training for parent in models for module in parent.modules()}
    before = [{key: tensor.clone() for key, tensor in parent.state_dict().items()} for parent in models]
    z = torch.randn(2, 4)
    rng = torch.random.get_rng_state().clone()
    a = beam_plan_v2(*models, z, steps=2)
    b = beam_plan_v2(*models, z, steps=2)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not a.requires_grad
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert all(module.training == flag for module, flag in flags.items())
    for parent, expected in zip(models, before):
        for name, tensor in parent.state_dict().items():
            torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)


def test_zero_steps_returns_initial_state_without_actions():
    z = torch.tensor([[0., 0.]])
    result = beam_plan_v2(TreeDynamics(), TreeValue(), ZeroHead(), z, steps=0, return_all=True)
    torch.testing.assert_close(result.final_state, z)
    assert len(result.trajectory) == 1 and result.actions == []
    assert result.score.item() == 0


@pytest.mark.parametrize("kwargs", [{"steps": -1}, {"beam_width": 0}, {"steps": True},
    {"temperature": 0}, {"temperature": float("nan")}, {"max_candidates": 1},
    {"max_state_elements": 1}, {"steps": 129}])
def test_invalid_parameters_and_budgets_fail_before_large_allocation(kwargs):
    with pytest.raises(ValueError):
        beam_plan_v2(TreeDynamics(), TreeValue(), ZeroHead(), torch.tensor([[0., 0.]]), **kwargs)


def test_failure_restores_training_flags():
    model, value, verifier = TreeDynamics(), TreeValue(), ZeroHead()
    value.eval()
    with pytest.raises(ValueError, match="budget"):
        beam_plan_v2(model, value, verifier, torch.tensor([[0., 0.]]), max_candidates=1)
    assert model.training and not value.training and verifier.training
