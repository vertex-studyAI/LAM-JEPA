"""Representable norm penalties should not fail due to squared-norm overflow."""
import math

import pytest
import torch
from torch import nn

from lam_jepa.planner_v2 import beam_plan_v2


class IdentityDynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_embed = nn.Embedding(2, 2)

    def step(self, z, *, action_override, **kwargs):
        return z.clone(), action_override, None, None, None, None


class ZeroHead(nn.Module):
    def forward(self, z):
        return z.new_zeros(len(z), 1)


@pytest.mark.parametrize("dtype,magnitude", [(torch.float16, 60000.), (torch.float32, 1e30), (torch.float64, 1e300)])
def test_large_finite_states_preserve_the_mathematical_norm_penalty(dtype, magnitude):
    z = torch.tensor([[magnitude, magnitude]], dtype=dtype)
    dynamics = IdentityDynamics()
    result = beam_plan_v2(dynamics, ZeroHead(), ZeroHead(), z, steps=2, beam_width=2, return_all=True)
    expected = -2 * .01 * math.hypot(float(z[0, 0]), float(z[0, 1]))
    assert torch.isfinite(result.score).all()
    assert float(result.score[0]) == pytest.approx(expected, rel=2e-3 if dtype == torch.float16 else 2e-7)
    torch.testing.assert_close(result.final_state, z, rtol=0, atol=0)
    assert [action.item() for action in result.actions] == [0, 0]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_norm_does_not_divide_by_zero(dtype):
    result = beam_plan_v2(IdentityDynamics(), ZeroHead(), ZeroHead(), torch.zeros(2, 2, dtype=dtype), steps=2, return_all=True)
    torch.testing.assert_close(result.score, torch.zeros(2, dtype=dtype), rtol=0, atol=0)


def test_zero_depth_output_is_detached_like_nonzero_search():
    z = torch.tensor([[1., 2.]], requires_grad=True)
    result = beam_plan_v2(IdentityDynamics(), ZeroHead(), ZeroHead(), z, steps=0, return_all=True)
    assert not result.final_state.requires_grad
    assert not result.trajectory[0].requires_grad
    assert z.requires_grad
    torch.testing.assert_close(result.final_state, z, rtol=0, atol=0)
