#!/usr/bin/env python3
"""Bounded first-forward identity check; synthetic input, no ARC training."""
from pathlib import Path
import json
import torch
from frozen_quantizer_excerpt import EMAQuantizer

def run_check() -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(0)  # fixture reproducibility, not an experimental seed search
    q = EMAQuantizer(32, 32)
    initial_weight = q.ema_weight.detach().clone()
    initial_code = q.codebook.detach().clone()
    z = initial_code[0].repeat(8, 1).clone()
    output, loss, indices = q(z)
    if not torch.equal(indices, torch.zeros(8,dtype=torch.long)):
        raise AssertionError('Synthetic fixture did not select code zero')
    unused = q.ema_count == 0
    expected = q.decay * initial_weight[unused] / q.eps
    torch.testing.assert_close(q.codebook.detach()[unused], expected, rtol=2e-6, atol=1e-6)
    torch.testing.assert_close(output.detach(), z, rtol=0, atol=2e-5)
    return {
        'check':'first_training_forward_unused_code_identity',
        'status':'PASS','scope':'Synthetic input to exact extracted quantizer, before any optimizer step.',
        'torch_version':torch.__version__,'codes':32,'latent_dimension':32,'synthetic_batch_size':8,
        'fixture_seed':0,'assigned_code':0,'unused_code_count':int(unused.sum()),
        'analytic_multiplier':q.decay/q.eps,
        'max_absolute_identity_error':float((q.codebook.detach()[unused]-expected).abs().max()),
        'assertion_rtol':2e-6,'assertion_atol':1e-6,
        'arc_data_loaded':False,'arc_training_runs':0,
        'establishes_cause_of_retained_arc_collapse':False,
        'proves_benefit_of_repair':False,
    }

if __name__=='__main__':
    result=run_check()
    out=Path(__file__).resolve().parents[1]/'verification/quantizer_identity_check.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
