"""Generated tensors only; no ARC data or retained predictions are accessed."""
import copy
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SOURCE = Path(__file__).resolve().parents[2] / 'src/lam_jepa/eval/classification_v2.py'
spec = importlib.util.spec_from_file_location('classification_under_test', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def summary(logits, labels, **kwargs):
    acc = m.ClassificationAccumulator(**kwargs)
    acc.update(logits, labels)
    return acc.result()


def test_hand_computable_metrics():
    p = torch.tensor([[0.8, 0.2], [0.25, 0.75]], dtype=torch.float64)
    out = summary(p.log(), torch.tensor([0, 0]), n_bins=2)
    assert out['accuracy'] == 0.5
    assert out['mean_top1_probability'] == pytest.approx(0.775)
    assert out['brier_multiclass'] == pytest.approx((0.08 + 1.125) / 2)
    assert out['negative_log_likelihood'] == pytest.approx(-math.log(0.8*0.25) / 2)
    assert out['ece_top_label'] == pytest.approx(0.275)
    assert out['ece_bins'][0]['accuracy'] is None
    assert out['ece_bins'][1]['count'] == 2
    assert out['scientific_clearance'] is False


@pytest.mark.parametrize('parts', [[7], [1, 6], [2, 3, 2], [1]*7])
def test_sample_weighted_partition_invariance(parts):
    logits = torch.tensor([[2,0,1], [0,3,1], [0,0,4], [1,1,1], [3,1,0], [0,2,0], [4,0,0]], dtype=torch.float64)
    y = torch.tensor([0, 1, 2, 0, 1, 0, 2])
    expected = summary(logits, y, n_bins=5)
    acc = m.ClassificationAccumulator(5)
    i = 0
    for n in parts:
        acc.update(logits[i:i+n], y[i:i+n]); i += n
    actual = acc.result()
    for field in ('accuracy','negative_log_likelihood','brier_multiclass','mean_top1_probability','ece_top_label'):
        assert actual[field] == pytest.approx(expected[field], abs=2e-15)
    assert actual['examples'] == 7
    assert [b['count'] for b in actual['ece_bins']] == [b['count'] for b in expected['ece_bins']]


def test_numpy_independent_reference():
    x = np.array([[0.4, -1.2, 2.1], [0., 0., 0.], [2.3, 0.5, -0.7]], dtype=np.float64)
    y = np.array([1, 0, 2])
    e = np.exp(x-x.max(axis=1,keepdims=True)); p = e/e.sum(axis=1,keepdims=True)
    out = summary(torch.from_numpy(x), torch.from_numpy(y))
    assert out['brier_multiclass'] == pytest.approx(np.mean(np.sum((p-np.eye(3)[y])**2,axis=1)))
    assert out['negative_log_likelihood'] == pytest.approx(-np.mean(np.log(p[np.arange(3),y])))


@pytest.mark.parametrize('dtype', [torch.float16,torch.bfloat16,torch.float32,torch.float64])
def test_input_dtype_and_no_autograd_retention(dtype):
    x = torch.tensor([[1,0],[0,1]], dtype=dtype,requires_grad=True)
    before = x.clone(); rng = torch.random.get_rng_state().clone()
    acc = m.ClassificationAccumulator(); acc.update(x,torch.tensor([0,1]))
    assert acc.result()['accuracy'] == 1
    assert torch.equal(x,before)
    assert x.grad is None and torch.equal(torch.random.get_rng_state(),rng)
    assert not any(isinstance(value,torch.Tensor) for value in vars(acc).values())


@pytest.mark.parametrize('labels', [torch.tensor([[0],[0]]),torch.tensor(0),torch.tensor([0]),torch.tensor([0.,0.]),torch.tensor([False,False]),torch.tensor([-1,0]),torch.tensor([0,2])])
def test_rejects_invalid_labels_transactionally(labels):
    acc=m.ClassificationAccumulator(); acc.update(torch.zeros((1,2)),torch.tensor([0]))
    before=copy.deepcopy(vars(acc))
    with pytest.raises(ValueError): acc.update(torch.zeros((2,2)),labels)
    assert vars(acc)==before


@pytest.mark.parametrize('logits', [torch.tensor([[float('nan'),0.]]),torch.tensor([[float('inf'),0.]]),torch.tensor([[-float('inf'),0.]]),torch.ones((1,2),dtype=torch.complex64),torch.ones((1,2),dtype=torch.int64),torch.zeros((0,2)),torch.zeros((1,1)),torch.zeros((1,2,1)),torch.zeros(2)])
def test_invalid_logits_do_not_mutate(logits):
    acc=m.ClassificationAccumulator(); before=copy.deepcopy(vars(acc))
    with pytest.raises(ValueError): acc.update(logits,torch.tensor([0]))
    assert vars(acc)==before


def test_changed_class_count_fails_without_mutating():
    acc=m.ClassificationAccumulator();acc.update(torch.zeros((1,2)),torch.tensor([0]))
    before=copy.deepcopy(vars(acc))
    with pytest.raises(ValueError, match='class count'):acc.update(torch.zeros((1,3)),torch.tensor([0]))
    assert vars(acc)==before


def test_empty_summary_is_not_zero():
    with pytest.raises(ValueError,match='no evaluated'):m.ClassificationAccumulator().result()


@pytest.mark.parametrize('kw', [{'n_bins':True},{'n_bins':0},{'n_bins':1.5},{'n_bins':10001},{'max_examples':False},{'max_batch_elements':0}])
def test_bad_configuration(kw):
    with pytest.raises(ValueError):m.ClassificationAccumulator(**kw)


def test_budget_rejection_preserves_aggregate():
    a=m.ClassificationAccumulator(max_examples=1,max_batch_elements=2)
    a.update(torch.zeros((1,2)),torch.tensor([0])); before=copy.deepcopy(vars(a))
    with pytest.raises(ValueError,match='budget'):a.update(torch.zeros((1,2)),torch.tensor([0]))
    assert vars(a)==before


def test_uniform_tie_and_probability_one_bin():
    a=m.ClassificationAccumulator(n_bins=4)
    a.update(torch.tensor([[0.,0.],[1000.,-1000.]]),torch.tensor([0,1]))
    r=a.result()
    assert [b['count'] for b in r['ece_bins']]==[0,0,1,1]
    assert r['accuracy']==0.5
    assert r['negative_log_likelihood']==pytest.approx((math.log(2)+2000)/2)


def test_representable_extreme_mean_without_raw_sum_overflow():
    r=summary(torch.tensor([[0.,-1e308]]*4,dtype=torch.float64),torch.tensor([1]*4))
    assert r['negative_log_likelihood']==pytest.approx(1e308)
    assert r['brier_multiclass']==2


def test_unrepresentable_log_probability_rejected():
    with pytest.raises(ValueError,match='representable'):
        summary(torch.tensor([[1e308,-1e308]],dtype=torch.float64),torch.tensor([0]))


class Classifier(torch.nn.Module):
    """Real tiny PyTorch network implementing LAM's inspected forward interface."""
    def __init__(self):
        super().__init__(); self.embed=torch.nn.Embedding(5,4); self.drop=torch.nn.Dropout(0.8); self.head=torch.nn.Linear(4,2); self.calls=0;self.fail=False
    def forward(self,tokens,numeric_x=None,sample_rollout=None,noise_std=None):
        self.calls+=1
        assert sample_rollout is False and noise_std==0.0
        assert not self.training and not self.drop.training and not torch.is_grad_enabled()
        if self.fail:raise RuntimeError('injected model failure')
        return {'logits':self.head(self.drop(self.embed(tokens).mean(1))), 'confidence':torch.full((len(tokens),1),float('nan'))}


def batch(n=3):
    return {'tokens':torch.zeros((n,2),dtype=torch.long),'numeric_x':torch.ones((n,4)),'labels':torch.zeros(n,dtype=torch.long)}


@pytest.mark.parametrize('error', [False,True])
def test_model_modes_and_parameters_preserved(error):
    model=Classifier();model.train();model.drop.eval();model.fail=error
    modes=[x.training for x in model.modules()];state={k:v.clone() for k,v in model.state_dict().items()};rng=torch.random.get_rng_state().clone()
    if error:
        with pytest.raises(RuntimeError):m.evaluate_classification_v2(model,[batch()])
    else:
        out=m.evaluate_classification_v2(model,[batch()]); assert out['examples']==3
        again=m.evaluate_classification_v2(model,[batch()]);assert out==again
        assert out['confidence_source']=='softmax(logits)'
    assert modes==[x.training for x in model.modules()]
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
    assert torch.equal(rng,torch.random.get_rng_state())


def test_column_label_refusal_precedes_forward():
    model=Classifier(); b=batch(); b['labels']=torch.zeros((3,1),dtype=torch.long)
    with pytest.raises(ValueError,match='broadcasting'):m.evaluate_classification_v2(model,[b])
    assert model.calls==0 and model.training


def test_model_input_budget_precedes_forward():
    model=Classifier()
    with pytest.raises(ValueError,match='budget'):m.evaluate_classification_v2(model,[batch()],max_examples=2)
    assert model.calls==0


def test_empty_loader_restores_modes():
    model=Classifier()
    with pytest.raises(ValueError,match='no evaluated'):m.evaluate_classification_v2(model,[])
    assert model.training


def test_loader_error_does_not_return_partial_summary():
    model=Classifier()
    def broken():
        yield batch()
        raise RuntimeError('loader failure')
    with pytest.raises(RuntimeError,match='loader failure'):m.evaluate_classification_v2(model,broken())
    assert model.calls==1 and model.training


def test_object_batch_and_optional_numeric():
    model=Classifier();b=batch();del b['numeric_x']
    assert m.evaluate_classification_v2(model,[SimpleNamespace(**b)])['examples']==3


@pytest.mark.parametrize('numeric', [torch.tensor([[float('nan')]]*3),torch.ones(2,4),torch.ones(3,0),torch.ones(3,2,2),torch.ones(3,4,dtype=torch.long)])
def test_bad_numeric_data_before_forward(numeric):
    model=Classifier();b=batch();b['numeric_x']=numeric
    with pytest.raises(ValueError,match='numeric_x'):m.evaluate_classification_v2(model,[b])
    assert model.calls==0


def test_no_synthetic_fallback():
    with pytest.raises(ValueError,match='explicit'):m.evaluate_classification_v2(Classifier(),None)


def test_distinct_near_tie_logits_keep_their_class_when_probabilities_round_equal():
    # Binary64 can represent this positive gap between logits, while the
    # subtraction of log(2) in log_softmax loses it. This fixture must expose
    # the rounded probability tie rather than silently testing an ordinary gap.
    gap = 2.0 ** -60
    logits = torch.tensor([[0.0, gap], [gap, 0.0]], dtype=torch.float64)
    labels = torch.tensor([1, 0], dtype=torch.int64)
    probabilities = torch.log_softmax(logits, dim=1).exp()
    assert torch.all(logits[:, 0] != logits[:, 1])
    assert torch.equal(logits.argmax(dim=1), labels)
    assert torch.equal(probabilities[:, 0], probabilities[:, 1])
    assert torch.equal(probabilities.argmax(dim=1), torch.zeros(2, dtype=torch.int64))

    out = summary(logits, labels, n_bins=2)
    assert out['examples'] == 2
    assert out['accuracy'] == 1.0
    selected_probability = probabilities[torch.arange(2), labels].mean().item()
    assert out['mean_top1_probability'] == selected_probability
    assert out['mean_top1_probability'] == pytest.approx(0.5)
    occupied = [bin_result for bin_result in out['ece_bins'] if bin_result['count']]
    assert len(occupied) == 1
    assert occupied[0]['count'] == 2
    assert occupied[0]['accuracy'] == 1.0
    assert occupied[0]['confidence'] == selected_probability
    assert out['ece_top_label'] == pytest.approx(0.5)
