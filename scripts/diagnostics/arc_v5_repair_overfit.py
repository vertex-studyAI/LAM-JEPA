from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from lam_jepa.benchmarking.arc_challenge import LAMARCClassifier, batchify, load_arc_split
from lam_jepa.benchmarking.arc_protocol import select_protocol_eligible_examples
from lam_jepa.benchmarking.arc_v5_repair import ARC_V5_REPAIR_ID, arc_v5_repair_spec, build_arc_v5_repaired_classifier
from lam_jepa.model import LAMJEPAConfig
from lam_jepa.utils import set_seed

CHECKPOINTS=(0,1,5,10,25,50,100,200,300)
SEEDS=(1,2)
OVERFIT_THRESHOLD=0.95
CONDITIONS=("legacy","repaired_v5","no_quantizer")

def balanced_subset(examples, per_class=8):
    buckets={label:[] for label in range(4)}
    for example in examples:
        if len(example.choices)!=4: continue
        if len(buckets[example.label])<per_class: buckets[example.label].append(example)
        if all(len(bucket)==per_class for bucket in buckets.values()): break
    if not all(len(bucket)==per_class for bucket in buckets.values()): raise RuntimeError("unable to construct balanced four-class subset")
    return [buckets[label][index] for index in range(per_class) for label in range(4)]

def build(condition):
    if condition=="legacy": return LAMARCClassifier(LAMJEPAConfig(),num_choices=4)
    if condition=="repaired_v5": return build_arc_v5_repaired_classifier(LAMJEPAConfig(),num_choices=4)
    if condition=="no_quantizer": return LAMARCClassifier(replace(LAMJEPAConfig(),use_quantizer=False),num_choices=4)
    raise ValueError(condition)

def quantizer_state(model,outputs):
    if not model.backbone.cfg.use_quantizer: return {"enabled":False,"code_support":None}
    q=model.backbone.quantizer; indices=outputs["indices"].detach().cpu().tolist(); norms=q.codebook.detach().float().norm(dim=1)
    return {"enabled":True,"code_support":len(set(indices)),"code_histogram":{str(k):int(v) for k,v in sorted(Counter(indices).items())},"codebook_max_norm":float(norms.max().item()),"ema_count_min":float(q.ema_count.min().item()),"ema_count_max":float(q.ema_count.max().item()),"finite":bool(torch.isfinite(q.codebook).all() and torch.isfinite(q.ema_count).all() and torch.isfinite(q.ema_weight).all())}
@torch.no_grad()
def evaluate(model,tokens,numeric_x,labels):
    model.eval(); logits,outputs=model(tokens,numeric_x,model_steps=1,deterministic=True); predictions=logits.argmax(dim=-1)
    return {"accuracy":float(predictions.eq(labels).float().mean().item()),"cross_entropy":float(F.cross_entropy(logits,labels).item()),"prediction_support":len(set(predictions.detach().cpu().tolist())),"z_feature_std":float(outputs["z"].float().std(dim=0,unbiased=False).mean().item()),"z_q_feature_std":float(outputs["z_q"].float().std(dim=0,unbiased=False).mean().item()),"latent_summary_feature_std":float(outputs["latent_summary"].float().std(dim=0,unbiased=False).mean().item()),"quantizer_state":quantizer_state(model,outputs)}
def train(condition,seed,tokens,numeric_x,labels):
    set_seed(seed); model=build(condition); optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4); history=[{"step":0,**evaluate(model,tokens,numeric_x,labels)}]
    for step in range(1,301):
        model.train(); optimizer.zero_grad(set_to_none=True); logits,_=model(tokens,numeric_x,model_steps=1,deterministic=False); loss=F.cross_entropy(logits,labels); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); model.backbone.update_target()
        if step in CHECKPOINTS: history.append({"step":step,**evaluate(model,tokens,numeric_x,labels)})
    return {"condition":condition,"seed":seed,"history":history}
def summarize(records):
    best=[max(float(point["accuracy"]) for point in record["history"]) for record in records]; final=[float(record["history"][-1]["accuracy"]) for record in records]
    return {"best_accuracy_by_seed":best,"final_accuracy_by_seed":final,"mean_best_accuracy":float(statistics.fmean(best)),"all_seeds_reach_overfit_threshold":all(value>=OVERFIT_THRESHOLD for value in best)}
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--train",type=Path,required=True); parser.add_argument("--out",type=Path,required=True); args=parser.parse_args()
    eligible=select_protocol_eligible_examples(load_arc_split(args.train)).eligible; subset=balanced_subset(eligible); cfg=LAMJEPAConfig(); tokens,numeric_x,labels=batchify(subset,vocab_size=cfg.vocab_size,device="cpu")
    records={condition:[] for condition in CONDITIONS}
    for seed in SEEDS:
        for condition in CONDITIONS: records[condition].append(train(condition,seed,tokens,numeric_x,labels))
    summaries={condition:summarize(records[condition]) for condition in CONDITIONS}; repaired_pass=bool(summaries["repaired_v5"]["all_seeds_reach_overfit_threshold"])
    payload={"artifact_type":"LAM-JEPA ARC v5 quantizer repair trainability gate","scope":"training-only primary repair acceptance; no validation or test access","repair_id":ARC_V5_REPAIR_ID,"repair_spec":arc_v5_repair_spec(),"subset":{"rows":32,"per_class":8,"ids":[e.item_id for e in subset]},"training":{"seeds":list(SEEDS),"steps":300,"learning_rate":3e-4,"overfit_threshold":OVERFIT_THRESHOLD},"records":records,"summaries":summaries,"repair_trainability_gate_pass":repaired_pass,"verdict":"PRIMARY_REPAIR_TRAINABILITY_GATE_PASSED" if repaired_pass else "PRIMARY_REPAIR_TRAINABILITY_GATE_FAILED","claim_boundary":{"validation_accessed":False,"test_accessed":False,"independent_reproduction_complete":False,"validation_authorized":False,"performance_claim_authorized":False,"research_complete":False}}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8"); print(json.dumps({"verdict":payload["verdict"],"summaries":summaries,"repair_spec":payload["repair_spec"]},indent=2))
    if not repaired_pass: raise SystemExit("repaired ARC v5 quantizer failed the frozen two-seed overfit gate")
if __name__=="__main__": main()
