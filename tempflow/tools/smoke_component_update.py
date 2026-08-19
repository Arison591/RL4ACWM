#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, tempfile
from pathlib import Path
import torch
from tempflow_video.core.policy_objective import component_policy_objective
from tempflow_video.core.reference_kl import FrozenReference
from tempflow_video.core.trainer import optimizer_step
from tempflow_video.rewards.component_advantage import component_advantages
from tempflow_video.runtime.checkpoint import load_checkpoint, save_checkpoint

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--updates",type=int,default=1); parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if not 1 <= args.updates <= 3: raise ValueError("smoke is limited to 1..3 updates")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=torch.nn.Linear(1,2,bias=False).to(device); reference_model=torch.nn.Linear(1,2,bias=False).to(device)
    with torch.no_grad(): model.weight.zero_()
    reference_model.load_state_dict(model.state_dict()); reference=FrozenReference(reference_model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3); rows=[]
    for step in range(args.updates):
        logits=model(torch.ones(1,1,device=device)).flatten()
        adv=component_advantages([.1,.4],[20.0,20.02],action_min_group_std=1e-4,
                                 psnr_min_group_std_db=2e-4)
        loss=component_policy_objective(log_probs=logits,old_log_probs=torch.zeros(2,device=device),
          action_advantages=adv.action.advantages.to(device,torch.float32),psnr_advantages=adv.psnr.advantages.to(device,torch.float32),
          noise_weights=torch.ones(2,device=device),policy_means=logits.reshape(2,1),
          reference_means=reference_model(torch.ones(1,1,device=device)).detach().reshape(2,1),
          transition_stds=torch.ones(2,device=device),clip_range=.2,kl_beta=.01)
        metrics=optimizer_step(loss,parameters=list(model.parameters()),optimizer=optimizer)
        if metrics["psnr_policy_grad_norm"] <= 0 or metrics["action_policy_grad_norm"] <= 0:
            raise RuntimeError("valid smoke group produced a zero component policy gradient")
        rows.append(metrics)
        reference.assert_unchanged()
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint=save_checkpoint(Path(tmp)/"checkpoint",policy=model,optimizer=optimizer,
                                   trainer_state={"optimizer_step":args.updates,"policy_version":args.updates})
        state=load_checkpoint(checkpoint,policy=model,optimizer=optimizer)
    report={"device":str(device),"updates":args.updates,"metrics":rows,
            "reference_unchanged":True,"checkpoint_resume":state["optimizer_step"]==args.updates,
            "scope":"algebraic component-loss smoke; no AWM generation"}
    text=json.dumps(report,indent=2)
    if args.output: args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(text+"\n")
    print(text)
if __name__=="__main__": main()
