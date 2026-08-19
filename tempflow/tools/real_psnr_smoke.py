#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
import cv2, numpy as np, torch, yaml

from tempflow_video.adapters.gesim_policy_adapter import GESimPolicyAdapter
from tempflow_video.core.policy_objective import ComponentPolicyLoss, component_policy_objective
from tempflow_video.core.transitions import (deterministic_edm_step,
    edm_sde_transition_with_logprob, edm_transition_mean, noise_aware_weights)
from tempflow_video.rewards.component_advantage import component_advantages
from tempflow_video.rewards.psnr_reward import compute_psnr_reward
from tempflow_video.runtime.checkpoint import save_adapter_checkpoint, load_adapter_checkpoint
from tempflow_video.runtime.integrity import audit_upstream
from tempflow_video.runtime.upstream_loader import import_upstream, upstream_root

def expand(value):
    if isinstance(value,str): return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value,dict): return {k:expand(v) for k,v in value.items()}
    if isinstance(value,list): return [expand(v) for v in value]
    return value
def merge(a,b):
    out=dict(a)
    for k,v in b.items(): out[k]=merge(out[k],v) if isinstance(v,dict) and isinstance(out.get(k),dict) else v
    return out
def sha_tensor(x): return hashlib.sha256(x.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()
def read_video(path):
    cap=cv2.VideoCapture(str(path)); frames=[]
    while True:
        ok,frame=cap.read()
        if not ok: break
        frames.append(frame)
    cap.release()
    if not frames: raise RuntimeError(f"no frames: {path}")
    return np.stack(frames)
def align_gt(gt,pred):
    if len(gt)!=len(pred):
        n=min(len(gt),len(pred)); gt,pred=gt[:n],pred[:n]
    if gt.shape[1:3]!=pred.shape[1:3]:
        gt=np.stack([cv2.resize(frame,(pred.shape[2],pred.shape[1]),interpolation=cv2.INTER_AREA) for frame in gt])
    return gt,pred
def aggregate(outputs):
    fields=ComponentPolicyLoss.__dataclass_fields__
    return ComponentPolicyLoss(**{name:sum(getattr(x,name) for x in outputs)/len(outputs) for name in fields})
def grad_norm(grads): return float(torch.sqrt(sum(g.detach().float().square().sum() for g in grads if g is not None)))
def assert_no_legacy_import():
    bad=[]
    for name,module in sys.modules.items():
        path=getattr(module,"__file__",None)
        if path and "legacy_source" in str(path): bad.append((name,str(path)))
    if bad: raise RuntimeError(f"legacy_source import detected: {bad[:3]}")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--branch-timestep",type=int,default=2)
    args=p.parse_args(); audit_upstream(); root=upstream_root()
    base=yaml.safe_load((root/"configs/awm_coca_train.yaml").read_text()); local=expand(yaml.safe_load(args.config.read_text()))
    config=merge(base,local); data=Path(os.environ["TEMPFLOW_DATA_ROOT"]); assets=Path(os.environ["AWM_ASSET_ROOT"])
    config["dataset"].update({"prep_root":str(data/"prep"),"limit":1,"shuffle":False,
                              "ids_file":str(root/"configs/awm_coca_overfit16_ids.txt")})
    config["model"].update({"gesim_config":str(root/"configs/cosmos_model/acwm_cosmos.yaml"),
                            "checkpoint_root":str(assets/"checkpoints"),"dtype":"bf16"})
    config.setdefault("optimizer",{}).update({
      "learning_rate":float(config.get("optimizer",{}).get("learning_rate",1e-6)),
      "betas":config.get("optimizer",{}).get("betas",[.9,.999]),
      "epsilon":float(config.get("optimizer",{}).get("epsilon",1e-8)),
      "weight_decay":float(config.get("optimizer",{}).get("weight_decay",0.0)),
      "max_grad_norm":float(config.get("optimizer",{}).get("max_grad_norm",1.0)),
      "clip_range":float(config.get("optimizer",{}).get("clip_range",1e-4)),
      "reference_kl_beta":float(config.get("optimizer",{}).get("reference_kl_beta",.01))})
    config["rollout"].update({"history_frames":4,"future_frames":25,"total_frames":29,"chunks":1,
                              "group_size":2,"reverse_denoise_steps":15,"rollout_batch_size":1})
    cd=import_upstream("experiments.awm_coca.condition_dataset"); gr=import_upstream("experiments.awm_coca.gesim_runtime")
    save_video=import_upstream("utils").save_video; assert_no_legacy_import()
    manifest,invalid=cd.build_manifest(config["dataset"]["prep_root"],validation_mode="strict")
    if invalid: raise RuntimeError(invalid)
    raw=cd.PrepConditionDataset(manifest)[0]; runtime=gr.PersistentGeSimRuntime(config,device="cuda")
    policy=GESimPolicyAdapter(runtime); prepared=runtime.prepare_condition(raw)
    _,artifacts=runtime.rollout_group(prepared,seeds=[123456],output_dir=args.output/"base",
      prompt=gr.DEFAULT_PROMPT,expected_group_size=1,rollout_batch_size=1)
    base_artifact=artifacts[0]; times=[float(s/(s+1)) for s in torch.as_tensor(runtime.scheduler.sigmas,dtype=torch.float64)]
    step=args.branch_timestep; current=base_artifact.trajectory[step]["latents"].to(runtime.device); condition=base_artifact.condition_template
    weights=noise_aware_weights(times,eta=.7,enabled=True,normalization="schedule_mean")
    branches=[]; runtime.transformer.eval()
    with torch.inference_mode():
      for bid,seed in enumerate((910000,910001)):
        velocity=policy.predict_velocity_or_noise(current,times[step],condition)
        tr=edm_sde_transition_with_logprob(current,velocity,flow_time=times[step],next_flow_time=times[step+1],eta=.7,
          generator=torch.Generator(device=runtime.device).manual_seed(seed))
        latent=tr.next_sample
        for i in range(step+1,len(times)-1):
          if times[i+1]==times[i]: continue
          velocity=policy.predict_velocity_or_noise(latent,times[i],condition)
          latent=deterministic_edm_step(latent,velocity,flow_time=times[i],next_flow_time=times[i+1])
        future=policy.decode_video(latent).detach().cpu(); full=torch.cat((prepared.observation,future),dim=2).clamp(-1,1)
        out=args.output/f"branch_{bid:03d}"; out.mkdir(parents=True)
        for vi,camera in enumerate(runtime.args.data["train"]["valid_cam"]): save_video(full[vi],str(out/f"{camera}_color.mp4"),fps=16)
        branches.append({"current":current.detach().cpu(),"next":tr.next_sample.detach().cpu(),"old":float(tr.log_prob.mean()),
                         "condition":condition,"dir":out,"noise":sha_tensor(tr.exploration_noise)})
    gt={}; psnr=[]
    for branch in branches:
      pred={}
      for camera in runtime.args.data["train"]["valid_cam"]:
        gt[camera]=read_video(data/"selected_samples/samples"/raw.condition_id/f"{camera}_29_frames.mp4")
        pred[camera]=read_video(branch["dir"]/f"{camera}_color.mp4")
        gt[camera],pred[camera]=align_gt(gt[camera],pred[camera])
      psnr.append(compute_psnr_reward(gt,pred,history_frames=int(config["rollout"]["history_frames"])))
    adv=component_advantages([0,0],[x.psnr_aggregate_future_db for x in psnr],action_min_group_std=1e-4,
      psnr_min_group_std_db=2e-4,formal_training=True)
    params=policy.trainable_parameters(); optimizer=torch.optim.AdamW(params,lr=float(config["optimizer"]["learning_rate"]),
      betas=tuple(config["optimizer"]["betas"]),eps=float(config["optimizer"]["epsilon"]),
      weight_decay=float(config["optimizer"]["weight_decay"])); before={id(p):p.detach().clone() for p in params}
    base_versions={name:int(p._version) for name,p in runtime.transformer.named_parameters() if "lora_" not in name}
    runtime.transformer.train(); outputs=[]
    for idx,branch in enumerate(branches):
      cur=branch["current"].to(runtime.device); nxt=branch["next"].to(runtime.device)
      velocity=policy.predict_velocity_or_noise(cur,times[step],branch["condition"])
      tr=edm_sde_transition_with_logprob(cur,velocity,flow_time=times[step],next_flow_time=times[step+1],eta=.7,next_sample=nxt)
      with torch.no_grad():
        rv=policy.predict_velocity_or_noise(cur,times[step],branch["condition"],reference=True)
        rm,_,_=edm_transition_mean(cur,rv,flow_time=times[step],next_flow_time=times[step+1],eta=.7)
      outputs.append(component_policy_objective(log_probs=tr.log_prob.mean().reshape(1),
        old_log_probs=torch.tensor([branch["old"]],device=cur.device),action_advantages=torch.zeros(1,device=cur.device),
        psnr_advantages=adv.psnr.advantages[idx:idx+1].to(cur.device,torch.float32),
        noise_weights=torch.tensor([float(weights[step])],device=cur.device),policy_means=tr.mean.unsqueeze(0),
        reference_means=rm.unsqueeze(0),transition_stds=tr.std.reshape(1),clip_range=float(config["optimizer"]["clip_range"]),
        lambda_action=0.,lambda_psnr=1.,kl_beta=float(config["optimizer"]["reference_kl_beta"])))
    loss=aggregate(outputs); psnr_grads=torch.autograd.grad(loss.psnr_policy_loss,params,retain_graph=True,allow_unused=True)
    psnr_grad=grad_norm(psnr_grads); optimizer.zero_grad(set_to_none=True); loss.total_loss.backward()
    total_before=grad_norm([p.grad for p in params]); torch.nn.utils.clip_grad_norm_(params,float(config["optimizer"]["max_grad_norm"])); optimizer.step()
    changed=sum(not torch.equal(before[id(p)],p.detach()) for p in params)
    reference_unchanged=all(int(p._version)==base_versions[name] for name,p in runtime.transformer.named_parameters() if "lora_" not in name)
    ckpt=save_adapter_checkpoint(args.output/"checkpoint_1",policy=runtime.transformer,optimizer=optimizer,
      trainer_state={"optimizer_step":1,"policy_version":1})
    checkpoint_snapshot={id(p):p.detach().cpu().clone() for p in params}
    with torch.no_grad(): params[0].add_(1)
    state=load_adapter_checkpoint(ckpt,policy=runtime.transformer,optimizer=optimizer)
    restored=all(torch.equal(p.detach().cpu(),checkpoint_snapshot[id(p)]) for p in params) and state["optimizer_step"]==1
    assert_no_legacy_import()
    report={"ok":psnr_grad>0 and changed>0 and reference_unchanged and restored,"condition_id":raw.condition_id,
      "branch_timestep":step,"branches":2,"prefix_hash":sha_tensor(current),"branch_noise_hashes":[b["noise"] for b in branches],
      "psnr_future_db":[x.psnr_aggregate_future_db for x in psnr],"psnr_full_db":[x.psnr_aggregate_full_db for x in psnr],
      "psnr_advantages":adv.psnr.advantages.tolist(),"psnr_policy_loss":float(loss.psnr_policy_loss.detach()),
      "raw_kl_loss":float(loss.raw_kl_loss.detach()),"psnr_policy_grad_norm":psnr_grad,
      "total_grad_norm_before_clip":total_before,"changed_lora_tensors":changed,"trainable_lora_tensors":len(params),
      "reference_unchanged":reference_unchanged,"checkpoint_restored":restored,"legacy_source_imports":0}
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))
    if not report["ok"]: raise SystemExit(1)
if __name__=="__main__": main()
