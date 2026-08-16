#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import torch, yaml
from real_psnr_smoke import expand, merge, assert_no_legacy_import
from tempflow_video.adapters.gesim_policy_adapter import GESimPolicyAdapter
from tempflow_video.runtime.checkpoint import load_adapter_checkpoint
from tempflow_video.runtime.integrity import audit_upstream
from tempflow_video.runtime.upstream_loader import import_upstream, upstream_root

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--checkpoint",type=Path,required=True)
    args=p.parse_args(); audit_upstream(); root=upstream_root(); assets=Path(os.environ["AWM_ASSET_ROOT"])
    config=merge(yaml.safe_load((root/"configs/awm_coca_train.yaml").read_text()),expand(yaml.safe_load(args.config.read_text())))
    config["model"].update({"gesim_config":str(root/"configs/cosmos_model/acwm_cosmos.yaml"),
                            "checkpoint_root":str(assets/"checkpoints"),"dtype":"bf16"})
    config["rollout"].update({"history_frames":4,"future_frames":25,"total_frames":29,"chunks":1,
                              "group_size":2,"reverse_denoise_steps":15,"rollout_batch_size":1})
    runtime=import_upstream("experiments.awm_coca.gesim_runtime").PersistentGeSimRuntime(config,device="cuda")
    params=GESimPolicyAdapter(runtime).trainable_parameters(); optimizer=torch.optim.AdamW(params,lr=1e-6)
    state=load_adapter_checkpoint(args.checkpoint,policy=runtime.transformer,optimizer=optimizer)
    expected={id(p):p.detach().cpu().clone() for p in params}
    with torch.no_grad():
        for p in params: p.add_(1)
    load_adapter_checkpoint(args.checkpoint,policy=runtime.transformer,optimizer=optimizer)
    restored=all(torch.equal(p.detach().cpu(),expected[id(p)]) for p in params)
    assert_no_legacy_import(); report={"checkpoint_restored":restored,"tensors_checked":len(params),
      "optimizer_step":state["optimizer_step"],"legacy_source_imports":0}
    print(json.dumps(report));
    if not restored: raise SystemExit(1)
if __name__=="__main__": main()
