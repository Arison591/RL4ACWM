#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import torch, yaml

def sha_file(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def sha_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--upstream-root",type=Path,required=True)
    p.add_argument("--effective-config",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seed",type=int,default=123456); args=p.parse_args()
    sys.path.insert(0,str(args.upstream_root.resolve()))
    from experiments.awm_coca.condition_dataset import PrepConditionDataset, build_manifest
    from experiments.awm_coca.gesim_runtime import DEFAULT_PROMPT, PersistentGeSimRuntime
    config=yaml.safe_load(args.effective_config.read_text())
    manifest,invalid=build_manifest(config["dataset"]["prep_root"],validation_mode="strict")
    if invalid: raise RuntimeError(invalid)
    raw=PrepConditionDataset(manifest)[0]
    runtime=PersistentGeSimRuntime(config,device="cuda")
    prepared=runtime.prepare_condition(raw)
    _,values=runtime.rollout_group(prepared,seeds=[args.seed],output_dir=args.output,
      prompt=DEFAULT_PROMPT,expected_group_size=1,rollout_batch_size=1)
    artifact=values[0]
    report={"condition_id":raw.condition_id,"seed":args.seed,
      "trajectory_hashes":[sha_tensor(row["latents"]) for row in artifact.trajectory],
      "video_hashes":{camera:sha_file(artifact.seed_dir/f"{camera}_color.mp4") for camera in runtime.args.data["train"]["valid_cam"]}}
    (args.output/"generation_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))
if __name__=="__main__": main()
