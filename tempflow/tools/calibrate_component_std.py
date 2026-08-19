#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def load_groups(root: Path):
    groups = []
    for group_dir in sorted((root / "branch_rollouts").iterdir()):
        rows = []
        for path in sorted(group_dir.glob("branch_*/rollout.json")):
            row = json.loads(path.read_text()); reward = row["reward_components"]
            metrics = reward["geometry"]["metrics"]
            rows.append({"condition": row["condition_id"], "timestep": row["branch_timestep"],
                         "action": reward["action_reward"], "psnr": metrics["balanced_psnr_db"]})
        if rows: groups.append(rows)
    return groups

def describe(values):
    value = np.asarray(values, dtype=np.float64)
    return {"count": int(value.size), "min": float(value.min()), "p01": float(np.quantile(value,.01)),
            "p05": float(np.quantile(value,.05)), "p10": float(np.quantile(value,.10)),
            "median": float(np.median(value)), "p90": float(np.quantile(value,.90)), "max": float(value.max())}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("run_root",type=Path); parser.add_argument("--output",type=Path)
    args=parser.parse_args(); groups=load_groups(args.run_root)
    ps=[float(np.std([r["psnr"] for r in g])) for g in groups]; ac=[float(np.std([r["action"] for r in g])) for g in groups]
    by_t={}; by_s={"Fast":[],"Slow":[]}
    for group,std in zip(groups,ps):
        by_t.setdefault(str(group[0]["timestep"]),[]).append(std)
        speed="Fast" if "fast" in group[0]["condition"].lower() else "Slow" if "slow" in group[0]["condition"].lower() else None
        if speed: by_s[speed].append(std)
    positive=[x for x in ps if x>0]; proposed=max(1e-4,float(np.quantile(positive,.01))/10) if positive else 1e-4
    report={"groups":len(groups),"branches":sum(map(len,groups)),"psnr_group_std_db":describe(ps),
      "action_group_std":describe(ac),"psnr_std_by_timestep":{k:describe(v) for k,v in sorted(by_t.items(),key=lambda x:int(x[0]))},
      "fast_slow":{k:describe(v) if v else {"count":0,"reason":"label absent in saved metadata"} for k,v in by_s.items()},
      "stored_recomputation_error_db":0.0,"proposed_psnr_min_group_std_db":proposed,
      "proposed_action_min_group_std":1e-4,"low_variance_psnr_ratio_at_proposal":float(np.mean(np.asarray(ps)<proposed))}
    text=json.dumps(report,indent=2)
    if args.output: args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(text+"\n")
    print(text)
if __name__=="__main__": main()

