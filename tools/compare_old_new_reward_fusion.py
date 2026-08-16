#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from calibrate_component_std import load_groups

def z(value,threshold=0.0):
    value=np.asarray(value,dtype=np.float64); std=value.std()
    return np.zeros_like(value) if std<threshold else np.clip((value-value.mean())/(std+1e-6),-1,1)
def corr(a,b): return float(np.corrcoef(a,b)[0,1])

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("run_root",type=Path)
    parser.add_argument("--psnr-threshold",type=float,required=True); parser.add_argument("--action-threshold",type=float,required=True)
    parser.add_argument("--output",type=Path,required=True); args=parser.parse_args(); groups=load_groups(args.run_root)
    old_a=[]; old_p=[]; actions=[]; psnrs=[]; totals=[]; new_a=[]; new_p=[]; conflicts=[]; skipped=[]
    for group in groups:
        a=np.asarray([r["action"] for r in group]); p=np.asarray([r["psnr"] for r in group])
        legacy=1/(1+np.exp(-(p-20.4)/1.8)); total=.5*a+.5*legacy; denom=total.std()+1e-6
        old_a.extend(.5*(a-a.mean())/denom); old_p.extend(.5*(legacy-legacy.mean())/denom)
        actions.extend(a); psnrs.extend(p); totals.extend(total)
        az,pz=z(a,args.action_threshold),z(p,args.psnr_threshold); new_a.extend(.5*az); new_p.extend(.5*pz)
        skipped.append(float(p.std()<args.psnr_threshold))
        conflicts.extend((a[i]-a[j])*(p[i]-p[j])<0 for i in range(len(a)) for j in range(i))
    rms=lambda x:float(np.sqrt(np.mean(np.square(x))))
    report={"groups":len(groups),"branches":len(actions),"old":{"action_contribution_rms":rms(old_a),
      "psnr_contribution_rms":rms(old_p),"psnr_to_action_contribution_ratio":rms(old_p)/rms(old_a),
      "corr_total_action":corr(totals,actions),"corr_total_psnr":corr(totals,psnrs)},
      "new":{"weighted_action_advantage_rms":rms(new_a),"weighted_psnr_advantage_rms":rms(new_p),
      "effective_psnr_group_ratio":1-float(np.mean(skipped)),"psnr_skipped_group_ratio":float(np.mean(skipped)),
      "action_psnr_pairwise_ranking_conflict_rate":float(np.mean(conflicts)),"psnr_ranking_changed_by_normalization":False},
      "note":"Advantage scaling does not equalize parameter gradients; use weighted gradient norms."}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
