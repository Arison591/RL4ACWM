#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.tempflow_video.config import load_tempflow_config
from experiments.tempflow_video.preflight import run_preflight, write_preflight_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate TempFlow video data, assets, schedule, and GPU")
    parser.add_argument("--config", required=True)
    parser.add_argument("--load-model", action="store_true", help="also load GE-Sim and install LoRA")
    parser.add_argument("--output", help="report JSON path (defaults under config output_dir)")
    args = parser.parse_args()
    config = load_tempflow_config(args.config)
    report = run_preflight(config, load_model=args.load_model)
    output = Path(args.output) if args.output else Path(config["output_dir"]) / "preflight.json"
    write_preflight_report(report, output)
    print(json.dumps({"ok": report["ok"], "report": str(output), "model_loaded": report["model_loaded"]}))


if __name__ == "__main__":
    main()
