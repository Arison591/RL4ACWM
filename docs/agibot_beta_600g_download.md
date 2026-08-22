# AgiBotWorld-Beta interaction subset downloader

This downloader fetches a curated **599,237,980,160-byte (~599.24 GB)** payload from the public ModelScope dataset `agibot_world/agibot_world_beta`. It is intended for GE-Sim/RL4ACWM experiments that need object-centric interaction rather than mostly empty robot motion.

## Selected tasks

- Task 327 and 354: supermarket item pickup and placement (`Pick`, `Place`)
- Task 359: warehouse sorting (`Pick`, `Place`, `HandOver`)
- Task 410: restaurant water pouring (`Pick`, `Pour`)

The manifest includes the corresponding raw observation archives, camera-parameter archives, and proprioception/action archive. AgiBotWorld observations contain head, left-hand, and right-hand camera streams; the parameter archives contain matching camera calibration data.

## Requirements

- Linux with Bash 4+
- Python 3
- ModelScope CLI 1.39.x
- At least 605 GB free for a fresh download (more space is recommended for extraction)

```bash
python -m pip install 'modelscope>=1.39,<2'
```

## Foreground download

```bash
bash scripts/download_agibot_beta_600g.sh \
  --target /data/AgiBotWorld-Beta-interaction-600G \
  --workers 15
```

The default mode removes `HTTP_PROXY`, `HTTPS_PROXY`, and `ALL_PROXY` only for the ModelScope child process, so ModelScope is contacted directly. Pass `--keep-proxy` when the new machine requires a proxy.

## Background download

```bash
target=/data/AgiBotWorld-Beta-interaction-600G
mkdir -p "$target"
setsid -f bash scripts/download_agibot_beta_600g.sh \
  --target "$target" --workers 15 \
  >"$target/download.log" 2>&1 < /dev/null
```

The ModelScope client resumes `.incomplete` files. The wrapper retries failed attempts every 20 seconds.

## Completion and verification

Successful completion creates two files in the target directory:

- `DOWNLOAD_COMPLETE`: completion timestamp
- `VERIFICATION.txt`: expected and actual byte sizes for all 15 payload archives

Check status with:

```bash
test -f /data/AgiBotWorld-Beta-interaction-600G/DOWNLOAD_COMPLETE \
  && echo complete || echo in-progress
```

ModelScope performs its own hash validation during download; the wrapper additionally validates every final archive against the repository sizes recorded in `scripts/agibot_beta_600g_manifest.tsv`.

## Data license

The script and manifest do not redistribute dataset content. Downloaded AgiBotWorld-Beta data remains subject to the license and terms published by the upstream dataset provider.
