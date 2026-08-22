#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MANIFEST="$SCRIPT_DIR/agibot_beta_600g_manifest.tsv"
REPO="agibot_world/agibot_world_beta"
REVISION="master"
WORKERS=15
TARGET="${AGIBOT_DATA_ROOT:-}"
KEEP_PROXY=0

usage() {
  cat <<'EOF'
Download the curated ~600 GB AgiBotWorld-Beta interaction subset.

Usage:
  download_agibot_beta_600g.sh --target PATH [options]

Options:
  --target PATH       Destination directory (or set AGIBOT_DATA_ROOT).
  --workers N         Parallel ModelScope downloads (default: 15).
  --revision NAME     Dataset revision (default: master).
  --keep-proxy        Preserve HTTP(S)/ALL_PROXY instead of using direct access.
  -h, --help          Show this help.
EOF
}

while (($#)); do
  case "$1" in
    --target)
      [[ $# -ge 2 ]] || { echo "--target requires a path" >&2; exit 2; }
      TARGET=$2
      shift 2
      ;;
    --workers)
      [[ $# -ge 2 ]] || { echo "--workers requires a value" >&2; exit 2; }
      WORKERS=$2
      shift 2
      ;;
    --revision)
      [[ $# -ge 2 ]] || { echo "--revision requires a value" >&2; exit 2; }
      REVISION=$2
      shift 2
      ;;
    --keep-proxy)
      KEEP_PROXY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ -n "$TARGET" ]] || { echo "Pass --target PATH or set AGIBOT_DATA_ROOT." >&2; exit 2; }
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "--workers must be a positive integer." >&2; exit 2; }
[[ -f "$MANIFEST" ]] || { echo "Manifest not found: $MANIFEST" >&2; exit 1; }
command -v modelscope >/dev/null 2>&1 || {
  echo "ModelScope CLI is required. Install it with: python -m pip install 'modelscope>=1.39,<2'" >&2
  exit 1
}

mkdir -p "$TARGET"

mapfile -t FILES < <(awk -F '\t' '!/^#/ && NF == 2 {print $2}' "$MANIFEST")
EXPECTED_BYTES=$(awk -F '\t' '!/^#/ && NF == 2 {sum += $1} END {printf "%.0f", sum}' "$MANIFEST")
AVAILABLE_BYTES=$(df -PB1 "$TARGET" | awk 'NR == 2 {print $4}')
EXISTING_BYTES=0

while IFS=$'\t' read -r expected rel; do
  [[ "$expected" =~ ^[0-9]+$ ]] || continue
  for candidate in "$TARGET/$rel" "$TARGET/$rel.incomplete"; do
    if [[ -f "$candidate" ]]; then
      actual=$(stat -c '%s' "$candidate")
      (( actual > expected )) && actual=$expected
      EXISTING_BYTES=$((EXISTING_BYTES + actual))
      break
    fi
  done
done < "$MANIFEST"

REMAINING_BYTES=$((EXPECTED_BYTES - EXISTING_BYTES))
RESERVE_BYTES=5368709120
if (( AVAILABLE_BYTES < REMAINING_BYTES + RESERVE_BYTES )); then
  echo "Insufficient free space: need approximately $((REMAINING_BYTES + RESERVE_BYTES)) bytes including reserve; available $AVAILABLE_BYTES." >&2
  exit 1
fi

echo "Repository: $REPO@$REVISION"
echo "Target: $TARGET"
echo "Selected payload: $EXPECTED_BYTES bytes"
echo "Already present: $EXISTING_BYTES bytes"
echo "Remaining estimate: $REMAINING_BYTES bytes"
echo "Workers: $WORKERS"
echo "Network mode: $([[ $KEEP_PROXY -eq 1 ]] && echo inherited-proxy || echo direct)"

download_once() {
  local args=(
    modelscope download --repo-type dataset "$REPO"
    --revision "$REVISION"
    --include README.md 'task_info/*' "${FILES[@]}"
    --local-dir "$TARGET"
    --max-workers "$WORKERS"
  )
  if (( KEEP_PROXY )); then
    "${args[@]}"
  else
    env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
      -u http_proxy -u https_proxy -u all_proxy "${args[@]}"
  fi
}

until download_once; do
  echo "Download attempt failed; resuming in 20 seconds." >&2
  sleep 20
done

python - "$TARGET" "$MANIFEST" <<'PY'
from datetime import datetime
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
manifest = Path(sys.argv[2]).resolve()
expected = {}
for raw in manifest.read_text().splitlines():
    if not raw or raw.startswith("#"):
        continue
    size, rel = raw.split("\t", 1)
    expected[rel] = int(size)

rows = []
ok = True
for rel, size in expected.items():
    path = root / rel
    actual = path.stat().st_size if path.is_file() else -1
    match = actual == size
    ok &= match
    rows.append(f'{"OK" if match else "FAIL"}\t{actual}\t{size}\t{rel}')

now = datetime.now().astimezone().isoformat()
report = [
    f"verified_at={now}",
    f"expected_payload_bytes={sum(expected.values())}",
    f'files_verified={sum(line.startswith("OK") for line in rows)}/{len(rows)}',
    *rows,
]
(root / "VERIFICATION.txt").write_text("\n".join(report) + "\n")
if not ok:
    raise SystemExit("Downloaded files did not match expected sizes")
(root / "DOWNLOAD_COMPLETE").write_text(now + "\n")
print(f"Download and size verification complete: {root}")
PY
