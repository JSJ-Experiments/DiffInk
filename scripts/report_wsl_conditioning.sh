#!/usr/bin/env bash
# CPU-only report waiter. NEVER starts/restarts training or calls Modal.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -f scripts/activate_rocm.sh && -f ../../scripts/activate_rocm.sh ]]; then cd ../..; fi
source scripts/activate_rocm.sh
if [[ $# != 1 ]]; then echo 'usage: scripts/report_wsl_conditioning.sh checkpoints/iam_wsl_conditioning/<study>' >&2; exit 2; fi
mkdir -p logs/native-study-locks
exec 9>"logs/native-study-locks/$(basename "$1").report.lock"
flock -n 9 || { echo 'Another CPU report waiter holds this study lock' >&2; exit 3; }
export OMP_NUM_THREADS=1
export STUDY="$1"
python - <<'PYWAIT'
import json,os,time
from pathlib import Path
from iam_tools.launch_ledger import study_path
from iam_tools.wsl_conditioning import validate
p=study_path('data',os.environ['STUDY'],'checkpoints/iam_wsl_conditioning/')
validate(json.loads((p/'config.json').read_text()))
start=time.monotonic()
while time.monotonic()-start < 10800:
    statuses=[json.loads(f.read_text()) if f.exists() else None for f in [p/'concat-run-status.json',p/'joint-run-status.json']]
    if all(s and s['state']=='complete' for s in statuses):break
    if any(s and s['state'] not in ['started','complete'] for s in statuses):raise SystemExit('non-success native status; no report or retry')
    for s in statuses:
        if s and s['state']=='started':
            try:os.kill(s['parent_pid'],0)
            except ProcessLookupError:raise SystemExit('native coordinator stopped with incomplete status; investigate, never restart')
    time.sleep(60)
else:raise SystemExit('report waiter time limit; no training restart')
PYWAIT
python -m iam_tools.report_wsl_conditioning "$STUDY" --arms concat joint --terminal
