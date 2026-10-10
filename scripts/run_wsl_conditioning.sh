#!/usr/bin/env bash
# Native WSL only. Serial arms on ONE AMD GPU, no Modal dependencies/launches.
set -euo pipefail
cd "$(dirname "$0")/.."
# The tracked copy also lives in the nested fork, but execution uses the full
# workstation workspace containing the ROCm activation script and data link.
if [[ ! -f scripts/activate_rocm.sh && -f ../../scripts/activate_rocm.sh ]]; then cd ../..; fi
source scripts/activate_rocm.sh
if [[ $# != 1 ]]; then echo 'usage: scripts/run_wsl_conditioning.sh checkpoints/iam_wsl_conditioning/<prepared-study>' >&2; exit 2; fi
study="$1"
# A persistent flock prevents duplicate coordinators. Already-started arms are
# never restarted automatically: investigate the saved status after a crash.
python - "$study" <<'PYGUARD'
import sys,json
from pathlib import Path
from iam_tools.launch_ledger import study_path
from iam_tools.wsl_conditioning import validate
p=study_path('data',sys.argv[1],'checkpoints/iam_wsl_conditioning/')
if not p.is_dir():raise SystemExit('Existing prepared native study required')
validate(json.loads((p/'config.json').read_text()))
PYGUARD
mkdir -p logs/native-study-locks
exec 9>"logs/native-study-locks/$(basename "$study").lock"
flock -n 9 || { echo 'Another native coordinator holds this study lock' >&2; exit 3; }
for arm in concat joint; do
  status="data/$study/$arm-run-status.json"
  if [[ -e "$status" ]]; then
    python - "$status" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);s=json.loads(p.read_text())
if s['state']!='complete':raise SystemExit('Previous arm start is not confirmed complete; investigate, do NOT auto-restart: '+str(p))
PY
    continue
  fi
  python - "$status" "$arm" <<'PY'
import json,sys,datetime,os
from pathlib import Path
p=Path(sys.argv[1])
with p.open('x') as f:json.dump(dict(state='started',arm=sys.argv[2],utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_pid=os.getppid()),f)
PY
  python -m iam_tools.wsl_conditioning --train --study "$study" --arm "$arm"
  python - "$status" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);s=json.loads(p.read_text());s['state']='complete';q=p.with_suffix('.pending');q.write_text(json.dumps(s,indent=2)+'\n');q.replace(p)
PY
done
