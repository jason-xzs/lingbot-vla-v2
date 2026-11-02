#!/usr/bin/env bash
set -euo pipefail
ROBOTWIN_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${EVAL_LAUNCH_PYTHON:-python}" "$ROBOTWIN_CODE/tools/eval_robotwin.py" "$@"
