#!/usr/bin/env bash
set -euo pipefail
ROBOTWIN_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${TRAIN_PYTHON:-python}" "$ROBOTWIN_CODE/tools/train_robotwin.py" "$@"
