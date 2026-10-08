#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [ -x .venv/bin/python ]; then
  exec .venv/bin/python -m kinect_scanner
elif [ -x ../.venv/bin/python ]; then
  exec ../.venv/bin/python -m kinect_scanner
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
  exec "$VIRTUAL_ENV/bin/python" -m kinect_scanner
else
  exec python3 -m kinect_scanner
fi
