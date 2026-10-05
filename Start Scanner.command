#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [ -x .venv/bin/python ]; then
  exec .venv/bin/python scripts/start_scanner.py
elif [ -x ../.venv/bin/python ]; then
  exec ../.venv/bin/python scripts/start_scanner.py
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
  exec "$VIRTUAL_ENV/bin/python" scripts/start_scanner.py
else
  exec python3 scripts/start_scanner.py
fi
