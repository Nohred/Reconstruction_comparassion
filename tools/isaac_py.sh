#!/usr/bin/env bash
# Python de Isaac Sim con la raiz del repo y pxr en el path (sin arrancar SimulationApp).
#   tools/isaac_py.sh -m eval.export_colmap --objeto bunny --fuente real --check
#   tools/isaac_py.sh -m unittest discover -s tests -t .
set -euo pipefail
ISAAC="${ISAACSIM_PATH:-$HOME/isaacsim}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USDLIBS="$(ls -d "$ISAAC"/extscache/omni.usd.libs-* 2>/dev/null | head -1)"
export PYTHONPATH="$ROOT${USDLIBS:+:$USDLIBS}${PYTHONPATH:+:$PYTHONPATH}"
[ -n "$USDLIBS" ] && export LD_LIBRARY_PATH="$USDLIBS/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONDONTWRITEBYTECODE=1
unset CONDA_PREFIX CONDA_DEFAULT_ENV
exec "$ISAAC/python.sh" "$@" 2> >(grep -v -E "^Warning: running in conda env" >&2)
