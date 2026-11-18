#!/usr/bin/env bash
# Ejecuta un script de tools/inspect con el Python de Isaac Sim (trae numpy, cv2, open3d, PIL)
# agregando pxr (omni.usd.libs) al path, sin arrancar SimulationApp.
#   tools/inspect/run.sh inspect_usd.py Objects/stanford_bunny.usd
# Variable opcional: ISAACSIM_PATH (default ~/isaacsim).
set -euo pipefail
ISAAC="${ISAACSIM_PATH:-$HOME/isaacsim}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USDLIBS="$(ls -d "$ISAAC"/extscache/omni.usd.libs-* 2>/dev/null | head -1)"
[ -n "$USDLIBS" ] || { echo "No encontre omni.usd.libs en $ISAAC/extscache" >&2; exit 1; }
SCRIPT="$1"; shift
# rutas relativas de los argumentos se resuelven desde el directorio donde se llama
export PYTHONPATH="$HERE:$USDLIBS${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$USDLIBS/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset CONDA_PREFIX CONDA_DEFAULT_ENV
exec "$ISAAC/python.sh" "$HERE/$SCRIPT" "$@" 2> >(grep -v -E "^Warning: running in conda env" >&2)
