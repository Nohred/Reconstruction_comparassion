#!/usr/bin/env bash
# Python (y ns-*) del entorno pixi de nerfstudio, con la raiz del repo en el path.
#   tools/ns_py.sh -m recon.profundidad Reconstrucciones/real/bunnny
#   tools/ns_py.sh --ns ns-train nerfacto ...          (ejecuta un comando del entorno)
# NERFSTUDIO_DIR sobrescribe configs/recon.yaml entornos.nerfstudio_dir.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS="${NERFSTUDIO_DIR:-$(sed -n 's/^ *nerfstudio_dir: *//p' "$ROOT/configs/recon.yaml" | head -1)}"
[ -f "$NS/pixi.toml" ] || { echo "No encuentro nerfstudio con pixi en '$NS' (NERFSTUDIO_DIR)" >&2; exit 1; }
command -v pixi >/dev/null || export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
unset CONDA_PREFIX CONDA_DEFAULT_ENV
# gsplat compila sus kernels CUDA la primera vez (cache en ~/.cache/torch_extensions): nvcc de
# CUDA 11.8 solo acepta gcc <= 11. GCC_BIN / GXX_BIN lo sobrescriben.
export CC="${GCC_BIN:-/usr/bin/gcc-11}" CXX="${GXX_BIN:-/usr/bin/g++-11}"
export CUDAHOSTCXX="$CXX" MAX_JOBS="${MAX_JOBS:-8}"
if [ "${1:-}" = "--ns" ]; then shift; exec pixi run --manifest-path "$NS/pixi.toml" "$@"; fi
exec pixi run --manifest-path "$NS/pixi.toml" python "$@"
