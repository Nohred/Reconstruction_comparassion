#!/usr/bin/env bash
# Python del entorno de deteccion/validacion: OpenCV IGUAL al de la captura real (session.json).
# El Python de Isaac (OpenCV 4.14) desplaza las esquinas ChArUco ~0.4 px respecto a 4.10.
#   tools/cv_py.sh -m eval.export_colmap --objeto bunny --fuente real --check
# Variable opcional: TWIN_CV_ENV (default twin_cv410). Crear con:
#   conda create -n twin_cv410 python=3.11 && pip install opencv-contrib-python==4.10.0.84 "numpy<2.3" pyyaml
set -euo pipefail
ENV="${TWIN_CV_ENV:-twin_cv410}"
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
PY="$CONDA_BASE/envs/$ENV/bin/python"
[ -x "$PY" ] || { echo "No existe el entorno conda $ENV ($PY)" >&2; exit 1; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
exec "$PY" "$@"
