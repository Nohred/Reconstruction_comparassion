"""Resumen compacto de imagenes PNG (color o profundidad). Solo lectura.

    tools/inspect/run.sh inspect_png.py Captura_real/bunnny/alta/stop_00/*.png \
        --calib Captura_real/bunnny/alta/calib.json

Una imagen uint16 de 1 canal se trata como profundidad: se reportan valores en metros
usando depth_scale (de --depth-scale o de depth_scale_m en --calib) y % de pixeles invalidos (0).
Con mas de 3 archivos se imprime una linea por archivo.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from _out import Out, add_max_lines

PCTS = (1, 50, 99)


def load_scale(args):
    if args.depth_scale is not None:
        return args.depth_scale
    if args.calib:
        with open(args.calib, encoding="utf-8") as f:
            return float(json.load(f)["depth_scale_m"])
    return None


def describe(path, scale, out, one_line):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        out(f"{path.name}: no se pudo leer")
        return
    h, w = img.shape[:2]
    ch = 1 if img.ndim == 2 else img.shape[2]
    is_depth = ch == 1 and img.dtype == np.uint16
    if is_depth:
        valid = img[img > 0]
        inval = 100.0 * (1 - valid.size / img.size)
        p = np.percentile(valid, PCTS) if valid.size else [0, 0, 0]
        raw = f"raw min {valid.min() if valid.size else 0} max {valid.max() if valid.size else 0}"
        met = ""
        if scale is not None and valid.size:
            m = np.asarray(p) * scale
            met = f" | m p{PCTS[0]}/p{PCTS[1]}/p{PCTS[2]} = {m[0]:.3f}/{m[1]:.3f}/{m[2]:.3f}"
        if one_line:
            out(f"{path.name:14s} {w}x{h} u16 depth inval {inval:5.1f}% {raw}{met}")
        else:
            out(f"== {path.name}: {w}x{h}, 1 canal, uint16 -> PROFUNDIDAD")
            out(f"   {raw}, invalidos (0) {inval:.1f}%")
            out(f"   raw p{PCTS[0]}/p{PCTS[1]}/p{PCTS[2]} = {p[0]:.0f}/{p[1]:.0f}/{p[2]:.0f}{met}")
            if scale is None:
                out("   (sin depth_scale: usa --calib o --depth-scale para ver metros)")
        return
    flat = img.reshape(-1, ch)
    mins, maxs = flat.min(0), flat.max(0)
    means = flat.mean(0)
    if one_line:
        out(f"{path.name:14s} {w}x{h} {ch}ch {img.dtype} min {mins.tolist()} max {maxs.tolist()} "
            f"media {np.round(means, 1).tolist()}")
    else:
        order = "BGR(A) segun OpenCV" if ch >= 3 else "gris"
        out(f"== {path.name}: {w}x{h}, {ch} canal(es), {img.dtype} ({order})")
        out(f"   min {mins.tolist()} max {maxs.tolist()} media {np.round(means, 1).tolist()}")
        p = np.percentile(flat.mean(1), PCTS)
        out(f"   luminancia p{PCTS[0]}/p{PCTS[1]}/p{PCTS[2]} = {p[0]:.0f}/{p[1]:.0f}/{p[2]:.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--calib", help="calib.json con depth_scale_m")
    ap.add_argument("--depth-scale", type=float, help="metros por unidad de profundidad")
    add_max_lines(ap)
    args = ap.parse_args()
    out = Out(args.max_lines)
    scale = load_scale(args)
    if scale is not None:
        out(f"depth_scale = {scale:.8g} m/unidad")
    files = sorted(Path(f) for f in args.files)
    for f in files:
        describe(f, scale, out, len(files) > 3)
    out.close()


if __name__ == "__main__":
    main()
