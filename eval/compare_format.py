"""Compara el formato de una captura sintetica con la real de referencia (resumen corto).

    tools/cv_py.sh -m eval.compare_format Captura_sintetica/bunny_x200

1. Estructura: mismos archivos por (camara, parada) [se ignoran gt/ y ground_truth.json].
2. JSON: mismo arbol de claves y tipos (session.json, rig.json, calib.json, meta.json).
3. PNG: mismo dtype / forma / canales (color.png y depth_XX.png).
4. Poses: meta.json sintetico (detector + rig, como la captura) vs pose GT del render, y
   esquinas detectadas sintetico vs real.
Sale con codigo 1 si 1-3 no coinciden.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from common.board import L
from common.config import load_config, repo_path
from common.geometry import as_T
from common.realdata import Capture, read_json

MAX_LINES = 40


def key_tree(x, path=""):
    """{ruta: tipo} de un JSON; listas se resumen por su primer elemento."""
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(key_tree(v, f"{path}.{k}"))
    elif isinstance(x, list):
        out[path] = "list"
        if x:
            out.update(key_tree(x[0], f"{path}[]"))
    else:
        t = "num" if isinstance(x, (int, float)) and not isinstance(x, bool) else type(x).__name__
        out[path] = t
    return out


def diff_keys(a, b, ignore=()):
    ta, tb = key_tree(a), key_tree(b)
    msgs = []
    for k in sorted(set(ta) | set(tb)):
        if any(k.startswith(i) for i in ignore):
            continue
        if k not in tb:
            msgs.append(f"falta {k}")
        elif k not in ta:
            msgs.append(f"sobra {k}")
        elif ta[k] != tb[k] and "NoneType" not in (ta[k], tb[k]):
            msgs.append(f"tipo {k}: real {ta[k]} vs sint {tb[k]}")
    return msgs


def png_sig(p):
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    return None if im is None else (str(im.dtype), im.shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sintetica")
    ap.add_argument("--config", default="twin")
    args = ap.parse_args()
    common, cfg = load_config("common"), load_config(args.config)
    lay = common["capture_layout"]
    sroot = repo_path(args.sintetica)
    gt = read_json(sroot / cfg["gt_file"])
    real = Capture.open(repo_path(gt["plantilla"]), lay)
    syn = Capture.open(sroot, lay)
    problems, lines = [], []

    # 1-2. archivos raiz y JSON
    for f in (lay["session_file"], lay["rig_file"]):
        problems += [f"{f}: {m}" for m in diff_keys(read_json(real.root / f), read_json(sroot / f))]
    for cam in syn.cameras:
        problems += [f"{cam}/calib.json: {m}" for m in diff_keys(real.calib[cam], syn.calib[cam])]
        for k in syn.stops[cam]:
            sd, rd = syn.stop_dir(cam, k), real.stop_dir(cam, k)
            fs = sorted(p.name for p in sd.iterdir())
            fr = sorted(p.name for p in rd.iterdir()) if rd.is_dir() else []
            if fs != fr:
                extra, miss = set(fs) - set(fr), set(fr) - set(fs)
                lines.append(f"  {cam}/stop_{k:02d} archivos: sobran {sorted(extra)} faltan {sorted(miss)}"
                             + (" (falta en la REAL)" if miss == set() else ""))
                if miss:
                    problems.append(f"{cam}/stop_{k:02d}: faltan {sorted(miss)}")
            ms, mr = read_json(sd / lay["meta_file"]), read_json(rd / lay["meta_file"])
            problems += [f"{cam}/stop_{k:02d}/meta.json: {m}" for m in diff_keys(mr, ms)]
            for name in fs:
                if name.endswith(".png") and name in fr:
                    a, b = png_sig(rd / name), png_sig(sd / name)
                    if a != b:
                        problems.append(f"{cam}/stop_{k:02d}/{name}: real {a} vs sint {b}")

    # 4. poses
    gt_pose = {(v["camara"], v["parada"]): np.array(v["T_color_from_board_mm"]) for v in gt["vistas"]}
    rows = []
    for v in syn.views():
        T_gt = gt_pose.get((v.camera, v.stop))
        rv = real.view(v.camera, v.stop) if v.stop in real.stops.get(v.camera, []) else None
        n_real = (rv.meta.get("directa") or {}).get("corners") if rv else None
        n_syn = (v.meta.get("directa") or {}).get("corners")
        dd = dm = np.nan
        if v.has_pose and T_gt is not None:
            dd, dm = L.pose_diff(v.T_color_from_board, T_gt)
        rows.append((v.camera, v.stop, n_syn, n_real, v.meta.get("pose_fuente"), v.meta.get("pose_calidad"), dd, dm))
    for cam in syn.cameras:
        r = [x for x in rows if x[0] == cam]
        dm = np.array([x[7] for x in r], float)
        ns = [x[2] or 0 for x in r]
        nr = [x[3] or 0 for x in r]
        q = [x[5] for x in r]
        lines.append(f"  {cam:6s} {len(r):2d} vistas | esquinas sint med {np.median(ns):.0f} vs real {np.median(nr):.0f}"
                     f" | pose meta vs GT: med {np.nanmedian(dm):.2f} mm max {np.nanmax(dm) if np.isfinite(dm).any() else np.nan:.2f} mm"
                     f" | calidad buena {q.count('buena')}/{len(q)}")
    print(f"formato {sroot.name} vs {real.root.name}: {'IDENTICO' if not problems else f'{len(problems)} diferencias'}")
    for p in problems[:MAX_LINES // 2]:
        print("  DIF", p)
    if len(problems) > MAX_LINES // 2:
        print(f"  ... {len(problems) - MAX_LINES // 2} mas")
    print("detalle:")
    for ln in lines[:MAX_LINES // 2]:
        print(ln)
    if len(lines) > MAX_LINES // 2:
        print(f"  ... {len(lines) - MAX_LINES // 2} lineas mas")
    raise SystemExit(0 if not problems else 1)


if __name__ == "__main__":
    main()
