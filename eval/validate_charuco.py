"""Valida el tablero sintetico: exactitud del render y paridad con el detector real.

    tools/cv_py.sh -m eval.validate_charuco Captura_sintetica/bunny_x200

Para cada vista de ground_truth.json (render limpio gt/<camara>/stop_XX/color.png):
  1. RENDER (criterio): cornerSubPix sobre color.png partiendo de las 54 esquinas proyectadas
     con la pose/K/dist del GT. Sesgo y mediana del error = que tan exacto es el render
     respecto a la proyeccion OpenCV (detecta errores de convencion: medio pixel, UV, signos).
  2. DETECTOR (paridad): BoardDetector.pose de l515_common (OpenCV de la captura) vs el GT:
     esquinas (sintetico vs real), RMS, error de pose y sesgo, que debe parecerse al sesgo propio
     del detector medido en datos reales (validacion.detector_bias_px).
"""
import argparse
import cv2
import numpy as np

from common.board import L
from common.config import load_config, repo_path
from common.intrinsics import K_dist_from_calib
from common.realdata import Capture, read_json

EDGE_PX = 10
SUBPIX_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)


def render_accuracy(gray, T, K, dist, corners3d, win, occluder=None):
    P, _ = cv2.projectPoints(corners3d, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, dist)
    P = P.reshape(-1, 2)
    h, w = gray.shape
    m = (P[:, 0] > EDGE_PX) & (P[:, 0] < w - EDGE_PX) & (P[:, 1] > EDGE_PX) & (P[:, 1] < h - EDGE_PX)
    if occluder is not None:            # esquinas tapadas por el objeto (con margen de la ventana)
        occ = cv2.dilate(occluder.astype(np.uint8), np.ones((2 * win + 5,) * 2, np.uint8)) > 0
        ij = np.round(P).astype(int).clip(0, [w - 1, h - 1])
        m &= ~occ[ij[:, 1], ij[:, 0]]
    if not m.any():
        return None
    r = cv2.cornerSubPix(gray, P[m].astype(np.float32).reshape(-1, 1, 2).copy(), (win, win), (-1, -1),
                         SUBPIX_CRIT).reshape(-1, 2)
    e = r - P[m]
    # sesgo = mediana por eje: un error de convencion (medio pixel) mueve TODAS las esquinas;
    # la mediana ignora las pocas afectadas por la sombra o el borde del objeto
    return np.median(e, 0), float(np.median(np.linalg.norm(e, axis=1))), int(m.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sintetica")
    ap.add_argument("--config", default="twin")
    args = ap.parse_args()
    common, cfg = load_config("common"), load_config(args.config)
    lay = common["capture_layout"]
    lim = cfg["validacion"]
    root = repo_path(args.sintetica)
    gtall = read_json(root / cfg["gt_file"])
    real = Capture.open(repo_path(gtall["plantilla"]), lay)
    det = L.BoardDetector()
    rows, malas = [], 0
    for gt in gtall["vistas"]:
        cam, k = gt["camara"], gt["parada"]
        K, dist = K_dist_from_calib(real.calib[cam]["color_intrinsics"])
        if not gtall["distorsion_color"]:
            dist = np.zeros_like(dist)
        T = np.array(gt["T_color_from_board_mm"])
        vdir = root / cfg["gt_dir"] / cam / lay["stop_dir"].format(k=k)
        gray = cv2.imread(str(vdir / "color.png"), cv2.IMREAD_GRAYSCALE)
        occ = cv2.imread(str(vdir / "mask.png"), cv2.IMREAD_GRAYSCALE)
        tag = f"{cam}/stop_{k:02d}"
        ra = render_accuracy(gray, T, K, dist, det.corners3d, lim["subpix_win"],
                             None if occ is None else occ > 0)
        if ra is None:
            print(f"{tag}: tablero fuera de la imagen o tapado")
            malas += 1
            continue
        rbias, rmed, rn = ra
        good = bool(np.all(np.abs(rbias) <= lim["max_render_bias_px"]) and rmed <= lim["max_render_median_px"])
        malas += not good
        line = (f"{tag}: RENDER sesgo ({rbias[0]:+.2f},{rbias[1]:+.2f}) med {rmed:.2f} px n={rn} "
                f"{'OK' if good else 'FALLA'}")
        p = det.pose(gray, K, dist)
        rd = (real.view(cam, k).meta.get("directa") or {}) if k in real.stops.get(cam, []) else {}
        if p is None:
            print(line + f" | DETECTOR: no detectado (real {rd.get('corners')} esq)")
            rows.append((rmed, np.nan, np.nan, np.nan))
            continue
        dd, dm = L.pose_diff(L.T_from(p["R"], p["t"]), T)
        obj = det.corners3d[p["ids"].ravel()] if p["ids"] is not None else np.zeros((0, 3))
        dbias = np.full(2, np.nan)
        if len(obj):
            proj, _ = cv2.projectPoints(obj, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, dist)
            dbias = (p["corners"].reshape(-1, 2) - proj.reshape(-1, 2)).mean(0)
        rows.append((rmed, dm, dd, float(np.mean(dbias))))
        print(line + f" | DETECTOR esq {p['n']:2d} (real {rd.get('corners', '?')}) reproj {p['reproj_px']:.2f}"
              f" pose {dm:.2f} mm {dd:.3f} deg sesgo ({dbias[0]:+.2f},{dbias[1]:+.2f})")
    if rows:
        r = np.array(rows, float)
        db = np.nanmedian(r[:, 3])
        parity = abs(db - lim["detector_bias_px"]) <= lim["max_detector_bias_dev_px"]
        print(f"resumen {len(r)} vistas: render med {np.median(r[:, 0]):.3f} px max {r[:, 0].max():.3f} | "
              f"detector pose mm med {np.nanmedian(r[:, 1]):.2f} max {np.nanmax(r[:, 1]):.2f}, deg med "
              f"{np.nanmedian(r[:, 2]):.3f}, sesgo med {db:+.2f} px (esperado {lim['detector_bias_px']:+.2f}: "
              f"{'OK' if parity else 'DIFIERE'})")
    # criterio global: una vista suelta fuera de tolerancia (pocas esquinas por oclusion, ruido de
    # 0.01 px sobre el limite) no invalida la captura; si muchas fallan o la mediana global, si
    n = len(gtall["vistas"])
    ok = bool(rows) and np.median(np.array(rows, float)[:, 0]) <= lim["max_render_median_px"] \
        and malas <= lim["max_fraccion_vistas_falla"] * n
    print(f"vistas fuera de tolerancia: {malas}/{n} (max {lim['max_fraccion_vistas_falla']:.0%}) -> "
          f"{'OK' if ok else 'FALLA'}")
    print(f"OpenCV {cv2.__version__}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
