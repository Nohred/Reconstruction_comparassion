"""Valida la geometria del render del gemelo contra su ground truth (sin detector).

    tools/cv_py.sh -m eval.validate_geometry Captura_sintetica/bunny_x200

Por vista de gt/<camara>/stop_XX/:
  PLANO   : depth.png perfecta retroproyectada (K de profundidad) y llevada al marco del tablero
            con T_depth<-board: los pixeles de la hoja deben dar z = 0 (RMS <= max_plano_rms_mm).
  SILUETA : la malla GT proyectada (fillPoly por triangulo) vs mask.png / mask_depth.png:
            CRITERIO: el desplazamiento entero que maximiza el IoU debe ser (0, 0) (si la pose,
            la lente o la escala estuvieran mal, la silueta GT quedaria corrida). Informativo: IoU,
            p99 de la distancia de los pixeles en desacuerdo al otro contorno (muestreo del borde,
            mayor en objetos con partes delgadas) y centroide.
"""
import argparse

import cv2
import numpy as np

from common.board import page_texture
from common.config import load_config, repo_path
from common.geometry import T_inv, rev_to_board, transform_points
from common.intrinsics import K_dist_from_calib
from common.meshio import read_mesh_ply
from common.realdata import Capture, read_json

SUB = 4                     # fillPoly con coordenadas en 1/16 px (shift=4)


def render_silhouette(P_board, F, T, K, dist, w, h):
    """Union de las proyecciones de todos los triangulos.

    Triangulo por triangulo: cv2.fillPoly con varios poligonos rellena par-impar (los solapes se
    cancelan), lo que abria huecos en mallas con caras superpuestas en la imagen (artefacto_pi2).
    """
    uv, _ = cv2.projectPoints(P_board, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, dist)
    uv = uv.reshape(-1, 2)
    tri = np.round(uv[F] * (1 << SUB)).astype(np.int32)
    lo, hi = tri.min(1) >> SUB, tri.max(1) >> SUB
    inside = (hi[:, 0] >= 0) & (hi[:, 1] >= 0) & (lo[:, 0] < w) & (lo[:, 1] < h)
    m = np.zeros((h, w), np.uint8)
    for t in tri[inside]:
        cv2.fillConvexPoly(m, t, 255, lineType=cv2.LINE_8, shift=SUB)
    return m > 0


def best_shift(a, b, r):
    """Desplazamiento entero (dx, dy) de `a` en [-r, r]^2 que maximiza el IoU con `b`."""
    ys, xs = np.nonzero(a | b)
    if not len(xs):
        return (0, 0)
    y0, y1 = max(ys.min() - r, 0), ys.max() + r + 1
    x0, x1 = max(xs.min() - r, 0), xs.max() + r + 1
    A, B = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
    best = None
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            S = np.roll(np.roll(A, dy, 0), dx, 1)
            iou = (S & B).sum() / max((S | B).sum(), 1)
            if best is None or iou > best[0] + 1e-9 or (abs(iou - best[0]) <= 1e-9 and abs(dx) + abs(dy) == 0):
                best = (iou, dx, dy)
    return best[1], best[2]


def sil_stats(a, b):
    """(IoU, p99 de distancia de los pixeles en desacuerdo al otro contorno, desplazamiento del centroide)."""
    inter, union = np.logical_and(a, b).sum(), np.logical_or(a, b).sum()
    dist = []
    for x, y in ((a, b), (b, a)):
        only = x & ~y
        if only.any():
            dist.append(cv2.distanceTransform((~y).astype(np.uint8), cv2.DIST_L2, 5)[only])
    p99 = float(np.percentile(np.concatenate(dist), 99)) if dist else 0.0
    ya, xa = np.nonzero(a)
    yb, xb = np.nonzero(b)
    if not len(xa) or not len(xb):
        return np.nan, np.inf, np.array([np.nan, np.nan])
    return inter / union, p99, np.array([xb.mean() - xa.mean(), yb.mean() - ya.mean()])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sintetica")
    ap.add_argument("--config", default="twin")
    args = ap.parse_args()
    common, cfg, scene = load_config("common"), load_config(args.config), load_config("scene")
    lay = common["capture_layout"]
    root = repo_path(args.sintetica)
    gt = read_json(root / cfg["gt_file"])
    real = Capture.open(repo_path(gt["plantilla"]), lay)
    lim = cfg["validacion"]
    _, (x0, y0, x1, y1) = page_texture(scene["board_page"])
    mesh = None
    if gt.get("objeto"):
        P, F = read_mesh_ply(root / gt["objeto"]["malla_rev"])
        mesh = (rev_to_board(P, common["conventions"]["revision_flip"]), F)
    ok, rows = True, []
    for v in gt["vistas"]:
        cam, k = v["camara"], v["parada"]
        d = root / cfg["gt_dir"] / cam / lay["stop_dir"].format(k=k)
        cal = real.calib[cam]
        Kd, _ = K_dist_from_calib(cal["depth_intrinsics"])
        Kc, dc = K_dist_from_calib(cal["color_intrinsics"])
        if not gt["distorsion_color"]:
            dc = np.zeros_like(dc)
        Td, Tc = np.array(v["T_depth_from_board_mm"]), np.array(v["T_color_from_board_mm"])
        z = cv2.imread(str(d / "depth.png"), cv2.IMREAD_UNCHANGED).astype(np.float64) * cal["depth_scale_m"] * 1000
        md = cv2.imread(str(d / "mask_depth.png"), cv2.IMREAD_GRAYSCALE)
        sel = (z > 0) & ((md == 0) if md is not None else True)
        vv, uu = np.nonzero(sel)
        zz = z[sel]
        Pc = np.stack([(uu - Kd[0, 2]) / Kd[0, 0] * zz, (vv - Kd[1, 2]) / Kd[1, 1] * zz, zz], 1)
        Pb = transform_points(T_inv(Td), Pc)
        page = (Pb[:, 0] > x0) & (Pb[:, 0] < x1) & (Pb[:, 1] > y0) & (Pb[:, 1] < y1) & (np.abs(Pb[:, 2]) < 10)
        rms = float(np.sqrt(np.mean(Pb[page, 2] ** 2))) if page.any() else np.nan
        good = bool(rms <= lim["max_plano_rms_mm"])
        line = f"{cam}/stop_{k:02d}: PLANO rms {rms:.3f} mm (n={int(page.sum())})"
        if mesh is not None:
            for name, T, K, dist, img in (("color", Tc, Kc, dc, "mask.png"), ("prof", Td, Kd, None, "mask_depth.png")):
                m = cv2.imread(str(d / img), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    continue
                s = render_silhouette(mesh[0], mesh[1], T, K, dist, m.shape[1], m.shape[0])
                iou, p99, dc_ = sil_stats(s, m > 0)
                sh = best_shift(s, m > 0, lim["silueta_busqueda_px"])
                rows.append((iou, p99, *dc_, abs(sh[0]) + abs(sh[1])))
                good &= sh == (0, 0)
                line += (f" | {name} mejor desplaz. {sh} IoU {iou:.3f} borde p99 {p99:.2f} px "
                         f"centroide ({dc_[0]:+.2f},{dc_[1]:+.2f})")
        ok &= good
        print(line + ("" if good else "  FALLA"))
    if rows:
        r = np.array(rows)
        print(f"resumen {len(gt['vistas'])} vistas: silueta desalineada en {int((r[:, 4] > 0).sum())} de {len(r)}, "
              f"borde p99 max {r[:, 1].max():.2f} px, IoU med "
              f"{np.median(r[:, 0]):.3f}, centroide medio ({r[:, 2].mean():+.2f},{r[:, 3].mean():+.2f}) px")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
