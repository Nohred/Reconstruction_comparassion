"""Exporta una captura (real o sintetica, mismo formato) para reconstruccion con poses conocidas.

    tools/cv_py.sh -m eval.export_recon Captura_real/bunny_x200 --check
    tools/cv_py.sh -m eval.export_recon Captura_sintetica/bunny_x200 --check

Salida en Reconstrucciones/<real|sintetica>/<captura>/entrada/ (la fuente se deduce de si existe
ground_truth.json). Es la MISMA para ambas fuentes, asi las tecnicas se comparan con igual entrada:
    images/<camara>/stop_XX.png   color sin distorsion (modelo PINHOLE)
    masks/<camara>/stop_XX.png    ROI: cilindro sobre el eje de la base (255 = dentro)
    sparse/0/cameras.txt          PINHOLE, una camara por L515
    sparse/0/images.txt           world->cam = T_color<-board de meta.json (pose FINAL), en metros
    sparse/0/points3D.txt         vacio (eval/triangulate.py lo llena con COLMAP)
    sparse/known/                 copia sin puntos (entrada fija de eval/triangulate.py)
    transforms.json               nerfstudio (NeRF / splatfacto): c2w OpenGL, intrinsecos por cuadro
    vistas.json                   K (OpenCV), T_color<-board (mm) y origen de cada imagen
Pixeles: cameras.txt / transforms.json usan cx, cy + recon.principal_point_offset_px (convencion
de COLMAP/nerfstudio: centro del pixel (0,0) en (0.5, 0.5)); vistas.json usa la de OpenCV.
Mundo = marco del tablero en metros (z hacia la mesa; "up" = -z). La malla GT del gemelo esta en
el marco de revision/: p_rev = diag(revision_flip) @ p_board (mm).

--check relee el modelo, detecta el ChArUco en las imagenes exportadas y reproyecta las esquinas
con la pose de images.txt: valida K, la sin-distorsion, los cuaterniones y las unidades.
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from common.board import L
from common.colmap import ColmapCamera, ColmapImage, read_model, write_model
from common.config import load_config, repo_path
from common.geometry import CV_FROM_USD_CAM, T_inv, fit_turntable_axis
from common.intrinsics import K_dist_from_calib
from common.realdata import Capture
from sim.poses import replay_turntable

SPARSE_DIR = "sparse/0"
KNOWN_DIR = "sparse/known"
IMAGES_DIR = "images"
MASKS_DIR = "masks"
GL_FROM_CV = np.diag([1.0, -1.0, -1.0, 1.0])     # ejes de camara OpenCV -> OpenGL (nerfstudio)
assert np.allclose(GL_FROM_CV[:3, :3], CV_FROM_USD_CAM)


def recon_dir(common, fuente, name):
    """Reconstrucciones/<fuente>/<captura>/"""
    return repo_path(common["paths"]["recon_root"]) / fuente / name


def roi_points(axis_xy, roi):
    """Puntos de dos circulos (base y tapa) del cilindro ROI en el marco del tablero (mm)."""
    a = np.linspace(0, 2 * np.pi, roi["muestras"], endpoint=False)
    ring = np.stack([axis_xy[0] + roi["radio_mm"] * np.cos(a), axis_xy[1] + roi["radio_mm"] * np.sin(a)], 1)
    zs = (roi["bajo_tablero_mm"], -roi["altura_mm"])          # z del tablero apunta a la mesa
    return np.vstack([np.column_stack([ring, np.full(len(ring), z)]) for z in zs])


def roi_mask(P, T, K, w, h):
    uv, _ = cv2.projectPoints(P, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, None)
    m = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(uv.reshape(-1, 2).astype(np.float32)).astype(np.int32), 255)
    return m


def capture_axis(cap):
    plan = replay_turntable(cap, cap.cameras, sorted(set().union(*cap.stops.values())))
    xy, _ = fit_turntable_axis([plan.T_b0_from_bk[k] for k in plan.stops])
    return xy


def export(cap, out_dir, rcfg):
    s = rcfg["units_scale_from_mm"]
    off = rcfg["principal_point_offset_px"]
    roi = roi_points(capture_axis(cap), rcfg["roi"])
    cams, images, frames, vistas, skipped = [], [], [], [], []
    for sub in (IMAGES_DIR, MASKS_DIR):
        shutil.rmtree(out_dir / sub, ignore_errors=True)
    for cid, cam in enumerate(cap.cameras, start=1):
        ci = cap.calib[cam]["color_intrinsics"]
        K, dist = K_dist_from_calib(ci)
        w, h = int(ci["width"]), int(ci["height"])
        cams.append(ColmapCamera(cid, w, h, K[0, 0], K[1, 1], K[0, 2] + off, K[1, 2] + off))
        for v in cap.views([cam]):
            if not v.has_pose:
                skipped.append(f"{cam}/{v.stop_dir.name} (sin pose)")
                continue
            img = cv2.imread(str(v.color_path), cv2.IMREAD_UNCHANGED) if v.color_path.exists() else None
            if img is None:
                skipped.append(f"{cam}/{v.stop_dir.name} (sin {v.color_path.name})")
                continue
            name = f"{cam}/{v.stop_dir.name}.{rcfg['image_ext']}"
            for sub, data in ((IMAGES_DIR, cv2.undistort(img, K, dist, None, K)
                               if rcfg["undistort"] and np.any(dist) else img),
                              (MASKS_DIR, roi_mask(roi, v.T_color_from_board, K, w, h))):
                dst = out_dir / sub / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dst), data)
            T = v.T_color_from_board
            images.append(ColmapImage(len(images) + 1, T[:3, :3], T[:3, 3] * s, cid, name))
            c2w = T_inv(T)
            c2w[:3, 3] *= s
            frames.append(dict(file_path=f"{IMAGES_DIR}/{name}", mask_path=f"{MASKS_DIR}/{name}",
                               transform_matrix=(c2w @ GL_FROM_CV).tolist(),
                               fl_x=K[0, 0], fl_y=K[1, 1], cx=K[0, 2] + off, cy=K[1, 2] + off, w=w, h=h))
            vistas.append(dict(nombre=name, camara=cam, parada=v.stop, K=K.tolist(), width=w, height=h,
                               T_color_from_board_mm=T.tolist(),
                               stop_dir=str(v.stop_dir.relative_to(repo_path(".")))))
    for sub in (SPARSE_DIR, KNOWN_DIR):
        shutil.rmtree(out_dir / sub, ignore_errors=True)
        write_model(out_dir / sub, cams, images)
    tj = dict(camera_model="OPENCV", k1=0.0, k2=0.0, p1=0.0, p2=0.0, frames=frames,
              nota="mundo = tablero ChArUco en metros (z hacia la mesa); camaras OpenGL; "
                   "cx/cy con centro de pixel en +0.5")
    (out_dir / "transforms.json").write_text(json.dumps(tj, indent=1), encoding="utf-8")
    (out_dir / "vistas.json").write_text(json.dumps(dict(
        captura=str(cap.root.relative_to(repo_path("."))), unidades_T="mm", marco="tablero",
        roi=dict(eje_xy_mm=capture_axis(cap).tolist(), **rcfg["roi"]),
        vistas=vistas), indent=1), encoding="utf-8")
    return cams, images, skipped


def check(out_dir, rcfg, cap):
    """RMS de reproyeccion por camara usando SOLO lo escrito en el modelo COLMAP."""
    s = rcfg["units_scale_from_mm"]
    det = L.BoardDetector()
    cams, imgs = read_model(out_dir / SPARSE_DIR)
    per_cam = {}
    for im in imgs:
        cam = cams[im.camera_id]
        gray = cv2.imread(str(out_dir / IMAGES_DIR / im.name), cv2.IMREAD_GRAYSCALE)
        ch_c, ch_ids, _, _ = det.detect_all(gray)
        if ch_ids is None:
            continue
        obj = det.corners3d[ch_ids.ravel()]
        rvec = cv2.Rodrigues(im.R)[0]
        Kcv = cam.K.copy()
        Kcv[:2, 2] -= rcfg["principal_point_offset_px"]
        proj, _ = cv2.projectPoints(obj, rvec, im.t / s, Kcv, None)
        err = np.linalg.norm(proj.reshape(-1, 2) - ch_c.reshape(-1, 2), axis=1)
        # misma metrica con la pose "directa" (deteccion propia, sin fusion del rig) como referencia
        label, stop = im.name.split("/")[0], int(Path(im.name).stem.split("_")[1])
        d = cap.view(label, stop).meta.get("directa")
        err_d = None
        if d is not None:
            pd, _ = cv2.projectPoints(obj, cv2.Rodrigues(np.array(d["R_board_to_color"]))[0],
                                      np.array(d["t_board_to_color_mm"]), Kcv, None)
            err_d = np.linalg.norm(pd.reshape(-1, 2) - ch_c.reshape(-1, 2), axis=1)
        per_cam.setdefault(label, []).append((np.sqrt(np.mean(err ** 2)), len(err),
                                              None if err_d is None else np.sqrt(np.mean(err_d ** 2)),
                                              (ch_c.reshape(-1, 2) - proj.reshape(-1, 2)).mean(0)))
    print(f"check (umbral RMS {rcfg['check_max_reproj_px']} px; pose final de meta.json vs 'directa'):")
    for label, rows in per_cam.items():
        r = np.array([x[0] for x in rows])
        rd = np.array([x[2] for x in rows if x[2] is not None])
        bad = int(np.sum(r > rcfg["check_max_reproj_px"]))
        # Criterio: la mediana con la pose DIRECTA (misma deteccion que la captura) valida K,
        # sin-distorsion, cuaterniones y unidades. La pose final (fusion del rig) es informativa.
        print(f"  {label:6s} n={len(rows):2d} esquinas~{int(np.median([x[1] for x in rows])):2d}  "
              f"RMS final mediana {np.median(r):.3f} max {r.max():.3f} px (>{rcfg['check_max_reproj_px']}: {bad})"
              f" | directa mediana {np.median(rd):.3f} max {rd.max():.3f} px"
              f" | sesgo medio ({np.mean([x[3][0] for x in rows]):+.2f}, {np.mean([x[3][1] for x in rows]):+.2f}) px")
    # Criterio: mediana GLOBAL con la pose directa. Un error de exportacion (medio pixel,
    # cuaterniones, unidades) desplaza todas las vistas; una camara con oclusion/angulo rasante
    # puede tener su propia mediana algo mayor sin que la exportacion este mal.
    # Solo vistas con >= check_min_esquinas: con el tablero muy tapado (objeto grande) quedan pocas
    # esquinas, casi todas junto al borde del objeto, y el ruido del detector domina la mediana.
    rows_all = [x for rows in per_cam.values() for x in rows if x[2] is not None]
    nmin = rcfg["check_min_esquinas"]
    sel = [x[2] for x in rows_all if x[1] >= nmin]
    if len(sel) < rcfg["check_min_vistas"]:
        print(f"  (solo {len(sel)} vistas con >= {nmin} esquinas: el criterio usa todas)")
        sel = [x[2] for x in rows_all]
    alld = np.array(sel)
    ok = bool(len(alld) and np.median(alld) <= rcfg["check_max_reproj_px"])
    print(f"  global: directa mediana {np.median(alld):.3f} px en {len(alld)} vistas con >= {nmin} esquinas "
          f"(todas: {np.median([x[2] for x in rows_all]):.3f} px; umbral {rcfg['check_max_reproj_px']}) -> "
          f"{'OK' if ok else 'FALLA'}")
    print(f"  (detector: OpenCV {cv2.__version__}; la captura uso OpenCV {cap.session.get('opencv', '?')})")
    return ok


def roi_for(roi_cfg, root, fuente):
    """ROI de config; la altura crece si el objeto GT es mas alto (p. ej. caesar, 200 mm). En la real se
    usa el GT de su gemelo (Captura_sintetica/<mismo nombre>) si ya existe."""
    roi = {k: v for k, v in roi_cfg.items() if k != "altura_margen_mm"}
    src = root if fuente == "sintetica" else repo_path(load_config("common")["paths"]["synthetic_root"]) / root.name
    gt_file = src / load_config("twin")["gt_file"]
    if gt_file.exists():
        gt = json.loads(gt_file.read_text("utf-8"))
        h = (gt.get("objeto") or {}).get("extension_mm", [0, 0, 0])[2]
        roi["altura_mm"] = float(max(roi["altura_mm"], h + roi_cfg["altura_margen_mm"]))
    return roi


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("captura", help="Captura_real/<captura> o Captura_sintetica/<captura>")
    ap.add_argument("--salida", help="por defecto Reconstrucciones/<fuente>/<captura>/entrada")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    cfg = load_config("common")
    root = repo_path(args.captura)
    fuente = "sintetica" if (root / load_config("twin")["gt_file"]).exists() else "real"
    cap = Capture.open(root, cfg["capture_layout"])
    out_dir = repo_path(args.salida) if args.salida else (
        recon_dir(cfg, fuente, root.name) / cfg["recon"]["entrada_dir"])
    rcfg = dict(cfg["recon"], roi=roi_for(cfg["recon"]["roi"], root, fuente))
    cams, images, skipped = export(cap, out_dir, rcfg)
    print(f"{out_dir} ({fuente}): {len(cams)} camaras, {len(images)} imagenes, "
          f"ROI r {rcfg['roi']['radio_mm']:g} mm, altura {rcfg['roi']['altura_mm']:g} mm")
    for s in skipped:
        print(f"  OMITIDA {s}")
    if args.check:
        raise SystemExit(0 if check(out_dir, cfg["recon"], cap) else 1)


if __name__ == "__main__":
    main()
