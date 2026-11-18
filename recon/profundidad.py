"""Tecnica 4: fusion de las nubes de profundidad de las L515.

    tools/ns_py.sh -m recon.profundidad Reconstrucciones/real/bunnny

1. Por vista (camara, parada): mediana de los N cuadros depth_XX.png (pixel valido si tiene
   valor en >= cuadros_validos_min de ellos), retroproyeccion con los intrinsecos de profundidad
   y T_depth<-board = inv(T_color<-depth) @ T_color<-board (pose FINAL de meta.json).
2. Recorte a la ROI (cilindro sobre el eje) y ICP multivista (point-to-plane, grafo de poses de
   open3d) entre camaras de la misma parada y paradas vecinas: corrige errores de pose/extrinseco.
3. TSDF con las poses refinadas -> malla.ply, resultado.ply (superficie muestreada), y
   nube_cruda.ply (todas las nubes ya alineadas).
Salida: Reconstrucciones/<fuente>/<captura>/profundidad/
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from common.config import load_config, repo_path
from common.geometry import T_inv, depth_to_color_T
from common.intrinsics import K_dist_from_calib
from common.realdata import Capture
from recon.comun import MM, TSDF, load_entrada, load_recon_config, roi_image_mask, roi_select, write_result


def median_depth(paths, scale_m, min_frac):
    D = np.stack([cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in paths]).astype(np.float32)
    valid = D > 0
    n = valid.sum(0)
    D[~valid] = np.nan
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # pixeles sin ningun valor
        z = np.nanmedian(D, axis=0) * scale_m
    z[(n < min_frac * len(paths)) | ~np.isfinite(z)] = 0.0
    return z


def backproject(z, K):
    v, u = np.nonzero(z > 0)
    d = z[v, u]
    return np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d], 1)


def views_with_depth(cap, roi, pcfg):
    out = []
    for v in cap.views():
        if not v.has_pose or not v.depth_paths:
            continue
        cal = cap.calib[v.camera]
        Kd, _ = K_dist_from_calib(cal["depth_intrinsics"])
        T_cd = depth_to_color_T(cal["depth_to_color"], 1.0)                 # m
        T_cb = v.T_color_from_board.copy()
        T_cb[:3, 3] *= MM
        T_db = T_inv(T_cd) @ T_cb
        z = median_depth(v.depth_paths, cal["depth_scale_m"], pcfg["cuadros_validos_min"])
        h, w = z.shape
        z[~roi_image_mask(roi, T_db, Kd, w, h)] = 0.0
        out.append(dict(nombre=f"{v.camera}/stop_{v.stop:02d}", camara=v.camera, parada=v.stop, K=Kd,
                        T=T_db, z=z))
    return out


def view_clouds(views, roi, vox, o3d, C=None):
    clouds = []
    for i, v in enumerate(views):
        Ti = T_inv(v["T"])
        if C is not None:
            Ti = C[i] @ Ti
        P = backproject(v["z"], v["K"]) @ Ti[:3, :3].T + Ti[:3, 3]
        P = P[roi_select(P, roi)]
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)).voxel_down_sample(vox)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4 * vox, max_nn=30))
        clouds.append(pc)
    return clouds


def icp_pairs(views, k):
    n = len(views)
    return [(i, j) for i in range(n) for j in range(i + 1, n)
            if views[i]["parada"] == views[j]["parada"] or (
                views[i]["camara"] == views[j]["camara"] and
                0 < abs(views[i]["parada"] - views[j]["parada"]) <= k)]


def consistency_mm(clouds, pairs, dmax):
    """Mediana de la distancia NN entre nubes de pares que se solapan (solo d < dmax)."""
    d = []
    for i, j in pairs:
        x = np.asarray(clouds[i].compute_point_cloud_distance(clouds[j]))
        d.append(x[x < dmax])
    d = np.concatenate(d) if d else np.zeros(1)
    return float(np.median(d) / MM)


def displacement_mm(T, center):
    return float(np.linalg.norm(T[:3, :3] @ center + T[:3, 3] - center) / MM)


def multiway_icp(views, roi, pcfg, o3d):
    """Correcciones C_i (tablero -> tablero refinado) por vista con un grafo de poses.

    Aristas = ICP point-to-plane robusto (Tukey) entre vistas que se solapan; se descartan las que
    mueven el centro del objeto mas de max_correccion_mm. Tras optimizar se re-ancla al promedio
    (el marco del tablero lo definen las poses ChArUco, no una vista arbitraria).
    """
    from common.geometry import mean_pose

    vox = pcfg["voxel_icp_mm"] * MM
    clouds = view_clouds(views, roi, vox, o3d)
    center = np.array([*np.array(roi["eje_xy_mm"]) * MM, -roi["altura_mm"] * MM / 2])
    pairs = icp_pairs(views, pcfg["vecinos_paradas"])
    pg = o3d.pipelines.registration.PoseGraph()
    for _ in views:
        pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))
    est = o3d.pipelines.registration.TransformationEstimationPointToPlane(
        o3d.pipelines.registration.TukeyLoss(k=pcfg["tukey_mm"] * MM))
    used, rejected, disp = 0, 0, []
    for i, j in pairs:
        T = np.eye(4)
        for d in pcfg["icp_dist_max_mm"]:
            T = o3d.pipelines.registration.registration_icp(clouds[i], clouds[j], d * MM, T, est).transformation
        disp.append(displacement_mm(T, center))
        big = disp[-1] > pcfg["max_correccion_mm"]
        rejected += int(big)
        used += int(not big)
        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            clouds[i], clouds[j], pcfg["icp_dist_max_mm"][-1] * MM, T)
        # T lleva la nube i sobre la j: N_i = N_j @ T (poses de nodo: nube -> marco comun).
        # Aristas grandes: "inciertas" (el line process de open3d las pondera o las poda).
        pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(i, j, T, info, uncertain=big))
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    o3d.pipelines.registration.global_optimization(
        pg, o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=pcfg["icp_dist_max_mm"][-1] * MM, edge_prune_threshold=0.25,
            reference_node=0))
    C = [np.asarray(nd.pose) for nd in pg.nodes]
    M = T_inv(mean_pose(C))
    C = [M @ c for c in C]
    before = consistency_mm(clouds, pairs, pcfg["consistencia_dist_mm"] * MM)
    after = consistency_mm(view_clouds(views, roi, vox, o3d, C), pairs, pcfg["consistencia_dist_mm"] * MM)
    stats = dict(pares=len(pairs), aristas=used, inciertas=rejected,
                 icp_par_desplazamiento_mm=np.percentile(disp, [10, 50, 90]).tolist() if disp else None, consistencia_antes_mm=before,
                 consistencia_despues_mm=after, aceptado=bool(after < before),
                 correccion_centro_mm=[displacement_mm(c, center) for c in C])
    if not stats["aceptado"]:
        C = [np.eye(4) for _ in views]
    return C, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recon_dir")
    args = ap.parse_args()
    import open3d as o3d

    rcfg, common = load_recon_config(), load_config("common")
    pcfg = rcfg["profundidad"]
    rdir = repo_path(args.recon_dir)
    entrada = load_entrada(rdir)
    roi = entrada["roi"]
    cap = Capture.open(repo_path(entrada["captura"]), common["capture_layout"])
    views = views_with_depth(cap, roi, pcfg)
    print(f"{len(views)} vistas de profundidad")
    C, stats = [np.eye(4) for _ in views], dict(aceptado=False, motivo="refinar_poses=false")
    if pcfg["refinar_poses"]:
        C, stats = multiway_icp(views, roi, pcfg, o3d)
        cc = np.array(stats["correccion_centro_mm"])
        print(f"ICP por pares: desplazamiento en el objeto p10/p50/p90 "
              f"{np.round(stats['icp_par_desplazamiento_mm'], 2)} mm")
        print(f"ICP multivista: {stats['aristas']}/{stats['pares']} aristas ciertas, consistencia entre vistas "
              f"{stats['consistencia_antes_mm']:.2f} -> {stats['consistencia_despues_mm']:.2f} mm, "
              f"correccion en el objeto mediana {np.median(cc):.2f} max {cc.max():.2f} mm -> "
              f"{'ACEPTADO' if stats['aceptado'] else 'descartado (no mejora)'}")
    tsdf = TSDF(rcfg["tsdf"])
    raw = []
    for v, Ci in zip(views, C):
        T = v["T"] @ T_inv(Ci)                       # camara <- tablero refinado
        tsdf.integrate(v["z"], v["K"], T)
        P = backproject(v["z"], v["K"])
        Ti = T_inv(T)
        P = P @ Ti[:3, :3].T + Ti[:3, 3]
        raw.append(P[roi_select(P, roi)])
    mesh, pcd = tsdf.extract(roi)
    out = rdir / "profundidad"
    cruda = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack(raw))).voxel_down_sample(
        pcfg["voxel_icp_mm"] * MM / 3)
    out.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out / "nube_cruda.ply"), cruda)
    write_result(out, mesh, pcd, dict(tecnica="profundidad", vistas=[v["nombre"] for v in views],
                                      icp=stats, triangulos=len(mesh.triangles)))
    print(f"-> {out}: malla {len(mesh.triangles)} triangulos, {len(pcd.points)} puntos")


if __name__ == "__main__":
    main()
