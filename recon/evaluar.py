"""Evalua las 4 tecnicas de una captura contra la malla GT (marco del tablero, mm).

    tools/ns_py.sh -m recon.evaluar Reconstrucciones/sintetica/bunnny
    tools/ns_py.sh -m recon.evaluar Reconstrucciones/real/bunnny [--gt otra_malla_rev.ply]

GT: gt/objeto_rev.ply de la captura sintetica (en la real, la del gemelo con el mismo nombre en
Captura_sintetica/, cuya colocacion no coincide con la real: se registra UNA vez sobre la
reconstruccion de profundidad con busqueda de yaw + ICP y esa pose sirve para todas las tecnicas).
Nube evaluada: <tecnica>/resultado.ply LIMPIA (recon/limpieza.py: ROI, piso = plano del tablero
ajustado con RANSAC, outliers y clusters flotantes) -> <tecnica>/resultado_limpio.ply. La base del
objeto (apoyada en el tablero) no es observable y se excluye del GT hasta la MISMA altura a la que
la limpieza corto el piso en esa tecnica (corte_mm, >= limpieza.sobre_plano_mm).
Ademas: evaluacion/<tecnica>_error.ply = nube limpia coloreada por su distancia al GT.
Metricas (mm): precision (nube -> malla GT), completitud (GT -> nube), chamfer, hausdorff (max
y p95), F-score a varios umbrales. "sin_icp" = tal cual; "con_icp" = tras ICP de la nube al GT.
Escribe evaluacion/metricas.json y actualiza Resultados/resumen.csv.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from common.config import load_config, repo_path
from common.realdata import read_json
from common.meshio import read_mesh_ply
from recon.comun import MM, load_entrada, load_recon_config
from recon.limpieza import clean_cloud, describe

TECNICAS = ["nerf", "gaussian", "fotogrametria", "profundidad"]


def gt_mesh_board_m(path, flip, o3d):
    V, F = read_mesh_ply(path)
    V = V * np.asarray(flip) * MM               # rev (mm) -> tablero (m); el reflejo invierte caras
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F[:, ::-1]))
    m.compute_vertex_normals()
    return m


def point_to_mesh_mm(P, mesh, o3d):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene.compute_distance(o3d.core.Tensor(P.astype(np.float32))).numpy() / MM


def error_colors(err_mm, vmax):
    """Mapa azul (0) -> verde -> amarillo -> rojo (>= vmax), sin depender de matplotlib."""
    t = np.clip(err_mm / vmax, 0, 1)[:, None]
    stops = np.array([[0.0, 0.2, 0.9], [0.0, 0.8, 0.3], [1.0, 0.85, 0.0], [0.9, 0.05, 0.05]])
    x = t * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    f = x - i
    return stops[i[:, 0]] * (1 - f) + stops[i[:, 0] + 1] * f


def metrics(P, mesh, gt_pts, ecfg, o3d):
    from scipy.spatial import cKDTree

    if len(P) == 0:
        return dict(puntos=0)
    acc = point_to_mesh_mm(P, mesh, o3d)
    comp = cKDTree(P).query(gt_pts)[0] / MM
    out = dict(puntos=int(len(P)),
               precision_media_mm=float(acc.mean()), precision_mediana_mm=float(np.median(acc)),
               precision_p90_mm=float(np.percentile(acc, 90)),
               completitud_media_mm=float(comp.mean()), completitud_mediana_mm=float(np.median(comp)),
               completitud_p90_mm=float(np.percentile(comp, 90)),
               chamfer_mm=float((acc.mean() + comp.mean()) / 2),
               hausdorff_mm=float(max(acc.max(), comp.max())),
               hausdorff_p95_mm=float(max(np.percentile(acc, 95), np.percentile(comp, 95))))
    for t in ecfg["umbrales_fscore_mm"]:
        p, r = float((acc < t).mean()), float((comp < t).mean())
        out[f"fscore_{t:g}mm"] = 2 * p * r / (p + r) if p + r else 0.0
        out[f"precision_{t:g}mm"], out[f"recall_{t:g}mm"] = p, r
    cv = ecfg["curva_umbrales_mm"]
    th = np.arange(cv["paso"], cv["max"] + 1e-9, cv["paso"])
    out["curva"] = dict(umbral_mm=th.round(4).tolist(),
                        precision=[float((acc < t).mean()) for t in th],
                        recall=[float((comp < t).mean()) for t in th])
    return out


def icp(P, gt_pcd, dists_mm, o3d, T0=np.eye(4)):
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    T = T0
    for d in dists_mm:
        T = o3d.pipelines.registration.registration_icp(
            src, gt_pcd, d * MM, T, o3d.pipelines.registration.TransformationEstimationPointToPlane()).transformation
    return T


def register_gt_to(target_P, gt_pcd, rcfg, o3d):
    """Pose del GT sobre una nube real: barrido de yaw (eje vertical por el centroide) + ICP."""
    rc = rcfg["evaluacion"]["registro_real"]
    G = np.asarray(gt_pcd.points)
    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_P))
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.006, max_nn=30))
    cg, ct = G.mean(0), target_P.mean(0)
    best = None
    for yaw in np.arange(0, 360, rc["yaw_paso_deg"]):
        a = np.radians(yaw)
        R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = np.array([ct[0], ct[1], cg[2]]) - R @ cg
        for d in rc["icp_dist_mm"]:
            r = o3d.pipelines.registration.registration_icp(
                gt_pcd, tgt, d * MM, T, o3d.pipelines.registration.TransformationEstimationPointToPlane())
            T = r.transformation
        score = (r.fitness, -r.inlier_rmse)
        if best is None or score > best[0]:
            best = (score, T, yaw)
    (fit, nrmse), T, yaw = best
    return T, dict(yaw_inicial_deg=float(yaw), fitness=float(fit), rmse_mm=float(-nrmse / MM))


def update_summary(results_root):
    """Resultados/resumen.csv con todas las evaluaciones existentes."""
    common = load_config("common")
    rows = []
    for mj in sorted(repo_path(common["paths"]["recon_root"]).glob("*/*/evaluacion/metricas.json")):
        m = json.loads(mj.read_text("utf-8"))
        for tec, r in m["tecnicas"].items():
            for modo in ("sin_icp", "con_icp"):
                if modo in r:
                    lp = (r.get("limpieza") or {}).get("pasos", {})
                    rows.append(dict(fuente=m["fuente"], captura=m["captura"], tecnica=tec, modo=modo,
                                     perfil=(r.get("info") or {}).get("perfil", ""),
                                     puntos_roi=lp.get("roi"), puntos_sin_piso=lp.get("sin_piso"),
                                     puntos_sin_outliers=lp.get("sin_outliers"),
                                     **{k: v for k, v in r[modo].items() if k != "curva"}))
    if not rows:
        return None
    keys = list(dict.fromkeys(k for r in rows for k in r))
    out = repo_path(results_root) / "resumen.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recon_dir")
    ap.add_argument("--gt", help="malla GT (marco de revision, mm); por defecto la del gemelo")
    args = ap.parse_args()
    import open3d as o3d

    common, rcfg = load_config("common"), load_recon_config()
    ecfg, lcfg = rcfg["evaluacion"], rcfg["limpieza"]
    tcfg = load_config("twin")
    rdir = repo_path(args.recon_dir)
    fuente, captura = rdir.parent.name, rdir.name
    entrada = load_entrada(rdir)
    roi = entrada["roi"]
    synth = repo_path(common["paths"]["synthetic_root"]) / captura
    gt_path = repo_path(args.gt) if args.gt else synth / tcfg["gt_dir"] / "objeto_rev.ply"
    if not gt_path.exists():
        raise SystemExit(f"no hay malla GT {gt_path} (para una captura real, genera antes su gemelo)")
    mesh = gt_mesh_board_m(gt_path, common["conventions"]["revision_flip"], o3d)
    res = dict(fuente=fuente, captura=captura, gt=str(gt_path.relative_to(repo_path("."))),
               perfil_reconstruccion={}, tecnicas={})
    if fuente == "real":
        ref = rdir / "profundidad" / "resultado.ply"
        if not ref.exists():
            raise SystemExit("captura real: falta profundidad/resultado.ply para registrar el GT")
        P, _ = clean_cloud(np.asarray(o3d.io.read_point_cloud(str(ref)).points), roi, lcfg, o3d)
        T, rinfo = register_gt_to(P, mesh.sample_points_uniformly(50000), rcfg, o3d)
        mesh.transform(T)
        res["registro_gt"] = dict(rinfo, T_tablero_m=T.tolist(), referencia="profundidad")
        print(f"GT registrado sobre profundidad: fitness {rinfo['fitness']:.2f}, rmse {rinfo['rmse_mm']:.2f} mm")
    gt_all = mesh.sample_points_uniformly(ecfg["muestras_gt"])
    gt_all.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))
    G = np.asarray(gt_all.points)
    h_gt = (G[:, 2].max() - G[:, 2]) / MM            # altura sobre la base del GT (z max, z hacia la mesa)
    out_err = rdir / "evaluacion"
    for tec in TECNICAS:
        f = rdir / tec / "resultado.ply"
        if not f.exists():
            continue
        P, linfo = clean_cloud(np.asarray(o3d.io.read_point_cloud(str(f)).points), roi, lcfg, o3d)
        o3d.io.write_point_cloud(str(rdir / tec / "resultado_limpio.ply"),
                                 o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)))
        print(f"  {tec:13s} limpieza: {describe(linfo)}")
        vis = np.nonzero(h_gt > linfo["corte_mm"])[0]   # GT recortado a la altura del corte del piso
        gt_pts, gt_pcd = G[vis], gt_all.select_by_index(vis)
        r = dict(limpieza=linfo, sin_icp=metrics(P, mesh, gt_pts, ecfg, o3d))
        if len(P):
            ep = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
            ep.colors = o3d.utility.Vector3dVector(
                error_colors(point_to_mesh_mm(P, mesh, o3d), ecfg["error_ply_max_mm"]))
            out_err.mkdir(exist_ok=True)
            o3d.io.write_point_cloud(str(out_err / f"{tec}_error.ply"), ep)
        if len(P):
            T = icp(P, gt_pcd, ecfg["icp_por_tecnica_mm"], o3d)
            Pc = P @ T[:3, :3].T + T[:3, 3]
            r["con_icp"] = metrics(Pc, mesh, gt_pts, ecfg, o3d)
            c = gt_pts.mean(0)
            r["icp_desplazamiento_mm"] = float(np.linalg.norm(T[:3, :3] @ c + T[:3, 3] - c) / MM)
        info = rdir / tec / "info.json"
        if info.exists():
            i = json.loads(info.read_text("utf-8"))
            r["info"] = {k: i[k] for k in ("perfil", "iteraciones", "entrenamiento_s", "fecha") if k in i}
        res["tecnicas"][tec] = r
        s = r["sin_icp"]
        if s.get("puntos"):
            print(f"  {tec:13s} chamfer {s['chamfer_mm']:6.2f} mm | precision med {s['precision_mediana_mm']:5.2f}"
                  f" | completitud med {s['completitud_mediana_mm']:5.2f} | F@2mm {s['fscore_2mm']:.3f}"
                  f" | con ICP chamfer {r['con_icp']['chamfer_mm']:6.2f} (mueve {r['icp_desplazamiento_mm']:.1f} mm)")
        else:
            print(f"  {tec:13s} sin puntos en la ROI")
    out = rdir / "evaluacion"
    out.mkdir(exist_ok=True)
    o3d.io.write_triangle_mesh(str(out / "gt_tablero_m.ply"), mesh)
    (out / "metricas.json").write_text(json.dumps(res, indent=1), "utf-8")
    s = update_summary(common["paths"]["results_root"])
    print(f"-> {out / 'metricas.json'}" + (f"; resumen {s}" if s else ""))


if __name__ == "__main__":
    main()
