"""Limpieza de las nubes reconstruidas antes de compararlas con el GT (configs/recon.yaml limpieza).

    tools/ns_py.sh -m recon.limpieza Reconstrucciones/sintetica/caesar [--tecnicas nerf gaussian]

Pasos (todos en el marco del tablero, metros):
  1) ROI: cilindro sobre el eje de la base (vistas.json de la entrada).
  2) Piso: el plano del tablero se ajusta con RANSAC en la PROPIA nube (cada tecnica tiene su
     sesgo de profundidad; en la real el plano no cae exacto en z = 0) con los puntos con
     |z| < busqueda_mm de la corona exterior de la ROI (sin objeto). Si el ajuste es dudoso (pocos
     puntos, muy inclinado o lejos de z = 0) se usa z = 0. En esa corona se mide el grosor del piso
     reconstruido (percentil de la altura sobre el plano) y se quita todo lo que quede por debajo de
     corte = clip(max(sobre_plano_mm, grosor + margen), corte_max_mm). El evaluador recorta el GT a
     la misma altura (info["corte_mm"]).
  3) Outliers estadisticos (distancia media a k vecinos > media + std * sigma).
  4) Clusters (DBSCAN en la nube submuestreada): se queda el mayor de los que tocan el piso (a menos
     de contacto_mm del corte; el objeto esta apoyado) y los que tengan >= min_fraccion
     de los puntos Y esten a menos de union_mm de lo ya aceptado (un objeto puede quedar en partes al
     cortar su base); el resto son flotantes de NeRF/3DGS o restos del piso/soporte.
Escribe <tecnica>/resultado_limpio.ply y devuelve un resumen de lo quitado en cada paso.
"""
import argparse
import json

import numpy as np

from common.config import repo_path
from recon.comun import MM, load_entrada, load_recon_config, roi_select

TECNICAS = ["nerf", "gaussian", "fotogrametria", "profundidad"]


def outer_ring(P, roi, pcfg):
    c = np.array(roi["eje_xy_mm"]) * MM
    return np.linalg.norm(P[:, :2] - c, axis=1) > pcfg["anillo_frac"] * roi["radio_mm"] * MM


def fit_board_plane(P, roi, pcfg, o3d):
    """(n, d) con n . p + d = altura sobre el tablero (m, positiva hacia la camara) y su info."""
    default = (np.array([0.0, 0.0, -1.0]), 0.0)
    near = np.abs(P[:, 2]) < pcfg["busqueda_mm"] * MM
    ring = near & outer_ring(P, roi, pcfg)
    cand = P[ring] if ring.sum() >= pcfg["min_puntos"] else P[near]
    info = dict(candidatos=int(len(cand)), zona="corona" if ring.sum() >= pcfg["min_puntos"] else "todo")
    if len(cand) < pcfg["min_puntos"]:
        return default, dict(info, metodo="z=0", motivo="pocos candidatos")
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cand))
    (a, b, c, d), inl = pc.segment_plane(pcfg["ransac_dist_mm"] * MM, 3, pcfg["ransac_iter"])
    Q = cand[inl]                                   # refinamiento por minimos cuadrados (SVD)
    cen = Q.mean(0)
    n = np.linalg.eigh(np.cov((Q - cen).T))[1][:, 0]   # menor autovalor = normal
    if n[2] > 0:
        n = -n                                      # normal hacia las camaras (z negativo)
    d = -n @ cen
    tilt = float(np.degrees(np.arccos(min(1.0, -n[2]))))
    axis = np.array([*np.array(roi["eje_xy_mm"]) * MM, 0.0])
    off = float((n @ axis + d) / MM)                # altura de z=0 (en el eje) sobre el plano
    info.update(inliers=int(len(inl)), inclinacion_deg=tilt, desplazamiento_mm=off,
                rms_mm=float(np.sqrt(np.mean((Q @ n + d) ** 2)) / MM))
    if tilt > pcfg["max_inclinacion_deg"] or abs(off) > pcfg["max_desplazamiento_mm"]:
        return default, dict(info, metodo="z=0", motivo="ajuste fuera de tolerancia")
    return (n, d), dict(info, metodo="ransac")


CONSERVADO, PISO, OUTLIER, FLOTANTE = 0, 1, 2, 3     # etiquetas de clean_cloud(..., etiquetas=True)


def clean_cloud(P, roi, lcfg, o3d, etiquetas=False):
    """Nube limpia (m) + resumen. P: (N,3) en el marco del tablero.

    etiquetas=True devuelve ademas (P_roi, etiqueta por punto: CONSERVADO/PISO/OUTLIER/FLOTANTE).
    """
    from scipy.spatial import cKDTree

    o3d.utility.random.seed(lcfg.get("semilla", 0))   # RANSAC reproducible
    pasos = dict(entrada=int(len(P)))
    P = P[roi_select(P, roi)]
    lab = np.full(len(P), CONSERVADO)
    pasos["roi"] = int(len(P))
    pc_ = lcfg["plano"]
    (n, d), pinfo = fit_board_plane(P, roi, pc_, o3d)
    h = (P @ n + d) / MM
    ring = outer_ring(P, roi, pc_) & (np.abs(h) < pc_["busqueda_mm"])
    grosor = float(np.percentile(h[ring], pc_["grosor_percentil"])) if ring.sum() >= pc_["min_puntos"] else 0.0
    corte = float(min(max(lcfg["sobre_plano_mm"], grosor + pc_["margen_mm"]), lcfg["corte_max_mm"]))
    pinfo.update(grosor_mm=grosor, corte_mm=corte)
    lab[h <= corte] = PISO
    pasos["sin_piso"] = int((lab == CONSERVADO).sum())
    idx = np.nonzero(lab == CONSERVADO)[0]
    if len(idx):
        s = lcfg["estadistico"]
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[idx]))
        _, keep = pc.remove_statistical_outlier(s["vecinos"], s["std"])
        bad = np.ones(len(idx), bool)
        bad[np.asarray(keep, int)] = False
        lab[idx[bad]] = OUTLIER
    pasos["sin_outliers"] = int((lab == CONSERVADO).sum())
    idx = np.nonzero(lab == CONSERVADO)[0]
    cinfo = {}
    if len(idx):
        c = lcfg["clusters"]
        Q = P[idx]
        down = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Q)).voxel_down_sample(c["voxel_mm"] * MM)
        D = np.asarray(down.points)
        cl = np.asarray(down.cluster_dbscan(c["eps_mm"] * MM, c["min_puntos"]))
        ids, cnt = np.unique(cl[cl >= 0], return_counts=True)
        good = []
        if len(ids):                                 # semilla + los grandes cercanos a lo ya aceptado
            order = ids[np.argsort(-cnt)]
            big = set(ids[cnt >= c["min_fraccion"] * len(D)].tolist())
            # semilla: el mayor de los que TOCAN el piso (el objeto esta apoyado en el tablero); una
            # "sabana" de flotantes de 3DGS puede tener mas puntos que un objeto chico
            hD = (D @ n + d) / MM
            toca = [i for i in order if hD[cl == i].min() <= corte + c["contacto_mm"]]
            seed = toca[0] if toca else order[0]
            good = [seed]
            pend = [i for i in order if i in big and i != seed]
            changed = True
            while pend and changed:
                tree = cKDTree(D[np.isin(cl, good)])
                changed = False
                for i in list(pend):
                    if tree.query(D[cl == i], distance_upper_bound=c["union_mm"] * MM)[0].min() < np.inf:
                        good.append(i)
                        pend.remove(i)
                        changed = True
                        break
            good = np.array(good)
        nearest = cKDTree(D).query(Q)[1]
        lab[idx[~np.isin(cl, good)[nearest]]] = FLOTANTE
        cinfo = dict(clusters=int(len(ids)), conservados=int(len(good)),
                     tamanos=sorted((int(x) for x in cnt), reverse=True)[:5])
    pasos["final"] = int((lab == CONSERVADO).sum())
    info = dict(pasos=pasos, plano=pinfo, plano_n_d=[*map(float, n), float(d)], corte_mm=corte, clusters=cinfo)
    if etiquetas:
        return P[lab == CONSERVADO], info, (P, lab)
    return P[lab == CONSERVADO], info


def describe(info):
    p, pl = info["pasos"], info["plano"]
    tot = max(p["roi"], 1)
    plano = (f"corte {info['corte_mm']:.1f} mm (grosor piso {pl['grosor_mm']:.1f}), plano {pl['metodo']}" + (f" (incl {pl['inclinacion_deg']:.2f} deg, z0 {pl['desplazamiento_mm']:+.2f} mm,"
                                       f" rms {pl['rms_mm']:.2f})" if pl["metodo"] == "ransac" else f" ({pl['motivo']})"))
    return (f"ROI {p['roi']} pts -> piso -{100 * (p['roi'] - p['sin_piso']) / tot:.1f}% "
            f"-> outliers -{100 * (p['sin_piso'] - p['sin_outliers']) / tot:.1f}% "
            f"-> clusters -{100 * (p['sin_outliers'] - p['final']) / tot:.2f}% = {p['final']} | {plano}")


def clean_technique(rdir, tec, roi, lcfg, o3d):
    """Limpia <tec>/resultado.ply -> <tec>/resultado_limpio.ply. None si no hay resultado."""
    f = rdir / tec / "resultado.ply"
    if not f.exists():
        return None, None
    P, info = clean_cloud(np.asarray(o3d.io.read_point_cloud(str(f)).points), roi, lcfg, o3d)
    o3d.io.write_point_cloud(str(rdir / tec / "resultado_limpio.ply"),
                             o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)))
    return P, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recon_dir")
    ap.add_argument("--tecnicas", nargs="*", default=TECNICAS)
    args = ap.parse_args()
    import open3d as o3d

    rdir = repo_path(args.recon_dir)
    lcfg = load_recon_config()["limpieza"]
    roi = load_entrada(rdir)["roi"]
    res = {}
    for tec in args.tecnicas:
        _, info = clean_technique(rdir, tec, roi, lcfg, o3d)
        if info:
            res[tec] = info
            print(f"  {tec:13s} {describe(info)}")
    print(json.dumps({t: r["pasos"] for t, r in res.items()}))


if __name__ == "__main__":
    main()
