"""Resumen compacto de uno o varios .ply (nube o malla). Solo lectura.

    tools/inspect/run.sh inspect_ply.py Captura_real/bunny_x200/revision/*.ply

Imprime: puntos/caras, bounding box, extension, centroide, espaciado medio entre vecinos
(escala estimada), colores/normales y una suposicion de unidades.
"""
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

from _out import Out, add_max_lines, fmt3

NN_SAMPLE = 2000          # puntos muestreados para estimar el espaciado entre vecinos
SEED = 0


def guess_units(extent_max):
    if extent_max < 5:
        return "m?"
    if extent_max < 500:
        return "mm? (o cm)"
    return "mm?"


def summarize(path, out, one_line):
    path = Path(path)
    mesh = o3d.io.read_triangle_mesh(str(path))
    is_mesh = len(mesh.triangles) > 0
    if is_mesh:
        pts = np.asarray(mesh.vertices)
        has_c, has_n = mesh.has_vertex_colors(), mesh.has_vertex_normals()
        n_faces = len(mesh.triangles)
    else:
        pc = o3d.io.read_point_cloud(str(path))
        pts = np.asarray(pc.points)
        has_c, has_n = pc.has_colors(), pc.has_normals()
        n_faces = 0
    if len(pts) == 0:
        out(f"{path.name}: VACIO")
        return
    lo, hi = pts.min(0), pts.max(0)
    ext = hi - lo
    cen = pts.mean(0)
    rng = np.random.default_rng(SEED)
    pc_tmp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    tree = o3d.geometry.KDTreeFlann(pc_tmp)
    idx = rng.choice(len(pts), min(NN_SAMPLE, len(pts)), replace=False)
    nn = [np.sqrt(tree.search_knn_vector_3d(pts[i], 2)[2][1]) for i in idx]
    spacing = float(np.median(nn))
    kind = f"malla {n_faces} caras" if is_mesh else "nube"
    if one_line:
        out(f"{path.name:28s} {kind:18s} n={len(pts):8d} ext={fmt3(ext)} cen={fmt3(cen)} "
            f"esp={spacing:.3g} c={'s' if has_c else 'n'} nor={'s' if has_n else 'n'}")
        return
    out(f"== {path.name} ({path.stat().st_size / 1e6:.1f} MB): {kind}, {len(pts)} vertices/puntos")
    out(f"   bbox min {fmt3(lo)}  max {fmt3(hi)}")
    out(f"   extension {fmt3(ext)}  diag {np.linalg.norm(ext):.4g}  -> unidades {guess_units(ext.max())}")
    out(f"   centroide {fmt3(cen)}  centro bbox {fmt3((lo + hi) / 2)}")
    out(f"   espaciado mediano vecino {spacing:.4g}  colores {has_c}  normales {has_n}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--compact", action="store_true", help="una linea por archivo")
    add_max_lines(ap)
    args = ap.parse_args()
    out = Out(args.max_lines)
    one_line = args.compact or len(args.files) > 5
    for f in args.files:
        summarize(f, out, one_line)
    out.close()


if __name__ == "__main__":
    main()
