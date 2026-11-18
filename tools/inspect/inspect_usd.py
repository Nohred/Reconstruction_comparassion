"""Resumen compacto de un .usd/.usdc/.usda. Solo lectura (no guarda el stage).

    tools/inspect/run.sh inspect_usd.py Objects/stanford_bunny.usd

Imprime metersPerUnit, upAxis, defaultPrim, jerarquia resumida, bounding box (unidades del stage
y metros), transformacion del prim raiz y donde queda el origen respecto a la base del objeto.
"""
import argparse

import numpy as np
from pxr import Gf, Usd, UsdGeom

from _out import Out, add_max_lines, fmt3

MAX_DEPTH = 3          # profundidad de la jerarquia a mostrar
MAX_CHILDREN = 4       # hijos por prim a mostrar


def hierarchy(prim, out, depth=0, max_depth=MAX_DEPTH):
    for i, child in enumerate(prim.GetChildren()):
        if i >= MAX_CHILDREN:
            out(f"{'  ' * (depth + 1)}... (+{len(prim.GetChildren()) - MAX_CHILDREN} hijos)")
            break
        extra = ""
        if child.IsA(UsdGeom.Mesh):
            m = UsdGeom.Mesh(child)
            pts = m.GetPointsAttr().Get()
            fc = m.GetFaceVertexCountsAttr().Get()
            extra = f" [{0 if pts is None else len(pts)} pts, {0 if fc is None else len(fc)} caras]"
        xf = UsdGeom.Xformable(child) if child.IsA(UsdGeom.Xformable) else None
        if xf is not None and xf.GetOrderedXformOps():
            extra += " ops=" + ",".join(op.GetOpName().replace("xformOp:", "")
                                        for op in xf.GetOrderedXformOps())
        out(f"{'  ' * (depth + 1)}{child.GetName()} <{child.GetTypeName()}>{extra}")
        if depth + 1 < max_depth:
            hierarchy(child, out, depth + 1, max_depth)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--depth", type=int, default=MAX_DEPTH, help="profundidad de jerarquia")
    add_max_lines(ap)
    args = ap.parse_args()
    out = Out(args.max_lines)

    stage = Usd.Stage.Open(args.file, Usd.Stage.LoadAll)
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    has_mpu = stage.HasAuthoredMetadata(UsdGeom.Tokens.metersPerUnit)
    up = UsdGeom.GetStageUpAxis(stage)
    dp = stage.GetDefaultPrim()
    out(f"== {args.file}")
    out(f"   metersPerUnit {mpu} ({'autorado' if has_mpu else 'NO autorado: default'}), "
        f"upAxis {up}, defaultPrim {dp.GetPath() if dp else None}")
    n_mesh = n_pts = 0
    for p in stage.Traverse():
        if p.IsA(UsdGeom.Mesh):
            n_mesh += 1
            pts = UsdGeom.Mesh(p).GetPointsAttr().Get()
            n_pts += 0 if pts is None else len(pts)
    out(f"   mallas {n_mesh}, puntos totales {n_pts}")

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    root = dp if dp else stage.GetPseudoRoot()
    box = cache.ComputeWorldBound(root).ComputeAlignedRange()
    if box.IsEmpty():
        out("   bbox vacio")
    else:
        lo, hi = np.array(box.GetMin()), np.array(box.GetMax())
        ext = hi - lo
        out(f"   bbox unidades: min {fmt3(lo)} max {fmt3(hi)}")
        out(f"   extension: {fmt3(ext)} u = {fmt3(ext * mpu)} m  (diag {np.linalg.norm(ext) * mpu:.4g} m)")
        ax = "XYZ".index(up)
        others = [i for i in range(3) if i != ax]
        c = (lo + hi) / 2
        out(f"   origen vs base: min_{up}={lo[ax] * mpu:+.4f} m (0 => origen en la base); "
            f"centro horiz. bbox ({c[others[0]] * mpu:+.4f}, {c[others[1]] * mpu:+.4f}) m")

    if dp and dp.IsA(UsdGeom.Xformable):
        xf = UsdGeom.Xformable(dp)
        M = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = M.ExtractTranslation()
        s = [Gf.Vec3d(M.GetRow3(i)).GetLength() for i in range(3)]
        out(f"   raiz: ops {[op.GetOpName() for op in xf.GetOrderedXformOps()]}")
        out(f"   raiz: traslacion {fmt3(t)} escala~{fmt3(s)}")
    out("   jerarquia:")
    hierarchy(stage.GetPseudoRoot(), out, 0, args.depth)
    out.close()


if __name__ == "__main__":
    main()
