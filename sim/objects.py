"""Objeto a reconstruir: se copia la malla del USD de Objects/ a la escena (sin referenciarlo).

Los USD de Objects/ declaran metersPerUnit=0.01 / upAxis=Y, pero su geometria esta en mm con Z
arriba. Referenciarlos dejaria que Kit "corrija" unidades con ese metadato erroneo; copiando
puntos/caras/normales y poniendo la transformacion explicita se evita cualquier ambiguedad.
"""
from dataclasses import dataclass

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdShade, Vt


@dataclass
class MeshData:
    points: np.ndarray            # (N, 3) en unidades del archivo (mm reales), marco local
    counts: np.ndarray
    indices: np.ndarray
    normals: np.ndarray = None
    normals_interp: str = None


def fan_triangulate(counts):
    """Indices (en la lista face-vertex) de la triangulacion en abanico de poligonos convexos."""
    out, start = [], 0
    for n in counts:
        n = int(n)
        for i in range(1, n - 1):
            out += [start, start + i, start + i + 1]
        start += n
    return np.array(out, np.int64)


def load_mesh(usd_path):
    """Une todas las mallas del USD (trianguladas) aplicando sus transformaciones (mismo archivo)."""
    stage = Usd.Stage.Open(str(usd_path))
    cache = UsdGeom.XformCache()
    pts, cnt, idx, nrm, interp, off = [], [], [], [], None, 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if UsdGeom.Imageable(prim).ComputePurpose() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
            continue
        m = UsdGeom.Mesh(prim)
        M = np.array(cache.GetLocalToWorldTransform(prim), float).T      # USD fila -> columna
        P = np.array(m.GetPointsAttr().Get(), float)
        pts.append(P @ M[:3, :3].T + M[:3, 3])
        counts = np.array(m.GetFaceVertexCountsAttr().Get(), np.int64)
        fv = np.array(m.GetFaceVertexIndicesAttr().Get(), np.int64)
        tri_fv = fan_triangulate(counts)                 # indices dentro de la lista face-vertex
        cnt.append(np.full(len(tri_fv) // 3, 3, np.int32))
        idx.append(fv[tri_fv] + off)
        off += len(P)
        N = m.GetNormalsAttr().Get()
        ni = m.GetNormalsInterpolation()
        if N is not None and len(N) and ni in (UsdGeom.Tokens.faceVarying, UsdGeom.Tokens.vertex):
            Rn = np.linalg.inv(M[:3, :3]).T
            Nn = np.array(N, float) @ Rn.T
            Nn /= np.linalg.norm(Nn, axis=1, keepdims=True).clip(1e-12)
            nrm.append(Nn[tri_fv] if ni == UsdGeom.Tokens.faceVarying else Nn)
            if interp not in (None, ni):
                nrm.append(None)                         # mezcla de interpolaciones: se descartan
            interp = ni
    if not pts:
        raise ValueError(f"{usd_path}: no hay mallas")
    normals = np.vstack(nrm) if len(nrm) == len(pts) and all(n is not None for n in nrm) else None
    return MeshData(np.vstack(pts), np.concatenate(cnt), np.concatenate(idx), normals,
                    interp if normals is not None else None)


def add_object_mesh(stage, path, mesh, T_parent_from_local, material):
    """Crea la malla en `path` con una transformacion 4x4 (puede incluir escala) hacia su padre."""
    um = UsdGeom.Mesh.Define(stage, path)
    um.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(mesh.points.astype(np.float32)))
    um.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(mesh.counts.astype(np.int32)))
    um.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(mesh.indices.astype(np.int32)))
    if mesh.normals is not None:
        um.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(mesh.normals.astype(np.float32)))
        um.SetNormalsInterpolation(mesh.normals_interp)
    um.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    lo, hi = mesh.points.min(0), mesh.points.max(0)
    um.CreateExtentAttr([Gf.Vec3f(*lo), Gf.Vec3f(*hi)])
    UsdGeom.Xformable(um).AddTransformOp().Set(Gf.Matrix4d(np.asarray(T_parent_from_local).T.tolist()))
    UsdShade.MaterialBindingAPI.Apply(um.GetPrim()).Bind(material)
    return um


# Ejes: lleva un USD con Y arriba a Z arriba (x, y, z) -> (x, -z, y). Convencion, no parametro.
Z_UP_FROM_Y_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


@dataclass
class PreparedObject:
    """Geometria del objeto en mm, Z arriba, ya escalada. p_mm = mm_per_unit * U @ p_asset."""
    usd_path: str
    mesh: MeshData                # puntos en unidades del archivo
    U: np.ndarray                 # 3x3 a Z arriba
    mm_per_unit: float            # incluye unidades reales y escala/normalizacion
    points_mm: np.ndarray
    has_materials: bool
    info: dict


def _has_bound_materials(stage):
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Material):
            return True
    return False


def prepare_object(obj):
    """Lee el USD, decide unidades/eje vertical/escala y devuelve la geometria en mm."""
    path = str(obj["usd"])
    stage = Usd.Stage.Open(path)
    mpu = obj["real_units_m"] or UsdGeom.GetStageMetersPerUnit(stage)
    up = (obj["up_axis"] or UsdGeom.GetStageUpAxis(stage)).upper()
    if up not in ("Y", "Z"):
        raise ValueError(f"{path}: upAxis {up} no soportado")
    U = np.eye(3) if up == "Z" else Z_UP_FROM_Y_UP
    mesh = load_mesh(path)
    P = mesh.points @ U.T * (mpu * 1000.0)              # mm reales
    ext = P.max(0) - P.min(0)
    if obj["escala"] is not None:
        s, criterio = float(obj["escala"]), "escala"
    elif obj["tamano_mm"] is not None:
        s, criterio = float(obj["tamano_mm"]) / float(ext.max()), "tamano_mm"
    else:
        s, criterio = 1.0, "real"
    k = mpu * 1000.0 * s
    info = dict(usd=path, metros_por_unidad=mpu, eje_vertical=up, escala=s, criterio_tamano=criterio,
                extension_real_mm=ext, extension_mm=ext * s)
    return PreparedObject(path, mesh, U, k, P * s, _has_bound_materials(stage), info)


def add_referenced_object(stage, path, prep, T_parent_from_board, T_board_from_obj_mm, cache_dir):
    """Referencia el USD (conserva materiales/texturas) con la transformacion completa.

    Prim `path` (Xform) = T_parent<-board @ T_board<-obj @ [k * U]; el asset (version aplanada y
    limpia, ver flatten_asset) cuelga de path/Asset.
    Verifica con el bbox compuesto que nada (p. ej. correccion automatica de unidades de Kit) cambie
    la escala; si cambia, falla.
    """
    A = np.eye(4)
    A[:3, :3] = prep.mm_per_unit * prep.U
    M = np.asarray(T_parent_from_board) @ np.asarray(T_board_from_obj_mm) @ A
    xf = UsdGeom.Xform.Define(stage, path)
    xf.AddTransformOp().Set(Gf.Matrix4d(M.T.tolist()))
    asset = stage.DefinePrim(f"{path}/Asset", "Xform")
    asset.GetReferences().AddReference(str(flatten_asset(prep.usd_path, cache_dir)))
    rng = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                            useExtentsHint=False).ComputeWorldBound(xf.GetPrim()).ComputeAlignedRange()
    lo, hi = np.array(rng.GetMin()), np.array(rng.GetMax())
    P = prep.points_mm @ np.asarray(T_board_from_obj_mm)[:3, :3].T + np.asarray(T_board_from_obj_mm)[:3, 3]
    Wb = np.asarray(T_parent_from_board)
    Pw = P @ Wb[:3, :3].T + Wb[:3, 3]
    elo, ehi = Pw.min(0), Pw.max(0)
    if not (np.allclose(lo, elo, atol=2e-3) and np.allclose(hi, ehi, atol=2e-3)):
        raise RuntimeError(f"{prep.usd_path}: bbox referenciado [{lo}, {hi}] != esperado [{elo}, {ehi}] "
                           "(¿correccion automatica de unidades o transformaciones fuera del defaultPrim?)")
    return xf


def flatten_asset(usd_path, cache_dir):
    """Version plana, limpia y autocontenida del USD para referenciar en la escena.

    - aplana el stage (USDZ anidados incluidos) a un .usdc propio; las texturas dentro de paquetes
      se extraen a archivos normales junto a el;
    - hornea las pilas de xformOps (bake_xforms) y quita la fisica (strip_physics: un
      PhysicsRigidBodyAPI hacia caer el objeto al hacer play()).
    Cache por ruta y fecha del original.
    """
    from pathlib import Path

    from pxr import Ar, Sdf

    src = Path(usd_path)
    out_dir = Path(cache_dir) / src.stem
    out = out_dir / f"{src.stem}.usdc"
    stamp = out_dir / "origen.txt"
    key = f"{src.resolve()} {src.stat().st_mtime_ns}"
    if out.exists() and stamp.exists() and stamp.read_text() == key:
        return out
    out_dir.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.Open(str(src))
    layer = stage.Flatten(addSourceFileComment=False)
    flat = Usd.Stage.Open(layer)
    resolver = Ar.GetResolver()
    tex_dir = out_dir / "texturas"
    done = {}
    for prim in flat.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            val = attr.Get()
            if not val or not val.path:
                continue
            res = val.resolvedPath or str(resolver.Resolve(val.path))
            if "[" not in res:            # archivo normal: se deja
                continue
            if res not in done:
                name = res.rstrip("]").split("[")[-1].split("/")[-1]
                dst = tex_dir / f"{len(done):03d}_{name}"
                dst.parent.mkdir(parents=True, exist_ok=True)
                asset = resolver.OpenAsset(Ar.ResolvedPath(res))
                dst.write_bytes(bytes(memoryview(asset.GetBuffer())[:asset.GetSize()]))
                done[res] = dst
            attr.Set(Sdf.AssetPath(str(done[res])))
    bake_xforms(flat)
    strip_physics(flat)
    layer.Export(str(out))
    stamp.write_text(key)
    return out


def bake_xforms(stage):
    """Reemplaza cada pila de xformOps por un unico xformOp:transform (la matriz local de pxr).

    Assets pasados por el Metrics Assembler de Omniverse traen ops "xformOp:*:unitsResolve" que
    pxr aplica pero Kit/RTX trata aparte: el objeto aparecia con otra escala/orientacion (o no
    aparecia). Con la matriz horneada render y GT usan exactamente la misma transformacion.
    """
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Xformable):
            continue
        x = UsdGeom.Xformable(prim)
        ops = x.GetOrderedXformOps()
        if not ops:
            continue
        if any(o.GetNumTimeSamples() > 1 for o in ops):
            raise NotImplementedError(f"{prim.GetPath()}: transformaciones animadas")
        M = x.GetLocalTransformation(Usd.TimeCode.Default())
        x.ClearXformOpOrder()
        for o in ops:
            prim.RemoveProperty(o.GetAttr().GetName())
        x.AddTransformOp().Set(M)


PHYSICS_PREFIXES = ("Physics", "Physx", "PhysX")
PHYSICS_ATTR_PREFIXES = ("physics:", "physx", "physxCollision:", "physxRigidBody:")


def strip_physics(stage):
    """Quita APIs y atributos de fisica: con play() PhysX moveria el objeto (cuerpo rigido que cae)
    y el GT dejaria de valer. El objeto del gemelo es estatico, pegado a la base."""
    n = 0
    for prim in stage.Traverse():
        for sch in list(prim.GetAppliedSchemas()):
            if sch.startswith(PHYSICS_PREFIXES):
                prim.RemoveAppliedSchema(sch)
                n += 1
        for a in prim.GetAuthoredProperties():
            if a.GetName().startswith(PHYSICS_ATTR_PREFIXES):
                prim.RemoveProperty(a.GetName())
    return n
