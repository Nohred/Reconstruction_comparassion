"""Construccion de la escena USD: hoja ChArUco texturizada, carton, mesa y luces.

Mundo de Isaac: metros, Z arriba; p_world = s * Rx(180) @ p_board (common.geometry).
Todo lo que gira con la base (hoja, carton, objeto) vive bajo TURNTABLE y se autora en la pose
de la parada base (hoja en z_world = 0). Por parada solo cambia la transformacion de TURNTABLE
(sim/poses.py); camaras, mesa y luces quedan fijas como en el laboratorio.
Solo se llama despues de crear SimulationApp.
"""
from pathlib import Path

import cv2
import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

from common.board import page_texture
from common.config import repo_path
from common.geometry import transform_points, world_from_board

LOOKS = "/World/Looks"
TURNTABLE = "/World/Turntable"
OBJECT = f"{TURNTABLE}/Object"


def configure_render(render_cfg):
    import carb

    s = carb.settings.get_settings()
    s.set("/rtx/post/tonemap/op", render_cfg["tonemap_op"])
    s.set("/rtx/post/tonemap/filmIso", render_cfg["film_iso"])
    s.set("/rtx/post/tonemap/whitepoint", render_cfg["whitepoint"])
    s.set("/rtx/post/tonemap/enabled", True)
    s.set("/rtx/post/aa/op", render_cfg["aa_op"])


def _preview_material(stage, name, color=None, roughness=0.5, texture=None):
    path = f"{LOOKS}/{name}"
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    if texture is None:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    else:
        st = UsdShade.Shader.Define(stage, f"{path}/StReader")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, f"{path}/Texture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture))
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _box(stage, path, center, size, material):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    return cube


def add_board_page(stage, page_cfg, scale):
    """Hoja carta texturizada en z_world = 0. Devuelve el rect de la hoja (mm, marco tablero)."""
    img, (x0, y0, x1, y1) = page_texture(page_cfg)
    tex_path = repo_path(page_cfg["texture_file"])
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(tex_path), img)
    corners_b = np.array([[x0, y0, 0], [x0, y1, 0], [x1, y1, 0], [x1, y0, 0]], float)
    pts = transform_points(world_from_board(scale), corners_b)      # orden CCW visto desde +Z
    # UV: s crece con x del tablero; t = 1 en la fila 0 de la imagen (y minima del tablero)
    uv = [((x - x0) / (x1 - x0), 1.0 - (y - y0) / (y1 - y0)) for x, y, _ in corners_b]
    mesh = UsdGeom.Mesh.Define(stage, f"{TURNTABLE}/Page")
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                  UsdGeom.Tokens.vertex)
    st.Set([Gf.Vec2f(*u) for u in uv])
    mat = _preview_material(stage, "ChArUcoPage", roughness=page_cfg["roughness"], texture=tex_path)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    return (x0, y0, x1, y1)


def add_cardboard_and_table(stage, scene_cfg, page_rect, scale):
    x0, y0, x1, y1 = page_rect
    cb = scene_cfg["cardboard"]
    if cb["enabled"]:
        mx, my = cb["margin_mm"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        c = transform_points(world_from_board(scale), [cx, cy, 0])[0]
        th = cb["thickness_mm"] * scale
        size = ((x1 - x0 + 2 * mx) * scale, (y1 - y0 + 2 * my) * scale, th)
        # tope del carton apenas bajo la hoja para evitar z-fighting
        center = (c[0], c[1], -th / 2 - 0.1 * scale)
        _box(stage, f"{TURNTABLE}/Cardboard", center, size,
             _preview_material(stage, "Cardboard", cb["color"], cb["roughness"]))
    tb = scene_cfg["table"]
    if tb["enabled"]:
        th = 0.02
        z = -tb["z_below_page_mm"] * scale - th / 2
        c = transform_points(world_from_board(scale), [(x0 + x1) / 2, (y0 + y1) / 2, 0])[0]
        _box(stage, "/World/Table", (c[0], c[1], z), (*tb["size_m"], th),
             _preview_material(stage, "Table", tb["color"], tb["roughness"]))


def add_lights(stage, lights_cfg):
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(lights_cfg["dome_intensity"])
    dome.CreateColorAttr(Gf.Vec3f(*lights_cfg["dome_color"]))
    # el domo ilumina pero no se ve como fondo (el laboratorio es un cuarto oscuro)
    dome.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(
        bool(lights_cfg["dome_visible"]))
    key = lights_cfg["key"]
    if key["type"] != "distant":
        raise NotImplementedError(key["type"])
    k = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
    k.CreateIntensityAttr(key["intensity"])
    k.CreateColorAttr(Gf.Vec3f(*key["color"]))
    UsdGeom.Xformable(k).AddRotateXYZOp().Set(Gf.Vec3f(*key["rotate_xyz_deg"]))


def build_scene(scene_cfg, scale):
    """Stage nuevo (m, Z arriba) con tablero, carton, mesa y luces. Devuelve (stage, page_rect)."""
    import isaacsim.core.experimental.utils.stage as stage_utils

    stage = stage_utils.create_new_stage()
    stage_utils.set_stage_units(meters_per_unit=1.0)
    stage_utils.set_stage_up_axis("Z")
    UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    configure_render(scene_cfg["render"])
    UsdGeom.Xform.Define(stage, TURNTABLE).AddTransformOp()
    rect = add_board_page(stage, scene_cfg["board_page"], scale)
    add_cardboard_and_table(stage, scene_cfg, rect, scale)
    add_lights(stage, scene_cfg["lights"])
    return stage, rect


def set_turntable(stage, T_b0_from_bk, scale):
    """Coloca la base en la parada k: M = W @ T_b0<-bk @ W^-1, con W = T_world<-board (escala)."""
    W = world_from_board(scale)
    M = W @ np.asarray(T_b0_from_bk, float) @ np.linalg.inv(W)
    op = UsdGeom.Xformable(stage.GetPrimAtPath(TURNTABLE)).GetOrderedXformOps()[0]
    op.Set(Gf.Matrix4d(M.T.tolist()))


def add_object(stage, mesh, T_board_from_local_mm, scale, mat_cfg, name):
    """Objeto bajo TURNTABLE: p_world = W @ T_board<-local @ p_local (unidades locales = mm)."""
    from sim.objects import add_object_mesh

    mat = _preview_material(stage, f"Obj_{name}", mat_cfg["color"], mat_cfg["roughness"])
    return add_object_mesh(stage, OBJECT, mesh, world_from_board(scale) @ T_board_from_local_mm, mat)


def texture_path(scene_cfg):
    return Path(repo_path(scene_cfg["board_page"]["texture_file"]))
