"""Etapa 1 de la captura sintetica (Isaac Sim, headless).

    isaac-python sim/capture.py gemelo Captura_real/bunny_x200
    isaac-python sim/capture.py modelo ~/Escritorio/3D_reconstruction/3D_Models/sneaker.usdz
    isaac-python sim/capture.py modelo bunny --semilla 3 --nombre bunny_s3 --tamano-mm 150

gemelo: reproduce una captura real (sus poses exactas; el objeto sale de su session.json).
modelo: cualquier objeto (clave de configs/objects.yaml o ruta .usd/.usdz) con la plantilla
        (twin.yaml: plantilla) y variabilidad aleatoria: camaras re-montadas una vez por captura
        y giro ruidoso de la base en cada parada (sim/poses.py).

Las camaras quedan fijas y gira la base (hoja, carton, objeto). Escribe en
<salida>/<gt_dir>/ (por defecto Captura_sintetica/<nombre>/gt/):
    <camara>/stop_XX/color.png       render limpio (BGR, misma lente/distorsion que la real)
    <camara>/stop_XX/depth.png       profundidad PERFECTA, uint16, escala depth_scale_m (0 = nada)
    <camara>/stop_XX/mask.png        mascara del objeto en la imagen de color (0/255)
    <camara>/stop_XX/mask_depth.png  mascara del objeto en la imagen de profundidad
    objeto_rev.ply                   malla GT en el marco de revision/ (mm)
y <salida>/ground_truth.json. La etapa 2 (sim/write_capture.py, OpenCV de la captura) arma el
formato real (calib/meta/session/rig, depth_XX.png ruidosas) a partir de esto.
"""
import argparse
import sys
import time
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gemelo", help="reproducir una captura real")
    g.add_argument("captura", help="carpeta de la captura real (p. ej. Captura_real/bunny_x200)")
    m = sub.add_parser("modelo", help="cualquier USD con la plantilla y variabilidad aleatoria")
    m.add_argument("modelo", help="clave de configs/objects.yaml o ruta a .usd/.usdz")
    m.add_argument("--nombre", help="nombre de la captura (por defecto el del modelo)")
    m.add_argument("--semilla", type=int, help="por defecto twin.yaml o hash del nombre")
    m.add_argument("--plantilla", help="captura real plantilla (por defecto twin.yaml)")
    m.add_argument("--material", help="material de scene.yaml (p. ej. pla_gris) en lugar del del USD")
    size = m.add_mutually_exclusive_group()
    size.add_argument("--tamano-mm", type=float, help="dimension mayor del objeto en la escena")
    size.add_argument("--escala", type=float, help="factor sobre el tamano real del USD")
    size.add_argument("--tamano-real", action="store_true", help="sin normalizar")
    for p in (g, m):
        p.add_argument("--config", default="twin")
        p.add_argument("--camaras", nargs="*", help="sobrescribe 'camaras' del config")
        p.add_argument("--paradas", nargs="*", type=int, help="sobrescribe 'paradas' del config")
        p.add_argument("--sin-objeto", action="store_true")
        p.add_argument("--sin-distorsion", action="store_true")
        p.add_argument("--salida", help="por defecto <synthetic_root>/<nombre>")
        p.add_argument("--set", action="append", default=[], metavar="clave.sub=valor",
                       help="sobrescribe configs/scene.yaml, p. ej. --set render.aa_op=0")
        p.add_argument("--solo-ruta", action="store_true",
                       help="imprime la carpeta de salida y termina (sin Isaac)")
    return ap.parse_args(argv)


def apply_overrides(cfg, items):
    import yaml

    for it in items:
        key, val = it.split("=", 1)
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d[p]
        if parts[-1] not in d:
            raise KeyError(f"--set: '{key}' no existe en la configuracion")
        d[parts[-1]] = yaml.safe_load(val)
    return cfg


def resolve_job(args, common, cfg):
    """Que se captura, con que plantilla, modo, semilla y donde se escribe (sin Isaac)."""
    from common.config import object_config, repo_path
    from common.realdata import Capture, read_json

    lay = common["capture_layout"]
    if args.cmd == "gemelo":
        template = repo_path(args.captura)
        obj = object_config(cfg["objeto"] or read_json(template / lay["session_file"])["objeto"])
        name, mode, seed = template.name, "replay", None
    else:
        template = repo_path(args.plantilla or cfg["plantilla"])
        obj = object_config(args.modelo)
        if args.tamano_mm is not None:
            obj.update(tamano_mm=args.tamano_mm, escala=None)
        elif args.escala is not None:
            obj.update(escala=args.escala, tamano_mm=None)
        elif args.tamano_real:
            obj.update(escala=None, tamano_mm=None)
        if args.material:
            obj["material"] = args.material
        name, mode = args.nombre or obj["name"], "aleatorio"
        seed = args.semilla if args.semilla is not None else cfg["aleatorio"]["semilla"]
        if seed is None:
            seed = zlib.crc32(name.encode("utf-8"))
    obj["usd"] = str(repo_path(obj["usd"]))
    out = repo_path(args.salida) if args.salida else (
        repo_path(common["paths"]["synthetic_root"]) / (name + cfg["sufijo_modo"][mode]))
    return dict(template=Capture.open(template, lay), obj=obj, name=name, mode=mode, seed=seed, out=out)


def object_mask(seg, info, prim_prefix):
    """Mascara (bool) de los pixeles cuyo prim esta bajo prim_prefix, desde instance_id_segmentation."""
    import numpy as np

    ids = [int(i) for i, p in (info.get("idToLabels") or {}).items() if str(p).startswith(prim_prefix)]
    seg = np.asarray(seg).reshape(seg.shape[:2])
    return np.isin(seg, ids) if ids else np.zeros(seg.shape, bool)


def main(argv=None):
    args = parse_args(argv)
    from common.config import load_config

    common = load_config("common")
    scene_cfg = apply_overrides(load_config("scene"), args.set)
    cfg = load_config(args.config)
    job = resolve_job(args, common, cfg)
    if args.solo_ruta:
        print(job["out"])
        return

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "renderer": scene_cfg["render"]["renderer"]})
    try:
        run(args, common, scene_cfg, cfg, job)
    finally:
        app.close()


def make_plan(job, cams, stops, cfg):
    import numpy as np

    from common.geometry import fit_turntable_axis
    from sim.poses import random_turntable, replay_turntable

    tpl = job["template"]
    if job["mode"] == "replay":
        all_stops = sorted(set().union(*(tpl.stops[c] for c in tpl.cameras)))
        full = replay_turntable(tpl, cams, all_stops)     # eje con TODAS las paradas reales
        axis_xy, axis_res = fit_turntable_axis([full.T_b0_from_bk[k] for k in full.stops])
        plan = replay_turntable(tpl, cams, stops if stops is not None else all_stops)
        print(f"gemelo: parada base {plan.base_stop}, eje ({axis_xy[0]:.1f}, {axis_xy[1]:.1f}) mm "
              f"(residuo {axis_res:.2f} mm), rig vs poses reales <= {plan.max_dev_mm:.2e} mm")
        return plan, axis_xy, dict(eje_residuo_mm=axis_res)
    plan = random_turntable(tpl, cams, cfg["aleatorio"], np.random.default_rng(job["seed"]))
    if stops is not None:
        plan.stops = [k for k in plan.stops if k in stops]
    point, axis = plan.axis
    print(f"aleatorio (semilla {job['seed']}): eje ({point[0]:.1f}, {point[1]:.1f}) mm, "
          f"inclinacion {np.degrees(np.arccos(abs(axis[2]))):.2f} deg, giros "
          f"{[round(plan.angles_deg[k], 1) for k in plan.stops]}")
    return plan, point[:2], dict(eje_direccion=axis, angulos_deg=plan.angles_deg)


def run(args, common, scene_cfg, cfg, job):
    import cv2
    import numpy as np
    import isaacsim.core.experimental.utils.app as app_utils

    from common.board import L
    from common.config import repo_path
    from common.geometry import board_to_rev, place_on_board, transform_points, world_from_board
    from common.meshio import write_mesh_ply
    from sim.cameras import SimCamera
    from sim.depth_noise import to_z16
    from sim.objects import MeshData, add_referenced_object, prepare_object
    from sim.scene import OBJECT, add_object, build_scene, set_turntable

    tpl, obj, out = job["template"], job["obj"], job["out"]
    lay = common["capture_layout"]
    scale = common["conventions"]["world_scale_from_board"]
    cams = args.camaras or (tpl.cameras if cfg["camaras"] == "all" else cfg["camaras"])
    stops = args.paradas if args.paradas is not None else (None if cfg["paradas"] == "all" else cfg["paradas"])
    include_obj = cfg["incluir_objeto"] and not args.sin_objeto
    distortion = cfg["distorsion_color"] and not args.sin_distorsion
    plan, axis_xy, plan_info = make_plan(job, cams, stops, cfg)

    stage, _ = build_scene(scene_cfg, scale)
    gt_obj = None
    if include_obj:
        pl = obj["placement"]
        if pl["center_on"] != "turntable_axis" or not pl["bottom_on_board"]:
            raise NotImplementedError(f"placement no soportado: {pl}")
        prep = prepare_object(obj)
        T_bo = place_on_board(prep.points_mm, axis_xy, pl["yaw_deg"])
        mat_key = obj["material"] or (None if prep.has_materials else obj["material_por_defecto"])
        if mat_key:          # material de config: se copia la malla (sin materiales del USD)
            normals = None if prep.mesh.normals is None else prep.mesh.normals @ prep.U.T
            mesh_mm = MeshData(prep.points_mm, prep.mesh.counts, prep.mesh.indices, normals,
                               prep.mesh.normals_interp)
            add_object(stage, mesh_mm, T_bo, scale, scene_cfg["materials"][mat_key], obj["name"])
        else:                # materiales/texturas del USD: se referencia
            add_referenced_object(stage, OBJECT, prep, world_from_board(scale), T_bo,
                                  repo_path(cfg["cache_modelos"]))
        gt_obj = (prep, T_bo, mat_key)
        print(f"objeto {obj['name']}: {prep.info['criterio_tamano']} escala {prep.info['escala']:.4g}, "
              f"extension {np.round(prep.info['extension_mm'], 1)} mm, "
              f"material {mat_key or 'del USD'}")

    import isaacsim.core.experimental.utils.stage as stage_utils

    n_wait = 0
    while stage_utils.is_stage_loading():     # referencias (USDZ, texturas) se cargan en async
        app_utils.update_app()
        n_wait += 1
    print(f"stage cargado ({n_wait} actualizaciones de espera)")
    rcfg = scene_cfg["render"]
    ann = cfg["anotadores"]
    sims = {}
    for c in cams:
        cal = tpl.calib[c]
        sims[c, "color"] = SimCamera(f"/World/Cameras/{c}_color", cal["color_intrinsics"], ann["color"],
                                     rcfg["clipping_m"], distortion, rcfg["opencv_pixel_offset_px"])
        sims[c, "depth"] = SimCamera(f"/World/Cameras/{c}_depth", cal["depth_intrinsics"],
                                     ann["profundidad"], rcfg["clipping_m"], True,
                                     rcfg["opencv_pixel_offset_px"])
        sims[c, "color"].set_pose(plan.T_color_from_b0[c], scale)
        sims[c, "depth"].set_pose(plan.T_depth_from_b0[c], scale)

    gt_root = out / cfg["gt_dir"]
    views = []
    app_utils.play(commit=True)
    t0 = time.time()
    for k in plan.stops:
        set_turntable(stage, plan.T_b0_from_bk[k], scale)
        app_utils.update_app(steps=rcfg["settle_frames"])
        for c in cams:
            data = {}
            for kind, name in (("color", "rgb"), ("depth", "distance_to_image_plane")):
                sc = sims[c, kind]
                while True:
                    d, _ = sc.sensor.get_data(name)
                    if d is not None:
                        break
                    app_utils.update_app()
                data[kind] = d.numpy()
                seg_name = "instance_id_segmentation"
                if seg_name in ann["color" if kind == "color" else "profundidad"]:
                    s, info = sc.sensor.get_data(seg_name)
                    data[kind + "_mask"] = object_mask(s.numpy(), info, OBJECT)
            d = gt_root / c / lay["stop_dir"].format(k=k)
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / "color.png"), cv2.cvtColor(data["color"], cv2.COLOR_RGB2BGR))
            z = np.asarray(data["depth"], np.float64).reshape(data["depth"].shape[:2])
            cv2.imwrite(str(d / "depth.png"), to_z16(z, tpl.calib[c]["depth_scale_m"]))
            for kind, fname in (("color_mask", "mask.png"), ("depth_mask", "mask_depth.png")):
                if kind in data:
                    cv2.imwrite(str(d / fname), data[kind].astype(np.uint8) * 255)
            views.append(dict(camara=c, parada=k,
                              T_color_from_board_mm=plan.T_color_from_board(c, k),
                              T_depth_from_board_mm=plan.T_depth_from_board(c, k),
                              objeto_px=int(data.get("color_mask", np.zeros(1)).sum())))
            print(f"  {c}/stop_{k:02d}: rgb media {data['color'].mean():.0f}, prof. valida "
                  f"{(z > 0).mean() * 100:.0f} %, objeto {views[-1]['objeto_px']} px")
    t_render = time.time() - t0
    if include_obj and views and not any(v["objeto_px"] for v in views):
        raise RuntimeError("el objeto no aparece en ninguna vista (¿fisica, transformacion o "
                           "material del USD?); revisar con --camaras media --paradas 0")

    gt = dict(
        version=2, nombre=job["name"], plantilla=str(Path(tpl.root).relative_to(repo_path("."))),
        modo=job["mode"], semilla=job["seed"], camaras=cams, paradas=plan.stops,
        paradas_sin_pose=plan.missing, distorsion_color=distortion, render_s=round(t_render, 1),
        convenciones=dict(
            poses="T_x<-board en mm: p_x = R @ p_board + t (igual que meta.json)",
            depth_png="uint16, escala calib.json depth_scale_m, 0 = sin superficie (perfecta)",
            malla_gt="marco de revision/: p_rev = diag(revision_flip) @ p_board, mm",
            pixel="OpenCV: centro del pixel (0,0) en (0,0)"),
        base=dict(parada_referencia=plan.base_stop, eje_xy_mm=axis_xy, **plan_info,
                  T_b0_from_bk_mm={str(k): plan.T_b0_from_bk[k] for k in plan.stops}),
        camaras_fijas={c: dict(T_color_from_b0_mm=plan.T_color_from_b0[c],
                               T_depth_from_b0_mm=plan.T_depth_from_b0[c]) for c in cams},
        vistas=views, escena=scene_cfg)
    if gt_obj is not None:
        prep, T_bo, mat_key = gt_obj
        P_rev = board_to_rev(transform_points(T_bo, prep.points_mm), common["conventions"]["revision_flip"])
        tri = prep.mesh.counts == 3
        if not tri.all():
            raise NotImplementedError("malla GT: solo triangulos (triangular el USD)")
        idx = prep.mesh.indices.reshape(-1, 3)[:, ::-1].ravel()    # el reflejo invierte la orientacion
        gt_root.mkdir(parents=True, exist_ok=True)
        write_mesh_ply(gt_root / "objeto_rev.ply", P_rev, prep.mesh.counts, idx)
        lo, hi = P_rev.min(0), P_rev.max(0)
        gt["objeto"] = dict(nombre=obj["name"], **prep.info, material=mat_key or "del USD",
                            T_board_from_objeto_mm=T_bo, malla_rev=f"{cfg['gt_dir']}/objeto_rev.ply",
                            bbox_rev_mm=[lo, hi])
    else:
        gt["objeto"] = None
    L.save_json(out / cfg["gt_file"], gt)
    print(f"{len(views)} vistas en {t_render:.1f} s -> {gt_root}")


if __name__ == "__main__":
    main()
