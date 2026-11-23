"""Tecnicas 1 y 2: NeRF (nerfacto) y Gaussian Splatting (splatfacto) con nerfstudio.

    tools/ns_py.sh -m recon.nerfstudio_tecnica nerf    Reconstrucciones/real/bunny_x200
    tools/ns_py.sh -m recon.nerfstudio_tecnica gaussian Reconstrucciones/real/bunny_x200

Entrena sobre entrada/transforms.json (poses e intrinsecos conocidos, mascaras ROI, todas las
imagenes en entrenamiento, sin optimizar camaras) y luego renderiza profundidad + color en las
MISMAS camaras de la captura y las fusiona con TSDF en el marco del tablero (igual que la tecnica
de profundidad): resultado.ply / malla.ply comparables entre tecnicas.
  - nerfacto: "depth" es distancia sobre el rayo -> z = d / |(x, y, 1)|.
  - splatfacto: "depth" ya es z de la camara. Ambos en unidades de nerfstudio -> / scale.
Salida: Reconstrucciones/<fuente>/<captura>/<nerf|gaussian>/ (ns/ = corrida de nerfstudio)
--solo-fusion: no entrena, reusa la ultima corrida.
"""
import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from common.config import load_config, repo_path
from recon.comun import TSDF, load_entrada, load_recon_config, roi_image_mask, write_result

TECNICAS = {"nerf": "nerf", "gaussian": "splat"}      # carpeta -> seccion de recon.yaml
RUN_NAME = "corrida"


def train(kind, rdir, rcfg, common, log):
    sec = rcfg[TECNICAS[kind]]
    ent = rdir / common["recon"]["entrada_dir"]
    ns_out = rdir / kind / "ns"
    shutil.rmtree(ns_out, ignore_errors=True)
    iters = rcfg["perfil"]["nerf_iteraciones" if kind == "nerf" else "splat_iteraciones"]
    cmd = ["ns-train", sec["metodo"], "--data", str(ent), "--output-dir", str(ns_out),
           "--experiment-name", RUN_NAME, "--timestamp", RUN_NAME, "--max-num-iterations", str(iters),
           "--viewer.quit-on-train-completion", "True", "--vis", "tensorboard",
           "--steps-per-save", str(iters)]
    if kind == "nerf":
        cmd += ["--pipeline.model.camera-optimizer.mode", sec["optimizar_camaras"]]
    else:
        cmd += ["--pipeline.model.random-init", str(not sec["init_puntos_colmap"])]
    cmd += [str(x) for x in sec["extra"]]
    cmd += ["nerfstudio-data", "--eval-mode", "all",
            "--load-3D-points", str(kind == "gaussian" and sec["init_puntos_colmap"])]
    with open(log, "a", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        t0 = time.time()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise SystemExit(f"fallo ns-train (ver {log})")
    return time.time() - t0, iters


def find_config(rdir, kind):
    cfgs = sorted((rdir / kind / "ns").glob(f"{RUN_NAME}/*/{RUN_NAME}/config.yml"))
    if not cfgs:
        raise SystemExit(f"no hay corrida de nerfstudio en {rdir / kind / 'ns'}")
    return cfgs[-1]


def render_and_fuse(kind, rdir, rcfg, entrada):
    import torch
    from nerfstudio.utils.eval_utils import eval_setup

    sec = rcfg[TECNICAS[kind]]
    cfg_path = find_config(rdir, kind)
    _, pipeline, _, _ = eval_setup(cfg_path, eval_num_rays_per_chunk=sec["rayos_por_bloque"],
                                   test_mode="inference")
    dp = json.loads((cfg_path.parent / "dataparser_transforms.json").read_text("utf-8"))
    scale = float(dp["scale"])
    ds = pipeline.datamanager.train_dataset
    names = [str(Path(f)) for f in ds._dataparser_outputs.image_filenames]
    by_name = {v["nombre"]: v for v in entrada["vistas"]}
    tsdf = TSDF(rcfg["tsdf"])
    roi = entrada["roi"]
    prev = Path(rdir / kind / "vistas_render")
    shutil.rmtree(prev, ignore_errors=True)
    prev.mkdir(parents=True)
    acc_list = []
    for i, fname in enumerate(names):
        key = "/".join(Path(fname).parts[-2:])
        v = by_name.get(key)
        if v is None:
            continue
        cam = ds.cameras[i:i + 1].to(pipeline.device)
        with torch.no_grad():
            out = pipeline.model.get_outputs_for_camera(cam)
        d = out["depth"].squeeze(-1).float().cpu().numpy() / scale
        acc = out["accumulation"].squeeze(-1).float().cpu().numpy()
        rgb = (out["rgb"].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        K, w, h = v["K"], v["width"], v["height"]
        if d.shape != (h, w):
            raise SystemExit(f"render {d.shape} != imagen {(h, w)}")
        if kind == "nerf":
            u, vv = np.meshgrid(np.arange(w), np.arange(h))
            d = d / np.sqrt(((u - K[0, 2]) / K[0, 0]) ** 2 + ((vv - K[1, 2]) / K[1, 1]) ** 2 + 1.0)
        ok = (acc >= sec["acumulacion_min"]) & roi_image_mask(roi, v["T"], K, w, h) & np.isfinite(d)
        d = np.where(ok, d, 0.0)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        tsdf.integrate(d, K, v["T"], bgr)
        acc_list.append(float(ok.mean()))
        cv2.imwrite(str(prev / f"{key.replace('/', '_')}.jpg"), cv2.resize(bgr, (w // 2, h // 2)))
    mesh, pcd = tsdf.extract(roi)
    return mesh, pcd, dict(config=str(cfg_path), escala_nerfstudio=scale, vistas_fusionadas=tsdf.n,
                           fraccion_pixeles_validos=float(np.median(acc_list)) if acc_list else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tecnica", choices=sorted(TECNICAS))
    ap.add_argument("recon_dir")
    ap.add_argument("--solo-fusion", action="store_true")
    args = ap.parse_args()
    rcfg, common = load_recon_config(), load_config("common")
    rdir = repo_path(args.recon_dir)
    entrada = load_entrada(rdir)
    out = rdir / args.tecnica
    out.mkdir(parents=True, exist_ok=True)
    info = dict(tecnica=args.tecnica, metodo=rcfg[TECNICAS[args.tecnica]]["metodo"],
                perfil=rcfg["perfil_nombre"])
    if not args.solo_fusion:
        log = out / "ns-train.log"
        log.write_text("", encoding="utf-8")
        dt, iters = train(args.tecnica, rdir, rcfg, common, log)
        info.update(entrenamiento_s=round(dt, 1), iteraciones=iters)
        print(f"entrenamiento {info['metodo']}: {iters} iteraciones en {dt / 60:.1f} min")
    mesh, pcd, finfo = render_and_fuse(args.tecnica, rdir, rcfg, entrada)
    prev = out / "info.json"
    if args.solo_fusion and prev.exists():
        old = json.loads(prev.read_text("utf-8"))
        info.update({k: old[k] for k in ("entrenamiento_s", "iteraciones") if k in old})
    write_result(out, mesh, pcd, dict(info, **finfo, triangulos=len(mesh.triangles)))
    print(f"-> {out}: malla {len(mesh.triangles)} triangulos, {len(pcd.points)} puntos")


if __name__ == "__main__":
    main()
