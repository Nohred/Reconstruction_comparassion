"""Tecnica 3: fotogrametria densa con COLMAP MVS y las poses FIJAS de la captura.

    tools/ns_py.sh -m recon.fotogrametria Reconstrucciones/real/bunnny

entrada/sparse/0 (poses conocidas + puntos triangulados por eval/triangulate.py) ->
image_undistorter -> patch_match_stereo (CUDA) -> stereo_fusion (con mascara ROI) ->
resultado.ply (nube fusionada recortada a la ROI) y malla.ply (Poisson de COLMAP).
Salida: Reconstrucciones/<fuente>/<captura>/fotogrametria/
"""
import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from common.config import load_config, repo_path
from eval.triangulate import colmap_bin
from recon.comun import load_entrada, load_recon_config, roi_select, write_result


def run(cmd, log):
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        r = subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise SystemExit(f"fallo {cmd[1]} (ver {log})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recon_dir")
    args = ap.parse_args()
    import open3d as o3d

    rcfg, common = load_recon_config(), load_config("common")
    fcfg = rcfg["fotogrametria"]
    rdir = repo_path(args.recon_dir)
    ent = rdir / common["recon"]["entrada_dir"]
    entrada = load_entrada(rdir)
    if not (ent / "sparse_pc.ply").exists():
        raise SystemExit("falta la triangulacion: corre eval/triangulate.py (./run.sh entrada)")
    out = rdir / "fotogrametria"
    work = out / "trabajo"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    log = out / "colmap.log"
    log.write_text("", encoding="utf-8")
    cb = colmap_bin(common["recon"]["triangulacion"])
    run([cb, "image_undistorter", "--image_path", ent / "images", "--input_path", ent / "sparse" / "0",
         "--output_path", work, "--output_type", "COLMAP",
         "--max_image_size", rcfg["perfil"]["mvs_max_imagen_px"]], log)
    run([cb, "patch_match_stereo", "--workspace_path", work, "--workspace_format", "COLMAP",
         "--PatchMatchStereo.geom_consistency", str(bool(fcfg["geom_consistency"])).lower()], log)
    fused = out / "fused.ply"
    cmd = [cb, "stereo_fusion", "--workspace_path", work, "--workspace_format", "COLMAP",
           "--input_type", "geometric" if fcfg["geom_consistency"] else "photometric",
           "--output_path", fused, "--StereoFusion.min_num_pixels", fcfg["fusion_min_pixeles"]]
    if fcfg["usar_mascara_roi"]:
        mdir = work / "mascaras"           # <mask_path>/<nombre_imagen>.png, al tamano sin distorsion
        for v in entrada["vistas"]:
            img = cv2.imread(str(work / "images" / v["nombre"]), cv2.IMREAD_GRAYSCALE)
            m = cv2.imread(str(ent / "masks" / v["nombre"]), cv2.IMREAD_GRAYSCALE)
            dst = mdir / f"{v['nombre']}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), cv2.resize(m, img.shape[::-1], interpolation=cv2.INTER_NEAREST))
        cmd += ["--StereoFusion.mask_path", mdir]
    run(cmd, log)
    pcd = o3d.io.read_point_cloud(str(fused))
    P = np.asarray(pcd.points)
    keep = np.nonzero(roi_select(P, entrada["roi"]))[0]
    pcd = pcd.select_by_index(keep)
    mesh = None
    if len(pcd.points) > 1000:
        run([cb, "poisson_mesher", "--input_path", fused, "--output_path", out / "malla_poisson.ply"], log)
    write_result(out, None, pcd, dict(tecnica="fotogrametria", puntos_fusionados=len(P),
                                      puntos_roi=len(pcd.points),
                                      max_imagen_px=rcfg["perfil"]["mvs_max_imagen_px"],
                                      perfil=rcfg["perfil_nombre"]))
    shutil.rmtree(work / "stereo" / "normal_maps", ignore_errors=True)   # pesados y no se usan
    print(f"-> {out}: {len(P)} puntos fusionados, {len(pcd.points)} en la ROI")


if __name__ == "__main__":
    main()
