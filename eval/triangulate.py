"""Triangula puntos 3D con COLMAP manteniendo FIJAS las poses e intrinsecos exportados.

    tools/cv_py.sh -m eval.triangulate Reconstrucciones/real/bunnny/entrada

Sobre la carpeta de eval/export_recon.py: feature_extractor (con la mascara ROI) ->
intrinsecos exactos escritos en la base de datos -> matcher -> point_triangulator con
sparse/known/ (poses conocidas sin puntos, lo escribe export_recon) como modelo de entrada.
Escribe sparse/0 con points3D: asi 3DGS / COLMAP-MVS usan las poses de la captura, no las de SfM.
Ademas sparse_pc.ply (misma nube) referenciada en transforms.json (ply_file_path) para splatfacto.
"""
import argparse
import json
import os
import shutil
import sqlite3
from dataclasses import replace
import subprocess
import sys
from pathlib import Path

import numpy as np

from common.colmap import ColmapImage, read_model, write_model
from common.config import load_config, repo_path
from common.meshio import write_points_ply

PINHOLE_ID = 1          # codigo de modelo de COLMAP


def colmap_bin(tcfg):
    """Binario de COLMAP con CUDA: COLMAP_BIN, o el del entorno conda tcfg['colmap_env']."""
    if os.environ.get("COLMAP_BIN"):
        return os.environ["COLMAP_BIN"]
    env = tcfg["colmap_env"]
    prefix = Path(sys.prefix)
    bases = [prefix.parent.parent] if prefix.parent.name == "envs" else []
    bases += [Path.home() / d for d in ("miniconda3", "anaconda3", "miniforge3", "mambaforge")]
    for base in bases:
        b = base / "envs" / env / "bin" / "colmap"
        if b.exists():
            return str(b)
    raise SystemExit(f"no encuentro COLMAP del entorno conda '{env}' (define COLMAP_BIN)")


def run(cmd, log):
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        r = subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise SystemExit(f"fallo {cmd[1]} (ver {log})")


def align_with_db(db, cams, imgs, out_dir):
    """Intrinsecos exactos en la base y modelo de entrada con los IDs de la base.

    COLMAP 4 compara los "frames" de la base con los del modelo: image_id y camera_id deben ser
    los mismos. La base (single_camera_per_folder) tiene una camara por L515; se le escriben los
    intrinsecos exactos y el modelo de poses conocidas se reescribe con sus IDs.
    """
    by_name = {im.name: im for im in imgs}
    con = sqlite3.connect(db)
    rows = con.execute("SELECT image_id, name, camera_id FROM images").fetchall()
    db_cams, new_imgs = {}, []
    for iid, name, cid in rows:
        im = by_name.get(name)
        if im is None:
            continue
        c = cams[im.camera_id]
        prev = db_cams.setdefault(cid, c)
        if (prev.fx, prev.fy, prev.cx, prev.cy) != (c.fx, c.fy, c.cx, c.cy):
            raise SystemExit(f"la camara {cid} de la base mezcla L515 distintas ({name})")
        new_imgs.append(ColmapImage(iid, im.R, im.t, cid, name))
    for cid, c in db_cams.items():
        params = np.array([c.fx, c.fy, c.cx, c.cy], np.float64).tobytes()
        con.execute("UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=1 "
                    "WHERE camera_id=?", (PINHOLE_ID, c.width, c.height, params, cid))
    con.commit()
    con.close()
    if len(new_imgs) != len(imgs):
        raise SystemExit(f"la base tiene {len(new_imgs)} de {len(imgs)} imagenes del modelo")
    write_model(out_dir, [replace(c, camera_id=cid) for cid, c in db_cams.items()], new_imgs)
    return len(db_cams)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recon_dir")
    args = ap.parse_args()
    tcfg = load_config("common")["recon"]["triangulacion"]
    d = repo_path(args.recon_dir)
    known, sparse = d / "sparse" / "known", d / "sparse" / "0"
    if not known.exists():
        raise SystemExit(f"falta {known}: corre primero eval/export_recon.py")
    cams, imgs = read_model(known)
    work = d / "colmap_work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    db, log = work / "database.db", work / "colmap.log"
    cb = colmap_bin(tcfg)
    ext = ["--ImageReader.camera_model", tcfg["camera_model"], "--ImageReader.single_camera_per_folder", 1]
    if tcfg["usar_mascara_roi"]:
        mdir = work / "masks"                    # COLMAP busca <mask_path>/<nombre_imagen>.png
        for im in imgs:
            dst = mdir / f"{im.name}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(d / "masks" / im.name, dst)
        ext += ["--ImageReader.mask_path", mdir]
    run([cb, "feature_extractor", "--database_path", db, "--image_path", d / "images", *ext], log)
    model_in = work / "entrada"
    n = align_with_db(db, cams, imgs, model_in)
    run([cb, tcfg["matcher"], "--database_path", db], log)
    out = work / "triangulado"
    out.mkdir()
    run([cb, "point_triangulator", "--database_path", db, "--image_path", d / "images",
         "--input_path", model_in, "--output_path", out], log)
    run([cb, "model_converter", "--input_path", out, "--output_path", out, "--output_type", "TXT"], log)
    for f in out.glob("*.txt"):
        shutil.copy(f, sparse / f.name)
    rows = [ln.split() for ln in (sparse / "points3D.txt").read_text().splitlines()
            if ln and not ln.startswith("#")]
    pts = len(rows)
    if rows:
        write_points_ply(d / "sparse_pc.ply", [[float(x) for x in r[1:4]] for r in rows],
                         [[int(x) for x in r[4:7]] for r in rows])
        tj_path = d / "transforms.json"
        tj = json.loads(tj_path.read_text(encoding="utf-8"))
        tj["ply_file_path"] = "sparse_pc.ply"
        tj_path.write_text(json.dumps(tj, indent=1), encoding="utf-8")
    # las poses no deben cambiar
    _, imgs2 = read_model(sparse)
    dt = max(np.abs(a.t - b.t).max() for a, b in zip(sorted(imgs, key=lambda i: i.name),
                                                    sorted(imgs2, key=lambda i: i.name)))
    print(f"{d}: {pts} puntos 3D, {n} camaras en la base, poses sin cambio (max dt {dt:.1e} m); log {log}")


if __name__ == "__main__":
    main()
