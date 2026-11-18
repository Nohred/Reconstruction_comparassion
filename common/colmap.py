"""Lectura y escritura del modelo de texto de COLMAP (cameras.txt, images.txt, points3D.txt).

images.txt guarda world->cam: X_cam = R(q) X_world + t, que es exactamente T_color<-board.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common.geometry import quat_wxyz_to_rot, rot_to_quat_wxyz


@dataclass
class ColmapCamera:
    camera_id: int
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str = "PINHOLE"

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])


@dataclass
class ColmapImage:
    image_id: int
    R: np.ndarray
    t: np.ndarray
    camera_id: int
    name: str


def write_model(out_dir, cameras, images):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "cameras.txt", "w", encoding="utf-8") as f:
        f.write("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n")
        for c in cameras:
            params = " ".join(repr(float(v)) for v in (c.fx, c.fy, c.cx, c.cy))
            f.write(f"{c.camera_id} {c.model} {int(c.width)} {int(c.height)} {params}\n")
    with open(out / "images.txt", "w", encoding="utf-8") as f:
        f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n# POINTS2D[] (vacio)\n")
        for im in images:
            q = rot_to_quat_wxyz(im.R)
            t = np.asarray(im.t, float)
            f.write(f"{im.image_id} {' '.join(repr(float(v)) for v in q)} "
                    f"{' '.join(repr(float(v)) for v in t)} {im.camera_id} {im.name}\n\n")
    (out / "points3D.txt").write_text("# POINT3D_ID X Y Z R G B ERROR TRACK[]\n", encoding="utf-8")


def _data_lines(path):
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        if ln.strip() and not ln.startswith("#"):
            yield ln


def read_model(model_dir):
    d = Path(model_dir)
    cams = {}
    for ln in _data_lines(d / "cameras.txt"):
        p = ln.split()
        if p[1] != "PINHOLE":
            raise NotImplementedError(f"modelo {p[1]}")
        cams[int(p[0])] = ColmapCamera(int(p[0]), int(p[2]), int(p[3]), *map(float, p[4:8]))
    imgs = []
    # images.txt alterna linea de pose y linea de POINTS2D (que puede estar vacia)
    lines = [ln for ln in (d / "images.txt").read_text(encoding="utf-8").splitlines()
             if not ln.startswith("#")]
    for ln in lines[0::2]:
        p = ln.split()
        imgs.append(ColmapImage(int(p[0]), quat_wxyz_to_rot(list(map(float, p[1:5]))),
                                np.array(list(map(float, p[5:8]))), int(p[8]), p[9]))
    return cams, imgs
