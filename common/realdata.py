"""Lectura de una captura en el formato de capture_turntable.py (real o sintetica).

    cap = Capture.open(Path("Captura_real/bunny_x200"))
    for v in cap.views():            # una vista por (camara, parada) con pose
        v.T_color_from_board, v.color_path, v.depth_paths
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from common.config import load_config
from common.geometry import as_T


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@dataclass
class View:
    camera: str
    stop: int
    serial: str
    stop_dir: Path
    color_path: Path
    depth_paths: list
    meta: dict
    T_color_from_board: np.ndarray = None      # None si no hubo pose

    @property
    def has_pose(self):
        return self.T_color_from_board is not None


@dataclass
class Capture:
    root: Path
    layout: dict
    session: dict
    rig: dict
    calib: dict = field(default_factory=dict)      # camara -> calib.json
    stops: dict = field(default_factory=dict)      # camara -> [indices de parada]

    @classmethod
    def open(cls, root, layout=None):
        root = Path(root)
        layout = layout or load_config("common")["capture_layout"]
        session = read_json(root / layout["session_file"])
        rig_path = root / layout["rig_file"]
        rig = read_json(rig_path) if rig_path.exists() else None
        cap = cls(root=root, layout=layout, session=session, rig=rig)
        for cam in layout["cameras"]:
            cdir = root / cam
            if not cdir.is_dir():
                continue
            cap.calib[cam] = read_json(cdir / layout["calib_file"])
            idx = []
            for d in sorted(cdir.iterdir()):
                if d.is_dir() and (d / layout["meta_file"]).exists():
                    idx.append(int(read_json(d / layout["meta_file"])["stop"]))
            cap.stops[cam] = sorted(idx)
        return cap

    @property
    def cameras(self):
        return [c for c in self.layout["cameras"] if c in self.calib]

    def stop_dir(self, cam, k):
        return self.root / cam / self.layout["stop_dir"].format(k=k)

    def view(self, cam, k):
        d = self.stop_dir(cam, k)
        meta = read_json(d / self.layout["meta_file"])
        T = None
        if meta.get("board_detected") and "R_board_to_color" in meta:
            T = as_T(meta["R_board_to_color"], meta["t_board_to_color_mm"])
        n_depth = len(meta.get("depth_timestamps_ms", []))
        depths = [d / self.layout["depth_file"].format(i=i) for i in range(n_depth)]
        return View(camera=cam, stop=k, serial=meta["serial"], stop_dir=d,
                    color_path=d / self.layout["color_file"], depth_paths=depths, meta=meta,
                    T_color_from_board=T)

    def views(self, cameras=None, stops=None):
        for cam in cameras or self.cameras:
            for k in self.stops[cam]:
                if stops is None or k in stops:
                    yield self.view(cam, k)

    def rig_transforms(self):
        """{label: T_color<-rig (mm)} desde rig.json."""
        if self.rig is None:
            return {}
        return {c["label"]: as_T(c["R_rig_to_color"], c["t_rig_to_color_mm"])
                for c in self.rig["camaras"].values()}
