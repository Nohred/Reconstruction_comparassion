"""Camaras RTX con intrinsecos exactos de la L515 (esquema OpenCvPinhole) y su pose."""
import numpy as np
from pxr import Gf, UsdGeom

from common.geometry import cam_pose_usd_in_world, rot_to_quat_wxyz
from common.intrinsics import (PINHOLE_NS, PINHOLE_SCHEMA, K_dist_from_calib, isaac_lens_attrs,
                               usd_camera_attrs)


class SimCamera:
    """Una camara (color o profundidad) de una L515: RtxCamera + CameraSensor.

    El esquema OmniLensDistortionOpenCvPinholeAPI se aplica SIEMPRE (con coeficientes 0 si no
    hay distorsion): CameraSensor fuerza pixeles cuadrados en las aperturas USD, asi que fx != fy
    y el punto principal solo son exactos a traves del esquema.
    """

    def __init__(self, path, intr, annotators, clipping_m, distortion=True, pixel_offset_px=0.0):
        from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera

        self.K, dist = K_dist_from_calib(intr)
        self.dist = dist if distortion else np.zeros_like(dist)
        self.width, self.height = int(intr["width"]), int(intr["height"])
        K_render = self.K.copy()                 # convencion de pixel de RTX (ver scene.yaml)
        K_render[0, 2] += pixel_offset_px
        K_render[1, 2] += pixel_offset_px
        lens = isaac_lens_attrs(K_render, self.dist, self.width, self.height)
        attrs = {f"{PINHOLE_NS}:{k}": (Gf.Vec2i(*v) if k == "imageSize" else float(v))
                 for k, v in lens.items()}
        self.rtx = RtxCamera(path, schemas=[PINHOLE_SCHEMA], attributes=attrs)
        cam = UsdGeom.Camera(self.rtx.prims[0])
        for name, val in usd_camera_attrs(self.K, self.width, self.height).items():
            getattr(cam, f"Create{name[0].upper()}{name[1:]}Attr")(val)
        cam.CreateClippingRangeAttr(Gf.Vec2f(*clipping_m))
        self.sensor = CameraSensor(self.rtx, resolution=(self.height, self.width), annotators=annotators)

    def set_pose(self, T_cam_from_board, scale):
        """Coloca la camara a partir de T_cam<-board (mm, OpenCV)."""
        M = cam_pose_usd_in_world(T_cam_from_board, scale)
        q = rot_to_quat_wxyz(M[:3, :3])
        self.rtx.set_world_poses(positions=[M[:3, 3].tolist()], orientations=[q.tolist()])

    def read(self, annotator):
        data, _ = self.sensor.get_data(annotator)
        return None if data is None else data.numpy()
