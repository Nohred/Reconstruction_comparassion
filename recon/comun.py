"""Utilidades comunes de las tecnicas de reconstruccion (entorno de nerfstudio: open3d, torch).

Marco de trabajo: tablero en METROS (z hacia la mesa). Las poses de vistas.json estan en mm.
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np

from common.config import load_config, repo_path
from common.geometry import T_inv

MM = 1e-3


def load_recon_config(perfil=None):
    cfg = load_config("recon")
    import os

    perfil = perfil or os.environ.get("PERFIL", "completo")
    cfg["perfil_nombre"] = perfil
    cfg["perfil"] = cfg["perfiles"][perfil]
    return cfg


def load_entrada(recon_dir):
    """vistas.json de la entrada: K (OpenCV), T_color<-board (m), ROI y captura de origen."""
    common = load_config("common")
    e = json.loads((Path(recon_dir) / common["recon"]["entrada_dir"] / "vistas.json").read_text("utf-8"))
    for v in e["vistas"]:
        v["K"] = np.array(v["K"])
        T = np.array(v["T_color_from_board_mm"])
        T[:3, 3] *= MM
        v["T"] = T
    return e


def roi_select(P_m, roi, extra_mm=0.0):
    """Puntos (m, marco tablero) dentro del cilindro ROI."""
    c = np.array(roi["eje_xy_mm"]) * MM
    r = np.linalg.norm(P_m[:, :2] - c, axis=1) <= (roi["radio_mm"] + extra_mm) * MM
    z = (P_m[:, 2] <= (roi["bajo_tablero_mm"] + extra_mm) * MM) & (P_m[:, 2] >= -(roi["altura_mm"] + extra_mm) * MM)
    return r & z


def roi_image_mask(roi, T_cam_from_board_m, K, w, h):
    from eval.export_recon import roi_mask, roi_points

    T = T_cam_from_board_m.copy()
    T[:3, 3] /= MM
    return roi_mask(roi_points(roi["eje_xy_mm"], roi), T, K, w, h) > 0


class TSDF:
    """TSDF de open3d en el marco del tablero (m)."""

    def __init__(self, tcfg):
        import open3d as o3d

        self.o3d = o3d
        self.cfg = tcfg
        self.vol = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=tcfg["voxel_mm"] * MM, sdf_trunc=tcfg["trunc_mm"] * MM,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        self.n = 0

    def integrate(self, depth_m, K, T_cam_from_board_m, color_bgr=None):
        o3d = self.o3d
        h, w = depth_m.shape
        if color_bgr is None:
            color = np.full((h, w, 3), 160, np.uint8)
        else:
            color = np.ascontiguousarray(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color), o3d.geometry.Image(depth_m.astype(np.float32)), depth_scale=1.0,
            depth_trunc=self.cfg["profundidad_max_m"], convert_rgb_to_intensity=False)
        intr = o3d.camera.PinholeCameraIntrinsic(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        self.vol.integrate(rgbd, intr, T_cam_from_board_m)
        self.n += 1

    def extract(self, roi):
        """(malla recortada a la ROI, nube muestreada de su superficie)."""
        mesh = self.vol.extract_triangle_mesh()
        V = np.asarray(mesh.vertices)
        if len(V):
            mesh.remove_vertices_by_mask(~roi_select(V, roi, extra_mm=5.0))
        mesh.compute_vertex_normals()
        n = self.cfg["puntos_superficie"]
        pcd = mesh.sample_points_uniformly(n) if len(mesh.triangles) else self.o3d.geometry.PointCloud()
        return mesh, pcd


def write_result(out_dir, mesh=None, pcd=None, info=None):
    """resultado.ply (nube, m, marco tablero) + malla.ply + info.json."""
    import open3d as o3d

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if mesh is not None:
        o3d.io.write_triangle_mesh(str(out_dir / "malla.ply"), mesh)
    if pcd is not None:
        o3d.io.write_point_cloud(str(out_dir / "resultado.ply"), pcd)
    if info is not None:
        info = dict(info, fecha=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    marco="tablero ChArUco, metros (z hacia la mesa)")
        (out_dir / "info.json").write_text(json.dumps(info, indent=1, default=float), "utf-8")


def T_board_to_cam_m(T_cam_from_board_mm):
    T = np.array(T_cam_from_board_mm, float)
    T[:3, 3] *= MM
    return T


def camera_center_m(T_cam_from_board_m):
    return T_inv(T_cam_from_board_m)[:3, 3]


def capture_path(entrada):
    return repo_path(entrada["captura"])
