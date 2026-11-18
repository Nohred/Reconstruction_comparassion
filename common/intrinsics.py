"""Intrinsecos: calib.json (librealsense) <-> OpenCV <-> parametros de camara de Isaac Sim / USD.

Isaac Sim 6.2 / RTX: la distorsion OpenCV se aplica con el esquema USD
OmniLensDistortionOpenCvPinholeAPI (atributos omni:lensdistortion:opencvPinhole:*), que toma
fx, fy, cx, cy, k1..k6, p1, p2 tal cual de OpenCV. Referencia: script de NVIDIA
omni.sensors.nv.camera/docs/intrinsic_calibration_script/calibrate_camera_to_usd.py.
Los atributos UsdGeom.Camera (focalLength, aperturas y offsets) se fijan con la misma regla del
script de referencia: focalLength = fx y horizontalAperture = ancho (unidades = pixeles).
Diferencia con el script de referencia: verticalApertureOffset se multiplica por fx/fy (exacto
cuando fx != fy; NVIDIA usa cy - h/2, diferencia < 0.001 px aqui). El signo de los offsets se
valida empiricamente en la fase 2 (deteccion ChArUco sobre el render).
"""
from types import SimpleNamespace

import numpy as np

from common.board import L

PINHOLE_SCHEMA = "OmniLensDistortionOpenCvPinholeAPI"
PINHOLE_NS = "omni:lensdistortion:opencvPinhole"

# librealsense guarda brown_conrady como [k1, k2, p1, p2, k3] = mismo orden que OpenCV.
_RS_OPENCV_COMPATIBLE = {"distortion.brown_conrady", "distortion.none"}


def K_dist_from_calib(intr):
    """(K, dist) OpenCV desde un bloque *_intrinsics de calib.json (reusa l515_common.intr_to_K)."""
    model = intr.get("model", "distortion.none")
    if model not in _RS_OPENCV_COMPATIBLE:
        raise NotImplementedError(f"Modelo de distorsion {model} no soportado (solo brown_conrady/none)")
    K, dist = L.intr_to_K(SimpleNamespace(**intr))
    return K, dist


def calib_from_K(K, dist, width, height, model="distortion.brown_conrady"):
    """Bloque *_intrinsics de calib.json desde K/dist (para escribir la captura sintetica)."""
    dist = np.zeros(5) if dist is None else np.asarray(dist, float).ravel()
    if np.allclose(dist, 0):
        model = "distortion.none"
    return dict(width=int(width), height=int(height), fx=float(K[0, 0]), fy=float(K[1, 1]),
                ppx=float(K[0, 2]), ppy=float(K[1, 2]), model=model,
                coeffs=[float(c) for c in dist[:5]])


def isaac_lens_attrs(K, dist, width, height):
    """Atributos del esquema OpenCvPinhole (nombre corto -> valor) para RTX."""
    d = np.zeros(12)
    if dist is not None:
        dd = np.asarray(dist, float).ravel()
        d[:len(dd)] = dd
    return dict(fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]),
                k1=d[0], k2=d[1], p1=d[2], p2=d[3], k3=d[4], k4=d[5], k5=d[6], k6=d[7],
                s1=d[8], s2=d[9], s3=d[10], s4=d[11], imageSize=(int(width), int(height)))


def usd_camera_attrs(K, width, height):
    """focalLength / apertures / offsets de UsdGeom.Camera que reproducen K (pinhole)."""
    # Unidad de apertura = 1 px horizontal; un px vertical mide fx/fy unidades.
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return dict(focalLength=float(fx), horizontalAperture=float(width),
                verticalAperture=float(fx * height / fy),
                horizontalApertureOffset=float(cx - width / 2.0),
                verticalApertureOffset=float((cy - height / 2.0) * fx / fy))


def K_from_usd_camera_attrs(a, width, height):
    """Inversa de usd_camera_attrs (para pruebas y para leer camaras ya creadas)."""
    fx = a["focalLength"] * width / a["horizontalAperture"]
    fy = fx * height / a["verticalAperture"] * (a["horizontalAperture"] / width)
    cx = a["horizontalApertureOffset"] * width / a["horizontalAperture"] + width / 2.0
    cy = a["verticalApertureOffset"] * height / a["verticalAperture"] + height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
