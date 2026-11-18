"""Geometria: transformaciones 4x4, cuaterniones y cambios de marco.

Marcos (ver CLAUDE.md):
  board  : tablero ChArUco de OpenCV, mm. Origen en una esquina; x a lo largo de 10 cuadros,
           y a lo largo de 7, z = x cross y apunta HACIA LA MESA (las camaras quedan en z < 0).
  color  : camara de color, convencion OpenCV (x der, y abajo, z adelante). meta.json guarda
           T_color<-board: p_color = R @ p_board + t (mm).
  rev    : marco de las nubes de revision/ y del GT sintetico: p_rev = diag(flip) @ p_board.
  world  : mundo de Isaac Sim, metros, Z arriba: p_world = s * Rx(180) @ p_board.
  cam_usd: camara USD/Isaac "usd" (x der, y arriba, mira hacia -z). La convencion "ros" de Isaac
           coincide con OpenCV.

Las matrices de cambio de ejes son definiciones de convencion, no parametros ajustables.
"""
import numpy as np

from common.board import L

T_from = L.T_from
T_inv = L.T_inv
mean_pose = L.mean_pose
pose_diff = L.pose_diff
rot_angle_deg = L.rot_angle_deg

# p_cv = CV_FROM_USD_CAM @ p_usd  (y y z invertidos). Es su propia inversa.
CV_FROM_USD_CAM = np.diag([1.0, -1.0, -1.0])
# Rotacion de 180 deg alrededor de x: lleva el tablero (z hacia la mesa) a un marco Z arriba.
ROT_UP_FROM_BOARD = np.diag([1.0, -1.0, -1.0])


def as_T(R, t):
    return T_from(np.asarray(R, float), np.asarray(t, float))


def transform_points(T, P):
    P = np.asarray(P, float).reshape(-1, 3)
    return P @ T[:3, :3].T + T[:3, 3]


def camera_center(T_cam_from_world):
    """Centro de la camara en el marco del mundo dado T_cam<-world."""
    R, t = T_cam_from_world[:3, :3], T_cam_from_world[:3, 3]
    return -R.T @ t


# ----------------------------------------------------------------------------- cuaterniones
def rot_to_quat_wxyz(R):
    """Matriz de rotacion -> cuaternion unitario (w, x, y, z) con w >= 0 (convencion COLMAP)."""
    R = np.asarray(R, float)
    K = np.array([
        [R[0, 0] - R[1, 1] - R[2, 2], 0, 0, 0],
        [R[1, 0] + R[0, 1], R[1, 1] - R[0, 0] - R[2, 2], 0, 0],
        [R[2, 0] + R[0, 2], R[2, 1] + R[1, 2], R[2, 2] - R[0, 0] - R[1, 1], 0],
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1], R[0, 0] + R[1, 1] + R[2, 2]],
    ]) / 3.0
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]
    return q if q[0] >= 0 else -q


def quat_wxyz_to_rot(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


# ----------------------------------------------------------------------------- marcos
def scale_T(T, s):
    """Cambia las unidades de la traslacion de T (p. ej. mm -> m con s=0.001)."""
    T = np.array(T, float)
    T[:3, 3] *= s
    return T


def world_from_board(scale):
    """T_world<-board (Isaac, Z arriba). `scale` convierte mm a unidades del mundo."""
    T = np.eye(4)
    T[:3, :3] = scale * ROT_UP_FROM_BOARD
    return T


def cam_pose_usd_in_world(T_color_from_board, scale):
    """Pose camera-to-world (marco de camara USD) para Isaac, a partir de T_color<-board en mm.

    Devuelve una 4x4 rigida en metros (si scale=0.001): p_world = M @ p_cam_usd.
    """
    T_wb = world_from_board(1.0)
    T_wc_cv = T_wb @ T_inv(np.asarray(T_color_from_board, float))   # cam(cv) -> world (mm)
    M = T_wc_cv.copy()
    M[:3, :3] = T_wc_cv[:3, :3] @ CV_FROM_USD_CAM
    M[:3, 3] *= scale
    return M


def cam_pose_ros_in_world(T_color_from_board, scale):
    """Igual que cam_pose_usd_in_world pero con ejes de camara OpenCV/"ros" de Isaac."""
    T_wc_cv = world_from_board(1.0) @ T_inv(np.asarray(T_color_from_board, float))
    T_wc_cv[:3, 3] *= scale
    return T_wc_cv


def board_to_rev(P, flip):
    """Puntos del marco del tablero al marco de revision/ (p_rev = diag(flip) @ p_board)."""
    return np.asarray(P, float).reshape(-1, 3) * np.asarray(flip, float)


def rev_to_board(P, flip):
    return board_to_rev(P, flip)          # diag(+-1) es su propia inversa


def depth_to_color_T(ext, scale_from_m):
    """T_color<-depth desde el extrinseco de librealsense guardado en calib.json.

    rs2_extrinsics.rotation es column-major: R = reshape(3,3).T. translation en metros.
    """
    R = np.asarray(ext["rotation_colmajor"], float).reshape(3, 3).T
    t = np.asarray(ext["translation_m"], float) * scale_from_m
    return T_from(R, t)


# ----------------------------------------------------------------------------- base giratoria
def rot_z(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def fit_turntable_axis(Ts_b0_from_bk):
    """Punto (x, y) del plano z=0 del tablero por el que pasa el eje de giro.

    Ts: transformaciones T_b0<-bk (tablero en la parada k expresado en el tablero de la parada 0).
    Un punto p del eje queda fijo: R_k p + t_k = p. Con p = (x, y, 0) se resuelve por minimos
    cuadrados apilando (R_k - I)[:, :2] [x y]^T = -t_k. Devuelve (xy, residuo RMS en mm).
    """
    A, b = [], []
    for T in Ts_b0_from_bk:
        A.append((T[:3, :3] - np.eye(3))[:, :2])
        b.append(-T[:3, 3])
    A, b = np.vstack(A), np.concatenate(b)
    xy, *_ = np.linalg.lstsq(A, b, rcond=None)
    res = float(np.sqrt(np.mean((A @ xy - b) ** 2)))
    return xy, res


def place_on_board(P_local, axis_xy, yaw_deg):
    """T_board<-objeto (mm) para un objeto con Z arriba en su marco local.

    El objeto se gira yaw_deg alrededor de su Z, se voltea a la convencion del tablero (z hacia la
    mesa) con Rx(180), su bbox se centra en XY sobre axis_xy y su punto mas bajo queda en z=0.
    """
    R = ROT_UP_FROM_BOARD @ rot_z(yaw_deg)       # ROT_UP_FROM_BOARD es su propia inversa
    Q = np.asarray(P_local, float).reshape(-1, 3) @ R.T
    lo, hi = Q.min(0), Q.max(0)
    t = np.array([axis_xy[0] - (lo[0] + hi[0]) / 2, axis_xy[1] - (lo[1] + hi[1]) / 2, -hi[2]])
    return T_from(R, t)


def rot_log(R):
    """(eje unitario, angulo en grados) de una rotacion; angulo en [0, 180]."""
    R = np.asarray(R, float)
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(v)
    return (v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])), float(np.degrees(ang))


def rot_axis_angle(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    t = np.radians(deg)
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def rot_about_line(point, axis, deg):
    """T (4x4) que gira `deg` alrededor de la recta que pasa por `point` con direccion `axis`."""
    R = rot_axis_angle(axis, deg)
    p = np.asarray(point, float)
    return T_from(R, p - R @ p)
