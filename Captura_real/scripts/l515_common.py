"""Utilidades comunes para capturar con varias Intel RealSense L515 y el tablero ChArUco.

Tablero: 10x7 cuadros de 24 mm, marcadores de 18 mm, DICT_5X5_250 (el mismo de make_charuco.py).
Todas las poses se expresan en milimetros: R, t llevan puntos del marco del tablero al marco
de la camara de color (p_cam = R @ p_tablero + t).

Funciona con OpenCV >= 4.7 (API CharucoDetector) y con versiones anteriores (opencv-contrib).
"""
import json
import time
import numpy as np
import cv2
import pyrealsense2 as rs

# --- Tablero (debe coincidir con make_charuco.py) ---
SQUARES_X, SQUARES_Y = 10, 7
SQUARE_MM, MARKER_MM = 24.0, 18.0
ARUCO_DICT = cv2.aruco.DICT_5X5_250
BOARD_CENTER_MM = np.array([SQUARES_X * SQUARE_MM / 2, SQUARES_Y * SQUARE_MM / 2, 0.0])

# --- Criterio de pose valida (se reporta en el paper) ---
MIN_CORNERS = 10        # esquinas ChArUco minimas
MIN_ROWS = MIN_COLS = 3  # las esquinas deben cubrir al menos 3 filas y 3 columnas (no colineales)
MAX_REPROJ_PX = 1.0     # error RMS de reproyeccion maximo para considerar la pose "buena"
MIN_MARKERS = 6         # marcadores minimos si se usan sus esquinas (cuando faltan esquinas ChArUco)

RIG_FILE = "rig.json"

# --- Streams (los mismos de la captura anterior con la L515) ---
DEPTH_W, DEPTH_H = 1024, 768
COLOR_W, COLOR_H = 1280, 720
FPS = 30


# ----------------------------------------------------------------------------- tablero
class BoardDetector:
    """Detecta el tablero y estima su pose.

    1) Primero intenta solo con esquinas ChArUco (las mas precisas).
    2) Si no alcanzan (p. ej. la base de un objeto grande tapa los cruces entre cuadros pero
       siguen visibles los marcadores del borde), agrega las 4 esquinas de cada marcador ArUco
       detectado. El campo "fuente" dice cual se uso: "charuco" o "charuco+marcadores".
    """

    def __init__(self):
        self.dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self.new_api = hasattr(cv2.aruco, "CharucoDetector")
        if self.new_api:
            self.board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_MM, MARKER_MM,
                                                self.dictionary)
            dparams = cv2.aruco.DetectorParameters()
            dparams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.detector = cv2.aruco.CharucoDetector(self.board, cv2.aruco.CharucoParameters(),
                                                      dparams)
            corners = self.board.getChessboardCorners()
            mk_obj, mk_ids = self.board.getObjPoints(), self.board.getIds()
        else:
            self.board = cv2.aruco.CharucoBoard_create(SQUARES_X, SQUARES_Y, SQUARE_MM, MARKER_MM,
                                                       self.dictionary)
            self.dparams = cv2.aruco.DetectorParameters_create()
            self.dparams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            corners = (self.board.getChessboardCorners() if hasattr(self.board, "getChessboardCorners")
                       else self.board.chessboardCorners)
            mk_obj = (self.board.getObjPoints() if hasattr(self.board, "getObjPoints")
                      else self.board.objPoints)
            mk_ids = self.board.getIds() if hasattr(self.board, "getIds") else self.board.ids
        self.corners3d = np.asarray(corners, dtype=np.float64).reshape(-1, 3)
        self.marker3d = {int(i): np.asarray(o, np.float64).reshape(4, 3)
                         for i, o in zip(np.asarray(mk_ids).ravel(), mk_obj)}

    def detect_all(self, gray):
        """(esquinas ChArUco, ids ChArUco, esquinas de marcadores, ids de marcadores); None si no hay."""
        if self.new_api:
            ch_c, ch_ids, m_c, m_ids = self.detector.detectBoard(gray)
        else:
            m_c, m_ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.dparams)
            ch_c = ch_ids = None
            if m_ids is not None and len(m_ids) > 0:
                _, ch_c, ch_ids = cv2.aruco.interpolateCornersCharuco(m_c, m_ids, gray, self.board)
        if ch_ids is None or len(ch_ids) == 0:
            ch_c = ch_ids = None
        else:   # misma forma en OpenCV 4.x y 5.x: (N,1,2) y (N,1)
            ch_c = np.asarray(ch_c, np.float32).reshape(-1, 1, 2)
            ch_ids = np.asarray(ch_ids, np.int32).reshape(-1, 1)
        if m_ids is None or len(m_ids) == 0:
            m_c = m_ids = None
        else:
            m_c = tuple(np.asarray(c, np.float32).reshape(1, 4, 2) for c in m_c)
            m_ids = np.asarray(m_ids, np.int32).reshape(-1, 1)
        return ch_c, ch_ids, m_c, m_ids

    def detect(self, gray):
        ch_c, ch_ids, _, _ = self.detect_all(gray)
        return ch_c, ch_ids

    @staticmethod
    def _solve(obj, img, K, dist):
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error:                          # p. ej. puntos casi alineados
            return None
        if not ok:
            return None
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        reproj = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
        return cv2.Rodrigues(rvec)[0], tvec.reshape(3), reproj

    def pose(self, gray, K, dist, min_corners=MIN_CORNERS):
        """Pose del tablero en la camara (mm), o None.

        Se descarta si los puntos no cubren MIN_ROWS x MIN_COLS cuadros (pose mal condicionada).
        "good" = cumple el criterio completo: (>= MIN_CORNERS esquinas ChArUco, o >= MIN_MARKERS
        marcadores) y error de reproyeccion <= MAX_REPROJ_PX.
        """
        ch_c, ch_ids, m_c, m_ids = self.detect_all(gray)
        n_ch = 0 if ch_ids is None else int(len(ch_ids))
        obj_ch, img_ch = np.zeros((0, 3)), np.zeros((0, 2))
        if n_ch:
            obj_ch = self.corners3d[np.asarray(ch_ids).ravel()]
            img_ch = np.asarray(ch_c, np.float64).reshape(-1, 2)

        best = None
        # 1) solo ChArUco
        if n_ch >= max(min_corners, 6):
            rows, cols = grid_spread(obj_ch)
            if rows >= MIN_ROWS and cols >= MIN_COLS:
                r = self._solve(obj_ch, img_ch, K, dist)
                if r is not None:
                    best = dict(R=r[0], t=r[1], reproj_px=r[2], rows=rows, cols=cols, n_markers=0,
                                fuente="charuco",
                                good=bool(n_ch >= MIN_CORNERS and r[2] <= MAX_REPROJ_PX))
        # 2) ChArUco + esquinas de marcadores
        if (best is None or not best["good"]) and m_ids is not None:
            ids = [int(i) for i in np.asarray(m_ids).ravel() if int(i) in self.marker3d]
            if len(ids) >= 3:
                sel = [k for k, i in enumerate(np.asarray(m_ids).ravel()) if int(i) in self.marker3d]
                obj = np.vstack([obj_ch] + [self.marker3d[i] for i in ids])
                img = np.vstack([img_ch] + [np.asarray(m_c[k], np.float64).reshape(4, 2) for k in sel])
                rows, cols = grid_spread(obj)
                if rows >= MIN_ROWS and cols >= MIN_COLS:
                    r = self._solve(obj, img, K, dist)
                    if r is not None:
                        cand = dict(R=r[0], t=r[1], reproj_px=r[2], rows=rows, cols=cols,
                                    n_markers=len(ids), fuente="charuco+marcadores",
                                    good=bool(len(ids) >= MIN_MARKERS and r[2] <= MAX_REPROJ_PX))
                        if best is None or cand["good"]:
                            best = cand
        if best is None:
            return None
        best.update(n=n_ch, weight=n_ch + best["n_markers"], corners=ch_c, ids=ch_ids,
                    marker_corners=m_c, marker_ids=m_ids)
        return best


def draw_detection(vis, p):
    """Dibuja lo que se uso para la pose (esquinas ChArUco en verde, marcadores en azul)."""
    if p.get("marker_ids") is not None and p["fuente"] != "charuco":
        cv2.aruco.drawDetectedMarkers(vis, p["marker_corners"], None, (255, 128, 0))
    if p.get("ids") is not None:
        cv2.aruco.drawDetectedCornersCharuco(vis, p["corners"], p["ids"], (0, 255, 0))


def grid_spread(obj):
    """Filas y columnas de cuadros (de SQUARE_MM) que cubren unos puntos del tablero (mm)."""
    obj = np.asarray(obj).reshape(-1, 3)
    cells = np.floor(obj[:, :2] / SQUARE_MM + 1e-6).astype(int)
    return len(np.unique(cells[:, 1])), len(np.unique(cells[:, 0]))


def corner_spread(ids):
    """Numero de filas y columnas distintas que cubren las esquinas ChArUco detectadas."""
    ids = np.asarray(ids).ravel()
    per_row = SQUARES_X - 1                        # esquinas interiores por fila
    return len(np.unique(ids // per_row)), len(np.unique(ids % per_row))


def aim_point(R, t, height_mm=0.0):
    """Punto a media altura del objeto, sobre el centro del tablero (marco del tablero, mm).

    El signo del eje z del tablero depende de la convencion de OpenCV; se elige el lado
    donde esta la camara para que el punto quede "arriba" del tablero.
    """
    cam_center = -R.T @ np.ravel(t)
    up = 1.0 if cam_center[2] > 0 else -1.0
    return BOARD_CENTER_MM + np.array([0.0, 0.0, up * height_mm / 2.0])


# ----------------------------------------------------------------------------- poses 4x4
def T_from(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.ravel(t)
    return T


def T_inv(T):
    R, t = T[:3, :3], T[:3, 3]
    return T_from(R.T, -R.T @ t)


def mean_pose(Ts, weights=None):
    """Promedio de transformaciones rigidas: rotacion por SVD (media cordal) y traslacion media."""
    w = np.ones(len(Ts)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    M = sum(wi * T[:3, :3] for wi, T in zip(w, Ts))
    U, _, Vt = np.linalg.svd(M)
    R = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    t = sum(wi * T[:3, 3] for wi, T in zip(w, Ts))
    return T_from(R, t)


def pose_diff(Ta, Tb):
    """(angulo en grados, distancia en mm) entre dos transformaciones."""
    return rot_angle_deg(Ta[:3, :3], Tb[:3, :3]), float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]))


def load_rig(path=RIG_FILE):
    """Lee rig.json. Devuelve {serial: X} con X = transformacion marco_rig -> color de esa camara."""
    import os
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rig = {s: T_from(np.array(c["R_rig_to_color"]), np.array(c["t_rig_to_color_mm"]))
           for s, c in data["camaras"].items()}
    return rig, data


def fuse_board_pose(poses, rig):
    """Pose comun del tablero en el marco del rig a partir de las camaras que lo vieron.

    poses: {serial: dict de BoardDetector.pose o None}. Usa solo las detecciones "good" si hay
    alguna; si no, todas las disponibles. Promedio ponderado por numero de esquinas.
    Devuelve (T_rig_board, [seriales usados], calidad) o (None, [], None).
    """
    avail = {s: p for s, p in poses.items() if p is not None and s in rig}
    good = {s: p for s, p in avail.items() if p["good"]}
    use, quality = (good, "buena") if good else (avail, "baja")
    if not use:
        return None, [], None
    Ts = [T_inv(rig[s]) @ T_from(p["R"], p["t"]) for s, p in use.items()]
    return mean_pose(Ts, [p["weight"] for p in use.values()]), list(use), quality


def view_geometry(R, t):
    """Elevacion (grados sobre el plano del tablero) y distancia (mm) al centro del tablero."""
    view_dir = R.T @ np.array([0.0, 0.0, 1.0])          # eje optico en el marco del tablero
    elev = float(np.degrees(np.arcsin(min(1.0, abs(view_dir[2])))))
    dist = float(np.linalg.norm(R @ BOARD_CENTER_MM + t))
    return elev, dist


def yaw_deg(R0, R):
    """Giro del tablero alrededor de su normal entre la pose R0 y la pose R (grados)."""
    M = R0.T @ R
    return float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))


def rot_angle_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


class YawTracker:
    """Acumula el giro sin saltos en +-180 y lo hace positivo en el sentido en que se gire."""

    def __init__(self):
        self.prev = None
        self.unwrapped = 0.0
        self.direction = 0

    def update(self, raw):
        if self.prev is None:
            self.unwrapped = raw
        else:
            self.unwrapped += (raw - self.prev + 180.0) % 360.0 - 180.0
        self.prev = raw
        if self.direction == 0 and abs(self.unwrapped) > 5.0:
            self.direction = 1 if self.unwrapped > 0 else -1
        return self.unwrapped * (self.direction or 1)


# ----------------------------------------------------------------------------- camara
def intr_to_K(intr):
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64)
    return K, np.array(intr.coeffs, dtype=np.float64)


def intr_to_dict(intr):
    return dict(width=intr.width, height=intr.height, fx=intr.fx, fy=intr.fy, ppx=intr.ppx,
                ppy=intr.ppy, model=str(intr.model), coeffs=list(intr.coeffs))


def discover(reset=False):
    ctx = rs.context()
    if reset:
        for d in ctx.query_devices():
            d.hardware_reset()
        print("Reiniciando camaras (8 s)...")
        time.sleep(8)
        ctx = rs.context()
    out = []
    for d in ctx.query_devices():
        name = d.get_info(rs.camera_info.name)
        if "L515" not in name:
            continue
        usb = (d.get_info(rs.camera_info.usb_type_descriptor)
               if d.supports(rs.camera_info.usb_type_descriptor) else "?")
        out.append(dict(serial=d.get_info(rs.camera_info.serial_number), name=name,
                        firmware=d.get_info(rs.camera_info.firmware_version), usb=usb))
    return sorted(out, key=lambda x: x["serial"])


class L515:
    def __init__(self, serial, label=None):
        self.serial = serial
        self.label = label or serial
        self.pipe = None
        self.profile = None

    def start(self, depth=True, color=True, retries=3):
        for attempt in range(1, retries + 1):
            try:
                cfg = rs.config()
                cfg.enable_device(self.serial)
                if depth:
                    cfg.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, FPS)
                if color:
                    cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, FPS)
                self.pipe = rs.pipeline()
                self.profile = self.pipe.start(cfg)
                if depth:
                    self._set_short_range()
                return
            except RuntimeError as e:
                print(f"  [{self.label}] no inicio ({e}); intento {attempt}/{retries}")
                self.stop()
                time.sleep(1.5)
        raise RuntimeError(f"La camara {self.serial} no pudo iniciar")

    def _set_short_range(self):
        s = self.profile.get_device().first_depth_sensor()
        try:
            s.set_option(rs.option.visual_preset, int(rs.l500_visual_preset.short_range))
        except Exception as e:  # noqa: BLE001
            print(f"  [{self.label}] aviso: no se pudo fijar el preset Short Range ({e})")

    def stop(self):
        if self.pipe is not None:
            try:
                self.pipe.stop()
            except Exception:  # noqa: BLE001
                pass
        self.pipe, self.profile = None, None

    def wait(self, timeout_ms=5000):
        return self.pipe.wait_for_frames(timeout_ms)

    def calibration(self):
        dprof = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
        cprof = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        ext = dprof.get_extrinsics_to(cprof)
        s = self.profile.get_device().first_depth_sensor()
        try:
            preset = str(rs.l500_visual_preset(int(s.get_option(rs.option.visual_preset))))
        except Exception:  # noqa: BLE001
            preset = "?"
        return dict(serial=self.serial, depth_scale_m=s.get_depth_scale(), visual_preset=preset,
                    depth_intrinsics=intr_to_dict(dprof.get_intrinsics()),
                    color_intrinsics=intr_to_dict(cprof.get_intrinsics()),
                    depth_to_color=dict(rotation_colmajor=list(ext.rotation),
                                        translation_m=list(ext.translation)))

    def temperatures(self):
        out = {}
        s = self.profile.get_device().first_depth_sensor()
        for opt in s.get_supported_options():
            name = str(opt).split(".")[-1]
            if "temperature" in name:
                try:
                    out[name] = float(s.get_option(opt))
                except Exception:  # noqa: BLE001
                    pass
        return out

    def grab(self, n_depth, warmup):
        """Enciende la camara sola, descarta warmup cuadros y guarda n_depth cuadros + 1 color."""
        self.start(depth=True, color=True)
        try:
            for _ in range(warmup):
                self.wait()
            depths, stamps, color, last = [], [], None, None
            while len(depths) < n_depth:
                fs = self.wait()
                d, c = fs.get_depth_frame(), fs.get_color_frame()
                if not d or not c:
                    continue
                if d.get_frame_number() == last:      # cuadro repetido: no aporta informacion
                    continue
                last = d.get_frame_number()
                depths.append(np.asanyarray(d.get_data()).copy())
                stamps.append(d.get_timestamp())
                color = np.asanyarray(c.get_data()).copy()
            calib, temps = self.calibration(), self.temperatures()
        finally:
            self.stop()
        return depths, color, calib, temps, stamps

    def color_pose(self, det, tries=60):
        """Pose del tablero vista por esta camara usando solo color (sin laser)."""
        self.start(depth=False, color=True)
        try:
            for _ in range(tries):
                c = self.wait().get_color_frame()
                if not c:
                    continue
                K, dist = intr_to_K(c.profile.as_video_stream_profile().get_intrinsics())
                img = np.asanyarray(c.get_data())
                p = det.pose(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), K, dist)
                if p is not None:
                    p["image"] = img.copy()
                    return p
        finally:
            self.stop()
        return None

    def color_pose_avg(self, det, n=15, tries=150):
        """Promedio de n poses "good" seguidas (solo color). Para calibrar el rig."""
        self.start(depth=False, color=True)
        poses = []
        try:
            for _ in range(tries):
                c = self.wait().get_color_frame()
                if not c:
                    continue
                K, dist = intr_to_K(c.profile.as_video_stream_profile().get_intrinsics())
                p = det.pose(cv2.cvtColor(np.asanyarray(c.get_data()), cv2.COLOR_BGR2GRAY), K, dist)
                if p is not None and p["good"] and p["fuente"] == "charuco":
                    poses.append(p)
                    if len(poses) >= n:
                        break
        finally:
            self.stop()
        if len(poses) < max(3, n // 2):
            return None
        Ts = [T_from(p["R"], p["t"]) for p in poses]
        T = mean_pose(Ts)
        jitter = max(pose_diff(T, Ti)[1] for Ti in Ts)
        return dict(R=T[:3, :3], t=T[:3, 3], n=int(np.median([p["n"] for p in poses])),
                    reproj_px=float(np.mean([p["reproj_px"] for p in poses])),
                    frames=len(poses), jitter_mm=jitter)


# ----------------------------------------------------------------------------- varios
def sdk_version():
    """Version de pyrealsense2 (algunas ruedas de Windows no traen __version__)."""
    v = getattr(rs, "__version__", None)
    if v:
        return v
    try:
        from importlib.metadata import version
        return version("pyrealsense2")
    except Exception:  # noqa: BLE001
        return "desconocida"


def to_jsonable(x):
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2, ensure_ascii=False)


def beep(ok=True):
    try:
        import winsound
        winsound.Beep(1200 if ok else 400, 150 if ok else 500)
    except Exception:  # noqa: BLE001
        print("\a", end="", flush=True)