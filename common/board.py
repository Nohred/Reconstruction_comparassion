"""Acceso al codigo real de captura (Captura_real/scripts/l515_common.py) sin modificarlo.

l515_common importa pyrealsense2 al inicio, pero lo que reutilizamos (parametros del ChArUco,
BoardDetector, T_from/T_inv, mean_pose, save_json, intr_to_K...) no lo necesita. Si
pyrealsense2 no esta instalado (Python de Isaac Sim) se registra un modulo vacio en su lugar.
Se carga sin escribir __pycache__ para no tocar Captura_real/.
"""
import importlib.util
import sys
import types

from common.config import load_config, repo_path

_MODULE_NAME = "l515_common"
STUB_ATTR = "_twin_stub"


def _ensure_realsense_importable():
    if "pyrealsense2" in sys.modules:
        return
    try:
        import pyrealsense2  # noqa: F401
    except ImportError:
        stub = types.ModuleType("pyrealsense2")
        setattr(stub, STUB_ATTR, True)
        sys.modules["pyrealsense2"] = stub


def load_real_script(name, scripts_dir=None):
    """Importa Captura_real/scripts/<name>.py como modulo `name` (una sola vez, sin __pycache__).

    Se usa para l515_common y capture_turntable (este importa `from l515_common import ...`,
    por eso l515_common se registra con su nombre real en sys.modules).
    """
    if name in sys.modules:
        return sys.modules[name]
    if scripts_dir is None:
        scripts_dir = load_config("common")["paths"]["real_scripts"]
    path = repo_path(scripts_dir) / f"{name}.py"
    _ensure_realsense_importable()
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise
    finally:
        sys.dont_write_bytecode = old
    return mod


def load_l515_common(scripts_dir=None):
    return load_real_script(_MODULE_NAME, scripts_dir)


L = load_l515_common()


def board_params():
    """Parametros del tablero tal como los define el codigo real."""
    return dict(squares_x=L.SQUARES_X, squares_y=L.SQUARES_Y, square_mm=L.SQUARE_MM,
                marker_mm=L.MARKER_MM, aruco_dict=int(L.ARUCO_DICT),
                center_mm=[float(v) for v in L.BOARD_CENTER_MM])


def board_size_mm():
    return L.SQUARES_X * L.SQUARE_MM, L.SQUARES_Y * L.SQUARE_MM


def page_texture(page_cfg):
    """Imagen (uint8, gris) de la hoja carta con el tablero, mismos parametros que el codigo real.

    Devuelve (imagen, rect_mm) con rect_mm = (x0, y0, x1, y1) de la hoja en el marco del tablero.
    Fila 0 de la imagen = borde de la hoja con y minima del tablero.
    """
    import cv2
    import numpy as np

    ppm = page_cfg["texture_px_per_mm"]
    pw, ph = page_cfg["page_size_mm"]
    ox, oy = page_cfg["board_offset_mm"]
    bw, bh = board_size_mm()
    dictionary = cv2.aruco.getPredefinedDictionary(L.ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((L.SQUARES_X, L.SQUARES_Y), L.SQUARE_MM, L.MARKER_MM, dictionary)
    bimg = board.generateImage((int(round(bw * ppm)), int(round(bh * ppm))), marginSize=0,
                               borderBits=page_cfg["border_bits"])
    page = np.full((int(round(ph * ppm)), int(round(pw * ppm))), 255, np.uint8)
    x0, y0 = int(round(ox * ppm)), int(round(oy * ppm))
    page[y0:y0 + bimg.shape[0], x0:x0 + bimg.shape[1]] = bimg
    # rect de la hoja en mm del tablero, usando el offset redondeado al pixel (textura exacta)
    rect = (-x0 / ppm, -y0 / ppm, -x0 / ppm + page.shape[1] / ppm, -y0 / ppm + page.shape[0] / ppm)
    return page, rect
