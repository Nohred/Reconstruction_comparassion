"""Vista en vivo de todas las L515 a la vez, para acomodarlas.

Solo usa el stream de COLOR (el laser no se enciende), asi que no hay interferencia.
En cada vista se marca: esquinas detectadas, elevacion y distancia al centro del tablero,
el centro de la imagen (cruz) y el centro del tablero (punto). Acomoda cada camara hasta que
el punto quede sobre la cruz y la elevacion se acerque a su objetivo.

    python preview_l515.py                    # cruz apuntando al centro del tablero
    python preview_l515.py --altura-mm 230    # cruz a media altura de un objeto de 230 mm

Teclas:  Q / ESC salir   |   S guardar captura de pantalla en check_out/
"""
import argparse
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from l515_common import (MAX_REPROJ_PX, MIN_CORNERS, MIN_MARKERS, BoardDetector, aim_point,
                         discover, draw_detection, intr_to_K, view_geometry)

PREV_W, PREV_H = 960, 540          # resolucion de color ligera para la vista previa
TILE_W, TILE_H = 640, 360
TARGETS = [17, 33, 65]             # elevaciones del montaje oficial: baja, media, alta
ELEV_TOL = 5
DIST_RANGE = (350, 600)            # mm al centro del tablero


def put(img, text, y, color=(255, 255, 255), scale=0.6):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--altura-mm", type=float, default=0.0,
                    help="altura del objeto; la cruz apunta a la mitad de esa altura")
    args = ap.parse_args()
    infos = discover()
    if not infos:
        raise SystemExit("No se encontro ninguna L515.")
    det = BoardDetector()
    cams = []
    for info in infos:
        cfg = rs.config()
        cfg.enable_device(info["serial"])
        cfg.enable_stream(rs.stream.color, PREV_W, PREV_H, rs.format.bgr8, 30)
        pipe = rs.pipeline()
        prof = pipe.start(cfg)
        intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        K, dist = intr_to_K(intr)
        cams.append(dict(serial=info["serial"], pipe=pipe, K=K, dist=dist, elev=None))
        print(f"Vista previa: {info['serial']}")

    Path("check_out").mkdir(exist_ok=True)
    win = "Vista previa L515 (Q salir, S guardar)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        while True:
            tiles = []
            for cam in cams:
                new = False
                try:
                    fs = cam["pipe"].poll_for_frames()
                    if fs:
                        c = fs.get_color_frame()
                        if c:
                            cam["img"] = np.asanyarray(c.get_data()).copy()
                            new = True
                except RuntimeError as e:
                    cam["error"] = str(e)
                    print(f"[{cam['serial']}] error de camara: {e}")
                if not new and "tile" in cam:      # sin cuadro nuevo: reusa la vista anterior
                    tiles.append(cam["tile"].copy())
                    continue
                img = cam.get("img")
                if img is None:
                    tiles.append(np.zeros((TILE_H, TILE_W, 3), np.uint8))
                    continue
                vis = img.copy()
                h, w = vis.shape[:2]
                cv2.drawMarker(vis, (w // 2, h // 2), (255, 255, 0), cv2.MARKER_CROSS, 40, 2)
                p = det.pose(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cam["K"], cam["dist"],
                             min_corners=6)
                cam["p"] = p
                if p is not None:
                    draw_detection(vis, p)
                    aim = aim_point(p["R"], p["t"], args.altura_mm)
                    cen, _ = cv2.projectPoints(aim.reshape(1, 3),
                                               cv2.Rodrigues(p["R"])[0], p["t"], cam["K"], cam["dist"])
                    cx, cy = cen.ravel().astype(int)
                    cv2.circle(vis, (cx, cy), 10, (0, 0, 255), -1)
                    cam["elev"], cam["dist_mm"] = view_geometry(p["R"], p["t"])
                    cam["off"] = (cx - w // 2, cy - h // 2)
                else:
                    cam["elev"] = None
                cam["tile"] = cv2.resize(vis, (TILE_W, TILE_H))
                tiles.append(cam["tile"].copy())

            # etiquetas baja/media/alta segun elevacion actual
            seen = sorted([c for c in cams if c["elev"] is not None], key=lambda c: c["elev"])
            names = ["BAJA", "MEDIA", "ALTA"]
            role = {c["serial"]: i for i, c in enumerate(seen)} if len(seen) == len(cams) == 3 else {}
            for cam, tile in zip(cams, tiles):
                put(tile, cam["serial"], 24, (255, 255, 0))
                p = cam.get("p")
                if p is None:
                    put(tile, "TABLERO NO DETECTADO", 52, (0, 0, 255))
                    continue
                txt = f"{p['n']} esq"
                if p["fuente"] != "charuco":
                    txt += f" + {p['n_markers']} marc"
                txt += f"  reproj {p['reproj_px']:.2f} px"
                if not p["good"]:
                    txt += "  (pose dudosa)"
                put(tile, txt, 52, (0, 255, 0) if p["good"] else (0, 165, 255))
                line = f"elev {cam['elev']:4.0f} deg   dist {cam['dist_mm']:4.0f} mm"
                col = (255, 255, 255)
                if cam["serial"] in role:
                    i = role[cam["serial"]]
                    tgt = TARGETS[i]
                    line += f"   [{names[i]} obj {tgt}]"
                    col = (0, 255, 0) if abs(cam["elev"] - tgt) <= ELEV_TOL else (0, 165, 255)
                put(tile, line, 80, col)
                dx, dy = cam["off"]
                ok_off = abs(dx) < 60 and abs(dy) < 60
                hint = "centrado" if ok_off else (
                    f"mueve: {'der' if dx > 0 else 'izq'} {abs(dx)} px, "
                    f"{'abajo' if dy > 0 else 'arriba'} {abs(dy)} px")
                put(tile, hint, 108, (0, 255, 0) if ok_off else (0, 165, 255))
                ok_d = DIST_RANGE[0] <= cam["dist_mm"] <= DIST_RANGE[1]
                if not ok_d:
                    put(tile, f"distancia fuera de {DIST_RANGE[0]}-{DIST_RANGE[1]} mm", 136,
                        (0, 165, 255))

            while len(tiles) < 4:
                tiles.append(np.zeros((TILE_H, TILE_W, 3), np.uint8))
            grid = np.vstack([np.hstack(tiles[0:2]), np.hstack(tiles[2:4])])
            if len(cams) <= 3:
                g = grid[TILE_H:, TILE_W:]
                put(g, "Objetivo por camara:", 30)
                put(g, f"- pose: {MIN_CORNERS}+ esq o {MIN_MARKERS}+ marc, reproj <= {MAX_REPROJ_PX}", 60)
                put(g, "- punto rojo sobre la cruz", 90)
                put(g, f"- elevacion {' / '.join(map(str, TARGETS))} deg (+-{ELEV_TOL})", 120)
                put(g, f"- distancia {DIST_RANGE[0]}-{DIST_RANGE[1]} mm", 150)
                put(g, "La cruz es el centro de la imagen;", 200, (255, 255, 0))
                put(g, ("el punto rojo, el centro del tablero." if args.altura_mm <= 0 else
                        f"el punto rojo, {args.altura_mm / 2:.0f} mm sobre el centro."), 228, (255, 255, 0))
            cv2.imshow(win, grid)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                fn = Path("check_out") / f"preview_{time.strftime('%H%M%S')}.png"
                cv2.imwrite(str(fn), grid)
                print(f"Guardado {fn}")
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:   # cerraste la ventana
                break
            time.sleep(0.005)
    except Exception:  # noqa: BLE001
        msg = traceback.format_exc()
        print("\nLa vista previa se detuvo por este error:\n" + msg)
        (Path("check_out") / "preview_error.txt").write_text(msg, encoding="utf-8")
    finally:
        for cam in cams:
            try:
                cam["pipe"].stop()
            except Exception:  # noqa: BLE001
                pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()