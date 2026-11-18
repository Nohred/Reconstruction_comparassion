"""Paso 2: captura automatica con 3 L515 fijas y base giratoria manual.

El tablero ChArUco va ENCIMA de la base giratoria y el objeto pegado al tablero: todo gira junto.
La camara de en medio vigila el giro del tablero (solo color, sin laser). Cuando la base llega al
angulo objetivo y deja de moverse, el script captura las tres camaras UNA POR UNA (para que sus
laseres no se interfieran), suena un pitido y te pide el siguiente angulo.

Antes: python calibrar_rig.py (tablero vacio). Con rig.json, la pose del tablero de cada parada
se calcula combinando las camaras que mejor lo ven y se pasa a las demas, asi una camara tapada por
el objeto sigue teniendo pose. Sin rig.json cada camara usa solo su propia deteccion.

    python capture_turntable.py bunny --paradas 12
    python capture_turntable.py busto_cesar --paradas 16 --notas "PLA gris, luz de techo"
    python capture_turntable.py busto_cesar --ref alta     # forzar camara que vigila el giro

Teclas en la ventana:  ESPACIO = capturar ya   |   Q o ESC = terminar y guardar

Salida:
  capturas/<objeto>/<fecha_hora>/
      session.json                  parametros, camaras, tiempos y resumen por parada
      <baja|media|alta>/calib.json  intrinsecos de color y profundidad, extrinseco, escala
      <baja|media|alta>/stop_XX/depth_00..NN.png (uint16 crudo), color.png, meta.json
      rig.json                      copia del rig usado (si existe)

meta.json: R_board_to_color / t_board_to_color_mm es la pose FINAL (la que debe usar la fusion);
pose_fuente dice si salio de la deteccion propia ("directa") o del rig ("rig"); el bloque
"directa" guarda la deteccion propia y su diferencia contra la pose del rig (diagnostico).
"""
import argparse
import re
import shutil
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from l515_common import (L515, RIG_FILE, BoardDetector, draw_detection, T_from, T_inv, YawTracker, beep,
                         discover, fuse_board_pose, intr_to_K, load_rig, pose_diff, rot_angle_deg,
                         save_json, sdk_version, view_geometry, yaw_deg)

WIN = "Captura L515 - base giratoria"
STABLE_FRAMES = 18          # ~0.6 s quieto a 30 fps
STABLE_DEG, STABLE_MM = 0.3, 1.0
MOVED_DEG, MOVED_MM = 0.5, 1.5
MAX_REDO = 2
RIG_WARN_DEG, RIG_WARN_MM = 1.0, 5.0   # directa vs rig: mas que esto sugiere que algo se movio


def put_lines(img, lines, color=(255, 255, 255)):
    y = 36
    for ln in lines:
        cv2.putText(img, ln, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, ln, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
        y += 34


def show(img):
    cv2.imshow(WIN, cv2.resize(img, (960, 540)))
    return cv2.waitKey(1) & 0xFF


def show_status(msg, color=(0, 255, 255)):
    img = np.zeros((540, 960, 3), np.uint8)
    put_lines(img, msg if isinstance(msg, list) else [msg], color)
    cv2.imshow(WIN, img)
    cv2.waitKey(1)


def monitor(ref, det, R0, tracker, target, tol, k, n):
    """Muestra el giro en vivo. Devuelve (pose, angulo) al llegar estable al objetivo, o None."""
    ref.start(depth=False, color=True)
    hist = deque(maxlen=STABLE_FRAMES)
    K = dist = None
    try:
        while True:
            c = ref.wait().get_color_frame()
            if not c:
                continue
            if K is None:
                K, dist = intr_to_K(c.profile.as_video_stream_profile().get_intrinsics())
            img = np.asanyarray(c.get_data())
            vis = img.copy()
            p = det.pose(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), K, dist)
            theta = None
            if p is not None:
                draw_detection(vis, p)
                if R0 is not None:
                    theta = tracker.update(yaw_deg(R0, p["R"]))
                    hist.append((theta, p["t"].copy()))
            else:
                hist.clear()

            stable = False
            if len(hist) == hist.maxlen:
                th = np.array([h[0] for h in hist])
                tt = np.array([h[1] for h in hist])
                stable = np.ptp(th) < STABLE_DEG and np.ptp(tt, axis=0).max() < STABLE_MM

            lines = [f"Parada {k + 1}/{n}   objetivo {target:5.1f} deg"]
            color = (255, 255, 255)
            if p is None:
                lines.append("Tablero NO detectado (mano o reflejo?)")
                color = (0, 0, 255)
            elif R0 is None:
                lines.append("Coloca el objeto, aleja las manos y presiona ESPACIO")
            else:
                delta = target - theta
                lines.append(f"Actual {theta:6.1f} deg    faltan {delta:+6.1f} deg")
                if abs(delta) < tol:
                    lines.append("Suelta la base..." if not stable else "Estable: capturando")
                    color = (0, 255, 0)
                elif delta < 0:
                    lines.append("Te pasaste: regresa un poco")
                    color = (0, 165, 255)
                else:
                    lines.append("Gira la base")
            lines.append((f"{p['n']} esq + {p['n_markers']} marc" if p else "0 esq") +
                         " | ESPACIO capturar ya | Q terminar")
            put_lines(vis, lines, color)
            key = show(vis)

            if key in (ord("q"), 27):
                return None
            if key == ord(" ") and p is not None:
                return p, (theta if theta is not None else 0.0)
            if R0 is not None and p is not None and stable and abs(target - theta) < tol:
                return p, theta
    finally:
        ref.stop()


def capture_stop(cams, ref, det, k, session_dir, args, R0_by_cam, calib_written, rig):
    """Captura todas las camaras en secuencia; la de referencia al final (detecta movimiento).

    Devuelve (directas, finales, info): poses detectadas por cada camara, poses finales
    (directa o derivada del rig) e info de la fusion de poses.
    """
    order = [c for c in cams if c is not ref] + [ref]
    direct, metas, dirs = {}, {}, {}
    for cam in order:
        show_status([f"Parada {k + 1}: capturando camara {cam.label}...",
                     "No toques la base"])
        depths, color, calib, temps, stamps = cam.grab(args.cuadros, args.warmup)
        cam_dir = session_dir / cam.label
        stop_dir = cam_dir / f"stop_{k:02d}"
        if stop_dir.exists():
            shutil.rmtree(stop_dir)
        stop_dir.mkdir(parents=True)
        if cam.label not in calib_written:
            save_json(cam_dir / "calib.json", calib)
            calib_written.add(cam.label)
        for i, d in enumerate(depths):
            cv2.imwrite(str(stop_dir / f"depth_{i:02d}.png"), d)
        cv2.imwrite(str(stop_dir / "color.png"), color)

        K, dist = intr_to_K_dict(calib["color_intrinsics"])
        direct[cam.serial] = det.pose(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), K, dist)
        metas[cam.serial] = dict(serial=cam.serial, label=cam.label, stop=k,
                                 depth_timestamps_ms=stamps, temperatures=temps,
                                 time=datetime.now().isoformat())
        dirs[cam.serial] = stop_dir

    # pose final por camara
    T_rig, used, quality = (fuse_board_pose(direct, rig) if rig else (None, [], None))
    final, info = {}, dict(rig=bool(rig), camaras_usadas=[], calidad=quality, max_diff_deg=0.0,
                           max_diff_mm=0.0)
    if T_rig is not None:
        info["camaras_usadas"] = [c.label for c in cams if c.serial in used]
    for cam in cams:
        p = direct[cam.serial]
        meta = metas[cam.serial]
        if T_rig is not None and cam.serial in rig:
            T = rig[cam.serial] @ T_rig
            src = "rig"
        elif p is not None:
            T = T_from(p["R"], p["t"])
            src = "directa"
        else:
            T, src = None, None
        if p is not None:
            meta["directa"] = dict(corners=p["n"], marcadores=p["n_markers"], fuente=p["fuente"],
                                   reproj_px=p["reproj_px"], filas=p["rows"],
                                   columnas=p["cols"], buena=p["good"],
                                   R_board_to_color=p["R"], t_board_to_color_mm=p["t"])
            if src == "rig":
                dd, dm = pose_diff(T_from(p["R"], p["t"]), T)
                meta["directa"].update(diff_vs_rig_deg=dd, diff_vs_rig_mm=dm)
                if p["good"]:
                    info["max_diff_deg"] = max(info["max_diff_deg"], dd)
                    info["max_diff_mm"] = max(info["max_diff_mm"], dm)
        meta.update(board_detected=T is not None, tablero_visto_directo=p is not None,
                    pose_fuente=src,
                    pose_camaras_usadas=info["camaras_usadas"] if src == "rig" else None,
                    pose_calidad=quality if src == "rig" else (
                        None if p is None else ("buena" if p["good"] else "baja")))
        if T is not None:
            R, t = T[:3, :3], T[:3, 3]
            elev, distance = view_geometry(R, t)
            if k == 0 or cam.label not in R0_by_cam:
                R0_by_cam[cam.label] = R
            meta.update(elevation_deg=elev, distance_mm=distance,
                        yaw_from_stop0_deg=yaw_deg(R0_by_cam[cam.label], R),
                        R_board_to_color=R, t_board_to_color_mm=t)
            final[cam.serial] = dict(R=R, t=t)
        else:
            final[cam.serial] = None
            print(f"  aviso: no hay pose para la camara {cam.label} en la parada {k + 1}")
        save_json(dirs[cam.serial] / "meta.json", meta)
    if info["max_diff_deg"] > RIG_WARN_DEG or info["max_diff_mm"] > RIG_WARN_MM:
        print(f"  AVISO: deteccion directa vs rig difiere {info['max_diff_deg']:.2f} deg / "
              f"{info['max_diff_mm']:.1f} mm. Si se repite, una camara se movio: "
              "deten la captura y corre calibrar_rig.py.")
    return direct, final, info


def intr_to_K_dict(d):
    K = np.array([[d["fx"], 0, d["ppx"]], [0, d["fy"], d["ppy"]], [0, 0, 1]], np.float64)
    return K, np.array(d["coeffs"], np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("objeto", help="nombre del objeto, p. ej. P1, busto_cesar, busto_deadpool")
    ap.add_argument("--paradas", type=int, default=12, help="paradas por vuelta (12 = cada 30 deg)")
    ap.add_argument("--tol", type=float, default=3.0, help="tolerancia al angulo objetivo (deg)")
    ap.add_argument("--cuadros", type=int, default=10, help="cuadros de profundidad por parada")
    ap.add_argument("--warmup", type=int, default=45, help="cuadros descartados al encender")
    ap.add_argument("--salida", default="capturas")
    ap.add_argument("--notas", default="", help="material, iluminacion, etc.")
    ap.add_argument("--reset", action="store_true", help="reinicia las camaras antes de empezar")
    ap.add_argument("--rig", default=RIG_FILE, help="archivo de calibrar_rig.py")
    ap.add_argument("--ref", choices=["baja", "media", "alta"], default=None,
                    help="camara que vigila el giro (por defecto la que ve mas esquinas)")
    args = ap.parse_args()

    infos = discover(reset=args.reset)
    if not infos:
        sys.exit("No se encontro ninguna L515.")
    det = BoardDetector()
    loaded = load_rig(args.rig)
    rig, rig_data = loaded if loaded else (None, None)
    if rig is not None:
        missing = [i["serial"] for i in infos if i["serial"] not in rig]
        if missing:
            sys.exit(f"{args.rig} no incluye las camaras {missing}. Corre calibrar_rig.py.")
        print(f"Rig: {args.rig} ({rig_data['fecha'][:16]})")
    else:
        print(f"AVISO: no existe {args.rig}; cada camara usara solo su propia deteccion.")

    cams = []
    print("Identificando camaras por su elevacion...")
    for info in infos:
        cam = L515(info["serial"])
        cam.info = info
        cam.p0 = cam.color_pose(det)
        cams.append(cam)
    direct0 = {c.serial: c.p0 for c in cams}
    T_rig0 = fuse_board_pose(direct0, rig)[0] if rig else None
    for cam in cams:
        if T_rig0 is not None:
            T = rig[cam.serial] @ T_rig0
            if cam.p0 is not None and cam.p0["good"]:
                dd, dm = pose_diff(T_from(cam.p0["R"], cam.p0["t"]), T)
                print(f"  {cam.serial}: directa vs rig {dd:.2f} deg / {dm:.1f} mm")
                if dd > RIG_WARN_DEG or dm > RIG_WARN_MM:
                    beep(ok=False)
                    if input("  Parece que esta camara se movio desde calibrar_rig.py. "
                             "Continuar de todos modos? [s/N] ").strip().lower() != "s":
                        sys.exit("Corre calibrar_rig.py con el tablero vacio y vuelve a intentar.")
        elif cam.p0 is not None:
            T = T_from(cam.p0["R"], cam.p0["t"])
        else:
            sys.exit(f"La camara {cam.serial} no ve el tablero. Ajusta su encuadre (check_out/).")
        cam.elev, cam.dist = view_geometry(T[:3, :3], T[:3, 3])
    cams.sort(key=lambda c: c.elev)
    labels = {1: ["unica"], 2: ["baja", "alta"], 3: ["baja", "media", "alta"]}.get(
        len(cams), [f"cam{i}" for i in range(len(cams))])
    for cam, lab in zip(cams, labels):
        cam.label = lab
        print(f"  {lab:6s} {cam.serial}  elevacion {cam.elev:4.0f} deg  distancia {cam.dist:4.0f} mm")
    if args.ref:
        ref = next(c for c in cams if c.label == args.ref)
    else:      # la que mejor ve el tablero CON el objeto puesto
        ref = max(cams, key=lambda c: (c.p0 is not None and c.p0["good"],
                                       c.p0["weight"] if c.p0 is not None else 0))
    if ref.p0 is None:
        sys.exit(f"La camara de referencia ({ref.label}) no ve el tablero; usa --ref con otra.")
    print(f"Camara de referencia para vigilar el giro: {ref.label} "
          f"({ref.p0['n']} esq + {ref.p0['n_markers']} marc)")

    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", args.objeto)
    session_dir = Path(args.salida) / slug / datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True)
    session = dict(objeto=args.objeto, notas=args.notas, inicio=datetime.now().isoformat(),
                   sdk=sdk_version(), opencv=cv2.__version__, parametros=vars(args),
                   camaras=[dict(label=c.label, serial=c.serial, firmware=c.info["firmware"],
                                 usb=c.info["usb"], elevacion_deg=c.elev, distancia_mm=c.dist)
                            for c in cams],
                   referencia=ref.label, rig=rig_data, paradas=[])
    save_json(session_dir / "session.json", session)
    if rig_data is not None:
        save_json(session_dir / "rig.json", rig_data)

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    tracker, R0 = YawTracker(), None
    R0_by_cam, calib_written = {}, set()
    step = 360.0 / args.paradas
    t_start, k, redo = time.time(), 0, 0
    while k < args.paradas:
        target = k * step
        res = monitor(ref, det, R0, tracker, target, args.tol, k, args.paradas)
        if res is None:
            print("Captura terminada por el usuario.")
            break
        p_mon, theta = res
        if R0 is None:
            R0, theta = p_mon["R"], 0.0
        t0 = time.time()
        direct, final, pinfo = capture_stop(cams, ref, det, k, session_dir, args, R0_by_cam,
                                            calib_written, rig)

        p_ref = direct[ref.serial] or final[ref.serial]
        moved = (p_ref is None or rot_angle_deg(p_mon["R"], p_ref["R"]) > MOVED_DEG or
                 np.linalg.norm(p_mon["t"] - p_ref["t"]) > MOVED_MM)
        if moved and redo < MAX_REDO:
            redo += 1
            beep(ok=False)
            print(f"  La base se movio durante la parada {k + 1}: se repite ({redo}/{MAX_REDO}).")
            continue
        entry = dict(indice=k, objetivo_deg=target, medido_deg=theta,
                     duracion_s=round(time.time() - t0, 1), repeticiones=redo,
                     movimiento_sin_resolver=bool(moved),
                     camaras_con_tablero=[c.label for c in cams if direct[c.serial] is not None],
                     camaras_con_pose=[c.label for c in cams if final[c.serial] is not None],
                     pose=pinfo)
        session["paradas"].append(entry)
        session["duracion_total_s"] = round(time.time() - t_start, 1)
        save_json(session_dir / "session.json", session)
        print(f"  Parada {k + 1}/{args.paradas} lista: {theta:6.1f} deg, {entry['duracion_s']} s, "
              f"tablero visto por {len(entry['camaras_con_tablero'])}/{len(cams)}, "
              f"pose en {len(entry['camaras_con_pose'])}/{len(cams)} camaras")
        beep(ok=True)
        k, redo = k + 1, 0

    cv2.destroyAllWindows()
    session["fin"] = datetime.now().isoformat()
    session["duracion_total_s"] = round(time.time() - t_start, 1)
    save_json(session_dir / "session.json", session)
    print(f"\nSesion guardada en {session_dir.resolve()}")
    print(f"Duracion total: {session['duracion_total_s'] / 60:.1f} min")
    for cam in cams:
        n_ok = sum(cam.label in s["camaras_con_tablero"] for s in session["paradas"])
        n_pose = sum(cam.label in s["camaras_con_pose"] for s in session["paradas"])
        print(f"  {cam.label:6s}: tablero visto en {n_ok}/{len(session['paradas'])}, "
              f"pose en {n_pose}/{len(session['paradas'])}")


if __name__ == "__main__":
    main()