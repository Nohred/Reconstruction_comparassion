"""Paso 1b: calibra la posicion relativa de las L515 (rig) con el tablero VACIO.

Como las camaras no se mueven entre objetos, basta medir una vez la transformacion fija entre
ellas. Despues, en cada parada de la captura, la pose del tablero se calcula con las camaras que
mejor lo ven y se pasa a las demas. Asi la pose ya no depende de cuanto tablero tape el objeto.

Solo usa el stream de color (sin laser). Gira la base a varias posiciones; en cada una se
promedian ~15 poses por camara.

    python calibrar_rig.py                 # 4 posiciones (cada ~90 deg)
    python calibrar_rig.py --posiciones 6

Repitelo cada dia de captura y siempre que alguien toque o mueva una camara o tripie.
Salida: rig.json (lo leen capture_turntable.py y la fusion).
"""
import argparse
import sys
from datetime import datetime

import numpy as np

from l515_common import (L515, MAX_REPROJ_PX, MIN_CORNERS, RIG_FILE, SQUARE_MM, SQUARES_X,
                         SQUARES_Y, BoardDetector, T_from, T_inv, beep, discover, mean_pose,
                         pose_diff, save_json, sdk_version, view_geometry)

WARN_DEG, WARN_MM = 0.5, 3.0      # dispersion maxima aceptable entre posiciones


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--posiciones", type=int, default=4)
    ap.add_argument("--min-esquinas", type=int, default=30,
                    help="esquinas minimas por camara con el tablero vacio")
    ap.add_argument("--salida", default=RIG_FILE)
    args = ap.parse_args()

    infos = discover()
    if len(infos) < 2:
        sys.exit("Se necesitan al menos 2 L515 conectadas.")
    det = BoardDetector()
    cams = [L515(i["serial"]) for i in infos]
    print(f"Camaras: {', '.join(c.serial for c in cams)}")
    print("Quita el objeto: el tablero debe estar VACIO y sin manos encima.\n")

    meas = []          # una entrada por posicion: {serial: pose promediada}
    step = 360.0 / args.posiciones
    while len(meas) < args.posiciones:
        j = len(meas)
        input(f"Posicion {j + 1}/{args.posiciones}: gira la base a ~{j * step:.0f} deg, "
              f"suelta y presiona ENTER...")
        row = {}
        for cam in cams:
            p = cam.color_pose_avg(det)
            if p is None or p["n"] < args.min_esquinas:
                n = 0 if p is None else p["n"]
                print(f"  {cam.serial}: solo {n} esquinas buenas (min {args.min_esquinas}). "
                      "Revisa reflejos o encuadre y repite esta posicion.")
                beep(ok=False)
                row = None
                break
            print(f"  {cam.serial}: {p['n']} esquinas, reproj {p['reproj_px']:.2f} px, "
                  f"jitter {p['jitter_mm']:.2f} mm ({p['frames']} cuadros)")
            row[cam.serial] = p
        if row is not None:
            meas.append(row)
            beep(ok=True)

    # roles por elevacion (promedio de las posiciones) y camara de referencia = la de en medio
    geo = {}
    for cam in cams:
        g = [view_geometry(m[cam.serial]["R"], m[cam.serial]["t"]) for m in meas]
        geo[cam.serial] = (float(np.mean([e for e, _ in g])), float(np.mean([d for _, d in g])))
    order = sorted(geo, key=lambda s: geo[s][0])
    labels = {2: ["baja", "alta"], 3: ["baja", "media", "alta"]}.get(
        len(order), [f"cam{i}" for i in range(len(order))])
    ref = order[len(order) // 2]

    out = dict(fecha=datetime.now().isoformat(), sdk=sdk_version(), referencia=ref,
               posiciones=args.posiciones,
               tablero=dict(squares_x=SQUARES_X, squares_y=SQUARES_Y, square_mm=SQUARE_MM),
               criterio_pose=dict(min_corners=MIN_CORNERS, max_reproj_px=MAX_REPROJ_PX,
                                  min_filas=3, min_columnas=3),
               nota="p_color = R_rig_to_color @ p_rig + t; el marco del rig es el de color de "
                    "la camara de referencia",
               camaras={})
    print("\nResultado:")
    worst = (0.0, 0.0)
    for s, lab in zip(order, labels):
        # X_j = T_s^j * inv(T_ref^j): transformacion ref -> s medida en cada posicion
        Xs = [T_from(m[s]["R"], m[s]["t"]) @ T_inv(T_from(m[ref]["R"], m[ref]["t"])) for m in meas]
        X = mean_pose(Xs)
        dev = [pose_diff(X, Xj) for Xj in Xs]
        dmax = (max(d[0] for d in dev), max(d[1] for d in dev))
        worst = (max(worst[0], dmax[0]), max(worst[1], dmax[1]))
        out["camaras"][s] = dict(label=lab, elevacion_deg=geo[s][0], distancia_mm=geo[s][1],
                                 R_rig_to_color=X[:3, :3], t_rig_to_color_mm=X[:3, 3],
                                 dispersion_max_deg=dmax[0], dispersion_max_mm=dmax[1],
                                 esquinas=[m[s]["n"] for m in meas],
                                 reproj_px=[m[s]["reproj_px"] for m in meas])
        tag = " (referencia)" if s == ref else ""
        print(f"  {lab:6s} {s}  elev {geo[s][0]:4.1f} deg  dist {geo[s][1]:4.0f} mm  "
              f"dispersion {dmax[0]:.2f} deg / {dmax[1]:.2f} mm{tag}")

    save_json(args.salida, out)
    print(f"\nGuardado {args.salida}")
    if worst[0] > WARN_DEG or worst[1] > WARN_MM:
        beep(ok=False)
        print(f"AVISO: la dispersion entre posiciones ({worst[0]:.2f} deg / {worst[1]:.2f} mm) "
              f"supera {WARN_DEG} deg / {WARN_MM} mm.\n"
              "  Causas tipicas: una camara o tripie se movio durante la calibracion, el tablero "
              "no es plano o tiene reflejos. Corrige y vuelve a correr el script.")
    else:
        print("Rig consistente. Ya puedes capturar sin mover las camaras.")


if __name__ == "__main__":
    main()