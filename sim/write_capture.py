"""Etapa 2 del gemelo: arma la captura sintetica con EXACTAMENTE el formato de la real.

    tools/cv_py.sh -m sim.write_capture Captura_sintetica/bunnny

Corre con el OpenCV de la captura (tools/cv_py.sh): la deteccion del ChArUco, la fusion con el
rig y la escritura de calib.json / stop_XX/{color.png, depth_XX.png, meta.json} las hace la
funcion capture_stop() de Captura_real/scripts/capture_turntable.py SIN modificar, alimentada
con camaras virtuales que devuelven los renders de la etapa 1 (gt/) en lugar de una L515.
session.json se arma igual que en main() de ese script. rig.json: el de la plantilla en modo
gemelo; en modo aleatorio se recalcula con las camaras re-montadas (mismo formato).
"""
import argparse
import json
import copy
import types
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from common.board import L, load_real_script
from common.config import load_config, repo_path
from common.geometry import T_inv
from common.realdata import Capture, read_json
from sim.depth_noise import noisy_frames

FRAME_MS = 1000.0 / L.FPS


class VirtualL515:
    """Imita lo que capture_stop() usa de una L515: label, serial y grab()."""

    def __init__(self, label, real, gt_root, layout, noise, rng, copy_fields):
        self.label = label
        self.real = real
        self.serial = real.calib[label]["serial"]
        self.calib = real.calib[label]
        self.gt_root, self.layout = gt_root, layout
        self.noise, self.rng, self.copy_fields = noise, rng, copy_fields
        self.stop = None
        self.t_ms = 0.0

    def gt_dir(self):
        return self.gt_root / self.label / self.layout["stop_dir"].format(k=self.stop)

    def grab(self, n_depth, warmup):
        d = self.gt_dir()
        color = cv2.imread(str(d / "color.png"), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(d / "depth.png"), cv2.IMREAD_UNCHANGED)
        scale = self.calib["depth_scale_m"]
        if self.noise["activo"]:
            depths = noisy_frames(depth.astype(np.float64) * scale, scale, n_depth,
                                  self.noise["sigma_mm"][self.label], self.noise["p_perdida"],
                                  self.noise["salto_borde_mm"], self.noise["dilatar_borde_px"], self.rng)
        else:
            depths = [depth.copy() for _ in range(n_depth)]
        self.t_ms += (warmup + 1) * FRAME_MS
        stamps = [self.t_ms + i * FRAME_MS for i in range(n_depth)]
        self.t_ms = stamps[-1]
        temps = {}
        if "temperatures" in self.copy_fields and self.stop in self.real.stops.get(self.label, []):
            temps = self.real.view(self.label, self.stop).meta.get("temperatures", {})
        return depths, color, self.calib, temps, stamps


def synthetic_rig(template_rig, gt, real):
    """rig.json de las camaras sinteticas: T_color<-rig con rig = color de la camara de referencia.

    Mismo formato que el de la plantilla (calibrar_rig.py); las cifras de calibracion
    (esquinas, reproj, dispersion) se dejan las de la plantilla como referencia.
    """
    data = copy.deepcopy(template_rig)
    T_c = {c: np.array(v["T_color_from_b0_mm"]) for c, v in gt["camaras_fijas"].items()}
    ref_serial = data["referencia"]
    ref = next(c["label"] for s, c in data["camaras"].items() if s == ref_serial)
    if ref not in T_c:
        raise SystemExit(f"la camara de referencia del rig ({ref}) no se renderizo")
    rig = {}
    for serial, c in data["camaras"].items():
        if c["label"] not in T_c:
            continue
        T = T_c[c["label"]] @ T_inv(T_c[ref])
        elev, dist = L.view_geometry(T_c[c["label"]][:3, :3], T_c[c["label"]][:3, 3])
        c.update(R_rig_to_color=T[:3, :3], t_rig_to_color_mm=T[:3, 3], elevacion_deg=elev,
                 distancia_mm=dist)
        rig[serial] = T
    data["fecha"] = datetime.now().isoformat()
    data["camaras"] = {s: c for s, c in data["camaras"].items() if s in rig}
    return rig, json.loads(json.dumps(L.to_jsonable(data)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sintetica", help="carpeta de la captura sintetica (con ground_truth.json)")
    ap.add_argument("--config", default="twin")
    args = ap.parse_args()
    common, cfg = load_config("common"), load_config(args.config)
    lay = common["capture_layout"]
    out = repo_path(args.sintetica)
    gt = read_json(out / cfg["gt_file"])
    real = Capture.open(repo_path(gt["plantilla"]), lay)
    CT = load_real_script("capture_turntable")
    CT.show_status = lambda *a, **k: None            # sin ventana: solo dibuja el estado en pantalla

    rig, rig_data = L.load_rig(str(real.root / lay["rig_file"]))
    if gt["modo"] != "replay":
        rig, rig_data = synthetic_rig(rig_data, gt, real)
    noise = cfg["ruido_profundidad"]
    rng = np.random.default_rng(noise["semilla"])
    order = [c["label"] for c in real.session["camaras"] if c["label"] in gt["camaras"]]
    cams = [VirtualL515(c, real, out / cfg["gt_dir"], lay, noise, rng, cfg["copiar_de_real"]) for c in order]
    ref = next((c for c in cams if c.label == real.session["referencia"]), cams[-1])
    params = dict(real.session["parametros"])
    params.update(salida=str(out.parent), notas=cfg["notas_sesion"].format(captura=real.root.name, modo=gt["modo"]))
    run_args = types.SimpleNamespace(**params)

    det = L.BoardDetector()
    session = dict(objeto=gt["objeto"]["nombre"] if gt.get("objeto") else gt["nombre"], notas=params["notas"], inicio=datetime.now().isoformat(),
                   sdk=real.session.get("sdk"), opencv=cv2.__version__, parametros=params,
                   camaras=[dict(c) for c in real.session["camaras"] if c["label"] in gt["camaras"]],
                   referencia=ref.label, rig=rig_data, paradas=[])
    L.save_json(out / lay["session_file"], session)
    L.save_json(out / lay["rig_file"], rig_data)

    n_stops = params["paradas"]
    step = 360.0 / n_stops
    tracker, R0, R0_by_cam, calib_written = L.YawTracker(), None, {}, set()
    for k in gt["paradas"]:
        for c in cams:
            c.stop = k
        t0 = datetime.now()
        direct, final, pinfo = CT.capture_stop(cams, ref, det, k, out, run_args, R0_by_cam,
                                               calib_written, rig)
        p_ref = final[ref.serial]
        theta = None
        if p_ref is not None:
            if R0 is None:
                R0 = p_ref["R"]
            theta = tracker.update(L.yaw_deg(R0, p_ref["R"]))
        session["paradas"].append(dict(
            indice=k, objetivo_deg=k * step, medido_deg=theta,
            duracion_s=round((datetime.now() - t0).total_seconds(), 1), repeticiones=0,
            movimiento_sin_resolver=False,
            camaras_con_tablero=[c.label for c in cams if direct[c.serial] is not None],
            camaras_con_pose=[c.label for c in cams if final[c.serial] is not None],
            pose=pinfo))
        print(f"  parada {k:2d}: tablero en {len(session['paradas'][-1]['camaras_con_tablero'])}/{len(cams)}, "
              f"pose {pinfo['calidad']}, directa vs rig max {pinfo['max_diff_mm']:.2f} mm")
    # elevacion/distancia de cada camara con la pose final de la primera parada (como main())
    for entry in session["camaras"]:
        m = read_json(out / entry["label"] / lay["stop_dir"].format(k=gt["paradas"][0]) / lay["meta_file"])
        if "elevation_deg" in m:
            entry.update(elevacion_deg=m["elevation_deg"], distancia_mm=m["distance_mm"])
    session["fin"] = datetime.now().isoformat()
    session["duracion_total_s"] = round(sum(p["duracion_s"] for p in session["paradas"]), 1)
    L.save_json(out / lay["session_file"], session)

    gt["ruido_profundidad"] = noise
    gt["copiado_de_real"] = dict(campos_meta=cfg["copiar_de_real"],
                                 session=["sdk", "parametros", "camaras (serial, firmware, usb)", "referencia"],
                                 rig="rig.json real (las camaras sinteticas se colocaron con el)")
    L.save_json(out / cfg["gt_file"], gt)
    print(f"captura sintetica lista: {out}")


if __name__ == "__main__":
    main()
