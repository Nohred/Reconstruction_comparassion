"""Poses de la base giratoria y de las camaras. Numpy puro: se prueba sin Isaac.

Como en la captura real, las camaras quedan FIJAS (rig) y lo que gira es la base: tablero, carton
y objeto. Mesa, luces y fondo quedan quietos respecto a las camaras, igual que en el laboratorio.

Marco de referencia de la escena: el tablero en la primera parada con pose (b0).
  camaras : T_color<-b0 = rig[c] @ T_rig<-b0           (fijas)
  base    : T_b0<-bk   = inv(T_rig<-b0) @ T_rig<-bk     (una por parada)
  vista   : T_color<-bk = T_color<-b0 @ T_b0<-bk = rig[c] @ T_rig<-bk  (= pose final real)

replay   : T_rig<-bk sale de las poses FINALES de meta.json real (promedio entre camaras).
aleatorio: variabilidad de captura a partir de una plantilla real. Las camaras se perturban UNA vez
           por captura (quedan fijas en todas las paradas, como un rig re-montado) y en cada
           parada solo gira la base: angulo nominal k*360/N + ruido, alrededor del eje (con su
           inclinacion) ajustado a la plantilla.
"""
from dataclasses import dataclass, field

import numpy as np

from common.geometry import (T_inv, depth_to_color_T, fit_turntable_axis, mean_pose,
                             rot_about_line, rot_axis_angle, rot_log)


@dataclass
class TurntablePlan:
    stops: list                                     # paradas a renderizar
    T_color_from_b0: dict                           # camara -> 4x4 (mm)
    T_depth_from_b0: dict                           # camara -> 4x4 (mm)
    T_b0_from_bk: dict                              # parada -> 4x4 (mm)
    base_stop: int = 0
    missing: list = field(default_factory=list)     # paradas sin pose real
    max_dev_mm: float = 0.0                         # consistencia rig vs poses finales reales

    def T_color_from_board(self, cam, k):
        return self.T_color_from_b0[cam] @ self.T_b0_from_bk[k]

    def T_depth_from_board(self, cam, k):
        return self.T_depth_from_b0[cam] @ self.T_b0_from_bk[k]


def replay_turntable(real_cap, cameras, stops):
    """Plan de render que reproduce las poses finales reales moviendo la base."""
    rig = real_cap.rig_transforms()
    if not rig:
        raise ValueError(f"{real_cap.root}: sin rig.json no se pueden fijar las camaras")
    T_rig_bk, missing, dev = {}, [], 0.0
    for k in stops:
        Ts = []
        for cam in real_cap.cameras:
            if k in real_cap.stops[cam]:
                v = real_cap.view(cam, k)
                if v.has_pose:
                    Ts.append(T_inv(rig[cam]) @ v.T_color_from_board)
        if not Ts:
            missing.append(k)
            continue
        T_rig_bk[k] = mean_pose(Ts)
        dev = max(dev, max(float(np.linalg.norm(T[:3, 3] - T_rig_bk[k][:3, 3])) for T in Ts))
    if not T_rig_bk:
        raise ValueError("ninguna parada tiene pose")
    k0 = min(T_rig_bk)
    T_rig_b0 = T_rig_bk[k0]
    T_c, T_d = {}, {}
    for cam in cameras:
        T_c[cam] = rig[cam] @ T_rig_b0
        T_cd = depth_to_color_T(real_cap.calib[cam]["depth_to_color"], 1000.0)   # m -> mm
        T_d[cam] = T_inv(T_cd) @ T_c[cam]
    T_b = {k: T_inv(T_rig_b0) @ T for k, T in T_rig_bk.items()}
    return TurntablePlan(stops=sorted(T_b), T_color_from_b0=T_c, T_depth_from_b0=T_d,
                         T_b0_from_bk=T_b, base_stop=k0, missing=missing, max_dev_mm=dev)


def turntable_axis(plan, n_stops):
    """Eje de giro de la base en el marco b0: (punto en z=0, direccion unitaria).

    La direccion se orienta para que el giro de la secuencia sea POSITIVO alrededor de ella; se
    promedia con las paradas de la primera media vuelta (angulo bien condicionado).
    """
    xy, _ = fit_turntable_axis([plan.T_b0_from_bk[k] for k in plan.stops])
    axes = []
    for k in plan.stops:
        if 0 < k < n_stops / 2:
            a, ang = rot_log(plan.T_b0_from_bk[k][:3, :3])
            if ang > 10:
                axes.append(a)
    if not axes:
        raise ValueError("la plantilla no tiene paradas utiles para ajustar el eje")
    d = np.mean(axes, 0)
    return np.array([xy[0], xy[1], 0.0]), d / np.linalg.norm(d)


def _clipped_normal(rng, sigma, max_sigmas):
    return float(np.clip(rng.normal(0.0, sigma), -max_sigmas * sigma, max_sigmas * sigma)) if sigma else 0.0


def perturb_camera(T_c_b0, target, up, cfg, rng):
    """Re-monta una camara: azimut/elevacion alrededor del objetivo, distancia, punteria y roll."""
    ms = cfg["max_sigmas"]
    c2b = T_inv(T_c_b0)
    C = c2b[:3, 3]
    M = rot_about_line(target, up, _clipped_normal(rng, cfg["azimut_sigma_deg"], ms))
    d = C - target
    h = np.cross(d, up)
    M = rot_about_line(target, h / np.linalg.norm(h), _clipped_normal(rng, cfg["elevacion_sigma_deg"], ms)) @ M
    c2b = M @ c2b
    v = c2b[:3, 3] - target
    c2b[:3, 3] += v / np.linalg.norm(v) * _clipped_normal(rng, cfg["distancia_sigma_mm"], ms)
    R = (rot_axis_angle([1, 0, 0], _clipped_normal(rng, cfg["punteria_sigma_deg"], ms))
         @ rot_axis_angle([0, 1, 0], _clipped_normal(rng, cfg["punteria_sigma_deg"], ms))
         @ rot_axis_angle([0, 0, 1], _clipped_normal(rng, cfg["roll_sigma_deg"], ms)))
    c2b[:3, :3] = c2b[:3, :3] @ R
    return T_inv(c2b)


def random_turntable(template, cameras, cfg, rng):
    """Plan aleatorio con la plantilla: camaras re-montadas + N paradas con giro ruidoso."""
    n = int(template.session["parametros"]["paradas"])
    all_stops = sorted(set().union(*(template.stops[c] for c in template.cameras)))
    base = replay_turntable(template, cameras, all_stops)
    point, axis = turntable_axis(base, n)
    up = np.array([0.0, 0.0, -1.0])                    # z del tablero apunta a la mesa
    T_c, T_d = {}, {}
    for cam in cameras:
        T_c[cam] = perturb_camera(base.T_color_from_b0[cam], point, up, cfg["camaras"], rng)
        T_cd = depth_to_color_T(template.calib[cam]["depth_to_color"], 1000.0)
        T_d[cam] = T_inv(T_cd) @ T_c[cam]
    g = cfg["giro"]
    T_b, angles = {}, {}
    for k in range(n):
        angles[k] = 0.0 if k == 0 else k * 360.0 / n + float(np.clip(rng.normal(0, g["sigma_deg"]),
                                                                      -g["max_deg"], g["max_deg"]))
        T_b[k] = rot_about_line(point, axis, angles[k])
    plan = TurntablePlan(stops=list(range(n)), T_color_from_b0=T_c, T_depth_from_b0=T_d,
                         T_b0_from_bk=T_b, base_stop=0)
    plan.angles_deg = angles
    plan.axis = (point, axis)
    return plan
