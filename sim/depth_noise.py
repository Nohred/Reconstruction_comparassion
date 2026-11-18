"""Modelo de ruido de profundidad de la L515 (provisional; se calibra mejor en la fase 6).

Numpy puro. Parte de la profundidad perfecta (m) del render y genera N cuadros uint16 como los de
la camara real (z16, escala depth_scale_m de calib.json, 0 = invalido):
  - ruido gaussiano temporal independiente por pixel y cuadro, sigma por camara (mm);
  - pixeles invalidos en discontinuidades de profundidad (bordes del objeto/tablero);
  - perdida aleatoria por cuadro (pixeles que "parpadean" a 0).
"""
import numpy as np


def depth_edges(z_m, jump_mm, dilate_px):
    """Mascara de discontinuidades: salto con algun vecino 4-conexo mayor a jump_mm."""
    z = np.where(z_m > 0, z_m * 1000.0, np.nan)
    e = np.zeros(z.shape, bool)
    with np.errstate(invalid="ignore"):
        dx = np.abs(np.diff(z, axis=1)) > jump_mm
        dy = np.abs(np.diff(z, axis=0)) > jump_mm
    e[:, 1:] |= dx
    e[:, :-1] |= dx
    e[1:, :] |= dy
    e[:-1, :] |= dy
    for _ in range(int(dilate_px)):
        d = e.copy()
        d[1:, :] |= e[:-1, :]
        d[:-1, :] |= e[1:, :]
        d[:, 1:] |= e[:, :-1]
        d[:, :-1] |= e[:, 1:]
        e = d
    return e


def to_z16(z_m, depth_scale_m):
    """Metros -> uint16 con la escala de la camara (0 si invalido o fuera de rango)."""
    z = np.where(np.isfinite(z_m) & (z_m > 0), z_m, 0.0)
    q = np.round(z / depth_scale_m)
    return np.where(q <= np.iinfo(np.uint16).max, q, 0).astype(np.uint16)


def noisy_frames(z_m, depth_scale_m, n_frames, sigma_mm, p_dropout, jump_mm, edge_dilate_px, rng):
    """Lista de n_frames imagenes uint16 ruidosas a partir de la profundidad perfecta z_m."""
    valid = np.isfinite(z_m) & (z_m > 0)
    valid &= ~depth_edges(z_m, jump_mm, edge_dilate_px)
    out = []
    for _ in range(n_frames):
        z = z_m + rng.normal(0.0, sigma_mm / 1000.0, z_m.shape)
        keep = valid & (rng.random(z_m.shape) >= p_dropout)
        out.append(to_z16(np.where(keep, z, 0.0), depth_scale_m))
    return out
