"""Limpieza de nubes (recon/limpieza.py) con una escena sintetica: tablero grueso + objeto + flotantes."""
import unittest

import numpy as np

try:
    import open3d as o3d
except ImportError:      # el entorno twin_cv410 no tiene open3d: corre con tools/ns_py.sh
    o3d = None

from recon.comun import load_recon_config


def scene(rng, piso_sigma_mm=0.5, piso_z0_mm=0.0):
    roi = dict(eje_xy_mm=[120.0, 110.0], radio_mm=160.0, altura_mm=190.0, bajo_tablero_mm=5.0)
    c = np.array(roi["eje_xy_mm"]) * 1e-3
    n = 60000                                            # tablero: disco de la ROI, z hacia la mesa
    r = 0.155 * np.sqrt(rng.random(n))
    a = rng.random(n) * 2 * np.pi
    piso = np.c_[c[0] + r * np.cos(a), c[1] + r * np.sin(a),
                 piso_z0_mm * 1e-3 + rng.normal(0, piso_sigma_mm * 1e-3, n)]
    v = rng.normal(size=(40000, 3))                      # objeto: esfera de 50 mm apoyada
    obj = 0.05 * v / np.linalg.norm(v, axis=1, keepdims=True) + [c[0], c[1], -0.05]
    obj = obj[obj[:, 2] < 0]
    flot = np.c_[c[0] + 0.12 + rng.random(300) * 0.01, c[1] + rng.random(300) * 0.01,
                 -0.12 - rng.random(300) * 0.01]           # flotante compacto lejos del objeto
    sueltos = np.c_[c + (rng.random((50, 2)) - 0.5) * 0.2, -rng.random(50) * 0.18]
    return roi, piso, obj, flot, sueltos


@unittest.skipIf(o3d is None, "requiere open3d (tools/ns_py.sh -m unittest tests.test_limpieza)")
class TestLimpieza(unittest.TestCase):
    def setUp(self):
        from recon.limpieza import clean_cloud

        self.clean = clean_cloud
        self.lcfg = load_recon_config("completo")["limpieza"]

    def run_scene(self, **kw):
        roi, piso, obj, flot, sueltos = scene(np.random.default_rng(0), **kw)
        P = np.vstack([piso, obj, flot, sueltos])
        Q, info = self.clean(P, roi, self.lcfg, o3d)
        return Q, info, obj

    def test_quita_piso_y_flotantes(self):
        Q, info, obj = self.run_scene()
        self.assertEqual(info["plano"]["metodo"], "ransac")
        self.assertAlmostEqual(info["corte_mm"], self.lcfg["sobre_plano_mm"])
        self.assertTrue(np.all(-Q[:, 2] > info["corte_mm"] * 1e-3 - 1e-6))     # nada del piso
        d = np.linalg.norm(Q[:, :2] - (np.array([120.0, 110.0]) * 1e-3), axis=1)
        self.assertLess(d.max(), 0.06)                                         # sin el flotante
        kept = (-obj[:, 2] > info["corte_mm"] * 1e-3).sum()
        self.assertGreater(len(Q), 0.97 * kept)                                # el objeto se queda

    def test_piso_grueso_y_desplazado(self):
        Q, info, _ = self.run_scene(piso_sigma_mm=2.0, piso_z0_mm=2.0)
        self.assertAlmostEqual(info["plano"]["desplazamiento_mm"], 2.0, delta=0.3)
        self.assertGreater(info["corte_mm"], self.lcfg["sobre_plano_mm"])     # corte adaptativo
        h = (Q @ np.array(info["plano_n_d"][:3]) + info["plano_n_d"][3]) * 1e3
        self.assertGreater(h.min(), info["corte_mm"] - 1e-6)


if __name__ == "__main__":
    unittest.main()
