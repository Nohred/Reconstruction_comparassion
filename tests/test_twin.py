"""Piezas numpy del gemelo: base giratoria, colocacion del objeto, ruido de profundidad, PLY."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from common import geometry as g
from common.meshio import read_mesh_ply, write_mesh_ply
from sim.depth_noise import depth_edges, noisy_frames, to_z16


def rot_about_axis(point_xy, deg, tilt_deg=0.0):
    """T que gira `deg` alrededor de un eje (casi) normal al tablero que pasa por point_xy."""
    a = np.radians(tilt_deg)
    axis = np.array([np.sin(a), 0.0, np.cos(a)])
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    th = np.radians(deg)
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    p = np.array([*point_xy, 0.0])
    return g.T_from(R, p - R @ p)


class TestTurntableAxis(unittest.TestCase):
    def test_recovers_axis(self):
        Ts = [rot_about_axis((122.5, 112.5), 22.5 * k) for k in range(16)]
        xy, res = g.fit_turntable_axis(Ts)
        np.testing.assert_allclose(xy, [122.5, 112.5], atol=1e-9)
        self.assertLess(res, 1e-9)

    def test_tilted_axis_small_bias(self):
        Ts = [rot_about_axis((100.0, 80.0), 22.5 * k, tilt_deg=0.5) for k in range(16)]
        xy, _ = g.fit_turntable_axis(Ts)
        np.testing.assert_allclose(xy, [100.0, 80.0], atol=0.5)


class TestPlacement(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.P = rng.uniform([20, 40, 0], [200, 180, 160], (500, 3))   # objeto Z arriba, mm

    def test_bbox_centered_and_on_board(self):
        T = g.place_on_board(self.P, (122.5, 112.5), 0.0)
        Q = g.transform_points(T, self.P)
        lo, hi = Q.min(0), Q.max(0)
        np.testing.assert_allclose((lo[:2] + hi[:2]) / 2, [122.5, 112.5], atol=1e-9)
        self.assertAlmostEqual(hi[2], 0.0)            # apoyado en el tablero
        self.assertLess(lo[2], -100)                  # hacia las camaras (z del tablero < 0)
        self.assertAlmostEqual(np.linalg.det(T[:3, :3]), 1.0)

    def test_up_is_minus_z_board(self):
        T = g.place_on_board(self.P, (0, 0), 30.0)
        np.testing.assert_allclose(T[:3, :3] @ [0, 0, 1], [0, 0, -1], atol=1e-12)


class TestDepthNoise(unittest.TestCase):
    def test_z16_roundtrip_and_invalid(self):
        z = np.array([[0.5, 0.0], [np.inf, 20.0]])
        q = to_z16(z, 0.00025)
        self.assertEqual(q.dtype, np.uint16)
        self.assertEqual(q[0, 0], 2000)
        self.assertEqual(q[0, 1], 0)
        self.assertEqual(q[1, 0], 0)
        self.assertEqual(q[1, 1], 0)                  # fuera del rango de uint16

    def test_edges(self):
        z = np.full((10, 10), 0.5)
        z[:, 5:] = 0.6
        e = depth_edges(z, jump_mm=20, dilate_px=0)
        self.assertTrue(e[:, 4].all() and e[:, 5].all())
        self.assertFalse(e[:, :4].any() or e[:, 6:].any())

    def test_noise_statistics(self):
        rng = np.random.default_rng(0)
        z = np.full((200, 200), 0.5)
        fr = noisy_frames(z, 0.00025, 10, sigma_mm=2.0, p_dropout=0.1, jump_mm=20, edge_dilate_px=1, rng=rng)
        D = np.stack(fr).astype(float) * 0.25
        self.assertEqual(len(fr), 10)
        self.assertAlmostEqual((D == 0).mean(), 0.1, delta=0.01)
        v = D[D > 0]
        self.assertAlmostEqual(v.mean(), 500.0, delta=0.05)
        self.assertAlmostEqual(v.std(), 2.0, delta=0.05)


class TestMeshIO(unittest.TestCase):
    def test_roundtrip(self):
        P = np.random.default_rng(0).normal(size=(30, 3))
        F = np.arange(30).reshape(10, 3)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.ply"
            write_mesh_ply(p, P, np.full(10, 3), F.ravel())
            P2, F2 = read_mesh_ply(p)
        np.testing.assert_allclose(P2, P.astype(np.float32))
        np.testing.assert_array_equal(F2, F)


if __name__ == "__main__":
    unittest.main()


class TestObjectConfig(unittest.TestCase):
    def test_key_and_path(self):
        from common.config import object_config

        b = object_config("bunny")
        self.assertEqual(b["name"], "bunny")
        self.assertIsNone(b["tamano_mm"])                 # tamano impreso real
        self.assertEqual(b["real_units_m"], 0.001)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "mi_objeto.usdz"
            p.write_bytes(b"")
            o = object_config(str(p))
            self.assertEqual(o["name"], "mi_objeto")
            self.assertEqual(o["tamano_mm"], 170.0)       # normalizado por defecto
            self.assertIsNone(o["real_units_m"])          # del metadato del USD
        with self.assertRaises(KeyError):
            object_config("no_existe")
