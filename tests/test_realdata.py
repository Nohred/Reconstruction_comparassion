"""Pruebas sobre la captura real del bunny (se omiten si no esta disponible)."""
import unittest

import numpy as np

from common import geometry as g
from common.board import L, board_params
from common.config import load_config, repo_path
from common.realdata import Capture

CFG = load_config("common")
ROOT = repo_path(CFG["paths"]["real_root"]) / "bunnny"


@unittest.skipUnless(ROOT.exists(), f"no existe {ROOT}")
class TestRealCapture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cap = Capture.open(ROOT, CFG["capture_layout"])

    def test_layout(self):
        self.assertEqual(self.cap.cameras, ["alta", "media", "baja"])
        for cam in self.cap.cameras:
            self.assertEqual(self.cap.stops[cam], list(range(16)))
            v = self.cap.view(cam, 0)
            self.assertEqual(len(v.depth_paths), 10)
            self.assertTrue(all(p.exists() for p in v.depth_paths + [v.color_path]))

    def test_board_matches_rig_json(self):
        bp, rt = board_params(), self.cap.rig["tablero"]
        self.assertEqual((bp["squares_x"], bp["squares_y"], bp["square_mm"]),
                         (rt["squares_x"], rt["squares_y"], rt["square_mm"]))

    def test_cameras_above_board_and_ordered(self):
        """Convencion: camaras en z_board < 0 y alturas baja < media < alta."""
        h = {c: np.mean([-g.camera_center(v.T_color_from_board)[2] for v in self.cap.views([c])])
             for c in self.cap.cameras}
        self.assertTrue(all(x > 0 for x in h.values()), h)
        self.assertLess(h["baja"], h["media"])
        self.assertLess(h["media"], h["alta"])

    def test_final_pose_is_rig_composition(self):
        X = self.cap.rig_transforms()
        for k in range(16):
            Tm = self.cap.view("media", k).T_color_from_board
            for c in ("alta", "baja"):
                pred = X[c] @ g.T_inv(X["media"]) @ Tm
                np.testing.assert_allclose(pred, self.cap.view(c, k).T_color_from_board, atol=1e-6)

    def test_view_geometry_matches_meta(self):
        v = self.cap.view("alta", 3)
        elev, dist = L.view_geometry(v.T_color_from_board[:3, :3], v.T_color_from_board[:3, 3])
        self.assertAlmostEqual(elev, v.meta["elevation_deg"], places=6)
        self.assertAlmostEqual(dist, v.meta["distance_mm"], places=6)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(ROOT.exists(), f"no existe {ROOT}")
class TestReplayTurntable(unittest.TestCase):
    """Camaras fijas + base que gira deben reproducir EXACTAMENTE las poses finales reales."""

    @classmethod
    def setUpClass(cls):
        from sim.poses import replay_turntable

        cls.cap = Capture.open(ROOT, CFG["capture_layout"])
        cls.plan = replay_turntable(cls.cap, cls.cap.cameras, list(range(16)))

    def test_reproduces_final_poses(self):
        for v in self.cap.views():
            T = self.plan.T_color_from_board(v.camera, v.stop)
            np.testing.assert_allclose(T[:3, :3], v.T_color_from_board[:3, :3], atol=1e-9)
            np.testing.assert_allclose(T[:3, 3], v.T_color_from_board[:3, 3], atol=1e-6)   # mm

    def test_depth_pose_uses_extrinsic(self):
        cam = "media"
        T_cd = g.depth_to_color_T(self.cap.calib[cam]["depth_to_color"], 1000.0)
        np.testing.assert_allclose(T_cd @ self.plan.T_depth_from_board(cam, 3),
                                   self.plan.T_color_from_board(cam, 3), atol=1e-9)

    def test_axis_and_base(self):
        xy, res = g.fit_turntable_axis([self.plan.T_b0_from_bk[k] for k in self.plan.stops])
        self.assertEqual(self.plan.base_stop, 0)
        np.testing.assert_allclose(self.plan.T_b0_from_bk[0], np.eye(4), atol=1e-12)
        self.assertLess(res, 1.0)
        # eje cerca del centro del tablero (240 x 168 mm), medido 2026-11-17: ~(122, 111)
        self.assertLess(np.linalg.norm(xy - [122.0, 111.0]), 5.0)


@unittest.skipUnless(ROOT.exists(), f"no existe {ROOT}")
class TestRandomTurntable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sim.poses import random_turntable, replay_turntable, turntable_axis

        cls.cap = Capture.open(ROOT, CFG["capture_layout"])
        cls.cfg = load_config("twin")["aleatorio"]
        cls.base = replay_turntable(cls.cap, cls.cap.cameras, list(range(16)))
        cls.axis = turntable_axis(cls.base, 16)
        cls.rand = staticmethod(random_turntable)

    def test_axis_tilt_and_direction(self):
        _, d = self.axis
        tilt = np.degrees(np.arccos(abs(d[2])))
        self.assertLess(tilt, 2.0)                        # medido ~0.45 deg
        # parada 1 gira ~+22.5 deg alrededor del eje orientado
        a, ang = g.rot_log(self.base.T_b0_from_bk[1][:3, :3])
        self.assertGreater(a @ d, 0.99)
        self.assertAlmostEqual(ang, 22.9, delta=1.0)

    def test_random_plan(self):
        rng = np.random.default_rng(3)
        p = self.rand(self.cap, self.cap.cameras, self.cfg, rng)
        self.assertEqual(p.stops, list(range(16)))
        np.testing.assert_allclose(p.T_b0_from_bk[0], np.eye(4), atol=1e-12)
        for k in range(1, 16):
            _, ang = g.rot_log(p.T_b0_from_bk[k][:3, :3])
            nominal = min(22.5 * k, 360 - 22.5 * k)
            self.assertLessEqual(abs(ang - nominal), self.cfg["giro"]["max_deg"] + 1e-6)
            # el punto del eje en el tablero no se mueve
            pt = self.axis[0]
            np.testing.assert_allclose(g.transform_points(p.T_b0_from_bk[k], pt)[0], pt, atol=1e-6)
        for cam in self.cap.cameras:
            dd, dm = g.pose_diff(p.T_color_from_b0[cam], self.base.T_color_from_b0[cam])
            self.assertGreater(dm, 0.0)
            self.assertLess(dd, 20.0)
            self.assertLess(dm, 120.0)
            # sigue mirando al objeto: el eje proyecta delante de la camara
            self.assertGreater(g.transform_points(p.T_color_from_b0[cam], self.axis[0])[0, 2], 200)

    def test_seed_reproducible(self):
        a = self.rand(self.cap, self.cap.cameras, self.cfg, np.random.default_rng(7))
        b = self.rand(self.cap, self.cap.cameras, self.cfg, np.random.default_rng(7))
        np.testing.assert_allclose(a.T_color_from_b0["alta"], b.T_color_from_b0["alta"])
