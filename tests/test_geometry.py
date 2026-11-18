import unittest

import numpy as np

from common import geometry as g

RNG = np.random.default_rng(0)


def random_T(scale=500.0):
    q = RNG.normal(size=4)
    return g.as_T(g.quat_wxyz_to_rot(q), RNG.normal(size=3) * scale)


class TestTransforms(unittest.TestCase):
    def test_inverse_roundtrip(self):
        for _ in range(20):
            T = random_T()
            np.testing.assert_allclose(T @ g.T_inv(T), np.eye(4), atol=1e-9)

    def test_camera_center_projects_to_origin(self):
        T = random_T()
        c = g.camera_center(T)
        np.testing.assert_allclose(g.transform_points(T, c), 0, atol=1e-9)

    def test_quaternion_roundtrip_and_sign(self):
        for _ in range(50):
            R = g.quat_wxyz_to_rot(RNG.normal(size=4))
            q = g.rot_to_quat_wxyz(R)
            self.assertGreaterEqual(q[0], 0)
            self.assertAlmostEqual(np.linalg.norm(q), 1.0, places=12)
            np.testing.assert_allclose(g.quat_wxyz_to_rot(q), R, atol=1e-9)

    def test_quaternion_known(self):
        Rz90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
        np.testing.assert_allclose(g.rot_to_quat_wxyz(Rz90), [np.sqrt(.5), 0, 0, np.sqrt(.5)], atol=1e-12)


class TestFrames(unittest.TestCase):
    def test_world_from_board_is_rigid_z_up(self):
        T = g.world_from_board(1.0)
        self.assertAlmostEqual(np.linalg.det(T[:3, :3]), 1.0)
        # un punto "arriba" del tablero (z_board < 0) queda con z_world > 0
        self.assertGreater(g.transform_points(T, [0, 0, -100])[0, 2], 0)

    def test_revision_flip_is_reflection_and_involution(self):
        flip = [1, 1, -1]
        P = RNG.normal(size=(10, 3))
        np.testing.assert_allclose(g.rev_to_board(g.board_to_rev(P, flip), flip), P)
        self.assertAlmostEqual(np.linalg.det(np.diag(flip)), -1.0)

    def test_usd_camera_pose_looks_at_same_point(self):
        """Un punto delante de la camara OpenCV (+z) esta en -z de la camara USD."""
        T_cb = random_T()
        s = 0.001
        M_usd = g.cam_pose_usd_in_world(T_cb, s)
        M_ros = g.cam_pose_ros_in_world(T_cb, s)
        for M in (M_usd, M_ros):
            self.assertAlmostEqual(np.linalg.det(M[:3, :3]), 1.0)
        p_board = g.transform_points(g.T_inv(T_cb), [0, 0, 300.0])     # 300 mm frente a la camara
        p_world = g.transform_points(g.world_from_board(s), p_board)
        np.testing.assert_allclose(g.transform_points(g.T_inv(M_usd), p_world), [[0, 0, -0.3]], atol=1e-9)
        np.testing.assert_allclose(g.transform_points(g.T_inv(M_ros), p_world), [[0, 0, 0.3]], atol=1e-9)

    def test_depth_to_color_colmajor(self):
        R = g.quat_wxyz_to_rot([0.9, 0.1, -0.2, 0.05])
        ext = dict(rotation_colmajor=list(R.T.ravel()), translation_m=[0.001, 0.014, -0.004])
        T = g.depth_to_color_T(ext, 1000.0)
        np.testing.assert_allclose(T[:3, :3], R)
        np.testing.assert_allclose(T[:3, 3], [1, 14, -4])


if __name__ == "__main__":
    unittest.main()
