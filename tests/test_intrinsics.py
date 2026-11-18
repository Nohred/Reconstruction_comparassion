import unittest

import cv2
import numpy as np

from common import intrinsics as ins

COLOR = dict(width=1280, height=720, fx=906.8161, fy=907.5136, ppx=649.5724, ppy=360.9671,
             model="distortion.brown_conrady", coeffs=[0.1656, -0.5050, -0.00088, -0.00063, 0.4607])


class TestIntrinsics(unittest.TestCase):
    def test_K_from_calib(self):
        K, d = ins.K_dist_from_calib(COLOR)
        self.assertEqual(K[0, 0], COLOR["fx"])
        self.assertEqual(K[1, 2], COLOR["ppy"])
        np.testing.assert_allclose(d, COLOR["coeffs"])

    def test_rs_brown_conrady_matches_opencv(self):
        """librealsense brown_conrady [k1,k2,p1,p2,k3] = orden OpenCV: proyeccion identica."""
        K, d = ins.K_dist_from_calib(COLOR)
        x, y = 0.3, -0.2
        r2 = x * x + y * y
        k1, k2, p1, p2, k3 = d
        rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        manual = [K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]]
        cvp, _ = cv2.projectPoints(np.array([[x, y, 1.0]]), np.zeros(3), np.zeros(3), K, d)
        np.testing.assert_allclose(cvp.ravel(), manual, atol=1e-9)

    def test_usd_attrs_roundtrip(self):
        K, _ = ins.K_dist_from_calib(COLOR)
        a = ins.usd_camera_attrs(K, COLOR["width"], COLOR["height"])
        np.testing.assert_allclose(ins.K_from_usd_camera_attrs(a, COLOR["width"], COLOR["height"]), K,
                                   atol=1e-9)

    def test_lens_attrs(self):
        K, d = ins.K_dist_from_calib(COLOR)
        a = ins.isaac_lens_attrs(K, d, 1280, 720)
        self.assertEqual((a["k1"], a["k2"], a["p1"], a["p2"], a["k3"]), tuple(d))
        self.assertEqual(a["k4"], 0.0)
        self.assertEqual(a["imageSize"], (1280, 720))

    def test_calib_roundtrip(self):
        K, d = ins.K_dist_from_calib(COLOR)
        c = ins.calib_from_K(K, d, 1280, 720)
        self.assertEqual(c["model"], COLOR["model"])
        self.assertEqual(ins.calib_from_K(K, None, 1280, 720)["model"], "distortion.none")

    def test_unsupported_model(self):
        with self.assertRaises(NotImplementedError):
            ins.K_dist_from_calib({**COLOR, "model": "distortion.inverse_brown_conrady"})


if __name__ == "__main__":
    unittest.main()
