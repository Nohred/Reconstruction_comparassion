import tempfile
import unittest

import numpy as np

from common import geometry as g
from common.colmap import ColmapCamera, ColmapImage, read_model, write_model


class TestColmapIO(unittest.TestCase):
    def test_roundtrip(self):
        rng = np.random.default_rng(1)
        # valores numpy a proposito: repr(np.float64) no es texto valido para COLMAP
        K = np.array([[906.8, 0, 649.6], [0, 907.5, 361.0], [0, 0, 1]])
        cams = [ColmapCamera(1, 1280, 720, K[0, 0], K[1, 1], K[0, 2], K[1, 2])]
        imgs = [ColmapImage(i + 1, g.quat_wxyz_to_rot(rng.normal(size=4)), rng.normal(size=3), 1,
                            f"alta/stop_{i:02d}.png") for i in range(5)]
        with tempfile.TemporaryDirectory() as d:
            write_model(d, cams, imgs)
            c2, i2 = read_model(d)
        self.assertEqual(c2[1], cams[0])
        for a, b in zip(imgs, i2):
            np.testing.assert_allclose(a.R, b.R, atol=1e-12)
            np.testing.assert_allclose(a.t, b.t, atol=1e-12)
            self.assertEqual(a.name, b.name)


if __name__ == "__main__":
    unittest.main()
