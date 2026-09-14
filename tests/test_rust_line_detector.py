"""Opt-in: RUN_RUST_TESTS=1 after compiling the release library."""
import os
import unittest
import numpy as np
from core.line_follower import LineDetector as PythonDetector


@unittest.skipUnless(os.environ.get('RUN_RUST_TESTS') == '1', 'optional Rust build')
class RustParityTests(unittest.TestCase):
    def test_seeded_consensus_parity(self):
        from core.rust_line_detector import LineDetector as RustDetector
        rng = np.random.default_rng(741)
        cases = [[], [(1, 0)] * 4, [(0, 0), (1, 1), (2, 2), (99, 3)]]
        for n in (3, 6, 12, 24, 60, 120):
            for _ in range(20):
                y = np.arange(n, dtype=float)
                x = y * rng.uniform(-2, 2) + rng.normal(0, 2, n)
                x[rng.random(n) < .2] += 35
                cases.append(list(zip(x, y)))
        for points in cases:
            for limit in (0., 3., 8.):
                a = PythonDetector._batch_corner_stem_fit(points, limit)
                b = RustDetector._batch_corner_stem_fit(points, limit)
                self.assertEqual(a[1], b[1])
                if a[0] is None:
                    self.assertIsNone(b[0])
                else:
                    np.testing.assert_allclose(a[0], b[0], rtol=1e-7, atol=1e-7)
