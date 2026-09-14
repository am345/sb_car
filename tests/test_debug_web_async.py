import time
import unittest
from unittest.mock import patch

import numpy as np

from debug_web import DebugWebServer


class DebugWebAsyncTests(unittest.TestCase):
    def test_debug_rendering_does_not_block_control_update(self):
        dashboard = DebugWebServer(port=0, stream_fps=30)
        dashboard.start()
        frame = np.zeros((480, 640, 3), np.uint8)

        def slow_compose(source, _det):
            time.sleep(0.2)
            return source

        try:
            with patch.object(DebugWebServer, '_compose_debug_frame',
                              side_effect=slow_compose):
                started = time.monotonic()
                dashboard.update(frame, {'is_valid': True}, {})
                update_seconds = time.monotonic() - started
                sequence, jpeg, running = dashboard.wait_for_frame(0, timeout=1)

            self.assertLess(update_seconds, 0.05)
            self.assertEqual(sequence, 1)
            self.assertTrue(jpeg)
            self.assertTrue(running)
        finally:
            dashboard.stop()


if __name__ == '__main__':
    unittest.main()
