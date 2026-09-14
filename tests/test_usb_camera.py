import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from comm.usb_camera import USBCamera


class UsbCameraTests(unittest.TestCase):
    def test_latest_frame_is_delivered_once_and_timeout_returns_none(self):
        camera = USBCamera(latest_frame=True)
        frame = np.zeros((4, 4, 3), np.uint8)
        with camera._condition:
            camera._latest = frame
            camera._sequence = 3
        self.assertIs(camera.read(), frame)
        self.assertEqual(camera._delivered, 3)
        self.assertIsNone(camera.read())

    def test_capture_failure_closes_device_and_does_not_replay_old_frame(self):
        camera = USBCamera(latest_frame=True)
        capture = Mock()
        capture.read.return_value = (False, None)
        camera.cap = capture
        camera._latest = np.zeros((4, 4, 3), np.uint8)
        camera._sequence = 1
        camera._capture_latest()
        capture.release.assert_called_once()
        self.assertIsNone(camera.read())

    @patch('comm.usb_camera.sys.platform', 'linux')
    @patch('comm.usb_camera.cv2.VideoCapture')
    def test_linux_device_path_uses_v4l2_backend(self, video_capture):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (
            True, np.zeros((480, 640, 3), dtype=np.uint8))
        video_capture.return_value = capture
        path = '/dev/v4l/by-id/camera-index0'

        camera = USBCamera(device=path)
        self.assertTrue(camera.open())

        video_capture.assert_called_once_with(path, cv2.CAP_V4L2)


if __name__ == '__main__':
    unittest.main()
