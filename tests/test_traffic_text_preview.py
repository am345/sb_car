import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from core.traffic_signs import TrafficSignWorker


class TrafficTextPreviewTests(unittest.TestCase):
    def test_text_only_keeps_detection_and_three_frame_confirmation(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        output = np.zeros((1, 5, 8400), np.float32)
        output[0, :, 0] = [320, 320, 80, 80, .9]
        worker = TrafficSignWorker('fixture.onnx')
        session = Mock()
        session.get_inputs.return_value = [SimpleNamespace(shape=[1, 3, 640, 640], name='images')]
        session.get_modelmeta.return_value = SimpleNamespace(custom_metadata_map={'names': "{0: 'right'}"})
        count = 0

        def infer(*args, **kwargs):
            nonlocal count
            count += 1
            if count < 3:
                worker.submit(frame)
            else:
                worker._stop.set()
            return [output]

        session.run.side_effect = infer
        module = SimpleNamespace(SessionOptions=SimpleNamespace,
                                 InferenceSession=Mock(return_value=session))
        worker.submit(frame)
        with patch.dict(sys.modules, {'onnxruntime': module}), \
                patch('core.traffic_signs.cv2.imencode') as encode, \
                patch('core.traffic_signs.cv2.rectangle') as draw:
            worker._run()
        status = worker.snapshot()
        self.assertEqual(status['state'], 'ready')
        self.assertEqual(status['sequence'], 3)
        self.assertEqual(status['confirmed'], 'right')
        self.assertEqual(status['confirm_count'], 3)
        self.assertAlmostEqual(status['detections'][0]['confidence'], .9, places=6)
        self.assertEqual(status['detections'][0]['box'], [280., 200., 360., 280.])
        self.assertIsNone(worker.jpeg())
        encode.assert_not_called()
        draw.assert_not_called()


if __name__ == '__main__':
    unittest.main()
