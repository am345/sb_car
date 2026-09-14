"""Display-only YOLO11 detection; no chassis access or movement decisions."""
import ast
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np


CONFIDENCE_THRESHOLD = 0.6


class TrafficSignWorker:
    def __init__(self, model_path, interval=0.0):
        self.model_path = model_path
        self.interval = interval
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._frame = None
        self._jpeg = None
        self._result_time = None
        self._status = {'state': 'loading', 'detections': [], 'sequence': 0}
        self._thread = threading.Thread(target=self._run, name='traffic-signs', daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread.ident is not None:
            self._thread.join(timeout=3)

    def submit(self, frame):
        if frame is None or not frame.size or self._stop.is_set():
            return
        with self._condition:
            self._frame = (frame.copy(), time.monotonic())
            self._condition.notify_all()

    def snapshot(self):
        with self._condition:
            result = dict(self._status)
            age = None if self._result_time is None else time.monotonic() - self._result_time
            result['age_sec'] = age
            if age is not None and age > 2 and result['state'] == 'ready':
                result.update(state='stale', detections=[], confirmed=None, confirm_count=0)
            return result

    def jpeg(self):
        with self._condition:
            return self._jpeg

    def wait_for_frame(self, previous, timeout=2.0):
        with self._condition:
            self._condition.wait_for(lambda: self._status['sequence'] != previous or self._stop.is_set(), timeout)
            return self._status['sequence'], self._jpeg, not self._stop.is_set()

    @staticmethod
    def preprocess(frame, npu=False):
        h, w = frame.shape[:2]
        ratio = min(640 / h, 640 / w)
        nw, nh = round(w * ratio), round(h * ratio)
        left, top = (640 - nw) // 2, (640 - nh) // 2
        resized = cv2.resize(frame, (nw, nh))
        padded = cv2.copyMakeBorder(resized, top, 640-nh-top, left, 640-nw-left,
                                    cv2.BORDER_CONSTANT, value=(114,114,114))
        if npu:
            return np.ascontiguousarray(padded[:, :, ::-1][None]), ratio, left, top
        tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2,0,1)[None], dtype=np.float32)
        tensor /= 255.0
        return tensor, ratio, left, top

    @staticmethod
    def decode(output, ratio, left, top, shape, names):
        if (set(names) != set(range(len(names))) or
                output.shape != (1,4+len(names),8400) or not np.isfinite(output).all()):
            raise ValueError('模型输出与类别元数据不匹配（需要640输入的YOLO11格式）')
        rows = output[0].T
        labels, scores = rows[:,4:].argmax(axis=1), rows[:,4:].max(axis=1)
        mask = scores > CONFIDENCE_THRESHOLD
        rows, labels, scores = rows[mask], labels[mask], scores[mask]
        boxes = rows[:,:4].copy()
        boxes[:,:2] -= boxes[:,2:] / 2
        result = []
        for label in np.unique(labels):
            indices = np.flatnonzero(labels == label)
            kept = cv2.dnn.NMSBoxes(
                boxes[indices].tolist(), scores[indices].tolist(),
                CONFIDENCE_THRESHOLD, 0.45)
            for k in np.asarray(kept).reshape(-1):
                i = indices[k]
                x,y,w,h = boxes[i]
                xyxy = np.clip([(x-left)/ratio,(y-top)/ratio,(x+w-left)/ratio,(y+h-top)/ratio],
                               [0,0,0,0], [shape[1],shape[0],shape[1],shape[0]])
                result.append({'class_id': int(label), 'label': names[int(label)],
                               'confidence': float(scores[i]), 'box': xyxy.tolist()})
        return sorted(result, key=lambda d: d['confidence'], reverse=True)[:30]

    def _run(self):
        runtime = None
        try:
            npu = Path(self.model_path).suffix.lower() == '.rknn'
            if npu:
                from core.npu_runtime import private_runtime_library
                with private_runtime_library():
                    from rknnlite.api import RKNNLite
                metadata = json.loads(Path(self.model_path).with_suffix('.json').read_text())
                names = {int(k):v for k,v in metadata['names'].items()}
                expected_output = [1, 4 + len(names), 8400]
                if (metadata['input_size'] != [640,640] or
                        metadata['output_shape'] != expected_output):
                    raise ValueError('RKNN 模型元数据不匹配')
                runtime = RKNNLite(verbose=False)
                with private_runtime_library():
                    if runtime.load_rknn(self.model_path) != 0 or runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
                        raise RuntimeError('NPU 模型加载失败，请检查 RKNN 环境和运行库')
                def infer(tensor):
                    outputs = runtime.inference(inputs=[tensor], data_format=['nhwc'])
                    if outputs is None or len(outputs) != 1:
                        raise RuntimeError('NPU 推理未返回有效输出')
                    return outputs[0]
            else:
                import onnxruntime as ort
                options = ort.SessionOptions()
                # Reserve CPU capacity for the timing-critical line-following
                # loop. Sign inference is asynchronous and does not benefit
                # the chassis from monopolising all four performance cores.
                options.intra_op_num_threads = 2
                options.inter_op_num_threads = 1
                session = ort.InferenceSession(self.model_path, sess_options=options,
                                               providers=['CPUExecutionProvider'])
                names = ast.literal_eval(session.get_modelmeta().custom_metadata_map['names'])
                if session.get_inputs()[0].shape != [1,3,640,640]:
                    raise ValueError('需要 640×640 输入的 ONNX 模型')
                input_name = session.get_inputs()[0].name
                def infer(tensor):
                    return session.run(None, {input_name: tensor})[0]
            if isinstance(names, list):
                names = dict(enumerate(names))
            if not names or set(names) != set(range(len(names))):
                raise ValueError('YOLO11 类别元数据必须从0连续编号')
            last_label, last_finish, count, sequence = None, None, 0, 0
            deadline = 0.0
            while not self._stop.is_set():
                with self._condition:
                    while not self._stop.is_set():
                        wait = deadline - time.monotonic()
                        if self._frame is not None and wait <= 0:
                            break
                        self._condition.wait(timeout=max(0.01,wait) if self._frame is not None else None)
                    if self._stop.is_set():
                        break
                    frame, captured_at = self._frame
                    self._frame = None
                started = time.monotonic()
                deadline = started + self.interval
                tensor, ratio, left, top = self.preprocess(frame, npu=npu)
                t0 = time.monotonic()
                output = infer(tensor)
                infer_ms = (time.monotonic()-t0)*1000
                detections = self.decode(output, ratio, left, top, frame.shape, names)
                valid = [d for d in detections
                         if d['confidence'] > CONFIDENCE_THRESHOLD]
                label = valid[0]['label'] if valid else None
                now = time.monotonic()
                count = count+1 if label and label == last_label and last_finish is not None and now-last_finish < 1.5 else (1 if label else 0)
                last_label = label
                # Text-only preview: keep detection/confirmation unchanged and
                # omit annotation and JPEG encoding from every inference.
                finished = time.monotonic()
                sequence += 1
                status = {'state':'ready', 'detections':detections, 'sequence':sequence,
                          'backend':'RKNN NPU FP16' if npu else 'ONNX Runtime CPU',
                          'confirmed':label if count >= 3 else None, 'confirm_count':min(count,3),
                          'frame_width':int(frame.shape[1]),
                          'frame_height':int(frame.shape[0]),
                          'inference_ms':infer_ms, 'inference_fps':1000/max(infer_ms,0.001),
                          'result_fps':0.0 if last_finish is None else 1/max(finished-last_finish,0.001),
                          'processing_ms':(finished-started)*1000, 'classes':names,
                          'control_enabled':False}
                last_finish = finished
                with self._condition:
                    self._status, self._jpeg, self._result_time = status, None, captured_at
                    self._condition.notify_all()
        except Exception as exc:
            self._stop.set()
            with self._condition:
                self._status = {'state':'error', 'error':str(exc), 'detections':[], 'sequence':0}
                self._jpeg = None
                self._condition.notify_all()
        finally:
            if runtime is not None:
                runtime.release()
