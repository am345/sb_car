#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""USB 摄像头封装（本地组件，可独立运行）。"""

import time
import sys
import threading

import cv2


class USBCamera:
    """USB 摄像头/本地视频源封装。"""

    def __init__(self, device=None, width=640, height=480, fps=30,
                 verify_reads=8, latest_frame=False):
        self.device = device          # int(设备号) 或 str(视频文件路径)
        self.width = width
        self.height = height
        self.fps = fps
        self.verify_reads = verify_reads  # 首帧读取验证重试次数(相机预热)
        self.cap = None
        self.actual_size = None       # (w, h) 实际分辨率
        self.latest_frame = bool(latest_frame)
        self._condition = threading.Condition()
        self._reader = None
        self._stopping = False
        self._latest = None
        self._sequence = 0
        self._delivered = 0

    @property
    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()

    def open(self, device=None):
        """打开摄像头。device 为空时自动尝试 0~3 号设备。

        部分 UVC 摄像头刚打开时首帧读不到图像（需要预热），
        因此每个候选设备会重试读取 verify_reads 次再判定可用。
        """
        if device is None:
            device = self.device

        if device is not None:
            candidates = [device]
        else:
            candidates = [0, 1, 2, 3]

        for idx in candidates:
            # Linux's default backend selected GStreamer on the IPC.  When a
            # UVC device disconnected, gst_app_sink blocked forever inside
            # cap.read(), freezing line detection, serial feedback and safe
            # restart together.  Direct V4L2 returns a failed read instead.
            is_v4l2 = (sys.platform.startswith('linux') and
                       (isinstance(idx, int) or
                        str(idx).startswith('/dev/')))
            cap = cv2.VideoCapture(
                idx, cv2.CAP_V4L2 if is_v4l2 else cv2.CAP_ANY)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            if is_v4l2:
                # Prefer fresh frames when processing briefly misses a period.
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # 重试读取若干帧，容忍首帧失败/预热
            ok = False
            for _ in range(self.verify_reads):
                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    ok = True
                    break
                time.sleep(0.15)
            if ok:
                self.cap = cap
                self.device = idx
                h, w = frame.shape[:2]
                self.actual_size = (w, h)
                if self.latest_frame:
                    self._stopping = False
                    self._latest = None
                    self._sequence = self._delivered = 0
                    self._reader = threading.Thread(
                        target=self._capture_latest, daemon=True,
                        name='camera-latest-frame')
                    self._reader.start()
                return True
            cap.release()
        return False

    def read(self):
        """读取一帧 BGR 图像；失败返回 None。"""
        if self.latest_frame:
            with self._condition:
                fresh = self._condition.wait_for(
                    lambda: self._stopping or self._sequence > self._delivered,
                    timeout=0.5)
                if not fresh or self._stopping:
                    return None
                self._delivered = self._sequence
                return self._latest
        if not self.is_opened:
            return None
        ret, frame = self.cap.read()
        if not ret or frame is None or frame.size == 0:
            return None
        return frame

    def _capture_latest(self):
        # One owner reads/releases VideoCapture. Publishing replaces old frames
        # instead of queuing them, and read() never delivers a frame twice.
        capture = self.cap
        try:
            while not self._stopping:
                ok, frame = capture.read()
                if not ok or frame is None or frame.size == 0:
                    break
                with self._condition:
                    self._latest = frame
                    self._sequence += 1
                    self._condition.notify_all()
        finally:
            capture.release()
            with self._condition:
                self._stopping = True
                self._condition.notify_all()

    def release(self):
        if self._reader is not None:
            with self._condition:
                self._stopping = True
                self._condition.notify_all()
            self._reader.join(timeout=1.0)
            self._reader = None
            self.cap = None
            return
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
