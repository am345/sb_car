#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""USB 摄像头封装（本地组件，可独立运行）。"""

import time

import cv2


class USBCamera:
    """USB 摄像头/本地视频源封装。"""

    def __init__(self, device=None, width=640, height=480, fps=30,
                 verify_reads=8):
        self.device = device          # int(设备号) 或 str(视频文件路径)
        self.width = width
        self.height = height
        self.fps = fps
        self.verify_reads = verify_reads  # 首帧读取验证重试次数(相机预热)
        self.cap = None
        self.actual_size = None       # (w, h) 实际分辨率

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
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
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
                return True
            cap.release()
        return False

    def read(self):
        """读取一帧 BGR 图像；失败返回 None。"""
        if not self.is_opened:
            return None
        ret, frame = self.cap.read()
        if not ret or frame is None or frame.size == 0:
            return None
        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass