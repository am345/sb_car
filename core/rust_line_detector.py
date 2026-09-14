"""Opt-in Rust experiment; production imports remain unchanged.

Build: cargo build --release --manifest-path rust/line_kernel/Cargo.toml
Set LINE_KERNEL_LIBRARY to override the platform-specific library path.
Final least-squares refitting deliberately remains NumPy for parity.
"""
import ctypes
import os
from pathlib import Path
import sys

import numpy as np
from core.line_follower import LineDetector as PythonLineDetector


def _load():
    name = ('line_kernel.dll' if sys.platform == 'win32' else
            'libline_kernel.dylib' if sys.platform == 'darwin' else 'libline_kernel.so')
    path = Path(__file__).resolve().parents[1] / 'rust/line_kernel/target/release' / name
    lib = ctypes.CDLL(os.environ.get('LINE_KERNEL_LIBRARY', str(path)))
    lib.line_consensus.argtypes = [ctypes.POINTER(ctypes.c_double)] * 2 + [
        ctypes.c_size_t, ctypes.c_double, ctypes.POINTER(ctypes.c_size_t)]
    lib.line_consensus.restype = ctypes.c_size_t
    return lib


_library = None


class LineDetector(PythonLineDetector):
    @staticmethod
    def _batch_corner_stem_fit(points, residual_limit):
        global _library
        if len(points) < 3:
            return None, []
        if _library is None:
            _library = _load()
        xs = np.asarray([p[0] for p in points], dtype=np.float64)
        ys = np.asarray([p[1] for p in points], dtype=np.float64)
        indices = np.empty(len(points), dtype=np.uintp)
        n = _library.line_consensus(
            xs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ys.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(points), residual_limit,
            indices.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)))
        if not n:
            return None, []
        selected = indices[:n]
        try:
            slope, intercept = np.polyfit(ys[selected], xs[selected], 1)
        except (ValueError, np.linalg.LinAlgError):
            return None, []
        return ((0.0, float(slope), float(intercept)), [points[int(i)] for i in selected])

    @staticmethod
    def _robust_linear_fit(points, residual_limit=8.0):
        return LineDetector._batch_corner_stem_fit(points, residual_limit)
