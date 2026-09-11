"""Load the private RKNN runtime without replacing the board's system library."""
from contextlib import contextmanager
import ctypes
import os
from pathlib import Path
import sys
import threading

_loader_lock = threading.Lock()


@contextmanager
def private_runtime_library():
    # Lite2 2.3.2 opens /usr/lib/librknnrt.so by absolute path, ignoring
    # LD_LIBRARY_PATH. Redirect only this library during runtime initialization.
    # Also wrap the FIRST rknnlite import: its extension caches ctypes symbols.
    # The CDLL object keeps the handle after the normal loaders are restored.
    configured = os.environ.get('RKNN_RUNTIME_LIBRARY')
    path = Path(configured) if configured else Path(sys.prefix)/'librknnrt.so'
    if not path.is_file():
        if configured:
            raise FileNotFoundError(path)
        yield
        return
    with _loader_lock:
        original = ctypes.CDLL
        original_loader = ctypes.cdll._dlltype

        class PrivateCDLL(original):
            def __init__(self, name, *args, **kwargs):
                if name and os.path.basename(os.fsdecode(name)) == 'librknnrt.so':
                    name = str(path.resolve())
                super().__init__(name, *args, **kwargs)

        try:
            ctypes.CDLL = PrivateCDLL
            ctypes.cdll._dlltype = PrivateCDLL
            yield
        finally:
            ctypes.CDLL = original
            ctypes.cdll._dlltype = original_loader
