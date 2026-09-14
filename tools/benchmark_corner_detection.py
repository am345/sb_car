"""Compare batched L stem fitting with the previous scalar implementation."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.line_follower import LineDetector


class ScalarDetector(LineDetector):
    _batch_corner_stem_fit = staticmethod(LineDetector._robust_linear_fit)


def detector(cls):
    return cls(crop_bottom_frac=.70, crop_top_frac=.90, track_half=60,
               enforce_width=False, line_width_model={
                   'horizontal_fov_deg': 100.0, 'camera_height_m': .23,
                   'pitch_down_deg': 8.0, 'segmentation_scale': .55,
                   'min_width_mm': 10.0, 'max_width_mm': 100.0})


def clean(image):
    if image.shape[1] >= 2 * image.shape[0]:
        image = image[:, :image.shape[1]//2].copy()
    channels = image.astype(np.int16)
    mask = (((channels.max(2)-channels.min(2)) > 55) &
            (channels.max(2) > 85)).astype(np.uint8)*255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sources', nargs='+', type=Path)
    parser.add_argument('--before', type=Path)
    parser.add_argument('--rust', action='store_true', help='Compare current Python with opt-in Rust')
    args = parser.parse_args()
    old_class = ScalarDetector
    if args.before:
        spec = importlib.util.spec_from_file_location('before_detector', args.before)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        old_class = module.LineDetector
    new_class = LineDetector
    if args.rust:
        from core.rust_line_detector import LineDetector as RustDetector
        old_class, new_class = LineDetector, RustDetector
    failures = 0
    for source in args.sources:
        before, after = detector(old_class), detector(new_class)
        paths = sorted(source.rglob('*.jpg')) if source.is_dir() else [source]
        times = [[], []]
        counts = [0, 0]
        unreadable, changed, numeric_changed, detail_changed = 0, [], [], []
        for path in paths:
            data = np.fromfile(str(path), dtype=np.uint8)
            image = cv2.imdecode(data, 1) if data.size else None
            if image is None:
                unreadable += 1
                continue
            frame = clean(image)
            results = [None, None]
            # Alternate order to avoid consistently giving one backend a warm CPU/cache.
            order = ((0, before), (1, after))
            if len(times[0]) % 2:
                order = order[::-1]
            for i, d in order:
                start = time.perf_counter()
                result = d.process(frame)
                times[i].append((time.perf_counter()-start)*1000)
                counts[i] += int(bool(result['corner_dir']))
                results[i] = result
            old, new = results
            detail_keys = ('binary', 'points', 'fit_coeffs', 'branch_candidates',
                           'selected_branch_direction', 'corner_point',
                           'junction_left', 'junction_right', 'junction_straight')
            different = [k for k in detail_keys if not same(old.get(k), new.get(k))]
            if different:
                detail_changed.append({'file': str(path), 'fields': different})
            if any(old[k] != new[k] for k in ('is_valid', 'corner_dir')):
                changed.append({'file': str(path), 'old': int(old['corner_dir']),
                                'new': int(new['corner_dir'])})
            keys = ('error_px', 'angle_deg', 'corner_y_ratio', 'corner_span')
            if not np.allclose([old[k] for k in keys], [new[k] for k in keys],
                               rtol=1e-7, atol=1e-7):
                numeric_changed.append(str(path))
        print(json.dumps({'source': str(source), 'readable': len(times[0]),
                          'unreadable': unreadable, 'L_counts': counts,
                          'changed': changed, 'numeric_changed': numeric_changed,
                          'detail_changed': detail_changed,
                          'mean_ms': [float(np.mean(t)) for t in times],
                          'p95_ms': [float(np.percentile(t, 95)) for t in times]},
                         ensure_ascii=True), flush=True)
        failures += len(changed) + len(numeric_changed) + len(detail_changed)
        failures += int(not times[0])
    if failures:
        raise SystemExit(1)


def same(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and
                a.shape == b.shape and np.allclose(a, b, rtol=1e-7, atol=1e-7))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
        return bool(np.isclose(a, b, rtol=1e-7, atol=1e-7))
    return a == b


if __name__ == '__main__':
    main()
