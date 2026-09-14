# Rust line kernel experiment

Baseline: `2a36ce2` on `refactor/original-line-following`.
Branch: `codex/rust-line-kernel`.

This is an **opt-in computational kernel**, not a complete Rust controller.
`run.py`, Webots, chassis commands, camera handling, YOLO and production imports
are unchanged. `core.rust_line_detector.LineDetector` subclasses the existing
detector and substitutes maximum-consensus scoring for line/L-stem fitting.
Thresholds and NumPy's final least-squares fit remain unchanged.

## Build and validate

Requires Rust/Cargo, a native linker, Python, NumPy and OpenCV on the same platform.
The Rust crate has no external dependencies. Linux example:

```sh
cargo test --manifest-path rust/line_kernel/Cargo.toml
cargo build --release --manifest-path rust/line_kernel/Cargo.toml
RUN_RUST_TESTS=1 python3 -m unittest discover -s tests -p test_rust_line_detector.py -v
python3 tools/benchmark_corner_detection.py --rust artifacts/ipc_image_logs_current_20260911
```

The shared library is loaded with ctypes; the call releases the Python GIL.
Override its path with `LINE_KERNEL_LIBRARY`. Build locally for the target CPU/OS;
the WSL x86-64 `.so` cannot run on the ARM IPC or Windows Python.
A missing library fails explicitly, with no silent Python fallback.

Replay compares two independent stateful detectors on the same ordered frames,
alternating timing order. It checks validity, L direction, geometry, binary masks,
points, fit coefficients and branches (numeric tolerance 1e-7), and exits nonzero
on differences. Existing replay cleans colored annotations from logged composite
images and disables physical-width enforcement; this is not raw-camera/live-control
validation. No timing includes image loading, camera wait, YOLO, WebUI or serial I/O.

## Initial validation (2026-09-14)

WSL Ubuntu x86-64, Rust 1.75 release build, OpenCV 4.6.0.
292 readable logged images across five sources matched all compared outputs;
32 unreadable source files were excluded. Seeded differential cases and two Rust
unit tests passed. Floating-point reduction order can still differ for untested
near-ties; matching these samples is not a universal equivalence proof.

Initial fixed-order 268-image run: Python mean 17.43 ms, Rust-backed mean 16.09 ms.
Second run with alternating order: **14.10 ms vs 13.43 ms (~4.7% less time)**,
p95 22.31 ms vs 21.82 ms. One seven-image subset was slightly slower with Rust;
timing noise and scene dependence matter, so these are preliminary measurements.
This is a partial migration and a WSL measurement, **not an
IPC FPS result or proof of stable 30 FPS**. Production deployment has not changed.
Repeat on the IPC before deciding on further migration.
