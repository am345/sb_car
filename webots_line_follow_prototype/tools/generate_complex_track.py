#!/usr/bin/env python3
"""Generate the static Webots tape geometry for the complex test track."""

from math import atan2, hypot
from pathlib import Path


TAPE_WIDTH_M = 0.030
TAPE_HEIGHT_M = 0.002
TAPE_Z_M = 0.013
SEGMENT_OVERLAP_M = 0.008


# Smooth portion: bottom start -> left polygon -> upper bends -> roundabout.
CURVE_CONTROLS = [
    (-0.20, -2.90), (-1.10, -2.95), (-2.30, -2.95), (-3.05, -2.70),
    (-3.28, -2.00), (-3.32, -0.90), (-3.45, 0.20), (-3.35, 1.25),
    (-2.95, 2.10), (-2.35, 2.62), (-1.72, 2.58), (-1.30, 2.20),
    (-1.05, 1.55), (-1.45, 0.72), (-1.90, -0.12), (-1.55, -0.72),
    (-1.05, -1.18), (-0.62, -0.55), (-0.62, 0.55), (-0.25, 1.48),
    (0.12, 2.35), (0.55, 2.90), (1.02, 3.02), (1.38, 2.70),
    (1.52, 2.20), (1.95, 1.92), (2.48, 1.98), (3.02, 1.78),
    (3.35, 1.35), (3.18, 0.92), (2.72, 0.72), (2.24, 0.86),
    (1.76, 1.10), (1.28, 0.82), (1.04, 0.38), (1.00, -0.02),
    (1.14, -0.48), (1.50, -0.78), (1.96, -0.80), (2.34, -0.56),
    (2.53, -0.18), (2.43, 0.26), (2.16, 0.58), (2.65, 0.62),
]

# This part is intentionally not smoothed.  It adds three exact 90-degree
# corners.  The following zigzag is smoothed separately so this test changes
# one geometric feature at a time.
HARD_CONTROLS = [
    (3.55, 0.62), (3.55, -0.55), (2.70, -0.55), (2.70, -1.55),
]

RETURN_CONTROLS = [
    (2.70, -1.55),
    (2.85, -1.82), (3.20, -2.05), (2.72, -2.32),
    (3.10, -2.55), (2.60, -2.72), (1.80, -2.82), (0.82, -2.82),
]

# A deliberately short transverse branch.  With no sign or route instruction,
# the generic follower should preserve the longer, current heading through it.
CROSS_BRANCH = [(0.30, -3.45), (0.30, -2.28)]


def chaikin_open(points, iterations=2):
    result = list(points)
    for _ in range(iterations):
        refined = [result[0]]
        for first, second in zip(result, result[1:]):
            refined.extend((
                (0.75 * first[0] + 0.25 * second[0],
                 0.75 * first[1] + 0.25 * second[1]),
                (0.25 * first[0] + 0.75 * second[0],
                 0.25 * first[1] + 0.75 * second[1]),
            ))
        refined.append(result[-1])
        result = refined
    return result


def segments(points, closed):
    pairs = list(zip(points, points[1:]))
    if closed:
        pairs.append((points[-1], points[0]))
    for first, second in pairs:
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        yield {
            "x": (first[0] + second[0]) / 2.0,
            "y": (first[1] + second[1]) / 2.0,
            "length": hypot(dx, dy) + SEGMENT_OVERLAP_M,
            "angle": atan2(dy, dx),
        }


def build_proto():
    return_path = chaikin_open(RETURN_CONTROLS, 2)
    main = chaikin_open(CURVE_CONTROLS, 2) + HARD_CONTROLS + return_path[1:]
    all_segments = list(segments(main, closed=True))
    main_count = len(all_segments)
    all_segments.extend(segments(CROSS_BRANCH, closed=False))
    main_length = sum(item["length"] - SEGMENT_OVERLAP_M
                      for item in all_segments[:main_count])

    lines = [
        "#VRML_SIM R2025a utf8",
        "",
        "# 30 mm single-route track with exact right-angle bends and a crossroad.",
        f"# Generated main centreline length: {main_length:.3f} m.",
        "PROTO ComplexCompetitionTrack [",
        "]",
        "{",
        "  Group {",
        "    children [",
    ]
    for index, item in enumerate(all_segments):
        lines.extend((
            "      Transform {",
            f"        translation {item['x']:.6f} {item['y']:.6f} {TAPE_Z_M:.6f}",
            f"        rotation 0 0 1 {item['angle']:.6f}",
            f"        scale {item['length']:.6f} {TAPE_WIDTH_M:.6f} {TAPE_HEIGHT_M:.6f}",
            "        children [",
        ))
        if index == 0:
            lines.extend((
                "          DEF TRACK_SEGMENT Shape {",
                "            appearance PBRAppearance {",
                "              baseColor 0.012 0.012 0.015",
                "              roughness 0.96",
                "            }",
                "            geometry Box { size 1 1 1 }",
                "          }",
            ))
        else:
            lines.append("          USE TRACK_SEGMENT")
        lines.extend(("        ]", "      }"))
    lines.extend(("    ]", "  }", "}", ""))
    return "\n".join(lines)


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "protos" / "ComplexCompetitionTrack.proto"
    target.write_text(build_proto(), encoding="utf-8", newline="\n")
    print(target)
