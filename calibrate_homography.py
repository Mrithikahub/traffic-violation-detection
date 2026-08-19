"""
Ground-plane calibration from lane markings.

Produces a 3x3 homography mapping image pixels to real-world metres on the road
surface, for speed_violation.py --homography.

METHOD
    Lane markings are detected in a chosen frame, grouped into two lane lines,
    and their endpoints matched to world coordinates implied by standard road
    geometry. The homography is then fitted by least squares over ALL dash
    endpoints rather than 4 hand-picked points -- a 4-point fit has zero
    redundancy, so it cannot reveal its own errors, and an early attempt here
    looked plausible while being badly wrong (dash lengths came out 3.3-5.5 m
    instead of a constant 3 m).

WHAT IS ASSUMED VS MEASURED
    Assumed (IRC standards, NOT site-verified):
        lane width  3.5 m   (IRC 86, urban)
        dash pitch  7.5 m   (IRC 35: 3.0 m dash + 4.5 m gap)
    Solved from the image:
        dash length (the dash:gap split varies by road class)
        along-road offset between the two lane lines' dash patterns

    Pitch anchors the along-road scale, so it is the assumption that matters
    most: every speed scales linearly with it. Forcing the IRC 3 m dash length
    as well produced inconsistent geometry, so that one is solved instead.

VALIDATION
    The fit is scored on quantities it must reproduce, then re-checked: the
    recovered pitch should come back at ~7.5 m across intervals. If it does not,
    the correspondences are wrong and the calibration must not be used.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="Fit an image->ground-plane homography from lane markings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dashes", required=True,
                   help="JSON: {'line1': [[[fx,fy],[nx,ny]], ...], "
                        "'line2': [...]} -- dash endpoints far->near per lane "
                        "line, ordered by distance from camera.")
    p.add_argument("--out", default="outputs/homography.npy")
    p.add_argument("--lane-width-m", type=float, default=3.5,
                   help="IRC 86 urban lane width.")
    p.add_argument("--pitch-m", type=float, default=7.5,
                   help="IRC 35 dash pitch (3.0 m dash + 4.5 m gap). This sets "
                        "the along-road scale.")
    p.add_argument("--dash-min", type=float, default=2.0)
    p.add_argument("--dash-max", type=float, default=5.5)
    p.add_argument("--max-pitch-error-m", type=float, default=0.6,
                   help="Refuse to write the homography if the recovered pitch "
                        "deviates from --pitch-m by more than this.")
    return p.parse_args()


def build(L1, L2, offset, dash, pitch, lane):
    src, dst = [], []
    for i, (f, n) in enumerate(L1):
        src += [f, n]
        dst += [[0.0, i * pitch], [0.0, i * pitch + dash]]
    for j, (f, n) in enumerate(L2):
        src += [f, n]
        dst += [[lane, offset + j * pitch], [lane, offset + j * pitch + dash]]
    return np.float32(src), np.float32(dst)


def measure(Hm, L1, L2):
    def w(p):
        return cv2.perspectiveTransform(
            np.float32(p).reshape(-1, 1, 2), Hm).reshape(-1, 2)
    lens, pitches = [], []
    for line in (L1, L2):
        for f, n in line:
            a, b = w([f, n])
            lens.append(float(np.linalg.norm(b - a)))
        for (f1, _), (f2, _) in zip(line, line[1:]):
            a, b = w([f1, f2])
            pitches.append(float(np.linalg.norm(b - a)))
    return np.array(lens), np.array(pitches)


def main():
    args = parse_args()
    # utf-8-sig: PowerShell's Out-File -Encoding utf8 prepends a BOM, which
    # plain utf-8 decoding rejects.
    spec = json.loads(Path(args.dashes).read_text(encoding="utf-8-sig"))
    L1 = [tuple(map(tuple, d)) for d in spec["line1"]]
    L2 = [tuple(map(tuple, d)) for d in spec["line2"]]
    if len(L1) < 2 or len(L2) < 2:
        raise SystemExit("need >=2 dashes on each lane line")

    best = None
    for dash in np.arange(args.dash_min, args.dash_max + 1e-9, 0.05):
        for off in np.arange(-15.0, 15.01, 0.25):
            src, dst = build(L1, L2, off, dash, args.pitch_m, args.lane_width_m)
            Hm, _ = cv2.findHomography(src, dst, method=0)
            if Hm is None:
                continue
            lens, pitches = measure(Hm, L1, L2)
            # score on pitch fidelity + dash self-consistency (not dash value,
            # which is what we are solving for)
            err = float(np.sqrt(np.mean((pitches - args.pitch_m) ** 2)
                                + lens.var()))
            if best is None or err < best[0]:
                best = (err, off, dash, Hm, lens, pitches)

    err, off, dash, Hm, lens, pitches = best
    src, dst = build(L1, L2, off, dash, args.pitch_m, args.lane_width_m)
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), Hm).reshape(-1, 2)
    resid = np.linalg.norm(proj - dst, axis=1)

    print(f"solved dash length : {dash:.2f} m  (gap {args.pitch_m - dash:.2f} m)")
    print(f"solved L2 offset   : {off:+.2f} m")
    print(f"dash lengths       : mean {lens.mean():.2f}  sd {lens.std():.2f}")
    print(f"pitch (IRC {args.pitch_m})   : mean {pitches.mean():.2f}  "
          f"sd {pitches.std():.2f}")
    print(f"reprojection       : mean {resid.mean():.3f} m  max {resid.max():.3f} m")

    pitch_err = abs(pitches.mean() - args.pitch_m)
    if pitch_err > args.max_pitch_error_m:
        raise SystemExit(
            f"\nVALIDATION FAILED: recovered pitch {pitches.mean():.2f} m "
            f"differs from {args.pitch_m} m by {pitch_err:.2f} m "
            f"(limit {args.max_pitch_error_m}). The dash correspondences are "
            f"probably wrong -- check the ordering and that no dash was missed. "
            f"Homography NOT written.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, Hm)
    meta = {
        "homography": Hm.tolist(),
        "assumed_lane_width_m": args.lane_width_m,
        "assumed_pitch_m": args.pitch_m,
        "solved_dash_length_m": round(float(dash), 3),
        "solved_line2_offset_m": round(float(off), 3),
        "recovered_pitch_mean_m": round(float(pitches.mean()), 3),
        "recovered_pitch_sd_m": round(float(pitches.std()), 3),
        "reprojection_mean_m": round(float(resid.mean()), 4),
        "reprojection_max_m": round(float(resid.max()), 4),
        "PROVENANCE": "IRC standard road geometry, NOT site-verified. No "
                      "distance in this scene was physically measured.",
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2),
                                        encoding="utf-8")
    print(f"\nVALIDATION PASSED (pitch within {pitch_err:.2f} m)")
    print(f"Wrote {out}\nWrote {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
