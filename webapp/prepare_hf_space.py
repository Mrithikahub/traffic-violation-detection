"""
Assemble a Hugging Face Space folder from this repo.

    python webapp/prepare_hf_space.py --out ../hf_space

Copies only what the demo actually needs, flattened into the layout a Streamlit
Space expects (app.py and requirements.txt at the root). The file list is
derived from what run_pipeline.py imports and invokes, so it stays small.

npr_module is deliberately excluded: the demo runs with --skip-npr, and the NPR
import is lazy and wrapped in try/except, so its absence costs nothing.
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (source relative to repo root, destination relative to Space root)
FILES = [
    ("webapp/streamlit_app.py", "app.py"),
    ("webapp/requirements.txt", "requirements.txt"),
    ("webapp/assets/sample_clip.mp4", "assets/sample_clip.mp4"),
    # pipeline entry point and the scripts it invokes
    ("run_pipeline.py", "run_pipeline.py"),
    ("detect_track.py", "detect_track.py"),
    ("speed_violation.py", "speed_violation.py"),
    ("wrong_lane.py", "wrong_lane.py"),
    ("lane_change.py", "lane_change.py"),
    ("tailgating.py", "tailgating.py"),
    ("combine_violations.py", "combine_violations.py"),
    # fine estimation package (namespace package, no __init__.py at top level)
    ("fine_estimation/src/__init__.py", "fine_estimation/src/__init__.py"),
    ("fine_estimation/src/fine_calculator.py", "fine_estimation/src/fine_calculator.py"),
    ("fine_estimation/src/rule_engine.py", "fine_estimation/src/rule_engine.py"),
    ("fine_estimation/config/fine_rules.json", "fine_estimation/config/fine_rules.json"),
    # camera calibration for the sample clip
    ("outputs_demo/homography.npy", "outputs_demo/homography.npy"),
    ("outputs_demo/zones.json", "outputs_demo/zones.json"),
    ("outputs_demo/lanes.json", "outputs_demo/lanes.json"),
    ("outputs_demo/site_calibration.json", "outputs_demo/site_calibration.json"),
    # fine-tuned detector
    ("runs/detect/runs/bmd45_ft/weights/best.pt",
     "runs/detect/runs/bmd45_ft/weights/best.pt"),
    # docs worth shipping with the demo
    ("KNOWN_LIMITATIONS.md", "KNOWN_LIMITATIONS.md"),
]

SPACE_README = """---
title: Traffic Violation Detection
emoji: 🚦
colorFrom: yellow
colorTo: red
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
---

# AI-Based Intelligent Vehicle Monitoring and Traffic Violation Detection

Team Tech Titans. Vehicle detection and tracking with a YOLOv8s detector
fine-tuned on Indian CCTV footage (BMD-45) and ByteTrack, followed by four
violation layers and rule-based fine estimation. Runs on CPU.

Pick the bundled sample clip to run all four violation layers (speeding,
wrong-side, lane change, tailgating) plus fine estimation. Uploaded clips run
detection, tracking and tailgating only, because the other three layers depend
on a homography and zones fitted to one specific camera view.

Processing is CPU-only and takes roughly 7-14 seconds per second of video on a
laptop, and longer on a free shared tier. A 5 second clip typically needs one
to three minutes.

Fine amounts are configurable prototype rules for demonstration and are not
legally enforceable. Speeds rest on IRC standard road geometry that was not
site-verified. See KNOWN_LIMITATIONS.md.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Target folder for the Space files.")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    total, missing = 0, []
    for src_rel, dst_rel in FILES:
        src = ROOT / src_rel
        if not src.exists():
            missing.append(src_rel)
            continue
        dst = out / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        total += dst.stat().st_size
        print(f"  {dst.stat().st_size / 1048576:8.2f} MB  {dst_rel}")

    (out / "README.md").write_text(SPACE_README, encoding="utf-8")
    print(f"  {(out / 'README.md').stat().st_size / 1048576:8.2f} MB  README.md")

    print(f"\nSpace folder ready: {out}")
    print(f"Total size: {total / 1048576:.1f} MB")
    if missing:
        print("\nMISSING (demo will not work without these):")
        for m in missing:
            print("  -", m)


if __name__ == "__main__":
    main()
