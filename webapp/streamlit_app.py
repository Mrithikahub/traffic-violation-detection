"""
Live demo for the AI-Based Intelligent Vehicle Monitoring and Traffic Violation
Detection System.

Runs the real pipeline (fine-tuned YOLOv8s + ByteTrack, then the violation
layers), not a mock. CPU only.

IMPORTANT ON CALIBRATION
------------------------
Speed, wrong-side and lane-change detection all depend on a homography and hand
marked zones fitted to ONE camera view. Applying that calibration to footage
from any other camera would produce numbers that look precise and mean nothing.
So the full four-layer pipeline runs only on the bundled sample clip, and
uploaded clips get detection, tracking and tailgating, which is scale-free and
transfers across cameras.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

_HERE = Path(__file__).resolve().parent
# Works from two layouts: this repo (webapp/streamlit_app.py, pipeline scripts
# one level up) and a Hugging Face Space (app.py beside the pipeline scripts).
ROOT = _HERE if (_HERE / "detect_track.py").exists() else _HERE.parent
SAMPLE = next((p for p in (_HERE / "assets" / "sample_clip.mp4",
                           ROOT / "webapp" / "assets" / "sample_clip.mp4")
               if p.exists()), _HERE / "assets" / "sample_clip.mp4")
FINETUNED = ROOT / "runs" / "detect" / "runs" / "bmd45_ft" / "weights" / "best.pt"

st.set_page_config(page_title="Traffic Violation Detection", page_icon="🚦",
                   layout="wide")


def weights():
    """Fine-tuned weights if shipped, else stock YOLOv8s (auto-downloaded)."""
    if FINETUNED.exists():
        return str(FINETUNED), "fine-tuned YOLOv8s (BMD-45)"
    return "yolov8s.pt", "stock YOLOv8s (fine-tuned weights not bundled)"


def trim(src: Path, dst: Path, seconds: int) -> tuple[int, float]:
    """Copy the first N seconds so a demo run stays inside free-tier limits."""
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n, limit = 0, int(fps * seconds)
    while n < limit:
        ok, f = cap.read()
        if not ok:
            break
        out.write(f)
        n += 1
    cap.release()
    out.release()
    return n, fps


def to_h264(src: Path) -> Path | None:
    """Browsers will not play OpenCV's mp4v output; re-encode to H.264."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
    dst = src.with_name(src.stem + "_h264.mp4")
    r = subprocess.run([exe, "-y", "-i", str(src), "-vcodec", "libx264",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        str(dst)], capture_output=True)
    return dst if r.returncode == 0 and dst.exists() else None


@st.cache_resource
def _run_lock():
    """Serialises pipeline subprocesses across all viewers of this server.

    Community Cloud serves everyone from one container and guarantees only
    690MB. Measured here: the app itself is ~100MB, one pipeline subprocess
    peaks at ~520MB. Two at once is what would actually exhaust the container,
    so only one torch process is allowed to run at a time.
    """
    import threading
    return threading.Lock()


def run(cmd: list[str], log: list[str]) -> bool:
    env = dict(os.environ)
    # Community Cloud allocates as little as 0.078 CPU cores. Letting torch
    # spawn a thread per core it thinks it has causes contention, not speed.
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")

    lock = _run_lock()
    if not lock.acquire(timeout=900):
        log.append("timed out waiting for another run to finish")
        return False
    try:
        r = subprocess.run([sys.executable] + cmd, cwd=str(ROOT),
                           capture_output=True, text=True, env=env)
    finally:
        lock.release()
    log.append(f"$ {' '.join(cmd[:2])} ... rc={r.returncode}")
    if r.returncode != 0:
        log.append((r.stderr or "")[-1500:])
    return r.returncode == 0


st.title("🚦 Traffic Violation Detection")
st.caption("Vehicle detection and tracking, four violation layers, and rule-based "
           "fine estimation. Team Tech Titans.")

wpath, wlabel = weights()
with st.sidebar:
    st.header("Input")
    mode = st.radio("Source", ["Sample clip (full pipeline)", "Upload a clip"])
    seconds = st.slider("Seconds to process", 3, 8, 3,
                        help="Processing time scales linearly with clip length.")
    st.divider()
    st.write(f"**Detector:** {wlabel}")
    st.write("**Tracker:** ByteTrack")
    # Measured on an i5-1235U, CPU only: 5 s of video took 34 s (warm) and 8 s
    # took 109 s (cold start, model load included), i.e. 7-14 s of compute per
    # second of video. Shared cloud CPUs are slower again.
    st.caption(f"CPU only. Measured at 7-14 s of processing per second of video "
               f"on a laptop CPU, and slower on a free shared tier, so expect "
               f"roughly {max(1, seconds * 7 // 60)}-{seconds * 30 // 60 + 1} "
               f"minutes for {seconds} s of video.")

uploaded = None
if mode == "Upload a clip":
    uploaded = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv"])
    st.info("Speed, wrong-side and lane-change need a homography and zones fitted "
            "to one specific camera, which does not exist for an uploaded clip. "
            "This mode runs detection, tracking and tailgating, which is "
            "scale-free. Pick the sample clip to see all four layers.")

go = st.button("Run pipeline", type="primary", use_container_width=True)

if go:
    if mode == "Upload a clip" and uploaded is None:
        st.error("Please choose a video file first.")
        st.stop()
    if mode.startswith("Sample") and not SAMPLE.exists():
        st.error(f"Sample clip missing at {SAMPLE}.")
        st.stop()

    work = Path(tempfile.mkdtemp(prefix="demo_"))
    src = work / "input.mp4"
    if uploaded is not None:
        src.write_bytes(uploaded.read())
    else:
        shutil.copy(SAMPLE, src)

    log: list[str] = []
    t0 = time.time()
    with st.status("Running the pipeline…", expanded=True) as status:
        st.write("Trimming clip")
        nframes, fps = trim(src, work / "clip.mp4", seconds)
        if nframes == 0:
            st.error("Could not read any frames from that file.")
            st.stop()
        clip = work / "clip.mp4"
        stem = clip.stem

        full = mode.startswith("Sample")
        if full:
            st.write(f"Detecting, tracking and scoring 4 violation layers on {nframes} frames")
            ok = run(["run_pipeline.py", "--video", str(clip), "--out-dir", str(work),
                      "--model", wpath, "--skip-npr", "--skip-crops", "--save-video"], log)
        else:
            st.write(f"Detecting and tracking {nframes} frames")
            ok = run(["detect_track.py", "--video", str(clip), "--model", wpath,
                      "--conf", "0.10", "--imgsz", "640", "--out-dir", str(work)], log)
            if ok:
                st.write("Checking tailgating")
                ok = run(["tailgating.py", "--video", str(clip), "--tracks",
                          str(work / f"{stem}_tracks.csv"), "--out-dir", str(work),
                          "--save-video"], log)
        if not ok:
            status.update(label="Pipeline failed", state="error")
            st.code("\n".join(log))
            st.stop()
        status.update(label=f"Done in {time.time() - t0:.0f} s", state="complete")

    elapsed = time.time() - t0
    st.success(f"Processed {nframes} frames ({nframes / fps:.1f} s of video) in "
               f"{elapsed:.0f} s — {nframes / elapsed:.1f} FPS on this machine.")

    vid = (work / f"{stem}_all_violations.mp4") if full else (work / f"{stem}_tailgating.mp4")
    if vid.exists():
        play = to_h264(vid) or vid
        st.video(str(play))
        st.download_button("Download annotated video", vid.read_bytes(),
                           file_name=vid.name, mime="video/mp4")
    else:
        st.warning("No annotated video was produced.")

    tracks = work / f"{stem}_tracks.csv"
    vehicles = work / f"{stem}_vehicles.csv"
    c1, c2, c3 = st.columns(3)
    if tracks.exists():
        df = pd.read_csv(tracks)
        c1.metric("Detections", f"{len(df):,}")
        c2.metric("Vehicles tracked", df[df.vehicle_id != -1].vehicle_id.nunique())
    if vehicles.exists():
        vdf = pd.read_csv(vehicles)
        c3.metric("Classes seen", vdf["class"].nunique())
        st.subheader("Vehicles")
        st.dataframe(vdf, use_container_width=True, height=240)

    if full:
        fines = work / f"{stem}_fines.csv"
        kpi = work / f"{stem}_summary_kpi.json"
        if kpi.exists():
            k = json.loads(kpi.read_text(encoding="utf-8"))
            st.subheader("Violations")
            cols = st.columns(5)
            for col, (name, key) in zip(cols, [("Speeding", "speeding"),
                                               ("Wrong side", "wrong_side"),
                                               ("Lane change", "lane_change"),
                                               ("Tailgating", "tailgating")]):
                col.metric(name, k.get("violations_by_type", {}).get(key, 0))
            cols[4].metric("Total fines (INR)", f"{k.get('total_prototype_fines', 0):,.0f}")
        if fines.exists():
            fdf = pd.read_csv(fines)
            flagged = fdf[fdf.total_fine > 0]
            st.subheader(f"Fines — {len(flagged)} vehicles")
            st.dataframe(flagged, use_container_width=True, height=280)
        st.caption("Fine amounts are configurable prototype rules for demonstration "
                   "only and are not legally enforceable. Speeds rest on IRC standard "
                   "road geometry that was not site-verified.")
    else:
        ev = work / f"{stem}_tailgating_events.csv"
        if ev.exists():
            edf = pd.read_csv(ev)
            st.subheader(f"Tailgating events — {len(edf)}")
            st.dataframe(edf, use_container_width=True, height=280)

    with st.expander("Run log"):
        st.code("\n".join(log))


def plate_samples() -> None:
    """Pre-computed plate recognition results, read from files.

    Live plate recognition (YOLOv8n + EasyOCR) needs roughly 1.5 GB on its own,
    more than Community Cloud guarantees, and the sample clip's plates are too
    small to read anyway. These results come from webapp/precompute_plates.py,
    which runs the real recognizer on still photos and keeps a result only if
    its OCR matches the plate text a person read off the photo.
    """
    d = next((p for p in (_HERE / "assets" / "plates",
                          ROOT / "webapp" / "assets" / "plates")
              if (p / "results.json").exists()), None)
    if d is None:
        return
    samples = json.loads((d / "results.json").read_text(encoding="utf-8"))["samples"]
    st.divider()
    st.subheader("Sample plate recognition result (pre-computed, not run live)")
    st.caption("These photos were run through the project's plate pipeline "
               "(YOLOv8n plate detector, then EasyOCR) offline, and the output "
               "is shown as it came out. The text was checked against each "
               "plate by eye. Plate recognition does not run on the clips above: "
               "it needs more memory than this free host provides, and plates "
               "in the sample clip are too small to read.")
    for s in samples:
        c1, c2 = st.columns([3, 2])
        c1.image(str(d / s["photo"]), caption="Detected plate (green box)",
                 use_container_width=True)
        c2.image(str(d / s["crop"]), caption="Plate crop passed to OCR",
                 use_container_width=True)
        c2.metric("OCR text", s["plate_text"])
        c2.write(f"OCR confidence **{s['ocr_confidence']:.2f}** · plate detection "
                 f"confidence **{s['detection_confidence']:.2f}**")
        if s.get("is_valid_indian_format"):
            c2.write(f"Valid Indian registration format"
                     + (f" ({s['state_name']})" if s.get("state_name") else ""))
        c2.caption(f"Raw OCR output before cleaning: `{s['raw_ocr_text']}`")


plate_samples()
