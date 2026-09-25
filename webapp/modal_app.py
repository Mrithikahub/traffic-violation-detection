"""
Deploy the demo app on Modal, with live number plate reading switched on.

Streamlit Community Cloud guarantees 690 MB, and plate reading alone needs about
1.5 GB, so the Community Cloud deployment shows pre-computed plate samples only.
This file runs the same Streamlit app on Modal with room for it, and sets
TVD_LIVE_PLATES=1 so plates are read from every clip.

Deploy from the repo root, after `pip install modal` and `modal setup`:

    modal deploy webapp/modal_app.py

Cost: Modal's Starter plan includes $30 of compute a month, and with no payment
method on the account, usage stops there rather than being billed. At the sizes
below (2 CPU cores, 4 GiB) a running container costs about $0.13 an hour, so
roughly 230 hours a month. The container shuts down 5 minutes after the last
visitor disconnects, but an open browser tab keeps it running, so close the
tab when you are done.
"""
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
APP_DIR = "/root/app"

# Everything the app and the pipeline scripts it runs need, in the repo layout.
FILES = [
    "run_pipeline.py", "detect_track.py", "speed_violation.py", "wrong_lane.py",
    "lane_change.py", "tailgating.py", "combine_violations.py",
    "fine_estimation/src/__init__.py", "fine_estimation/src/fine_calculator.py",
    "fine_estimation/src/rule_engine.py", "fine_estimation/config/fine_rules.json",
    "outputs_demo/homography.npy", "outputs_demo/zones.json",
    "outputs_demo/lanes.json", "outputs_demo/site_calibration.json",
    "runs/detect/runs/bmd45_ft/weights/best.pt",
    "npr_module/runs/detect/npr_yolov8n_baseline/weights/best.pt",
    "webapp/streamlit_app.py", "webapp/plate_stage.py",
    "webapp/assets/sample_clip.mp4",
]
DIRS = ["npr_module/src", "webapp/assets/plates"]


def _fetch_easyocr_weights() -> None:
    """Bake EasyOCR's ~100 MB of weights into the image instead of downloading
    them on the first plate read."""
    import easyocr
    easyocr.Reader(["en"], gpu=False)


image = (
    modal.Image.debian_slim(python_version="3.11")
    # CPU-only torch: the default PyPI build also loads ~400 MB of CUDA
    # libraries that a CPU container never uses.
    .pip_install("torch==2.8.0", "torchvision==0.23.0",
                 index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("streamlit>=1.40", "ultralytics>=8.3.0",
                 "opencv-python-headless>=4.10", "numpy>=1.24", "pandas>=2.0",
                 "lap>=0.5.12", "imageio-ffmpeg>=0.5", "easyocr>=1.7")
    .run_function(_fetch_easyocr_weights)
    .env({"TVD_LIVE_PLATES": "1"})
)
for f in FILES:
    image = image.add_local_file(REPO / f, f"{APP_DIR}/{f}")
for d in DIRS:
    image = image.add_local_dir(REPO / d, f"{APP_DIR}/{d}")

app = modal.App("traffic-violation-demo", image=image)


@app.function(
    cpu=2.0,
    memory=4096,           # plate stage peaks ~1.5 GB, plus the Streamlit server
    max_containers=1,      # one container; the app queues pipeline runs itself
    scaledown_window=300,  # shut down 5 min after the last connection closes
    timeout=3600,
)
@modal.concurrent(max_inputs=50)
@modal.web_server(8000, startup_timeout=120)
def ui() -> None:
    import subprocess
    subprocess.Popen(
        "streamlit run webapp/streamlit_app.py --server.port 8000 "
        "--server.address 0.0.0.0 --server.headless true "
        "--server.enableCORS false --server.enableXsrfProtection false "
        "--browser.gatherUsageStats false",
        shell=True, cwd=APP_DIR,
    )
