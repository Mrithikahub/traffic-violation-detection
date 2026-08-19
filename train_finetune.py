"""
Fine-tune the detector on the canonical class scheme.

Starts from pretrained COCO weights: the backbone transfers, but the detection
head is rebuilt for our class count, so classification is learned from scratch
and needs real epochs before the numbers mean anything. A 2-epoch model will
look far WORSE than the pretrained one -- that is the head converging, not a
failure.

Usage:
    python train_finetune.py --epochs 20
"""

import argparse
import time
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune YOLO on the canonical vehicle classes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", default="data/bmd45_yolo/data.yaml")
    p.add_argument("--model", default="yolov8s.pt",
                   help="Starting weights. Pretrained COCO weights transfer the "
                        "backbone; the head is resized to the new class count.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5,
                   help="Stop early if val fitness has not improved for this many "
                        "epochs. On CPU each epoch is expensive, so this matters.")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Keep at 640: lowering it shrinks two-wheelers, which are "
                        "the class we are trying to recover.")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--name", default="bmd45_ft")
    p.add_argument("--project", default="runs")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"start weights : {args.model}")
    print(f"data          : {args.data}")
    print(f"epochs        : {args.epochs} (patience {args.patience})")
    print(f"imgsz/batch   : {args.imgsz} / {args.batch}   device={args.device}\n",
          flush=True)

    t0 = time.time()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        pretrained=True,
        val=True,          # per-epoch val curves, so training can be judged
        plots=True,
        seed=args.seed,
        verbose=True,
    )
    el = time.time() - t0
    out = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\ntrained in {el/60:.1f} min")
    print(f"best weights: {out.resolve()}")


if __name__ == "__main__":
    main()
