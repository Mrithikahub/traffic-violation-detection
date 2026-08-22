import os
import sys
import argparse

# Ensure OpenMP runtime compatibility on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train_baseline(
    data_yaml: str = r"e:\CDAC Dataset\CDAC_Workspace\npr_module\configs\dataset_combined.yaml",
    weights: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    workers: int = 4,
    device: str = "cpu",
    project: str = r"e:\CDAC Dataset\CDAC_Workspace\npr_module\runs\detect",
    name: str = "npr_yolov8n_baseline"
):
    """
    Executes baseline YOLOv8n training on the combined Indian + Foreign dataset manifests.
    """
    from ultralytics import YOLO

    print("=" * 80)
    print("CDAC NPR MODULE - YOLOV8N BASELINE TRAINING")
    print("=" * 80)
    print(f"Data Config  : {data_yaml}")
    print(f"Base Weights : {weights}")
    print(f"Epochs       : {epochs}")
    print(f"Image Size   : {imgsz}")
    print(f"Batch Size   : {batch}")
    print(f"Workers      : {workers}")
    print(f"Device       : {device}")
    print(f"Output Path  : {os.path.join(project, name)}")
    print("=" * 80)

    # Initialize model
    model = YOLO(weights)

    # Launch training
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=device,
        seed=42,
        deterministic=True,
        cache=False,
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        # Augmentation settings
        hsv_h=0.015,           # HSV-Hue augmentation (fraction)
        hsv_s=0.7,             # HSV-Saturation augmentation (fraction)
        hsv_v=0.4,             # HSV-Value augmentation (fraction)
        degrees=5.0,           # Rotation (+/- deg)
        translate=0.1,         # Translation (+/- fraction)
        scale=0.5,             # Image scale (+/- gain)
        fliplr=0.5,            # Horizontal flip probability
        flipud=0.0,            # Vertical flip probability
        mosaic=1.0,            # Mosaic augmentation probability
        close_mosaic=10,       # Disable mosaic for last 10 epochs
        # Training mechanics
        project=project,
        name=name,
        exist_ok=True,
        save=True,
        save_period=10,
        val=True,
        patience=12
    )

    print("\n[SUCCESS] Baseline Training Complete!")
    print(f"Best model weights saved to: {os.path.join(project, name, 'weights', 'best.pt')}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8n NPR baseline")
    parser.add_argument("--data", type=str, default=r"e:\CDAC Dataset\CDAC_Workspace\npr_module\configs\dataset_combined.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16, help="Batch size (reduce to 8 if CPU memory pressure occurs)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    train_baseline(
        data_yaml=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=args.device
    )
