"""
4-Way Disaggregated Evaluation Script for NPR Baseline (YOLOv8n best.pt)
=========================================================================
Runs Ultralytics native model.val() on each of the 4 evaluation splits:
  A. Indian Validation Set     (val_indian.txt)
  B. Foreign Validation Set    (val_foreign.txt)
  C. Indian Unseen Test Set    (test_indian.txt)   <- TRULY UNSEEN
  D. Foreign Unseen Test Set   (test_foreign.txt)  <- TRULY UNSEEN

Also performs a sanity check (via file path intersection) to confirm
that ZERO test-set images appear in the training manifest.

Saves per-split results to:
  npr_module/eval_outputs/4way_eval_results.json
  npr_module/eval_outputs/4way_eval_report.txt

NO model weights are modified. NO dataset files are touched.
"""

import os, sys, json, hashlib, datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

WORKSPACE  = r"e:\CDAC Dataset\CDAC_Workspace"
NPR_ROOT   = os.path.join(WORKSPACE, "npr_module")
WEIGHTS    = os.path.join(NPR_ROOT, "runs", "detect", "npr_yolov8n_baseline", "weights", "best.pt")
MANIFEST_DIR = os.path.join(NPR_ROOT, "configs", "manifests")
EVAL_OUT   = os.path.join(NPR_ROOT, "eval_outputs")
os.makedirs(EVAL_OUT, exist_ok=True)

SPLITS = {
    "A_indian_val":   ("Indian Validation Set   [VAL]",    "val_indian.txt",   "VALIDATION"),
    "B_foreign_val":  ("Foreign Validation Set  [VAL]",    "val_foreign.txt",  "VALIDATION"),
    "C_indian_test":  ("Indian Unseen Test Set  [TEST]",   "test_indian.txt",  "UNSEEN_TEST"),
    "D_foreign_test": ("Foreign Unseen Test Set [TEST]",   "test_foreign.txt", "UNSEEN_TEST"),
}

TRAIN_MANIFEST = os.path.join(MANIFEST_DIR, "train_combined.txt")

# ---------------------------------------------------------------------------
# STEP 1: Sanity check — confirm ZERO test image paths exist in train manifest
# ---------------------------------------------------------------------------
def run_sanity_check():
    print("\n" + "="*80)
    print("STEP 1: SANITY CHECK — Test images must NOT appear in training set")
    print("="*80)

    with open(TRAIN_MANIFEST, "r", encoding="utf-8") as f:
        train_paths = set(l.strip().replace("\\", "/").lower() for l in f if l.strip())

    test_manifests = ["test_indian.txt", "test_foreign.txt"]
    violations = []
    for mf in test_manifests:
        mpath = os.path.join(MANIFEST_DIR, mf)
        with open(mpath, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip().replace("\\", "/").lower()
                if p and p in train_paths:
                    violations.append((mf, p))

    if violations:
        print(f"  [FAIL] {len(violations)} test image(s) found in training manifest!")
        for mf, p in violations[:10]:
            print(f"    {mf}: {p}")
    else:
        print("  [PASS] Zero path overlaps found between test splits and training manifest.")

    return len(violations) == 0


# ---------------------------------------------------------------------------
# STEP 2: Build a temporary per-split YAML for model.val()
# ---------------------------------------------------------------------------
def build_split_yaml(split_key, manifest_fname):
    yaml_content = f"""path: {WORKSPACE}
train: {TRAIN_MANIFEST}
val: {os.path.join(MANIFEST_DIR, manifest_fname)}
test: {os.path.join(MANIFEST_DIR, manifest_fname)}

nc: 1
names:
  0: license_plate
"""
    yaml_path = os.path.join(EVAL_OUT, f"eval_{split_key}.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    return yaml_path


# ---------------------------------------------------------------------------
# STEP 3: Run Ultralytics model.val() on each split
# ---------------------------------------------------------------------------
def run_evaluation():
    from ultralytics import YOLO

    print("\n" + "="*80)
    print("STEP 2: RUNNING 4-WAY DISAGGREGATED EVALUATION")
    print(f"  Weights : {WEIGHTS}")
    print(f"  Device  : CPU")
    print("="*80)

    model = YOLO(WEIGHTS)
    all_results = {}

    for split_key, (display_name, manifest_fname, split_type) in SPLITS.items():
        manifest_path = os.path.join(MANIFEST_DIR, manifest_fname)

        # Count images and instances
        with open(manifest_path, "r", encoding="utf-8") as f:
            img_paths = [l.strip() for l in f if l.strip()]
        n_images = len(img_paths)

        n_instances = 0
        for ip in img_paths:
            lp = ip.replace("\\images\\", "\\labels\\").replace("/images/", "/labels/")
            lp = os.path.splitext(lp)[0] + ".txt"
            if os.path.exists(lp):
                with open(lp, "r", encoding="utf-8") as lf:
                    n_instances += sum(1 for ln in lf if ln.strip())

        print(f"\n  [{split_key}] {display_name}")
        print(f"    Images: {n_images} | Instances: {n_instances}")

        yaml_path = build_split_yaml(split_key, manifest_fname)

        # Run validation using the split manifest as the 'val' set
        metrics = model.val(
            data=yaml_path,
            split="val",        # uses the 'val:' key in the yaml
            device="cpu",
            imgsz=640,
            batch=16,
            workers=4,
            verbose=False,
            plots=False,
            save_json=False,
            name=f"eval_{split_key}",
            exist_ok=True,
        )

        p   = float(metrics.box.mp)
        r   = float(metrics.box.mr)
        m50 = float(metrics.box.map50)
        m95 = float(metrics.box.map)

        print(f"    Precision : {p:.4f}")
        print(f"    Recall    : {r:.4f}")
        print(f"    mAP@50    : {m50:.4f}")
        print(f"    mAP@50-95 : {m95:.4f}")

        all_results[split_key] = {
            "display_name": display_name.strip(),
            "split_type":   split_type,
            "manifest":     manifest_fname,
            "n_images":     n_images,
            "n_instances":  n_instances,
            "precision":    round(p,   4),
            "recall":       round(r,   4),
            "map50":        round(m50, 4),
            "map5095":      round(m95, 4),
        }

    return all_results


# ---------------------------------------------------------------------------
# STEP 4: Print + Save summary report
# ---------------------------------------------------------------------------
def print_report(results, sanity_ok):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("="*100)
    lines.append("CDAC NPR MODULE — 4-WAY DISAGGREGATED EVALUATION REPORT")
    lines.append(f"Generated : {ts}")
    lines.append(f"Weights   : {WEIGHTS}")
    lines.append(f"Sanity Check (No test-train overlap): {'PASS' if sanity_ok else 'FAIL *** ALERT ***'}")
    lines.append("="*100)
    lines.append("")

    # Validation splits first
    lines.append("--- VALIDATION SPLITS (seen during training) ---------------------------------")
    lines.append(f"{'Split':<38} {'Images':>7} {'Instances':>10} {'Precision':>11} {'Recall':>8} {'mAP@50':>8} {'mAP@50-95':>11}")
    lines.append("-"*100)
    for k, r in results.items():
        if r["split_type"] == "VALIDATION":
            lines.append(f"  {r['display_name']:<36} {r['n_images']:>7d} {r['n_instances']:>10d} "
                         f"{r['precision']:>11.4f} {r['recall']:>8.4f} {r['map50']:>8.4f} {r['map5095']:>11.4f}")

    lines.append("")
    lines.append("--- UNSEEN TEST SPLITS (never seen during training) --------------------------")
    lines.append(f"{'Split':<38} {'Images':>7} {'Instances':>10} {'Precision':>11} {'Recall':>8} {'mAP@50':>8} {'mAP@50-95':>11}")
    lines.append("-"*100)
    for k, r in results.items():
        if r["split_type"] == "UNSEEN_TEST":
            lines.append(f"  {r['display_name']:<36} {r['n_images']:>7d} {r['n_instances']:>10d} "
                         f"{r['precision']:>11.4f} {r['recall']:>8.4f} {r['map50']:>8.4f} {r['map5095']:>11.4f}")

    lines.append("")
    lines.append("="*100)
    report_str = "\n".join(lines)
    print("\n" + report_str)

    # Save text report
    rpt_path = os.path.join(EVAL_OUT, "4way_eval_report.txt")
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write(report_str)

    # Save JSON
    json_path = os.path.join(EVAL_OUT, "4way_eval_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"generated_at": ts, "weights": WEIGHTS, "sanity_ok": sanity_ok, "results": results}, f, indent=2)

    print(f"\n  [SAVED] Text report : {rpt_path}")
    print(f"  [SAVED] JSON data   : {json_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sanity_ok = run_sanity_check()
    results   = run_evaluation()
    print_report(results, sanity_ok)
    print("\n[DONE] 4-way evaluation complete.\n")
