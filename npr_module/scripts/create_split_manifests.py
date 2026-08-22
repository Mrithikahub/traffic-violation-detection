"""
Dataset Split Manifest Generator & Deduplication Engine
CDAC AI-Based Intelligent Vehicle Monitoring & Traffic Violation Detection System
"""

import os
import sys
import random
import hashlib
from typing import List, Set, Dict, Tuple


def calculate_md5(file_path: str) -> str:
    """Calculates MD5 content hash of a file."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_manifests(
    workspace_root: str,
    output_manifest_dir: str,
    seed: int = 42
) -> Dict[str, any]:
    """
    Generates deterministic train/val/test split manifests for Foreign and Indian datasets.
    1. Excludes foreign training images whose content hash exists in the foreign test dataset.
    2. Splits clean foreign training set into 90% Train / 10% Val.
    3. Keeps foreign test set 100% untouched.
    4. Preserves the original 169-image Indian validation set.
    5. Splits the original 1,526-image Indian training set into 90% Train / 10% Test.
    6. Asserts zero overlap between all sets.
    """
    random.seed(seed)
    os.makedirs(output_manifest_dir, exist_ok=True)

    # -------------------------------------------------------------
    # 1. PATH DEFINITIONS
    # -------------------------------------------------------------
    foreign_train_img_dir = os.path.join(workspace_root, "no_plates_train_C", "images")
    foreign_test_img_dir  = os.path.join(workspace_root, "no_plates_test_C", "images")
    indian_train_img_dir  = os.path.join(workspace_root, "indian_no_plates_B", "images", "train")
    indian_val_img_dir    = os.path.join(workspace_root, "indian_no_plates_B", "images", "val")

    # -------------------------------------------------------------
    # 2. FOREIGN DATASET PROCESSING
    # -------------------------------------------------------------
    foreign_test_files = sorted([
        os.path.join(foreign_test_img_dir, f)
        for f in os.listdir(foreign_test_img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    foreign_train_raw_files = sorted([
        os.path.join(foreign_train_img_dir, f)
        for f in os.listdir(foreign_train_img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    # Hash foreign test files
    foreign_test_hashes = {calculate_md5(p): p for p in foreign_test_files}

    # Deduplicate foreign train against foreign test
    foreign_train_clean = []
    excluded_foreign_duplicates = []

    for fpath in foreign_train_raw_files:
        h = calculate_md5(fpath)
        if h in foreign_test_hashes:
            excluded_foreign_duplicates.append({
                "train_file": fpath,
                "train_basename": os.path.basename(fpath),
                "matched_test_file": foreign_test_hashes[h],
                "matched_test_basename": os.path.basename(foreign_test_hashes[h]),
                "md5_hash": h
            })
        else:
            foreign_train_clean.append(fpath)

    # Deterministic 90/10 split on clean foreign train
    shuffled_ftrain = list(foreign_train_clean)
    random.shuffle(shuffled_ftrain)

    n_f_clean = len(shuffled_ftrain)
    n_f_val = int(round(n_f_clean * 0.10))
    n_f_train = n_f_clean - n_f_val

    foreign_train_split = sorted(shuffled_ftrain[:n_f_train])
    foreign_val_split   = sorted(shuffled_ftrain[n_f_train:])
    foreign_test_split  = sorted(foreign_test_files)

    # -------------------------------------------------------------
    # 3. INDIAN DATASET PROCESSING
    # -------------------------------------------------------------
    indian_train_raw_files = sorted([
        os.path.join(indian_train_img_dir, f)
        for f in os.listdir(indian_train_img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    # Preserve original validation split
    indian_val_split = sorted([
        os.path.join(indian_val_img_dir, f)
        for f in os.listdir(indian_val_img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    # Split original Indian train into 90% train / 10% test
    shuffled_ind_tr = list(indian_train_raw_files)
    random.shuffle(shuffled_ind_tr)

    n_ind_raw = len(shuffled_ind_tr)
    n_ind_test = int(round(n_ind_raw * 0.10))
    n_ind_train = n_ind_raw - n_ind_test

    indian_train_split = sorted(shuffled_ind_tr[:n_ind_train])
    indian_test_split  = sorted(shuffled_ind_tr[n_ind_train:])

    # -------------------------------------------------------------
    # 4. COMBINED SETS
    # -------------------------------------------------------------
    train_combined = sorted(foreign_train_split + indian_train_split)
    val_combined   = sorted(foreign_val_split + indian_val_split)

    # -------------------------------------------------------------
    # 5. INTEGRITY & ZERO-OVERLAP ASSERTIONS
    # -------------------------------------------------------------
    set_ftrain = set(foreign_train_split)
    set_fval   = set(foreign_val_split)
    set_ftest  = set(foreign_test_split)

    set_itrain = set(indian_train_split)
    set_ival   = set(indian_val_split)
    set_itest  = set(indian_test_split)

    set_train_all = set(train_combined)
    set_val_all   = set(val_combined)
    set_test_all  = set(foreign_test_split + indian_test_split)

    # Internal Foreign disjointness
    assert len(set_ftrain.intersection(set_fval)) == 0, "Foreign Train and Val overlap!"
    assert len(set_ftrain.intersection(set_ftest)) == 0, "Foreign Train and Test overlap!"
    assert len(set_fval.intersection(set_ftest)) == 0, "Foreign Val and Test overlap!"

    # Internal Indian disjointness
    assert len(set_itrain.intersection(set_ival)) == 0, "Indian Train and Val overlap!"
    assert len(set_itrain.intersection(set_itest)) == 0, "Indian Train and Test overlap!"
    assert len(set_ival.intersection(set_itest)) == 0, "Indian Val and Test overlap!"

    # Cross-domain disjointness
    assert len(set_train_all.intersection(set_val_all)) == 0, "Combined Train and Val overlap!"
    assert len(set_train_all.intersection(set_test_all)) == 0, "Combined Train and Test overlap!"
    assert len(set_val_all.intersection(set_test_all)) == 0, "Combined Val and Test overlap!"

    # Hash disjointness check
    hash_ftrain = set(calculate_md5(f) for f in foreign_train_split)
    hash_fval   = set(calculate_md5(f) for f in foreign_val_split)
    hash_ftest  = set(calculate_md5(f) for f in foreign_test_split)
    assert len(hash_ftrain.intersection(hash_ftest)) == 0, "Foreign Train and Test hash collision!"
    assert len(hash_fval.intersection(hash_ftest)) == 0, "Foreign Val and Test hash collision!"

    # -------------------------------------------------------------
    # 6. WRITE MANIFEST FILES
    # -------------------------------------------------------------
    manifest_paths = {
        "train_combined": os.path.join(output_manifest_dir, "train_combined.txt"),
        "val_combined":   os.path.join(output_manifest_dir, "val_combined.txt"),
        "val_foreign":    os.path.join(output_manifest_dir, "val_foreign.txt"),
        "val_indian":     os.path.join(output_manifest_dir, "val_indian.txt"),
        "test_foreign":   os.path.join(output_manifest_dir, "test_foreign.txt"),
        "test_indian":    os.path.join(output_manifest_dir, "test_indian.txt"),
    }

    def write_list_to_file(file_path: str, lines: List[str]):
        with open(file_path, "w", encoding="utf-8") as f:
            for l in lines:
                f.write(l.replace("\\", "/") + "\n")

    write_list_to_file(manifest_paths["train_combined"], train_combined)
    write_list_to_file(manifest_paths["val_combined"], val_combined)
    write_list_to_file(manifest_paths["val_foreign"], foreign_val_split)
    write_list_to_file(manifest_paths["val_indian"], indian_val_split)
    write_list_to_file(manifest_paths["test_foreign"], foreign_test_split)
    write_list_to_file(manifest_paths["test_indian"], indian_test_split)

    results = {
        "manifest_paths": manifest_paths,
        "counts": {
            "foreign_train_raw": len(foreign_train_raw_files),
            "foreign_excluded_duplicates": len(excluded_foreign_duplicates),
            "foreign_clean_pool": n_f_clean,
            "foreign_train": len(foreign_train_split),
            "foreign_val": len(foreign_val_split),
            "foreign_test": len(foreign_test_split),
            "indian_train_raw": n_ind_raw,
            "indian_train": len(indian_train_split),
            "indian_val": len(indian_val_split),
            "indian_test": len(indian_test_split),
            "combined_train": len(train_combined),
            "combined_val": len(val_combined),
            "total_test": len(foreign_test_split) + len(indian_test_split),
        },
        "excluded_duplicates": excluded_foreign_duplicates
    }
    return results


if __name__ == "__main__":
    workspace = r"e:\CDAC Dataset\CDAC_Workspace"
    output_dir = os.path.join(workspace, "npr_module", "configs", "manifests")
    res = generate_manifests(workspace, output_dir, seed=42)

    print("=" * 80)
    print("MANIFEST GENERATION & INTEGRITY REPORT")
    print("=" * 80)
    print(f"Foreign Initial Train Count       : {res['counts']['foreign_train_raw']}")
    print(f"Foreign Excluded Duplicates       : {res['counts']['foreign_excluded_duplicates']}")
    print(f"Foreign Clean Training Pool       : {res['counts']['foreign_clean_pool']}")
    print(f"  -> Foreign Train (90%)          : {res['counts']['foreign_train']}")
    print(f"  -> Foreign Val (10%)            : {res['counts']['foreign_val']}")
    print(f"Foreign Unseen Test (Untouched)   : {res['counts']['foreign_test']}")
    print("-" * 80)
    print(f"Indian Original Train Pool        : {res['counts']['indian_train_raw']}")
    print(f"  -> Indian Train (90%)           : {res['counts']['indian_train']}")
    print(f"  -> Indian Unseen Test (10%)     : {res['counts']['indian_test']}")
    print(f"Indian Validation (Original 169)  : {res['counts']['indian_val']}")
    print("-" * 80)
    print(f"Combined Training Manifest        : {res['counts']['combined_train']} images")
    print(f"Combined Validation Manifest      : {res['counts']['combined_val']} images")
    print(f"Total Unseen Test Images          : {res['counts']['total_test']} images (1,020 Foreign + 153 Indian)")
    print("=" * 80)
    print("\nEXCLUDED DUPLICATES (Train -> Matched Test Image):")
    for idx, d in enumerate(res["excluded_duplicates"]):
        print(f"  [{idx+1:02d}] Train: {d['train_basename']} -> Test: {d['matched_test_basename']}")
    print("\n[PASS] All 6 Manifests Created & Zero-Overlap Successfully Verified!")
