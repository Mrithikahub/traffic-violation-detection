"""
Generate Comparison Report for Preprocessing & OCR Improvements
"""
import json, os

json_path = r"e:\CDAC Dataset\CDAC_Workspace\npr_module\audit_outputs\preprocessing_ocr_audit.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

results = data["results"]
variants = data["variants_tested"]

print("=" * 110)
print("PREPROCESSING & OCR IMPROVEMENT AUDIT: DETAILED BEFORE/AFTER COMPARISON")
print("=" * 110)

for idx, r in enumerate(results, 1):
    fname = r["image_file"]
    grp = r["group"]
    dim = r["crop_dimensions_wh"]
    ar = r["crop_aspect_ratio"]
    
    print(f"\n[{idx:02d}] {fname} | Group: {grp} | Dimensions: {dim[0]}x{dim[1]} (AR: {ar:.2f})")
    print(f"{'Variant':<22} | {'Raw OCR Output':<30} | {'Cleaned Text':<18} | {'Type':<8} | {'Valid IN':<8} | {'Conf':<6} | {'ms':<6}")
    print("-" * 110)
    for var in variants:
        vres = r["variant_results"][var]
        raw = vres["raw_ocr_text"][:28]
        clean = vres["cleaned_text"][:16]
        ptype = vres["plate_type"]
        valid = "YES" if vres["is_valid_indian"] else "NO"
        conf = f"{vres['ocr_confidence']:.3f}"
        ms = f"{vres['processing_time_ms']:.1f}"
        print(f"{var:<22} | {raw:<30} | {clean:<18} | {ptype:<8} | {valid:<8} | {conf:<6} | {ms:<6}")

print("\n" + "=" * 110)
