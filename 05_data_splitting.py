#!/usr/bin/env python3
"""
05_data_splitting.py

Preprocessing + leakage-safe train/val/test splitting for all 4 tasks.

Pipeline per task
─────────────────
0. Binary Tumor   (Figshare)      → patient-ID-based 70/15/15 split
1. Binary Tumor   (Br35H)         → use official folder split if present,
                                    else stratified 70/15/15
2. Multiclass     (17c + 44c)     → MD5 dedup across both datasets,
                                    then stratified 70/15/15
3. MS             (sclerosis)     → stratified 70/15/15
4. CT Stroke      (aisd + stroke) → AISD test patients locked by scan ID,
                                    CT Stroke used for train/val only

All images preprocessed to grayscale 224×224 uint8 before copying.

Output layout (matches 04_models_training_checkpoint.py expectations):
  data/split/MRI_tumor_binary_figshare_norm/{train,val,test}/{class}/
  data/split/MRI_tumor_binary_norm/{train,val,test}/{class}/
  data/split/MRI_tumor_multiclass_norm/{train,val,test}/{class}/
  data/split/MRI_ms_norm/{train,val,test}/{class}/
  data/split/CT_stroke_binary_norm/{train,val,test}/{class}/
"""

import os
import glob
import hashlib
import shutil
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False
    print("Warning: imagehash not installed. Only MD5 dedup will be used.")
    print("Install with: pip install imagehash")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR  = os.environ.get("THESIS_DIR",
            os.path.expanduser("~/Documents/MSc_Thesis_Neuroimaging"))
DATA_DIR  = os.path.join(BASE_DIR, "data")
SPLIT_DIR = os.path.join(DATA_DIR, "split")

# Raw source folders inside DATA_DIR
RAW = {
    "br35h":    os.path.join(DATA_DIR, "Br35H", "Br35H-Mask-RCNN"),
    "br35h_no": os.path.join(DATA_DIR, "Br35H", "no"),
    "17c":      os.path.join(DATA_DIR, "images-17"),
    "44c":      os.path.join(DATA_DIR, "images-44c"),
    "ms":       os.path.join(DATA_DIR, "sclerosis"),
    "stroke":   os.path.join(DATA_DIR, "stroke", "Brain_Stroke_CT_Dataset"),
    "aisd":     os.path.join(DATA_DIR, "aisd"),
    "figshare": os.path.join(DATA_DIR, "figshare"),  # experiment 1: binary tumor with patient IDs
}

# AISD test scan IDs — taken from https://github.com/GriffinLiang/AISD
# Place a file at DATA_DIR/aisd/test_ids.txt (one ID per line) to override.
# Fallback: last 52 patient folders by sorted order.
AISD_TEST_IDS_FILE = os.path.join(RAW["aisd"], "test_ids.txt")

# Split ratios (only used where we do stratified splitting)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15   # TEST_RATIO = 1 - TRAIN - VAL = 0.15
SEED        = 42

# Target image size (must match training script)
IMG_SIZE = (224, 224)

# Class name normalisation map for multiclass tumor
# (raw folder names → canonical class names used in training)
TUMOR_CLASS_MAP = {
    "glioma":          "Glioma",
    "meningioma":      "Meningioma",
    "schwannoma":      "Schwannoma",
    "neurocitoma":     "Neurocitoma",
    "neurocytoma":     "Neurocitoma",
    "carcinoma":       "Carcinoma",
    "germinoma":       "Germinoma",
    "granuloma":       "Granuloma",
    "tuberculoma":     "Tuberculoma",
    "ttuberculoma":    "Tuberculoma",
    "papiloma":        "Papiloma",
    "papilloma":       "Papiloma",
    "meduloblastoma":  "Meduloblastoma",
    "medulloblastoma": "Meduloblastoma",
    "outros":          "Other",
    "other":           "Other",
    "normal":          "Normal",
    "_normal":         "Normal",
    "control":         "Normal",
    "pituitary":       "Pituitary",
    "ependymoma":      "Ependymoma",
    "ependimoma":      "Ependymoma",   # Portuguese spelling (44c)
    "ganglioglioma":   "Glioma",
    "oligodendroglioma": "Glioma",
    "astrocytoma":     "Glioma",
    "astrocitoma":     "Glioma",       # Portuguese spelling (44c)
    "glioblastoma":    "Glioma",
}

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def list_images(folder: str) -> list[str]:
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(paths)


def preprocess_image(src_path: str, dst_path: str, size: tuple = IMG_SIZE) -> bool:
    """Grayscale + resize → dst_path. Returns True on success."""
    try:
        img = Image.open(src_path)
        if img.mode != "L":
            img = img.convert("L")
        img = img.resize(size, Image.Resampling.LANCZOS)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        img.save(dst_path)
        return True
    except Exception as e:
        print(f"  ⚠ Failed {src_path}: {e}")
        return False


def md5_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def phash(path: str):
    """Perceptual hash using imagehash (returns None if unavailable)."""
    if not HAS_IMAGEHASH:
        return None
    try:
        return str(imagehash.phash(Image.open(path)))
    except Exception:
        return None


def copy_split(files: list[str], split_dir: str, class_name: str,
               desc: str = "") -> int:
    """Preprocess and copy files into split_dir/class_name/."""
    dst_class = os.path.join(split_dir, class_name)
    os.makedirs(dst_class, exist_ok=True)
    ok = 0
    for src in tqdm(files, desc=desc, leave=False):
        fname = os.path.basename(src)
        dst = os.path.join(dst_class, fname)
        # Avoid silent overwrites from filename collisions across source datasets
        if os.path.exists(dst):
            stem, ext = os.path.splitext(fname)
            h8 = md5_hash(src)[:8]
            dst = os.path.join(dst_class, f"{stem}_{h8}{ext}")
        if preprocess_image(src, dst):
            ok += 1
    return ok


def stratified_split(files: list[str], seed: int = SEED):
    """Return (train, val, test) lists with 70/15/15 split."""
    if len(files) < 5:
        return files, [], []
    train, temp = train_test_split(files, train_size=TRAIN_RATIO,
                                   random_state=seed, shuffle=True)
    val, test   = train_test_split(temp,  test_size=0.5,
                                   random_state=seed, shuffle=True)
    return train, val, test


def save_split_stats(rows: list[dict], out_dir: str, name: str):
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"{name}_split_stats.csv"), index=False)
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    summary = df.groupby("dataset")[["train", "val", "test", "total"]].sum()
    print(summary.to_string())
    print(f"{'─'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 0: BINARY TUMOR — Figshare (experiment 1)
# Strategy: patient-ID-based split using cjdata.mat PID field.
#   • 233 patients, 3064 images (tumor only — no normal class).
#   • Split patients 70/15/15 → assign all slices per patient to that split.
#   Expected structure: figshare/*.mat  OR  figshare/{1,2,3}/*.jpg
#   (the .mat files contain cjdata.image and cjdata.PID)
# ─────────────────────────────────────────────────────────────────────────────

def _load_figshare_records() -> list[tuple[str, str, str]]:
    """
    Return [(pid, class_label, file_path), ...] for the Figshare dataset.

    Supports two layouts:
      A) .mat files at figshare/*.mat  — reads cjdata.PID and cjdata.label
         (label 1=Meningioma, 2=Glioma, 3=Pituitary; saves image as PNG)
      B) JPEG images pre-exported to figshare/{tumorType}/{pid}_{n}.jpg
         where folder name is the class and filename prefix is the PID.
    """
    src = RAW["figshare"]
    if not os.path.isdir(src):
        return []

    records = []

    # ── Layout B: pre-exported images ────────────────────────────────────────
    # e.g. figshare/meningioma/1_1.jpg  (PID = first token before '_')
    for class_folder in sorted(os.listdir(src)):
        class_path = os.path.join(src, class_folder)
        if not os.path.isdir(class_path):
            continue
        imgs = list_images(class_path)
        for img in imgs:
            stem = os.path.splitext(os.path.basename(img))[0]
            pid  = stem.split("_")[0]   # filename convention: {PID}_{slice}.jpg
            records.append((pid, class_folder.lower(), img))

    if records:
        return records

    # ── Layout A: .mat files (MATLAB v7.3 / HDF5 format) ─────────────────────
    try:
        import h5py
    except ImportError:
        print("  ⚠ h5py not installed — cannot read .mat files. "
              "Install with: pip install h5py")
        return []

    label_map = {1: "meningioma", 2: "glioma", 3: "pituitary"}
    mat_files = sorted(Path(src).rglob("*.mat"))
    if not mat_files:
        print(f"  ⚠ No .mat files found in {src}")
        return []

    img_out_dir = os.path.join(src, "_exported_images")
    os.makedirs(img_out_dir, exist_ok=True)

    n_failed = 0
    for mat_path in tqdm(mat_files, desc="  reading .mat files"):
        try:
            with h5py.File(str(mat_path), "r") as f:
                cj = f["cjdata"]

                def _read(name):
                    raw = cj[name][()]
                    # Dereference HDF5 object references (MATLAB v7.3 struct fields)
                    if raw.dtype == h5py.ref_dtype or raw.dtype.kind == 'O':
                        raw = np.array(f[raw.flat[0]])
                    return raw

                pid   = str(int(_read("PID").flat[0]))
                label = int(_read("label").flat[0])
                arr   = _read("image").astype(np.float64)

            cls = label_map.get(label, f"label{label}")
            # README normalisation: uint8(255 / (max-min) * (im - min))
            mn, mx = arr.min(), arr.max()
            arr = (arr - mn) / (mx - mn + 1e-8) * 255
            # MATLAB stores arrays column-major; h5py transposes → rotate back
            img_pil = Image.fromarray(arr.T.astype(np.uint8))
            fname   = f"{mat_path.stem}.png"
            dst     = os.path.join(img_out_dir, fname)
            if not os.path.exists(dst):
                img_pil.save(dst)
            records.append((pid, cls, dst))
        except Exception as e:
            n_failed += 1
            tqdm.write(f"  ⚠ Failed {mat_path.name}: {e}")

    if n_failed:
        print(f"  {n_failed} files failed (skipped)")
    return records


def split_figshare():
    print("\n" + "="*60)
    print("TASK 0 — Binary Tumor Figshare (patient-ID split)")
    print("="*60)

    records = _load_figshare_records()
    if not records:
        print(f"  ⚠ Figshare data not found at {RAW['figshare']} — skipping.")
        print("    (Re-run after the dataset finishes downloading.)")
        return

    out = os.path.join(SPLIT_DIR, "MRI_tumor_binary_figshare_norm")

    # Group files by patient ID
    pid_files: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for pid, cls, path in records:
        pid_files[pid].append((cls, path))

    all_pids = sorted(pid_files.keys())
    print(f"  {len(all_pids)} patients, {len(records):,} images")

    # Patient-level split (stratified by majority class per patient)
    pid_labels = []
    for pid in all_pids:
        classes = [cls for cls, _ in pid_files[pid]]
        majority = max(set(classes), key=classes.count)
        pid_labels.append(majority)

    train_pids, temp_pids, train_labels, temp_labels = train_test_split(
        all_pids, pid_labels, train_size=TRAIN_RATIO,
        stratify=pid_labels, random_state=SEED)
    # Stratify val/test only if every class has ≥2 members in temp
    from collections import Counter
    temp_counts = Counter(temp_labels)
    temp_stratify = temp_labels if min(temp_counts.values()) >= 2 else None
    if temp_stratify is None:
        print(f"  ⚠ Too few patients per class in temp split for stratification "
              f"— using random val/test split. Counts: {dict(temp_counts)}")
    val_pids, test_pids = train_test_split(
        temp_pids, test_size=0.5,
        stratify=temp_stratify, random_state=SEED)

    split_map = {"train": set(train_pids), "val": set(val_pids),
                 "test":  set(test_pids)}

    rows_by_class: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for split_name, pid_set in split_map.items():
        for pid in pid_set:
            for cls, path in pid_files[pid]:
                dst_dir = os.path.join(out, split_name)
                dst = os.path.join(dst_dir, cls,
                                   f"{pid}_{os.path.basename(path)}")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                preprocess_image(path, dst)
                rows_by_class[cls][split_name] += 1

    rows = []
    for cls, counts in rows_by_class.items():
        rows.append({"dataset": "MRI_tumor_binary_figshare_norm",
                     "class": cls,
                     "train": counts.get("train", 0),
                     "val":   counts.get("val",   0),
                     "test":  counts.get("test",  0),
                     "total": sum(counts.values())})

    save_split_stats(rows, SPLIT_DIR, "MRI_tumor_binary_figshare_norm")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1: BINARY TUMOR — Br35H
# Strategy:
#   • Tumor class  → use predefined Br35H-Mask-RCNN/{TRAIN,VAL,TEST}/ split
#                    exactly as provided (500/201/100)
#   • Normal class → stratified 70/15/15 from Br35H/no/
# ─────────────────────────────────────────────────────────────────────────────

def split_br35h():
    print("\n" + "="*60)
    print("TASK 1 — Binary Tumor (Br35H)")
    print("="*60)

    out = os.path.join(SPLIT_DIR, "MRI_tumor_binary_norm")
    rows = []

    # ── Tumor: use predefined split ───────────────────────────────────────────
    mask_rcnn_dir = RAW["br35h"]
    for split_name, folder in [("train", "TRAIN"), ("val", "VAL"), ("test", "TEST")]:
        split_src = os.path.join(mask_rcnn_dir, folder)
        if not os.path.isdir(split_src):
            print(f"  ⚠ Missing {split_src} — skipping tumor {split_name}")
            continue
        files = list_images(split_src)
        n = copy_split(files, os.path.join(out, split_name), "tumor",
                       desc=f"  {split_name}/tumor")
        rows.append({"dataset": "MRI_tumor_binary_norm", "class": "tumor",
                     "train": n if split_name == "train" else 0,
                     "val":   n if split_name == "val"   else 0,
                     "test":  n if split_name == "test"  else 0,
                     "total": n})
    print(f"  Tumor: 500 train / 201 val / 100 test (predefined split)")

    # ── Normal: stratified 70/15/15 from Br35H/no/ ───────────────────────────
    no_dir = RAW["br35h_no"]
    if os.path.isdir(no_dir):
        normal_files = list_images(no_dir)
        train_f, val_f, test_f = stratified_split(normal_files)
        for split_name, split_files in [("train", train_f),
                                         ("val",   val_f),
                                         ("test",  test_f)]:
            copy_split(split_files, os.path.join(out, split_name), "normal",
                       desc=f"  {split_name}/normal")
        rows.append({"dataset": "MRI_tumor_binary_norm", "class": "normal",
                     "train": len(train_f), "val": len(val_f),
                     "test":  len(test_f),  "total": len(normal_files)})
        print(f"  Normal: {len(train_f)} train / {len(val_f)} val / "
              f"{len(test_f)} test (stratified from Br35H/no/)")
    else:
        print(f"  ⚠ Normal class folder not found at {no_dir} — skipping.")

    save_split_stats(rows, SPLIT_DIR, "MRI_tumor_binary_norm")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2: MULTICLASS TUMOR — 17c + 44c combined
# Strategy:
#   1. Collect all images from both datasets, mapping folder names to canonical
#      class names.
#   2. Deduplicate: first pass MD5 (exact), second pass perceptual hash
#      (near-duplicate within threshold). Keep one copy per duplicate group.
#   3. Stratified 70/15/15 split per class.
# ─────────────────────────────────────────────────────────────────────────────

PHASH_THRESHOLD = 8  # Hamming distance; ≤8 is considered near-duplicate

def _canonical_class(folder_name: str) -> str | None:
    """Map raw folder name to canonical class name, or None to skip."""
    low = folder_name.lower().strip()
    for key, val in TUMOR_CLASS_MAP.items():
        if key in low:
            return val
    return None


def collect_multiclass_images(dataset_dirs: list[str]) -> dict[str, list[str]]:
    """
    Walk each dataset directory, map class folders to canonical names,
    return {canonical_class: [file_paths]}.
    """
    collected: dict[str, list[str]] = defaultdict(list)
    for ds_dir in dataset_dirs:
        if not os.path.isdir(ds_dir):
            print(f"  ⚠ Not found: {ds_dir}")
            continue
        for class_folder in sorted(os.listdir(ds_dir)):
            class_path = os.path.join(ds_dir, class_folder)
            if not os.path.isdir(class_path):
                continue
            canonical = _canonical_class(class_folder)
            if canonical is None:
                print(f"  ⚠ Unmapped class folder '{class_folder}' in "
                      f"{os.path.basename(ds_dir)} — skipping")
                continue
            imgs = list_images(class_path)
            collected[canonical].extend(imgs)
    return dict(collected)


def deduplicate(files: list[str]) -> tuple[list[str], int]:
    """
    Remove exact (MD5) and near (perceptual hash) duplicates.
    Returns (deduplicated_list, n_removed).
    """
    seen_md5:   dict[str, str] = {}   # md5 → first path
    seen_phash: dict[str, str] = {}   # phash_str → first path
    unique = []
    removed = 0

    for path in tqdm(files, desc="  deduplicating", leave=False):
        # Exact duplicate check
        m = md5_hash(path)
        if m in seen_md5:
            removed += 1
            continue
        seen_md5[m] = path

        # Near-duplicate check (perceptual hash)
        if HAS_IMAGEHASH:
            ph = phash(path)
            if ph is not None:
                matched = False
                for existing_ph_str in seen_phash:
                    existing_ph = imagehash.hex_to_hash(existing_ph_str)
                    candidate_ph = imagehash.hex_to_hash(ph)
                    if (existing_ph - candidate_ph) <= PHASH_THRESHOLD:
                        matched = True
                        removed += 1
                        break
                if matched:
                    continue
                seen_phash[ph] = path

        unique.append(path)

    return unique, removed


def split_multiclass_tumor():
    print("\n" + "="*60)
    print("TASK 2 — Multiclass Tumor (17c + 44c)")
    print("="*60)

    out = os.path.join(SPLIT_DIR, "MRI_tumor_multiclass_norm")

    # 1. Collect
    print("  Collecting images from 17c and 44c...")
    class_images = collect_multiclass_images([RAW["17c"], RAW["44c"]])

    total_raw = sum(len(v) for v in class_images.values())
    print(f"  Raw total: {total_raw:,} images across {len(class_images)} classes")

    # 2. Deduplicate per class (images from same class across the two datasets
    #    are most likely to be near-duplicates)
    print("  Deduplicating...")
    dedup_log = []
    total_removed = 0
    for cls in sorted(class_images):
        files = class_images[cls]
        unique, n_removed = deduplicate(files)
        class_images[cls] = unique
        total_removed += n_removed
        dedup_log.append({"class": cls, "raw": len(files) + n_removed,
                          "removed": n_removed, "kept": len(unique)})

    dedup_df = pd.DataFrame(dedup_log)
    dedup_df.to_csv(os.path.join(SPLIT_DIR, "multiclass_dedup_log.csv"),
                    index=False)
    print(f"  Removed {total_removed:,} duplicates "
          f"({total_raw - total_removed:,} unique remain)")
    if total_removed > 0:
        print(dedup_df.to_string(index=False))

    # 3. Stratified split per class
    rows = []
    for cls in sorted(class_images):
        files = class_images[cls]
        if len(files) < 5:
            print(f"  ⚠ Skipping '{cls}' — only {len(files)} images after dedup")
            continue
        train_f, val_f, test_f = stratified_split(files)
        for split_name, split_files in [("train", train_f),
                                         ("val",   val_f),
                                         ("test",  test_f)]:
            copy_split(split_files, os.path.join(out, split_name), cls,
                       desc=f"  {split_name}/{cls}")
        rows.append({"dataset": "MRI_tumor_multiclass_norm",
                     "class": cls,
                     "train": len(train_f), "val": len(val_f),
                     "test":  len(test_f),  "total": len(files)})

    save_split_stats(rows, SPLIT_DIR, "MRI_tumor_multiclass_norm")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3: MS
# Strategy: stratified 70/15/15
# ─────────────────────────────────────────────────────────────────────────────

def _ms_canonical(folder_name: str) -> str | None:
    """
    Map a folder name (at any depth) to 'MS' or 'Control', or None to skip.
    Folder names follow the pattern: '<Class> <Plane>_crop'
    e.g. 'MS Axial_crop', 'Control Saggital_crop'
    """
    low = folder_name.lower()
    if low.startswith("control") or low.startswith("healthy") or low.startswith("normal"):
        return "Control"
    if low.startswith("ms") or low.startswith("multiple"):
        return "MS"
    return None


def split_ms():
    print("\n" + "="*60)
    print("TASK 3 — MS (sclerosis)")
    print("="*60)

    src = RAW["ms"]
    out = os.path.join(SPLIT_DIR, "MRI_ms_norm")

    # Dataset structure: sclerosis/MS/{Control Axial_crop, MS Axial_crop, ...}/
    # The class is determined by the LEAF subfolder name, not the intermediate
    # 'MS' container folder. Walk all subdirs and only assign class at the level
    # where folders have images directly inside them.
    class_files: dict[str, list[str]] = defaultdict(list)

    for root, dirs, files in os.walk(src):
        imgs = list_images(root)
        if not imgs:
            continue
        # This directory has images — determine its class from its own name
        folder_name = os.path.basename(root)
        canonical = _ms_canonical(folder_name)
        if canonical is None:
            print(f"  ⚠ Unmapped leaf folder '{folder_name}' — skipping")
            continue
        class_files[canonical].extend(imgs)

    rows = []
    for canonical in sorted(class_files):
        files = class_files[canonical]
        if not files:
            print(f"  ⚠ No images found for class '{canonical}'")
            continue
        print(f"  {canonical}: {len(files):,} images")
        train_f, val_f, test_f = stratified_split(files)
        for split_name, split_files in [("train", train_f),
                                         ("val",   val_f),
                                         ("test",  test_f)]:
            copy_split(split_files, os.path.join(out, split_name), canonical,
                       desc=f"  {split_name}/{canonical}")
        rows.append({"dataset": "MRI_ms_norm",
                     "class": canonical,
                     "train": len(train_f), "val": len(val_f),
                     "test":  len(test_f),  "total": len(files)})

    save_split_stats(rows, SPLIT_DIR, "MRI_ms_norm")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 4: CT STROKE — AISD + CT Stroke (Kaggle)
# Strategy:
#   • AISD has patient-level IDs.  Its 52 test scans (from the GitHub repo)
#     go to the test split. The remaining 345 scans go into the train/val pool.
#   • CT Stroke (Kaggle) has no patient IDs → used for train/val pool ONLY
#     (never test) to avoid unknown patient overlap with AISD test set.
#   • Pool (AISD train/val + all CT Stroke) → stratified 80/20 train/val split.
# ─────────────────────────────────────────────────────────────────────────────

def load_aisd_test_ids() -> set[str]:
    """
    Returns the set of AISD patient folder names reserved for testing.

    Priority:
      1. DATA_DIR/aisd/test_ids.txt  — one folder name per line
      2. Last 52 patient folders by sorted order (fallback)
    """
    if os.path.isfile(AISD_TEST_IDS_FILE):
        with open(AISD_TEST_IDS_FILE) as f:
            content = f.read()
        # Support both newline-separated and comma-separated IDs
        import re
        ids = {tok.strip() for tok in re.split(r"[,\n]", content) if tok.strip()}
        print(f"  Loaded {len(ids)} AISD test IDs from {AISD_TEST_IDS_FILE}")
        return ids

    # Fallback: last 52 by sorted order
    images_dir = os.path.join(RAW["aisd"], "image")
    if not os.path.isdir(images_dir):
        images_dir = RAW["aisd"]   # flat layout

    patient_folders = sorted([
        d for d in os.listdir(images_dir)
        if os.path.isdir(os.path.join(images_dir, d))
    ])
    test_ids = set(patient_folders[-52:])
    print(f"  ⚠ {AISD_TEST_IDS_FILE} not found.")
    print(f"    Falling back to last 52 patient folders as test set: "
          f"{sorted(test_ids)[:5]}...")
    print(f"    → Create {AISD_TEST_IDS_FILE} to use the official split.")
    return test_ids


def collect_aisd_slices(split: str, test_ids: set[str]) -> list[str]:
    """
    Collect all PNG slices from AISD.
    split = 'train_val' → exclude test patient folders
    split = 'test'      → only test patient folders
    """
    images_dir = os.path.join(RAW["aisd"], "image")
    if not os.path.isdir(images_dir):
        images_dir = RAW["aisd"]

    result = []
    for patient in sorted(os.listdir(images_dir)):
        patient_path = os.path.join(images_dir, patient)
        if not os.path.isdir(patient_path):
            continue
        is_test = patient in test_ids
        if split == "test"      and not is_test: continue
        if split == "train_val" and     is_test: continue

        # Each patient folder may contain slices directly or in sub-subfolders
        slices = list_images(patient_path)
        if not slices:
            for sub in os.listdir(patient_path):
                slices.extend(list_images(os.path.join(patient_path, sub)))
        result.extend(slices)
    return result


def collect_kaggle_stroke() -> dict[str, list[str]]:
    """
    Returns {class: [file_paths]} for the Kaggle CT Stroke dataset.
    Expected structure: stroke/{Bleeding,Ischemia,Normal}/PNG/*.png
    """
    src = RAW["stroke"]
    mapping = {
        "bleeding":  "stroke",
        "ischemia":  "stroke",
        "hemorrhagic": "stroke",
        "stroke":    "stroke",
        "normal":    "normal",
        "healthy":   "normal",
    }
    result: dict[str, list[str]] = defaultdict(list)
    for top in os.listdir(src):
        top_path = os.path.join(src, top)
        if not os.path.isdir(top_path):
            continue
        canonical = mapping.get(top.lower())
        if canonical is None:
            print(f"  ⚠ Unmapped CT Stroke folder '{top}' — skipping")
            continue
        # Try PNG subfolder first, then root
        png_sub = os.path.join(top_path, "PNG")
        imgs = list_images(png_sub) if os.path.isdir(png_sub) \
               else list_images(top_path)
        result[canonical].extend(imgs)
    return dict(result)


def split_ct_stroke():
    print("\n" + "="*60)
    print("TASK 4 — CT Stroke (AISD + Kaggle CT Stroke)")
    print("="*60)

    out = os.path.join(SPLIT_DIR, "CT_stroke_binary_norm")
    test_ids = load_aisd_test_ids()

    # ── AISD test slices (locked) ─────────────────────────────────────────────
    aisd_test = collect_aisd_slices("test", test_ids)
    print(f"  AISD test  slices (locked): {len(aisd_test):,}")

    # ── Pool: AISD train/val + all CT Stroke ──────────────────────────────────
    aisd_trainval = collect_aisd_slices("train_val", test_ids)
    print(f"  AISD train/val slices:      {len(aisd_trainval):,}")

    kaggle = collect_kaggle_stroke()
    kaggle_stroke = kaggle.get("stroke", [])
    kaggle_normal = kaggle.get("normal", [])
    print(f"  Kaggle stroke slices:       {len(kaggle_stroke):,}")
    print(f"  Kaggle normal slices:       {len(kaggle_normal):,}")

    # AISD is ischemic stroke only → goes to 'stroke' class
    pool_stroke = aisd_trainval + kaggle_stroke
    pool_normal = kaggle_normal

    # ── Stratified 70/15/15 split of pool into train/val/test ────────────────
    def pool_split(files):
        if not files:
            return [], [], []
        train_f, tmp = train_test_split(files, train_size=0.70,
                                        random_state=SEED, shuffle=True)
        val_f, test_f = train_test_split(tmp, test_size=0.50,
                                         random_state=SEED, shuffle=True)
        return train_f, val_f, test_f

    train_stroke, val_stroke, _ = pool_split(pool_stroke)   # stroke test = AISD locked set
    train_normal, val_normal, test_normal = pool_split(pool_normal)

    splits_map = {
        "stroke": {"train": train_stroke, "val": val_stroke,
                   "test":  aisd_test},
        "normal": {"train": train_normal, "val": val_normal,
                   "test":  test_normal},
    }

    rows = []
    for cls, split_dict in splits_map.items():
        for split_name, files in split_dict.items():
            if files:
                copy_split(files, os.path.join(out, split_name), cls,
                           desc=f"  {split_name}/{cls}")
        rows.append({"dataset": "CT_stroke_binary_norm",
                     "class": cls,
                     "train": len(split_dict["train"]),
                     "val":   len(split_dict["val"]),
                     "test":  len(split_dict["test"]),
                     "total": (len(split_dict["train"])
                               + len(split_dict["val"])
                               + len(split_dict["test"]))})

    save_split_stats(rows, SPLIT_DIR, "CT_stroke_binary_norm")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_final_summary():
    print("\n" + "="*60)
    print("FINAL SPLIT SUMMARY")
    print("="*60)
    all_rows = []
    for name in ["MRI_tumor_binary_figshare_norm", "MRI_tumor_binary_norm",
                 "MRI_tumor_multiclass_norm", "MRI_ms_norm",
                 "CT_stroke_binary_norm"]:
        csv_path = os.path.join(SPLIT_DIR, f"{name}_split_stats.csv")
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            all_rows.append(df)

    if not all_rows:
        print("  No split stats found.")
        return

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(os.path.join(SPLIT_DIR, "all_split_stats.csv"), index=False)

    summary = combined.groupby("dataset")[["train", "val", "test", "total"]].sum()
    print(summary.to_string())
    print(f"\nGrand total images: {combined['total'].sum():,}")
    print(f"  Train: {combined['train'].sum():,}")
    print(f"  Val:   {combined['val'].sum():,}")
    print(f"  Test:  {combined['test'].sum():,}")
    print(f"\nSplit stats saved to: {SPLIT_DIR}/all_split_stats.csv")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess and split neuroimaging datasets.")
    parser.add_argument("--tasks", nargs="+",
                        choices=["figshare", "binary", "multiclass", "ms",
                                 "stroke", "all"],
                        default=["all"],
                        help="Which tasks to run (default: all)")
    args = parser.parse_args()

    run_all = "all" in args.tasks
    tasks   = args.tasks

    os.makedirs(SPLIT_DIR, exist_ok=True)

    if run_all or "figshare"   in tasks: split_figshare()
    if run_all or "binary"     in tasks: split_br35h()
    if run_all or "multiclass" in tasks: split_multiclass_tumor()
    if run_all or "ms"         in tasks: split_ms()
    if run_all or "stroke"     in tasks: split_ct_stroke()

    print_final_summary()
