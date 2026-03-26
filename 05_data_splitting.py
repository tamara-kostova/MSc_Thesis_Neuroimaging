#!/usr/bin/env python3
"""
05_data_splitting.py

Preprocessing + leakage-safe train/val/test splitting for all 4 tasks.

Pipeline per task
─────────────────
1. Binary Tumor   (Br35H)         → use official folder split if present,
                                    else stratified 70/15/15
2. Multiclass     (17c + 44c)     → MD5 dedup across both datasets,
                                    then stratified 70/15/15
3. MS             (sclerosis)     → stratified 70/15/15
4. CT Stroke      (aisd + stroke) → AISD test patients locked by scan ID,
                                    CT Stroke used for train/val only

All images preprocessed to grayscale 224×224 uint8 before copying.

Output layout (matches 04_models_training_checkpoint.py expectations):
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
    "br35h":    os.path.join(DATA_DIR, "Br35H"),
    "17c":      os.path.join(DATA_DIR, "images-17c"),
    "44c":      os.path.join(DATA_DIR, "images-44c"),
    "ms":       os.path.join(DATA_DIR, "sclerosis"),
    "stroke":   os.path.join(DATA_DIR, "stroke"),
    "aisd":     os.path.join(DATA_DIR, "aisd"),
    "figshare": os.path.join(DATA_DIR, "figshare"),  # not used in final 4 tasks
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
    "tuberculoma":     "Ttuberculoma",
    "ttuberculoma":    "Ttuberculoma",
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
    "ganglioglioma":   "Glioma",
    "oligodendroglioma": "Glioma",
    "astrocytoma":     "Glioma",
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
# TASK 1: BINARY TUMOR — Br35H
# Strategy: use official folder split (train/val/test) if present;
#           fall back to stratified 70/15/15
# ─────────────────────────────────────────────────────────────────────────────

def split_br35h():
    print("\n" + "="*60)
    print("TASK 1 — Binary Tumor (Br35H)")
    print("="*60)

    src   = RAW["br35h"]
    out   = os.path.join(SPLIT_DIR, "MRI_tumor_binary_norm")

    # ── Detect folder layout ──────────────────────────────────────────────────
    # Layout A (official split): src/{train,val,test}/{yes,no}/
    # Layout B (flat):           src/{yes,no}/  or  src/{0,1}/  or  src/{tumor,normal}/
    has_official = all(
        os.path.isdir(os.path.join(src, s))
        for s in ["train", "val", "test"]
    )

    CLASS_REMAP = {
        "yes": "tumor", "1": "tumor", "tumor": "tumor",
        "no":  "normal", "0": "normal", "normal": "normal",
    }

    rows = []

    if has_official:
        print("  Using official train/val/test folder split.")
        for split_name in ["train", "val", "test"]:
            split_src = os.path.join(src, split_name)
            split_dst = os.path.join(out, split_name)
            for raw_cls in os.listdir(split_src):
                canonical = CLASS_REMAP.get(raw_cls.lower(), raw_cls.lower())
                files = list_images(os.path.join(split_src, raw_cls))
                n = copy_split(files, split_dst, canonical,
                               desc=f"  {split_name}/{canonical}")
                rows.append({"dataset": "MRI_tumor_binary_norm",
                             "class": canonical, "split": split_name,
                             "train": n if split_name == "train" else 0,
                             "val":   n if split_name == "val"   else 0,
                             "test":  n if split_name == "test"  else 0,
                             "total": n})
    else:
        print("  No official split found — using stratified 70/15/15.")
        # Collect all images per class
        class_dirs = [
            d for d in os.listdir(src)
            if os.path.isdir(os.path.join(src, d)) and d.lower() in CLASS_REMAP
        ]
        if not class_dirs:
            # Try direct image files at root level grouped by nothing — skip
            print(f"  ⚠ Cannot determine class structure in {src}. Skipping.")
            return

        for raw_cls in class_dirs:
            canonical = CLASS_REMAP[raw_cls.lower()]
            files = list_images(os.path.join(src, raw_cls))
            train_f, val_f, test_f = stratified_split(files)
            for split_name, split_files in [("train", train_f),
                                             ("val",   val_f),
                                             ("test",  test_f)]:
                copy_split(split_files, os.path.join(out, split_name),
                           canonical, desc=f"  {split_name}/{canonical}")
            rows.append({"dataset": "MRI_tumor_binary_norm",
                         "class": canonical,
                         "train": len(train_f), "val": len(val_f),
                         "test":  len(test_f),
                         "total": len(files)})

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

MS_CLASS_MAP = {
    "ms": "MS", "multiple_sclerosis": "MS", "multiple sclerosis": "MS",
    "control": "Control", "healthy": "Control", "normal": "Control",
}

def split_ms():
    print("\n" + "="*60)
    print("TASK 3 — MS (sclerosis)")
    print("="*60)

    src = RAW["ms"]
    out = os.path.join(SPLIT_DIR, "MRI_ms_norm")

    rows = []
    for raw_cls in sorted(os.listdir(src)):
        class_path = os.path.join(src, raw_cls)
        if not os.path.isdir(class_path):
            continue
        canonical = MS_CLASS_MAP.get(raw_cls.lower(), raw_cls)
        files = list_images(class_path)
        if not files:
            # One level deeper (e.g. sclerosis/MS/axial/*.png)
            for sub in os.listdir(class_path):
                files.extend(list_images(os.path.join(class_path, sub)))
        if not files:
            print(f"  ⚠ No images found in {class_path}")
            continue
        print(f"  {raw_cls} → {canonical}: {len(files):,} images")
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
            ids = {line.strip() for line in f if line.strip()}
        print(f"  Loaded {len(ids)} AISD test IDs from {AISD_TEST_IDS_FILE}")
        return ids

    # Fallback: last 52 by sorted order
    images_dir = os.path.join(RAW["aisd"], "images")
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
    images_dir = os.path.join(RAW["aisd"], "images")
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

    # ── Stratified 80/20 split of pool into train/val ────────────────────────
    def pool_split(files):
        if not files:
            return [], []
        train_f, val_f = train_test_split(files, train_size=0.80,
                                          random_state=SEED, shuffle=True)
        return train_f, val_f

    train_stroke, val_stroke = pool_split(pool_stroke)
    train_normal, val_normal = pool_split(pool_normal)

    splits_map = {
        "stroke": {"train": train_stroke, "val": val_stroke,
                   "test":  aisd_test},
        "normal": {"train": train_normal, "val": val_normal,
                   "test":  []},           # no locked normal test slices
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
    for name in ["MRI_tumor_binary_norm", "MRI_tumor_multiclass_norm",
                 "MRI_ms_norm", "CT_stroke_binary_norm"]:
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
                        choices=["binary", "multiclass", "ms", "stroke", "all"],
                        default=["all"],
                        help="Which tasks to run (default: all)")
    args = parser.parse_args()

    run_all = "all" in args.tasks
    tasks   = args.tasks

    os.makedirs(SPLIT_DIR, exist_ok=True)

    if run_all or "binary"     in tasks: split_br35h()
    if run_all or "multiclass" in tasks: split_multiclass_tumor()
    if run_all or "ms"         in tasks: split_ms()
    if run_all or "stroke"     in tasks: split_ct_stroke()

    print_final_summary()
