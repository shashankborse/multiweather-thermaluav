"""
MultiWeather-ThermalUAV Final Dataset Assembly
===============================================
Assembles the final_dataset folder from the generated
weather condition folders.

Run this after generate_dataset.py has completed.

Usage:
    python3 rebuild_final_dataset.py

Output:
    MultiWeather-ThermalUAV/final_dataset/
      {train, val, test}/
        {clear, fog, rain, snow}/
          {rgb, thermal, masks}/
            000000.png ...
"""

import os
import shutil
import random

# -----------------------------------------------
# Reproducibility
# -----------------------------------------------
random.seed(42)

# -----------------------------------------------
# Config
# -----------------------------------------------
BASE        = "MultiWeather-ThermalUAV"
OUTPUT_ROOT = os.path.join(BASE, "final_dataset")
WEATHERS    = ["clear", "fog", "rain", "snow"]
MODALITIES  = ["rgb", "thermal", "masks"]
DATASETS    = ["suburbA", "suburbB", "urbA", "urbB"]
VAL_SPLIT   = 0.5

# -----------------------------------------------
# Create directory structure
# -----------------------------------------------
if os.path.exists(OUTPUT_ROOT):
    print("Removing existing final_dataset...")
    shutil.rmtree(OUTPUT_ROOT)

for split in ["train", "val", "test"]:
    for w in WEATHERS:
        for m in MODALITIES:
            os.makedirs(
                os.path.join(OUTPUT_ROOT, split, w, m),
                exist_ok=True)

print("Directory structure created.")

# -----------------------------------------------
# Assembly loop
# -----------------------------------------------
counters = {"train": 0, "val": 0, "test": 0}
skipped  = 0

for dataset in DATASETS:
    print(f"\nProcessing {dataset}")

    for orig_split in ["train", "validation"]:
        ref_path = os.path.join(BASE, "clear", "rgb", dataset, orig_split)

        if not os.path.exists(ref_path):
            print(f"  Missing: {ref_path}, skipping.")
            continue

        files = sorted(os.listdir(ref_path))

        if orig_split == "train":
            split_map = {"train": files}
        else:
            random.shuffle(files)
            mid      = int(len(files) * VAL_SPLIT)
            split_map = {
                "val":  files[:mid],
                "test": files[mid:]
            }

        for target_split, file_list in split_map.items():
            print(f"  {orig_split} → {target_split}: {len(file_list)} samples")

            for fname in file_list:
                idx      = counters[target_split]
                new_name = f"{idx:06d}"

                # Check all modalities exist before copying
                sample_ok = True
                for w in WEATHERS:
                    for m in MODALITIES:
                        src = os.path.join(
                            BASE, w, m, dataset, orig_split, fname)
                        if not os.path.exists(src):
                            print(f"  Missing source: {src}, skipping.")
                            sample_ok = False
                            break
                    if not sample_ok:
                        break

                if not sample_ok:
                    skipped += 1
                    continue

                # Copy all modalities
                for w in WEATHERS:
                    for m in MODALITIES:
                        src = os.path.join(
                            BASE, w, m, dataset, orig_split, fname)
                        dst = os.path.join(
                            OUTPUT_ROOT, target_split, w, m,
                            new_name + ".png")
                        shutil.copy(src, dst)

                counters[target_split] += 1

# -----------------------------------------------
# Summary
# -----------------------------------------------
print("\n========== FINAL COUNTS ==========")
for k, v in counters.items():
    print(f"{k}: {v} samples")
print(f"Skipped: {skipped}")

print("\n========== VERIFYING ALIGNMENT ==========")
all_ok = True
for split in ["train", "val", "test"]:
    counts = {}
    for w in WEATHERS:
        for m in MODALITIES:
            path = os.path.join(OUTPUT_ROOT, split, w, m)
            counts[f"{w}/{m}"] = len(os.listdir(path))
    values = list(counts.values())
    if len(set(values)) != 1:
        print(f"MISMATCH in {split}: {counts}")
        all_ok = False
    else:
        print(f"OK: {split} — {values[0]} files each")

if all_ok:
    print("\nfinal_dataset is clean and ready.")
```

---

**3. `requirements.txt`**
```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.7.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
h5py>=3.8.0
huggingface_hub>=0.20.0
```

---

**4. `.gitignore`**
```
# Data
MultiWeather-ThermalUAV/
*.h5
*.pt
*.npy

# Python
__pycache__/
*.pyc
venv/
.DS_Store

# Jupyter
.ipynb_checkpoints/

# Results
experiment_results/
```

---

## GitHub Repository Structure

Once all files are in place your repository should look like this:
```
multiweather-thermaluav/
  README.md
  generate_dataset.py
  rebuild_final_dataset.py
  experiments.ipynb
  requirements.txt
  .gitignore
  experiment_results/
    results_final.csv
    results_bar.png
    per_class_heatmap.png
