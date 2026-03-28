"""
MultiWeather-ThermalUAV Dataset Generation Pipeline
====================================================
Generates synthetic thermal imagery and weather augmentations
from the Safe-UAV dataset (.h5 files).

Requirements:
    pip install opencv-python numpy h5py

Usage:
    Place suburbA.h5, suburbB.h5, urbA.h5, urbB.h5 in the
    same directory as this script, then run:
        python3 generate_dataset.py

Output:
    MultiWeather-ThermalUAV/
      {clear, fog, rain, snow}/
        {rgb, thermal, depth, masks}/
          {suburbA, suburbB, urbA, urbB}/
            {train, validation}/
              000000.png ...
"""

import os
import h5py
import numpy as np
import cv2

# -----------------------------------------------
# Reproducibility
# -----------------------------------------------
np.random.seed(42)

# -----------------------------------------------
# Config
# -----------------------------------------------
H5_FILES    = ["suburbA.h5", "suburbB.h5", "urbA.h5", "urbB.h5"]
BASE_OUTPUT = "MultiWeather-ThermalUAV"
WEATHERS    = ["clear", "fog", "rain", "snow"]
MODALITIES  = ["rgb", "thermal", "depth", "masks"]


# -----------------------------------------------
# Directory setup
# -----------------------------------------------
def setup_dirs():
    for w in WEATHERS:
        for m in MODALITIES:
            for f in H5_FILES:
                base = f.replace(".h5", "")
                for split in ["train", "validation"]:
                    os.makedirs(
                        os.path.join(BASE_OUTPUT, w, m, base, split),
                        exist_ok=True)

setup_dirs()


# -----------------------------------------------
# Path helper
# -----------------------------------------------
def get_path(weather, modality, dataset, split, idx):
    return os.path.join(
        BASE_OUTPUT, weather, modality, dataset, split, f"{idx:06d}")


# -----------------------------------------------
# Thermal generation
# -----------------------------------------------
def generate_thermal(rgb, depth, mask):
    """
    Generates a synthetic 16-bit thermal image from RGB,
    depth, and segmentation mask using a physically motivated
    luminance-depth model.
    """
    luminance = np.dot(
        rgb[..., :3],
        [0.299, 0.587, 0.114]).astype(np.float32) / 255.0

    depth     = depth.astype(np.float32)
    dmin, dmax = depth.min(), depth.max()
    depth_norm = (depth - dmin) / (dmax - dmin + 1e-6)

    depth_effect = 0.8 + 0.2 * (1 - depth_norm)
    thermal      = luminance * depth_effect

    fg       = (mask > 0).astype(np.float32)
    fg       = cv2.GaussianBlur(fg, (15, 15), 3)
    thermal *= (1 + 0.2 * fg)

    thermal  = cv2.GaussianBlur(thermal, (9, 9), 2)

    edges    = np.abs(cv2.Laplacian(luminance, cv2.CV_32F))
    edges    = cv2.GaussianBlur(edges, (9, 9), 2)
    thermal *= (1 - 0.15 * edges)

    thermal /= thermal.max() + 1e-6
    thermal  = np.power(thermal, 0.85)

    noise    = cv2.GaussianBlur(
        np.random.normal(0, 0.006, thermal.shape).astype(np.float32),
        (15, 15), 4)

    thermal  = np.clip(thermal + noise, 0, 1)
    return (thermal * 65535).astype(np.uint16)


# -----------------------------------------------
# Fog augmentation
# -----------------------------------------------
def apply_fog_rgb(rgb, depth_norm):
    """Beer-Lambert depth-based atmospheric scattering."""
    t   = np.exp(-2.5 * 1.2 * depth_norm)
    fog = rgb.astype(np.float32) * t[..., None]
    fog += 220 * (1 - t[..., None])
    fog  = fog * 0.85 + 20
    return np.clip(fog, 0, 255).astype(np.uint8)


def apply_fog_thermal(th, depth_norm):
    """Thermal fog with reduced scattering coefficient."""
    th  = th.astype(np.float32) / 65535.0
    t   = np.exp(-1.8 * 1.2 * depth_norm)
    fog = th * t + 0.25 * (1 - t)
    fog = np.power(fog, 0.9)
    fog = fog * 0.75 + 0.1
    return (np.clip(fog, 0, 1) * 65535).astype(np.uint16)


def apply_fog_depth(depth_norm):
    t   = np.exp(-2.0 * 1.2 * depth_norm)
    fog = depth_norm * t
    return np.power(fog, 0.5)


# -----------------------------------------------
# Rain augmentation
# -----------------------------------------------
def add_rain_streaks(shape):
    """Directional rain streak simulation at ~75 degrees."""
    h, w  = shape
    rain  = np.zeros((h, w), np.float32)
    for _ in range(int(h * w * 0.0005)):
        x, y   = np.random.randint(0, w), np.random.randint(0, h)
        length = np.random.randint(20, 40)
        angle  = np.deg2rad(75 + np.random.randint(-5, 5))
        x2     = int(x + length * np.cos(angle))
        y2     = int(y + length * np.sin(angle))
        cv2.line(rain, (x, y), (x2, y2), 1.0, 1)
    return cv2.GaussianBlur(rain, (7, 7), 2)


def apply_rain_rgb(rgb):
    rain = add_rain_streaks(rgb.shape[:2])
    img  = rgb.astype(np.float32) * 0.75 + rain[..., None] * 255
    k    = np.zeros((9, 9))
    k[4, :] = 1 / 9
    img  = cv2.filter2D(img, -1, k)
    return np.clip(img, 0, 255).astype(np.uint8)


def apply_rain_thermal(th):
    th    = th.astype(np.float32) / 65535.0
    th    = th * 0.8 + 0.1
    noise = cv2.GaussianBlur(
        np.random.normal(0, 0.02, th.shape).astype(np.float32),
        (15, 15), 4)
    return (np.clip(th + noise, 0, 1) * 65535).astype(np.uint16)


def apply_rain_depth(depth_norm):
    noise = cv2.GaussianBlur(
        np.random.normal(0, 0.02, depth_norm.shape).astype(np.float32),
        (15, 15), 4)
    return np.clip(depth_norm + noise, 0, 1)


# -----------------------------------------------
# Snow augmentation
# -----------------------------------------------
def generate_snow(depth_norm):
    """Depth-aware snow particle distribution."""
    h, w = depth_norm.shape
    snow = np.zeros((h, w), np.float32)
    for _ in range(int(h * w * 0.003)):
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        if np.random.rand() < depth_norm[y, x]:
            snow[y, x] += 1
    return cv2.GaussianBlur(snow, (7, 7), 2)


def apply_snow_rgb(rgb, depth_norm):
    snow = generate_snow(depth_norm)
    img  = rgb.astype(np.float32) * 1.04 + 8
    img += snow[..., None] * 200
    img  = cv2.GaussianBlur(img, (3, 3), 0.8)
    return np.clip(img, 0, 255).astype(np.uint8)


def apply_snow_thermal(th):
    th    = th.astype(np.float32) / 65535.0
    noise = cv2.GaussianBlur(
        np.random.normal(0, 0.005, th.shape).astype(np.float32),
        (31, 31), 10)
    return (np.clip(th + noise, 0, 1) * 65535).astype(np.uint16)


def apply_snow_depth(depth_norm):
    noise = cv2.GaussianBlur(
        np.random.normal(0, 0.01, depth_norm.shape).astype(np.float32),
        (25, 25), 8)
    return np.clip(depth_norm + noise, 0, 1)


# -----------------------------------------------
# Main generation loop
# -----------------------------------------------
total = 0

for file in H5_FILES:
    base = file.replace(".h5", "")
    print(f"\nProcessing {base}")

    with h5py.File(file, "r") as f:
        for split in ["train", "validation"]:
            grp = f[split]
            n   = grp["rgb"].shape[0]

            for i in range(n):
                if i % 200 == 0:
                    print(f"  {split}: {i}/{n}")

                rgb        = grp["rgb"][i]
                depth      = grp["depth"][i]
                mask       = grp["hvn_gt_p1"][i]
                depth_norm = (depth - depth.min()) / (
                    depth.max() - depth.min() + 1e-6)

                # Generate thermal from clear scene
                th = generate_thermal(rgb, depth, mask)

                # Clear
                cv2.imwrite(
                    get_path("clear", "rgb", base, split, i) + ".png", rgb)
                cv2.imwrite(
                    get_path("clear", "thermal", base, split, i) + ".png", th)
                np.save(
                    get_path("clear", "depth", base, split, i) + ".npy",
                    depth_norm)
                cv2.imwrite(
                    get_path("clear", "masks", base, split, i) + ".png", mask)

                # Fog
                cv2.imwrite(
                    get_path("fog", "rgb", base, split, i) + ".png",
                    apply_fog_rgb(rgb, depth_norm))
                cv2.imwrite(
                    get_path("fog", "thermal", base, split, i) + ".png",
                    apply_fog_thermal(th, depth_norm))
                np.save(
                    get_path("fog", "depth", base, split, i) + ".npy",
                    apply_fog_depth(depth_norm))
                cv2.imwrite(
                    get_path("fog", "masks", base, split, i) + ".png", mask)

                # Rain
                cv2.imwrite(
                    get_path("rain", "rgb", base, split, i) + ".png",
                    apply_rain_rgb(rgb))
                cv2.imwrite(
                    get_path("rain", "thermal", base, split, i) + ".png",
                    apply_rain_thermal(th))
                np.save(
                    get_path("rain", "depth", base, split, i) + ".npy",
                    apply_rain_depth(depth_norm))
                cv2.imwrite(
                    get_path("rain", "masks", base, split, i) + ".png", mask)

                # Snow
                cv2.imwrite(
                    get_path("snow", "rgb", base, split, i) + ".png",
                    apply_snow_rgb(rgb, depth_norm))
                cv2.imwrite(
                    get_path("snow", "thermal", base, split, i) + ".png",
                    apply_snow_thermal(th))
                np.save(
                    get_path("snow", "depth", base, split, i) + ".npy",
                    apply_snow_depth(depth_norm))
                cv2.imwrite(
                    get_path("snow", "masks", base, split, i) + ".png", mask)

                total += 1

print(f"\nDone. Total samples processed: {total}")
