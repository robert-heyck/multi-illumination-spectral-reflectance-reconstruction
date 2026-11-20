"""Processing of raw (NEF) photo-box images into normalized RGB images (Section 2.2.1).

Steps, identical for every image set:
  1. average the duplicate raw frames taken under the same LED
  2. subtract the averaged dark frame (no LED on), clip at zero
  3. demosaic + downscale: every 4 x 4 block of the Bayer mosaic becomes one RGB pixel,
     each channel being the median of that block's pixels of that colour (4 R, 8 G, 4 B)
  4. rotate by 180 degrees (the camera is mounted upside down relative to the sample)
  5. divide each channel by the mean pixel value of a Spectralon white standard imaged
     under the W LED (a 10 x 10 pixel region in the processed image)

No white balance or other camera processing is applied; pixel values stay linear.
"""
from pathlib import Path

import numpy as np
import rawpy

# rawpy colour labels for the Nikon D7100 Bayer pattern (raw_pattern [[0, 1], [3, 2]]):
# 0 = R, 1 = G (first green site), 2 = B, 3 = G (second green site).
# Labels belonging to each output channel (R, G, B); green uses both green sites.
CHANNEL_LABELS = ((0,), (1, 3), (2,))


def read_raw(path):
    """Raw Bayer mosaic (float64) and per-pixel colour labels of one NEF file."""
    with rawpy.imread(str(path)) as raw:
        return raw.raw_image.astype(np.float64), raw.raw_colors.copy()


def mean_raw(paths):
    """Pixel-wise mean of several raw frames, and their colour labels."""
    paths = list(paths)
    if not paths:
        raise ValueError("no raw files given")
    total, colors = read_raw(paths[0])
    for p in paths[1:]:
        img, _ = read_raw(p)
        total += img
    return total / len(paths), colors


def median_downscale(raw_image, raw_colors, d=4):
    """
    Demosaic and downscale in one step: each d x d block -> one pixel whose R, G, B values
    are the medians of the block's R, G (both green sites), B pixels.
    input:
        raw_image: 2d-array (H, W), raw Bayer mosaic.
        raw_colors: 2d-array (H, W), colour label of each raw pixel.
    return:
        3d-array of shape (H // d, W // d, 3).
    """
    H, W = raw_image.shape[0] // d, raw_image.shape[1] // d
    img = raw_image[:H * d, :W * d]
    block_colors = raw_colors[:d, :d]
    # the Bayer pattern repeats every 2 pixels, so every d x d block has the same layout
    if not np.array_equal(np.tile(block_colors, (H, W)), raw_colors[:H * d, :W * d]):
        raise ValueError("colour layout differs between blocks; use an even block size")

    blocks = img.reshape(H, d, W, d).transpose(0, 2, 1, 3).reshape(H, W, d * d)
    flat_colors = block_colors.ravel()
    out = np.empty((H, W, 3))
    for ch, labels in enumerate(CHANNEL_LABELS):
        out[..., ch] = np.median(blocks[..., np.isin(flat_colors, labels)], axis=-1)
    return out


def process_frames(paths, dark_raw=None, d=4):
    """Steps 1-4 for one set of duplicate frames: (H // d, W // d, 3), rotated by 180 degrees."""
    img, colors = mean_raw(paths)
    if dark_raw is not None:
        img = np.clip(img - dark_raw, 0.0, None)
    small = median_downscale(img, colors, d=d)
    return small[::-1, ::-1, :]


def white_reference(white_image, roi):
    """Mean R, G, B of a Spectralon region (y1, y2, x1, x2) in a processed image."""
    y1, y2, x1, x2 = roi
    return white_image[y1:y2, x1:x2, :].mean(axis=(0, 1))


def process_led_series(led_frame_groups, dark_paths, spectralon_paths, spectralon_roi, verbose=True):
    """
    Full pipeline for one sample imaged under every LED.
    input:
        led_frame_groups: list (one entry per LED, in LED order) of lists of raw file paths.
        dark_paths: raw dark frames (no LED on).
        spectralon_paths: raw frames of the Spectralon white standard under the W LED.
        spectralon_roi: (y1, y2, x1, x2) of the Spectralon in the processed (rotated) image.
    return:
        images: array (n_LEDs, H, W, 3) of normalized pixel values.
        white: per-channel normalization factors (raw counts of the Spectralon under W).
    """
    dark, _ = mean_raw(dark_paths)
    white_img = process_frames(spectralon_paths, dark)
    white = white_reference(white_img, spectralon_roi)
    if verbose:
        print(f"Spectralon (W LED) normalization factors R, G, B: {np.round(white, 2)}")

    images = []
    for k, paths in enumerate(led_frame_groups):
        images.append(process_frames(paths, dark) / white)
        if verbose:
            print(f"  processed LED {k + 1}/{len(led_frame_groups)} ({len(paths)} frame(s))")
    return np.asarray(images), white


def nef_files(folder):
    """Sorted list of the NEF files in a folder."""
    files = sorted(Path(folder).glob("*.NEF")) + sorted(Path(folder).glob("*.nef"))
    if not files:
        raise FileNotFoundError(f"no NEF files in {folder}")
    return sorted(set(files))
