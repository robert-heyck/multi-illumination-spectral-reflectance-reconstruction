"""Process the raw photo-box images of the ColorChecker and extract the 24 patches.

Input  (data/color_checker/raw/):
    dark/        dark frames (no LED on)
    spectralon/  Spectralon white standard under the W LED
    chart/       ColorChecker, one frame under each LED, in the order R, G, B, W, T, V, UV-1, UV-2
                 (no duplicate frames to average: the patches are uniform, so the 10 x 10 pixel
                 patch average below does the noise reduction instead)
Output (output/):
    color_checker_patches.npz
        names          patch names (24,)
        pixels_mean    mean normalized pixel values over each 10 x 10 patch region, (24, 8 LEDs, 3)
        pixels_std     standard deviation over the same regions, (24, 8, 3)
        profile_mean   mean LED spatial-profile value over each region, (24, 8)
        rois           (y1, y2, x1, x2) of each region in the processed image, (24, 4)
    color_checker_rois.png   W-LED image with the patch regions marked
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

import raw_processing as rp
from common import DATA, LED_NAMES, W_LED, led_spatial_profiles, output_path, save_figure

RAW = DATA / "color_checker" / "raw"

# Spectralon region (y1, y2, x1, x2) in the processed, 180-degree-rotated image
SPECTRALON_ROI = (497, 507, 757, 767)

# Patch names in the order of the 10 x 10 sample regions below (the card is imaged
# upside down: the first row of regions is the neutral row, Black ... White).
PATCH_NAMES = [
    "Black", "Neutral 3.5", "Neutral 5", "Neutral 6.5", "Neutral 8", "White",
    "Cyan", "Magenta", "Yellow", "Red", "Green", "Blue",
    "Orange Yellow", "Yellow Green", "Purple", "Moderate Red", "Purplish Blue", "Orange",
    "Bluish Green", "Blue Flower", "Foliage", "Blue Sky", "Light Skin", "Dark Skin",
]


def patch_rois(y0=300, x0=425, pitch=130, size=10, n_rows=4, n_cols=6):
    """10 x 10 pixel sample regions, one inside each patch (row-major over the card)."""
    return np.array([(y0 + r * pitch, y0 + r * pitch + size, x0 + c * pitch, x0 + c * pitch + size)
                     for r in range(n_rows) for c in range(n_cols)])


def main():
    chart = rp.nef_files(RAW / "chart")
    if len(chart) != len(LED_NAMES):
        raise ValueError(f"expected {len(LED_NAMES)} chart frames (one per LED), found {len(chart)}")

    # Unlike the butterflies, we took a single frame under each LED and skip frame averaging
    # (step 1 in raw_processing.py). It is not needed here: every patch is spectrally uniform,
    # so pixel noise is averaged out over the 10 x 10 pixel patch regions below instead.
    images, _ = rp.process_led_series(
        [[p] for p in chart],
        dark_paths=rp.nef_files(RAW / "dark"),
        spectralon_paths=rp.nef_files(RAW / "spectralon"),
        spectralon_roi=SPECTRALON_ROI,
    )

    rois = patch_rois()
    profiles = led_spatial_profiles()
    pixels_mean = np.zeros((24, 8, 3))
    pixels_std = np.zeros((24, 8, 3))
    profile_mean = np.zeros((24, 8))
    for i, (y1, y2, x1, x2) in enumerate(rois):
        region = images[:, y1:y2, x1:x2, :]
        pixels_mean[i] = region.mean(axis=(1, 2))
        pixels_std[i] = region.std(axis=(1, 2))
        profile_mean[i] = np.asarray(profiles[:, y1:y2, x1:x2]).mean(axis=(1, 2))

    out = output_path("color_checker_patches.npz")
    np.savez(out, names=np.array(PATCH_NAMES), pixels_mean=pixels_mean, pixels_std=pixels_std,
             profile_mean=profile_mean, rois=rois)
    print(f"saved {out}")

    fig, ax = plt.subplots(figsize=(9, 6))
    w_img = images[W_LED]
    ax.imshow(np.clip(w_img / np.percentile(w_img, 99.5), 0, 1))
    for (y1, y2, x1, x2), name in zip(rois, PATCH_NAMES):
        ax.add_patch(Rectangle((x1 - 0.5, y1 - 0.5), x2 - x1, y2 - y1, fill=False, edgecolor="red", lw=1))
        ax.text(x1, y1 - 6, name, color="white", fontsize=6)
    y1, y2, x1, x2 = SPECTRALON_ROI
    ax.add_patch(Rectangle((x1 - 0.5, y1 - 0.5), x2 - x1, y2 - y1, fill=False, edgecolor="cyan", lw=1))
    ax.set_title("ColorChecker, W LED: patch regions (red) and Spectralon region position (cyan)")
    ax.axis("off")
    save_figure(fig, output_path("color_checker_rois.png"), dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
