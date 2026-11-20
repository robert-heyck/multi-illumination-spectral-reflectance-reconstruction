"""Process the raw photo-box images of the butterflies.

Input (data/butterflies/):
    raw/dark/         5 dark frames (no LED on)
    raw/spectralon/   5 frames of the Spectralon white standard under the W LED
    raw/b<n>/         40 frames of butterfly n: 5 duplicates under each LED, in the order
                      R, G, B, W, T, V, UV-1, UV-2 (sorted by file name)
    butterflies.csv   crop region of each butterfly
All four butterflies were imaged in one session and share the dark and Spectralon frames.

Output (output/), for each butterfly n:
    butterfly<n>_rgb_8xHxWx3.npy   normalized pixel values (LED, row, column, channel), cropped to the
                                   specimen (H x W = crop size)
    butterfly<n>_rgb_W.png         preview of the W-LED image

Usage: python process_butterflies.py [n ...]     (default: all butterflies)
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import raw_processing as rp
from butterflies import BUTTERFLY_DATA, load_butterflies, rgb_file
from common import LED_NAMES, W_LED, output_path, save_figure

RAW = BUTTERFLY_DATA / "raw"
FRAMES_PER_LED = 5

# Spectralon region (y1, y2, x1, x2) in the processed, 180-degree-rotated image
SPECTRALON_ROI = (492, 502, 746, 756)


def main():
    config = load_butterflies()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("butterflies", type=int, nargs="*", default=sorted(config))
    args = parser.parse_args()

    # steps 1-5 of raw_processing.py; the dark frame and Spectralon are shared by all butterflies
    dark, _ = rp.mean_raw(rp.nef_files(RAW / "dark"))
    white = rp.white_reference(rp.process_frames(rp.nef_files(RAW / "spectralon"), dark), SPECTRALON_ROI)
    print(f"Spectralon (W LED) normalization factors R, G, B: {np.round(white, 2)}")

    for n in args.butterflies:
        frames = rp.nef_files(RAW / f"b{n}")
        if len(frames) != FRAMES_PER_LED * len(LED_NAMES):
            raise ValueError(f"expected {FRAMES_PER_LED * len(LED_NAMES)} frames for butterfly {n}, found {len(frames)}")
        y1, y2, x1, x2 = config[n]["crop"]
        images = []
        for k in range(len(LED_NAMES)):
            group = frames[FRAMES_PER_LED * k: FRAMES_PER_LED * (k + 1)]
            images.append(rp.process_frames(group, dark)[y1:y2, x1:x2] / white)
        images = np.asarray(images)

        out = output_path(rgb_file(n))
        np.save(out, images)
        print(f"butterfly {n}: saved {out}  shape {images.shape}", flush=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(np.clip(images[W_LED], 0, 1))
        ax.set_title(f"Butterfly {n}, W LED (normalized pixel values)")
        ax.axis("off")
        save_figure(fig, output_path(f"butterfly{n}_rgb_W.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
