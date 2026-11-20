"""Reconstruct the hyperspectral reflectance images of the butterflies from their 8-LED RGB images.

Requires output/butterfly<n>_rgb_*.npy (run process_butterflies.py first).
Each pixel gets its own forward model: the LED spectra are scaled by the LED spatial profiles at that
pixel (Section 2.2.2), taken from the same crop region as the image (data/butterflies/butterflies.csv).
Regularization parameter w = 1.0.

Output (output/), for each butterfly n:
    butterfly<n>_reconstruction_w1.0_370to720nm_HxWx71.npy   reflectance (row, column, wavelength)

Usage: python reconstruct_butterflies.py [n ...] [--jobs N]     (default: all butterflies)
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import reflectance_solver as rs
from butterflies import W, load_butterflies, recon_file, rgb_file
from common import OUTPUT, WL_RECON, constraint_matrices, led_spatial_profiles, limit_blas_threads, output_path, require


def reconstruct(pixels, crop, executor):
    """Pixel-by-pixel reconstruction of one cropped 8-LED image (8, H, W, 3) -> (H, W, n_wl)."""
    n_led, H, Wd, _ = pixels.shape
    y1, y2, x1, x2 = crop
    profiles = np.asarray(led_spatial_profiles()[:, y1:y2, x1:x2])            # (8, H, W)
    recon = np.zeros((H, Wd, WL_RECON.size))
    t0 = time.time()
    for i in range(H):
        C = constraint_matrices(profiles[:, i, :].T)                          # (W, 24, 71)
        P = pixels[:, i, :, :].transpose(1, 0, 2).reshape(Wd, n_led * 3)      # (W, 24)
        recon[i] = rs.solve_batch(W, P, C, WL_RECON, executor=executor)
        if (i + 1) % 50 == 0 or i == H - 1:
            el = time.time() - t0
            print(f"  row {i + 1}/{H}  elapsed {el:.0f} s  (remaining ~{el / (i + 1) * (H - i - 1):.0f} s)",
                  flush=True)
    return recon


def main():
    config = load_butterflies()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("butterflies", type=int, nargs="*", default=sorted(config))
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="processes for the bounded (L-BFGS-B) fallback")
    args = parser.parse_args()
    limit_blas_threads()

    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for n in args.butterflies:
            pixels = np.load(require(OUTPUT / rgb_file(n), "run process_butterflies.py first"))
            print(f"butterfly {n}: reconstructing {pixels.shape[1]} x {pixels.shape[2]} pixels", flush=True)
            recon = reconstruct(pixels, config[n]["crop"], ex)
            out = output_path(recon_file(n))
            np.save(out, recon)
            print(f"butterfly {n}: saved {out}  shape {recon.shape}", flush=True)


if __name__ == "__main__":
    main()
