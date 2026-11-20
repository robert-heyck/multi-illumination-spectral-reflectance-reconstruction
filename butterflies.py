"""The butterfly specimens (Figures 5 and S4): configuration, file names and shared helpers.

Configuration, in data/butterflies/:
    butterflies.csv     species and crop region of each butterfly
    spot_regions.csv    the four wing-spot regions of each butterfly (in the cropped image)
"""
import csv

import numpy as np
from matplotlib.patches import Rectangle

from common import DATA, WL_RECON, load_table, relative_rms_error, require, sample_at

BUTTERFLY_DATA = DATA / "butterflies"
W = 1.0                                   # regularization parameter of the butterfly reconstructions
SPOTS = ["orange", "black", "yellow", "gray"]
PANEL_ORDER = ["black", "orange", "gray", "yellow"]     # left to right in Figures 5a and S4
WL_PLOT = (380, 720)                      # wavelength range of the spot curves and of their RRMSE


def _read_config(path):
    """Rows of a data/butterflies/*.csv configuration table ('#' lines are comments)."""
    with open(require(path), newline="") as f:
        lines = [line for line in f if not line.startswith("#")]
    return list(csv.DictReader(lines))


def load_butterflies():
    """
    Configuration of every butterfly.
    return:
        {n: dict(species=str, crop=(y1, y2, x1, x2), spots={spot name: (y1, y2, x1, x2)})}
        crop is in the processed (rotated) 1005 x 1509 image; spots are in the cropped image.
    """
    out = {}
    for r in _read_config(BUTTERFLY_DATA / "butterflies.csv"):
        crop = tuple(int(r[k]) for k in ("crop_y1", "crop_y2", "crop_x1", "crop_x2"))
        out[int(r["butterfly"])] = dict(species=r["species"], crop=crop, spots={})
    for r in _read_config(BUTTERFLY_DATA / "spot_regions.csv"):
        out[int(r["butterfly"])]["spots"][r["spot"]] = tuple(int(r[k]) for k in ("y1", "y2", "x1", "x2"))
    return out


def _crop_shape(n):
    y1, y2, x1, x2 = load_butterflies()[n]["crop"]
    return y2 - y1, x2 - x1


def rgb_file(n):
    """Output file of process_butterflies.py for butterfly n."""
    H, Wd = _crop_shape(n)
    return f"butterfly{n}_rgb_8x{H}x{Wd}x3.npy"


def recon_file(n):
    """Output file of reconstruct_butterflies.py for butterfly n."""
    H, Wd = _crop_shape(n)
    return f"butterfly{n}_reconstruction_w{W}_370to720nm_{H}x{Wd}x{WL_RECON.size}.npy"


def spot_stats(cube_hw_l, roi):
    """Mean and standard deviation over a region (y1, y2, x1, x2) of a (row, column, wavelength) cube."""
    y1, y2, x1, x2 = roi
    region = np.asarray(cube_hw_l[y1:y2, x1:x2, :], dtype=float)
    return region.mean(axis=(0, 1)), region.std(axis=(0, 1))


def spot_references():
    """
    Spectrometer and hyperspectral reflectance of the wing spots, keyed 'b<butterfly>_<spot>'.
    return: dict(wl_spec, spec, wl_hsi, hsi_mean, hsi_std); spec, hsi_mean and hsi_std are dicts of 1d-arrays.
    """
    wl_s, spec, names_s = load_table(BUTTERFLY_DATA / "spectrometer_spot_reflectance_350to750nm.csv")
    wl_h, hsi_m, names_m = load_table(BUTTERFLY_DATA / "hyperspectral_spot_reflectance_mean_350to750nm.csv")
    _, hsi_s, names_d = load_table(BUTTERFLY_DATA / "hyperspectral_spot_reflectance_std_350to750nm.csv")
    assert names_m == names_d
    return dict(wl_spec=wl_s, spec=dict(zip(names_s, spec)),
                wl_hsi=wl_h, hsi_mean=dict(zip(names_m, hsi_m)), hsi_std=dict(zip(names_d, hsi_s)))


def in_plot_range(wl):
    return (wl >= WL_PLOT[0]) & (wl <= WL_PLOT[1])


def spot_rrmse(recon_mean, wl_spec, spec):
    """RRMSE of a reconstructed spot spectrum (on WL_RECON) against the spectrometer, over WL_PLOT."""
    wl = WL_RECON[in_plot_range(WL_RECON)]
    return relative_rms_error(recon_mean[in_plot_range(WL_RECON)], sample_at(wl_spec, spec, wl))


def draw_spot_markers(ax, regions, scale=1.3, color="#3cc24a"):
    """
    Filled green squares on the spot regions of an image drawn with imshow. If the image is stretched to
    fill its axes (aspect='auto'), the squares are widened or narrowed so that they still print square.
    """
    x_per_y = 1.0
    if ax.get_aspect() == "auto":
        bb = ax.get_window_extent()
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        x_per_y = (abs(x1 - x0) / bb.width) / (abs(y1 - y0) / bb.height)
    for y1, y2, x1, x2 in regions.values():
        cy, cx = 0.5 * (y1 + y2) - 0.5, 0.5 * (x1 + x2) - 0.5      # imshow centres pixel i at i
        side = scale * min(y2 - y1, x2 - x1)
        w = side * x_per_y
        ax.add_patch(Rectangle((cx - w / 2, cy - side / 2), w, side, facecolor=color,
                               edgecolor="black", linewidth=0.9, zorder=10))
