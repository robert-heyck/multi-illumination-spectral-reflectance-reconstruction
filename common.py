"""Paths, names and data loaders shared by the processing and figure scripts.

All paths are relative to this folder, so the scripts can be run from anywhere.
"""
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
EXTERNAL = DATA / "external"       # third-party data the user downloads (see README.md)
OUTPUT = ROOT / "output"           # everything the scripts generate

LED_NAMES = ["R", "G", "B", "W", "T", "V", "UV-1", "UV-2"]
W_LED = LED_NAMES.index("W")
CHANNEL_NAMES = ["R", "G", "B"]

# Wavelength grid used for all reconstructions from our own images (Section 2.2.4)
WL_RECON = np.arange(370, 725, 5)          # 370-720 nm, 5 nm steps, n = 71
DL_RECON = 5.0

# Box plots (Figs. 2-4): box = 25th-75th percentile, black line = median,
# whiskers = 10th and 90th percentiles. Points beyond the whiskers are not drawn.
BOX_COLOR = "#1f4e79"
BOX_WHISKERS = (10, 90)


def box_style(widths=0.6):
    """Keyword arguments for matplotlib's Axes.boxplot, shared by all box plots."""
    line = dict(color=BOX_COLOR, linewidth=1.5, alpha=0.7)
    return dict(widths=widths, whis=BOX_WHISKERS, patch_artist=True, showfliers=False,
                boxprops=dict(facecolor=BOX_COLOR, color=BOX_COLOR, alpha=0.7),
                medianprops=dict(color="black", linewidth=2),
                whiskerprops=line, capprops=line)


def limit_blas_threads():
    """
    Run BLAS/LAPACK single-threaded. The reconstruction solves many small (71 x 71) linear systems,
    which multithreaded OpenBLAS runs ~100x slower than a single thread. Also sets the environment
    variables so that worker processes started afterwards inherit the setting.
    """
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except ImportError:
        pass


def require(path, hint=""):
    """Raise a helpful error if an input file is missing."""
    path = Path(path)
    if not path.exists():
        msg = f"Required input not found: {path}"
        if hint:
            msg += f"\n  -> {hint}"
        raise FileNotFoundError(msg)
    return path


def output_path(name):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    return OUTPUT / name


def relative_rms_error(recon, actual):
    """RRMSE = RMS(recon - actual) / mean(actual), over the last axis (wavelength)."""
    return np.sqrt(np.mean((recon - actual) ** 2, axis=-1)) / np.mean(actual, axis=-1)


def style_legend(leg):
    """Legend frame shared by the figures: nearly opaque, thin dark-grey edge."""
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_edgecolor("0.3")
    leg.get_frame().set_linewidth(0.8)
    return leg


def rrmse_label(value):
    """Panel annotation for a relative RMS error, in the format of Figure 2 (a, b)."""
    return f"RRMSE = {value:.3f}"


def text_below_legend(legend, text, ha="left", pad_points=4, **kwargs):
    """Text just below a legend, aligned with its left (ha='left') or right (ha='right') edge."""
    kwargs.pop("transform", None)                  # positioned relative to the legend instead
    return legend.axes.annotate(text, xy=(0 if ha == "left" else 1, 0), xycoords=legend,
                                xytext=(0, -pad_points), textcoords="offset points", ha=ha, va="top",
                                **kwargs)


def save_figure(fig, path, retries=10, wait=1.0, **kwargs):
    """
    fig.savefig(path, **kwargs), retried for a few seconds if the file cannot be opened. On Windows
    an image viewer showing the previous version can briefly lock the file (OSError).
    """
    import time
    for attempt in range(retries):
        try:
            fig.savefig(path, **kwargs)
            return
        except OSError:
            if attempt == retries - 1:
                raise OSError(f"could not write {path}; close any program showing it and run again")
            time.sleep(wait)


def load_table(path):
    """
    Read one of the data/*.csv tables: a commented header, a column-name line,
    then rows of `wavelength, value, value, ...`.
    return:
        wl: 1d-array of wavelengths (nm).
        values: 2d-array of shape (n_columns, n_wavelengths).
        names: list of column names (without the wavelength column).
    """
    path = require(path)
    names = None
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            names = line.lstrip("#").strip().split(",")
    arr = np.loadtxt(path, delimiter=",", comments="#")
    return arr[:, 0], arr[:, 1:].T, names[1:]


def sample_at(wl_table, values, wl):
    """Pick the table columns at the requested wavelengths (exact matches only)."""
    wl_table = np.round(np.asarray(wl_table), 6)
    idx = []
    for w in np.round(np.asarray(wl, dtype=float), 6):
        hit = np.flatnonzero(wl_table == w)
        if hit.size != 1:
            raise ValueError(f"wavelength {w} nm not found exactly once in the table")
        idx.append(hit[0])
    return np.asarray(values)[..., idx]


def led_spectra(wl=None):
    """
    Spectrometer-corrected LED irradiance at the centre of the imaging area, (8, len(wl)),
    on a common scale with the V peak equal to 1. Default grid: 350-750 nm, 1 nm.
    """
    wl_t, vals, _ = load_table(DATA / "leds" / "led_spectra_350to750nm.csv")
    return vals if wl is None else sample_at(wl_t, vals, wl)


def led_spatial_profiles():
    """Relative LED intensity over the image grid, (8, 1005, 1509); equal to 1 at the box centre."""
    return np.load(require(DATA / "leds" / "led_spatial_profiles_8x1005x1509.npy"), mmap_mode="r")


def camera_sensitivity(wl=None, with_std=False):
    """
    RGB camera spectral sensitivity, (3, len(wl)); default grid 350-750 nm, 5 nm.
    Scaled to be used with led_spectra() and led_spatial_profiles() on Spectralon-normalized pixels.
    """
    wl_t, vals, _ = load_table(DATA / "camera" / "camera_sensitivity_350to750nm.csv")
    mean, std = vals[:3], vals[3:]
    if wl is not None:
        mean, std = sample_at(wl_t, mean, wl), sample_at(wl_t, std, wl)
    return (mean, std) if with_std else mean


def constraint_matrices(profile_values, wl=WL_RECON):
    """
    Forward-model matrices C of Equation (4) for photo-box images.
    input:
        profile_values: array of shape (..., 8), spatial-profile value of each LED at each pixel
                        (or averaged over a region).
        wl: wavelength grid (default 370-720 nm, 5 nm).
    return:
        C: array of shape (..., 24, len(wl)); rows ordered LED-major (R-LED R/G/B channels, G-LED ...),
           matching pixel arrays of shape (..., 8, 3) flattened to (..., 24).
    """
    wl = np.asarray(wl)
    dl = wl[1] - wl[0]
    S = led_spectra(wl)                                   # (8, n)
    c = camera_sensitivity(wl)                            # (3, n)
    base = S[:, None, :] * c[None, :, :] * dl             # (8, 3, n)
    pv = np.asarray(profile_values)
    C = pv[..., :, None, None] * base                     # (..., 8, 3, n)
    return C.reshape(pv.shape[:-1] + (24, wl.size))
