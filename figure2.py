"""Figure 2: reconstruction of the Munsell reflectance spectra from synthetic CIE XYZ values.

Synthetic stimulus values are computed with Equation (1) from the published Munsell reflectances, our
six non-UV LED spectra and the CIE 1931 XYZ colour-matching functions (400-700 nm, 10 nm), normalized
by the values of a 100% reflector under the W LED. The reflectance is then reconstructed
  (c) with 1-6 illuminants added in the order W, T, V, G, R, B, at w = 1e-6, and
  (d) with all six illuminants for w = 1e-12 ... 1e3.
Box plots: box = 25th-75th percentile, whiskers = 10th and 90th percentiles.

External data (see README.md), in data/external/:
    SpectralReflectancesOf2007MunsellBookOfColorGlossy.txt   (Centore 2012)
    CIE_xyz_1931_2deg.csv, CIE_std_illum_D65.csv

Output: output/figure2.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Rectangle
import numpy as np

import color_space as cs
import reflectance_solver as rs
from common import (EXTERNAL, LED_NAMES, box_style, led_spectra, limit_blas_threads, output_path,
                    relative_rms_error, require, save_figure, style_legend)

MUNSELL_FILE = EXTERNAL / "SpectralReflectancesOf2007MunsellBookOfColorGlossy.txt"

WL = np.arange(400, 710, 10)      # 400-700 nm, 10 nm (the range of the Munsell data used)
DL = 10.0
ILLUM_ORDER = ["W", "T", "V", "G", "R", "B"]
W_ILLUM_SWEEP = 1e-6
WS = np.array([1e-12, 1e-9, 1e-6, 1e-3, 1.0, 1e3])
EXAMPLES = ("5YR7/2", "5P3/10")   # panels (a) and (b)

BOX_WIDTH = 0.45


def load_munsell():
    """
    Munsell glossy reflectances (Centore 2012): one row per colour, `name, R(380), R(390), ...` at 10 nm.
    Returns names and the reflectance at 400-700 nm; curves outside [0, 1] (fluorescent) are dropped.
    """
    path = require(MUNSELL_FILE, "download it from munsellcolourscienceforpainters.com; see README.md")
    names, refl = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            try:
                vals = np.array([float(v) for v in parts[1:]])
            except ValueError:
                continue                                  # header or blank line
            names.append(parts[0])
            refl.append(vals[2:2 + WL.size])              # columns start at 380 nm -> 400 nm is the 3rd
    names, refl = np.array(names), np.array(refl)
    keep = (refl.max(axis=1) <= 1) & (refl.min(axis=1) >= 0)
    return names[keep], refl[keep]


def add_swatch(ax, color, x_ax, y_ax, px=26):
    da = DrawingArea(px, px, 0, 0)
    da.add_artist(Rectangle((0, 0), px, px, facecolor=color, edgecolor="k", linewidth=1))
    ax.add_artist(AnnotationBbox(da, xy=(x_ax, y_ax), xycoords="axes fraction", box_alignment=(0, 1),
                                 frameon=False, pad=0.0))


def boxplot(ax, data_rows, positions):
    """Box = 25th-75th percentile, black line = median, whiskers = 10th and 90th percentiles."""
    ax.boxplot(data_rows, positions=positions, **box_style(widths=BOX_WIDTH))


def main():
    limit_blas_threads()
    names, refs = load_munsell()
    print(f"{len(names)} Munsell reflectance curves")

    xyz = cs.cie_xyz_cmf(WL).T                                          # (3, n)
    leds = led_spectra(WL)[[LED_NAMES.index(c) for c in ILLUM_ORDER]]   # (6, n)

    # Equation (1), normalized by a 100% reflector under the W LED (the first illuminant)
    white_W = (xyz * leds[0][None, :]).sum(axis=1) * DL                 # (3,)
    C = (leds[:, None, :] * xyz[None, :, :] * DL / white_W[None, :, None]).reshape(18, WL.size)
    pixels = refs @ C.T                                                 # (N, 18)

    # (d) all six illuminants, varying w
    sols_w = np.array([rs.solve_batch(w, pixels, C, WL) for w in WS])                  # (6, N, n)
    rrmse_w = relative_rms_error(sols_w, refs[None])                                   # (6, N)
    # (a-c) increasing number of illuminants at fixed w
    sols_k = np.array([rs.solve_batch(W_ILLUM_SWEEP, pixels[:, :3 * k], C[:3 * k], WL) for k in range(1, 7)])
    rrmse_k = relative_rms_error(sols_k, refs[None])

    for k in range(6):
        print(f"  {k + 1} illuminant(s): median RRMSE {np.median(rrmse_k[k]):.4f}")
    for w, r in zip(WS, rrmse_w):
        print(f"  w = {w:g}: median RRMSE {np.median(r):.4f}")

    # ------------------------------------------------------------------ figure
    d65 = cs.cie_d65(WL)
    fig, ax = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
    recon_colors = ["tab:blue", "tab:orange", "tab:green"]
    panel_cfg = [(ax[0, 0], EXAMPLES[0], (0.33, 0.57), None), (ax[0, 1], EXAMPLES[1], (-0.03, 0.56),
                                                                [0.03, 0.15, 0.27, 0.39, 0.51])]
    for a, name, ylim, yticks in panel_cfg:
        i = int(np.flatnonzero(names == name)[0])
        a.plot(WL, refs[i], color="k", lw=3, label="true", linestyle="--", zorder=2)
        for j in range(3):
            a.plot(WL, sols_k[2 * j, i], lw=3, color=recon_colors[j], label=f"{2 * j + 2} illum.", zorder=1)
        swatch = cs.render_srgb(refs[i], d65, cs.cie_xyz_cmf(WL)) / 255.0
        add_swatch(a, swatch, x_ax=0.205, y_ax=0.975)
        a.text(0.02, 0.795, f"Munsell\n{name}", transform=a.transAxes, fontsize=12, va="bottom")
        a.text(0.02, 0.70, f"RRMSE = {rrmse_k[5, i]:.3f}", transform=a.transAxes, fontsize=10, va="bottom")
        a.set_xlabel("Wavelength (nm)", fontsize=12)
        a.tick_params(labelsize=12)
        a.set_xticks([400, 500, 600, 700])
        if yticks:
            a.set_yticks(yticks)
        a.set_ylim(*ylim)
        a.grid(True, which="major", alpha=0.25)
    ax[0, 0].set_ylabel("Reflectance", fontsize=13)
    style_legend(ax[0, 0].legend(fontsize=12, frameon=True, loc="lower right"))

    # (c) RRMSE vs number of illuminants
    a = ax[1, 0]
    pos = np.arange(1, 7)
    boxplot(a, list(rrmse_k), pos)
    a.text(0.78, 0.95, r"$w=10^{-6}$", transform=a.transAxes, fontsize=12, va="top", ha="left",
           bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.3", alpha=0.85))
    a.set_xticks(pos); a.set_xticklabels([str(k) for k in pos])
    a.set_xlabel("Number of illuminants", fontsize=12)
    a.set_ylabel("Relative RMS error", fontsize=12)
    a.tick_params(axis="both", which="major", labelsize=11)
    a.grid(True, alpha=0.3)
    a.set_yscale("log")

    # (d) RRMSE vs w
    a = ax[1, 1]
    pos = np.arange(1, WS.size + 1)
    boxplot(a, list(rrmse_w), pos)
    a.text(0.03, 0.95, "num. illum. = 6", transform=a.transAxes, fontsize=12, va="top", ha="left",
           bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.3", alpha=0.85))
    a.set_xlabel(r"Regularization parameter $w$", fontsize=12)
    a.set_xticks(pos)
    a.set_xticklabels([rf"$10^{{{int(np.log10(w))}}}$" for w in WS], fontsize=11)
    a.tick_params(axis="y", which="major", labelsize=11)
    a.grid(True, alpha=0.3)
    a.set_yscale("log")

    for axis, label in zip(ax.flat, ["(a)", "(b)", "(c)", "(d)"]):
        axis.text(-0.14, 1.02, label, transform=axis.transAxes, fontsize=14, fontweight="bold",
                  ha="left", va="bottom", clip_on=False)
    fig.align_xlabels(ax[1, :])

    out = output_path("figure2.png")
    save_figure(fig, out, dpi=300, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
