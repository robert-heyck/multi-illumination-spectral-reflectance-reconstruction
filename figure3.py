"""Figure 3 and Figure S2: reconstruction of the 24 ColorChecker patches from photo-box images.

Requires output/color_checker_patches.npz (run process_color_checker.py first).
Each patch is reconstructed (370-720 nm, 5 nm) from its mean pixel values under all 8 LEDs, with the
LED spectra scaled by the spatial profile averaged over the patch region. Reference reflectances:
hyperspectral imaging (HSI, used for all error metrics) and the spectrometer probe.

(a) "Orange Yellow" with the per-patch optimal w (Section 4.1, Figure S5), with the RRMSE of the
    reconstruction vs HSI (370-720 nm, as in (c))
(b) the same patch for w = 0.01, 1, 100
(c) RRMSE vs HSI for all 24 patches as a function of w
(d) CIEDE2000 between reconstruction (w = 0.1) and HSI under the R, G, B, W, T LEDs and D65;
    red lines: median CIEDE2000 between spectrometer and HSI
Figure S2: all 24 patches, w = 0.1, with the RRMSE of each reconstruction vs HSI (370-720 nm, as in (c)).
HSI error bars: standard deviation over the patch's 10 x 10 pixel region. Box plots: box = 25th-75th
percentile, whiskers = 10th and 90th percentiles.

External data: data/external/CIE_xyz_1931_2deg.csv and CIE_std_illum_D65.csv (see README.md).
Output: output/figure3.png, output/figureS2.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
import numpy as np

import color_space as cs
import reflectance_solver as rs
from common import (DATA, DL_RECON, LED_NAMES, OUTPUT, WL_RECON, box_style, constraint_matrices,
                    led_spectra, limit_blas_threads, load_table, output_path, relative_rms_error, require,
                    rrmse_label, sample_at, save_figure, style_legend, text_below_legend)

WS = np.array([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])
W_CONST = 0.1                      # panel (d) and Figure S2
W_CASES = [(1e-2, "tab:blue", "0.01"), (1.0, "tab:orange", "1.00"), (100.0, "tab:green", "100")]
SELECTED = "Orange Yellow"         # panels (a) and (b)
D_ILLUMS = ["R", "G", "B", "W", "T", "D65"]
JND = 1.0                          # CIEDE2000 just-noticeable difference

# sRGB swatches of the ColorChecker Classic (nominal values)
CC_HEX = {
    "Dark Skin": "#735244", "Light Skin": "#c29682", "Blue Sky": "#627a9d", "Foliage": "#576c43",
    "Blue Flower": "#8580b1", "Bluish Green": "#67bdaa", "Orange": "#d67e2c", "Purplish Blue": "#505ba6",
    "Moderate Red": "#c15a63", "Purple": "#5e3c6c", "Yellow Green": "#9dbc40", "Orange Yellow": "#e0a32e",
    "Blue": "#383d96", "Green": "#469449", "Red": "#af363c", "Yellow": "#e7c71f", "Magenta": "#bb5695",
    "Cyan": "#0885a1", "White": "#f3f3f2", "Neutral 8": "#c8c8c8", "Neutral 6.5": "#a0a0a0",
    "Neutral 5": "#7a7a7a", "Neutral 3.5": "#555555", "Black": "#343434",
}
GRID_ROWS = [                      # layout of Figure S2
    ["Bluish Green", "Orange Yellow", "Cyan", "Black"],
    ["Blue Flower", "Yellow Green", "Magenta", "Neutral 3.5"],
    ["Foliage", "Purple", "Yellow", "Neutral 5"],
    ["Blue Sky", "Moderate Red", "Red", "Neutral 6.5"],
    ["Light Skin", "Purplish Blue", "Green", "Neutral 8"],
    ["Dark Skin", "Orange", "Blue", "White"],
]

HSI_COLOR, SPEC_COLOR, RECON_COLOR = "black", "tab:blue", "tab:orange"
LFS, TFS, LGF, PLF, LW = 13, 11, 10.4, 14, 3.0


def add_swatch(ax, color, xy, box_alignment, px=40):
    da = DrawingArea(px, px, 0, 0)
    da.add_artist(Rectangle((0, 0), px, px, facecolor=color, edgecolor="k", linewidth=1))
    ax.add_artist(AnnotationBbox(da, xy=xy, xycoords="axes fraction", box_alignment=box_alignment,
                                 frameon=False, pad=0.0))


def plot_hsi(ax, mean, std, label="Hyperspectral", every=4):
    """HSI mean (markers) with error bars on every `every`-th sample, drawn on top. The error bars are
    the standard deviation over the patch's 10 x 10 pixel region."""
    x = WL_RECON
    sel = np.arange(0, x.size, every)
    err = std
    ax.plot(x, mean, "o", markersize=3.4, color=HSI_COLOR, zorder=5)
    return ax.errorbar(x[sel], mean[sel], yerr=err[sel], fmt="o", color=HSI_COLOR, ecolor=HSI_COLOR,
                       alpha=0.75, elinewidth=2.0, markersize=3.4, capsize=3.3, label=label, zorder=4)


def main():
    limit_blas_threads()
    patches = np.load(require(OUTPUT / "color_checker_patches.npz", "run process_color_checker.py first"))
    names = list(patches["names"])
    n_p = len(names)

    # references, on the reconstruction grid
    wl_h, hsi_mean, hsi_names = load_table(DATA / "color_checker" / "hyperspectral_reflectance_mean_350to750nm.csv")
    _, hsi_std, _ = load_table(DATA / "color_checker" / "hyperspectral_reflectance_std_350to750nm.csv")
    wl_s, spec, spec_names = load_table(DATA / "color_checker" / "spectrometer_reflectance_350to750nm.csv")
    assert hsi_names == names and spec_names == names
    hsi_mean, hsi_std = sample_at(wl_h, hsi_mean, WL_RECON), sample_at(wl_h, hsi_std, WL_RECON)
    spec = sample_at(wl_s, spec, WL_RECON)

    # ------------------------------------------------------------------ reconstruction
    C_all = constraint_matrices(patches["profile_mean"])            # (24, 24, 71)
    P_all = patches["pixels_mean"].reshape(n_p, 24)                  # (24, 24)
    recon_w = np.zeros((n_p, WS.size, WL_RECON.size))
    recon_opt = np.zeros((n_p, WL_RECON.size))
    opt_w = np.zeros(n_p)
    for i in range(n_p):
        opt_w[i] = rs.optimal_weight(P_all[i], C_all[i], WL_RECON)
        recon_opt[i] = rs.solve(opt_w[i], P_all[i], C_all[i], WL_RECON)
        for j, w in enumerate(WS):
            recon_w[i, j] = rs.solve(w, P_all[i], C_all[i], WL_RECON)
    sel = names.index(SELECTED)
    print(f"optimal w: median {np.median(opt_w):.3f}; {SELECTED}: {opt_w[sel]:.2f}")

    # ------------------------------------------------------------------ error metrics
    rrmse = relative_rms_error(recon_w, hsi_mean[:, None, :])        # (24, 6)
    j_const = int(np.argmin(np.abs(WS - W_CONST)))
    print(f"median RRMSE vs HSI at w = {W_CONST}: {np.median(rrmse[:, j_const]):.4f}")

    xyz = cs.cie_xyz_cmf(WL_RECON)
    d65 = cs.cie_d65(WL_RECON)
    leds = led_spectra(WL_RECON)
    de_recon = np.zeros((n_p, len(D_ILLUMS)))
    de_spec = np.zeros((n_p, len(D_ILLUMS)))
    for i in range(n_p):
        for j, name in enumerate(D_ILLUMS):
            # LED irradiance at this patch (spectrum x spatial profile), or D65
            illum = d65 if name == "D65" else patches["profile_mean"][i, LED_NAMES.index(name)] * leds[LED_NAMES.index(name)]
            de_recon[i, j] = cs.color_difference(recon_w[i, j_const], hsi_mean[i], illum, xyz, DL_RECON)
            de_spec[i, j] = cs.color_difference(spec[i], hsi_mean[i], illum, xyz, DL_RECON)
    for j, name in enumerate(D_ILLUMS):
        print(f"  {name:>4s}: median dE00 recon-HSI {np.median(de_recon[:, j]):.2f}   "
              f"spectrometer-HSI {np.median(de_spec[:, j]):.2f}")

    # ------------------------------------------------------------------ Figure 3
    fig, axs = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)

    def style_spectral(ax, panel):
        ax.grid(True, alpha=0.2)
        ax.set_ylim(-0.05, 0.85); ax.set_xlim(360, 730)
        ax.set_xlabel("Wavelength (nm)", fontsize=LFS); ax.set_ylabel("Reflectance", fontsize=LFS)
        ax.set_xticks([400, 500, 600, 700]); ax.tick_params(labelsize=TFS)
        ax.text(-0.12, 1.04, panel, transform=ax.transAxes, fontsize=PLF, fontweight="bold", va="bottom")

    # (a)
    ax = axs[0, 0]
    h_rec, = ax.plot(WL_RECON, recon_opt[sel], color=RECON_COLOR, linewidth=LW, solid_capstyle="butt",
                     zorder=3, label=r"Recon. (opt. $w$)")
    h_spec, = ax.plot(WL_RECON, spec[sel], "--", color=SPEC_COLOR, linewidth=LW, zorder=3, label="Spectrometer")
    h_hsi = plot_hsi(ax, hsi_mean[sel], hsi_std[sel])
    add_swatch(ax, CC_HEX[SELECTED], (0.82, 0.415), (0, 1), px=42)
    ax.text(0.70, 0.025, SELECTED, transform=ax.transAxes, ha="left", va="bottom", fontsize=11,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.2), zorder=20)
    style_spectral(ax, "(a)")
    leg = style_legend(ax.legend([h_hsi, h_spec, h_rec], ["Hyperspectral", "Spectrometer", r"Recon. (opt. $w$)"],
                                 loc="upper left", fontsize=LGF, handlelength=2.4))
    rrmse_opt = relative_rms_error(recon_opt[sel], hsi_mean[sel])     # vs hyperspectral, 370-720 nm
    print(f"(a) {SELECTED}, w = {opt_w[sel]:.2f}: RRMSE vs HSI = {rrmse_opt:.4f}")
    text_below_legend(leg, rrmse_label(rrmse_opt), ha="left", pad_points=7.5, fontsize=11, zorder=20,
                      bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.2))

    # (b)
    ax = axs[0, 1]
    handles = []
    for w, color, label in W_CASES:
        j = int(np.argmin(np.abs(WS - w)))
        h, = ax.plot(WL_RECON, recon_w[sel, j], color=color, linewidth=LW, solid_capstyle="butt", zorder=3,
                     label=f"$w$={label}")
        handles.append(h)
    h_hsi = plot_hsi(ax, hsi_mean[sel], hsi_std[sel])
    style_spectral(ax, "(b)")
    style_legend(ax.legend([h_hsi] + handles, ["Hyperspectral"] + [f"$w$={lbl}" for _, _, lbl in W_CASES],
                           loc="upper left", fontsize=LGF, handlelength=2.4))

    box_kw = box_style(widths=0.6)       # whiskers at the 10th and 90th percentiles

    # (c)
    ax = axs[1, 0]
    pos = np.arange(1, WS.size + 1)
    ax.boxplot([rrmse[:, j] for j in range(WS.size)], positions=pos, **box_kw)
    ax.set_yscale("log")                 # y range: matplotlib's default, fitted to the whiskers
    ax.yaxis.set_major_locator(FixedLocator([0.1, 0.5]))
    ax.yaxis.set_major_formatter(FixedFormatter(["0.1", "0.5"]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"Regularization parameter $w$", fontsize=LFS)
    ax.set_ylabel("Relative RMS error", fontsize=LFS)
    ax.set_xticks(pos); ax.set_xticklabels([rf"$10^{{{int(np.log10(w))}}}$" for w in WS], fontsize=TFS)
    ax.tick_params(axis="y", labelsize=TFS); ax.tick_params(axis="y", which="minor", length=3)
    ax.grid(True, which="both", alpha=0.3)
    ax.text(-0.12, 1.04, "(c)", transform=ax.transAxes, fontsize=PLF, fontweight="bold", va="bottom")

    # (d)
    ax = axs[1, 1]
    pos = np.arange(1, len(D_ILLUMS) + 1)
    ax.boxplot([de_recon[:, j] for j in range(len(D_ILLUMS))], positions=pos, **box_kw)
    for p, m in zip(pos, np.median(de_spec, axis=0)):
        ax.hlines(m, p - 0.30, p + 0.30, colors="tab:red", linestyles="--", linewidth=2.0, zorder=5)
    ax.axhline(JND, color="black", linestyle="--", linewidth=2.0)
    ax.set_xticks(pos); ax.set_xticklabels(D_ILLUMS, fontsize=TFS)
    ax.set_xlabel("Illuminant", fontsize=LFS); ax.set_ylabel(r"$\Delta E_{00}$", fontsize=LFS)
    ax.tick_params(axis="y", labelsize=TFS)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(-0.1, 7.0)
    style_legend(ax.legend(handles=[Line2D([0], [0], color="black", linestyle="--", linewidth=2.0),
                                    Line2D([0], [0], color="tab:red", linestyle="--", linewidth=2.0)],
                           labels=["JND", "HSI-Spec difference"], loc="upper left", fontsize=LGF,
                           handlelength=2.4, ncol=2))
    ax.text(-0.12, 1.04, "(d)", transform=ax.transAxes, fontsize=PLF, fontweight="bold", va="bottom")
    fig.align_xlabels(axs[1, :])
    fig.canvas.draw()      # constrained layout can need a second pass

    out = output_path("figure3.png")
    save_figure(fig, out, dpi=300, bbox_inches="tight")
    print(f"saved {out}")

    # ------------------------------------------------------------------ Figure S2
    fig, axes = plt.subplots(6, 4, figsize=(16, 18), sharey=True, constrained_layout=True)
    recon_label = rf"Recon. ($w$={W_CONST})"
    for r, row in enumerate(GRID_ROWS):
        for c, name in enumerate(row):
            ax = axes[r, c]
            i = names.index(name)
            h_rec, = ax.plot(WL_RECON, recon_w[i, j_const], color=RECON_COLOR, linewidth=3.0,
                             solid_capstyle="butt", zorder=3, label=recon_label)
            h_spec, = ax.plot(WL_RECON, spec[i], linestyle="--", color=SPEC_COLOR, linewidth=3.0, zorder=3,
                              label="Spectrometer")
            h_hsi = plot_hsi(ax, hsi_mean[i], hsi_std[i])
            text_kw = dict(transform=ax.transAxes, fontsize=14, zorder=20,
                           bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.2))
            label = rrmse_label(rrmse[i, j_const])          # reconstruction vs hyperspectral
            if (r, c) == (5, 3):          # White: swatch, name and RRMSE in the lower right
                add_swatch(ax, CC_HEX[name], (0.98, 0.02), (1, 0))
                ax.text(0.98, 0.23, name, ha="right", va="bottom", **text_kw)
                ax.text(0.98, 0.37, label, ha="right", va="bottom", **text_kw)
            else:
                add_swatch(ax, CC_HEX[name], (0.02, 0.88), (0, 1))
                ax.text(0.02, 0.90, name, ha="left", va="bottom", clip_on=False, **text_kw)
            if (r, c) == (0, 3):          # Black: RRMSE below the legend
                leg = ax.legend([h_hsi, h_spec, h_rec], ["Hyperspectral", "Spectrometer", recon_label],
                                loc="upper right", fontsize=14)
                text_below_legend(leg, label, ha="right", pad_points=9, **text_kw)
            elif (r, c) != (5, 3):
                ax.text(0.98, 0.96, label, ha="right", va="top", **text_kw)
            ax.grid(True, alpha=0.2)
            ax.set_ylim(-0.05, 1.05); ax.set_xlim(360, 730)
            ax.set_xticks([400, 500, 600, 700])
            if r < 5:
                ax.tick_params(labelbottom=False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Wavelength (nm)", fontsize=16)
        ax.tick_params(axis="x", labelsize=12)
    for ax in axes[:, 0]:
        ax.set_ylabel("Reflectance", fontsize=16)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(axis="y", labelsize=12)

    out = output_path("figureS2.png")
    save_figure(fig, out, dpi=300, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
