"""Figure S4: reconstruction of the four butterflies (butterfly 3 repeats Figure 5a).

Requires output/butterfly<n>_reconstruction_*.npy for n = 1-4
(run process_butterflies.py, then reconstruct_butterflies.py).

One row per butterfly:
    Sample   the reconstruction rendered in sRGB under D65 (human vision), with the four wing spots marked
    Black, Orange, Gray, Yellow
             reflectance of each spot: reconstruction (mean and standard deviation over the 10 x 10 pixel
             region), hyperspectral imaging (mean and standard deviation over the same spot) and
             spectrometer. RRMSE: reconstruction vs spectrometer, 380-720 nm.

External data: data/external/CIE_xyz_1931_2deg.csv and CIE_std_illum_D65.csv (see README.md).
Output: output/figureS4.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import color_space as cs
from butterflies import (PANEL_ORDER, draw_spot_markers, in_plot_range, load_butterflies, recon_file,
                         spot_references, spot_rrmse, spot_stats)
from common import OUTPUT, WL_RECON, output_path, require, rrmse_label, save_figure, text_below_legend

HSI_COLOR, SPEC_COLOR, RECON_COLOR = "black", "tab:blue", "tab:orange"
LINEWIDTH = 4.0
LABEL_FS, TICK_FS, TITLE_FS, LEGEND_FS, PANEL_LABEL_FS, SPECIES_FS = 20, 18, 20, 20, 22, 17
ROW_LABELS = ["(a)", "(b)", "(c)", "(d)"]

# Layout in inches: the figure is exactly the panels plus margins.
PW = 4.0                 # panel width (all panels)
PH = PW * 2 / 3          # panel height
L_MARGIN = 0.95          # row letter + species name
IMG_GAP = 1.10           # Sample -> Black: room for the y ticks and "Reflectance"
CURVE_GAP = 0.15         # between the curve panels (their y tick labels are hidden)
R_MARGIN = 0.15
T_MARGIN = 0.45          # column titles
ROW_GAP = 0.22
B_MARGIN = 0.75          # x tick labels + "Wavelength (nm)"


def plot_spot(ax, rec, hsi, wl_hsi, wl_spec, spec, ylim):
    """One spot: reconstruction (band = standard deviation), spectrometer, hyperspectral (every 4th error bar)."""
    ax.set_box_aspect(2 / 3)
    sel = in_plot_range(WL_RECON)
    m, e = rec
    ax.fill_between(WL_RECON[sel], (m - e)[sel], (m + e)[sel], color=RECON_COLOR, alpha=0.5, linewidth=0, zorder=1)
    h_rec, = ax.plot(WL_RECON[sel], m[sel], color=RECON_COLOR, linewidth=LINEWIDTH, zorder=3)
    sel = in_plot_range(wl_spec)
    h_spec, = ax.plot(wl_spec[sel], spec[sel], linestyle="--", color=SPEC_COLOR, linewidth=LINEWIDTH, zorder=3)
    sel = in_plot_range(wl_hsi)
    x, m, e = wl_hsi[sel], hsi[0][sel], hsi[1][sel]
    ax.plot(x, m, "o", markersize=3.4, color=HSI_COLOR, zorder=5)
    every = np.arange(0, x.size, 4)
    h_hsi = ax.errorbar(x[every], m[every], yerr=e[every], fmt="o", color=HSI_COLOR, ecolor=HSI_COLOR,
                        alpha=0.75, elinewidth=2.0, markersize=3.4, capsize=3.3, zorder=4)
    ax.set_ylim(*ylim)
    ax.set_xlim(370, 730)
    ax.set_xticks([400, 500, 600, 700])
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.grid(True, alpha=0.2)
    return h_hsi, h_spec, h_rec


def main():
    config = load_butterflies()
    refs = spot_references()
    d65, xyz_cmf = cs.cie_d65(WL_RECON), cs.cie_xyz_cmf(WL_RECON)

    rows = []
    for n in sorted(config):
        recon = np.load(require(OUTPUT / recon_file(n), "run process_butterflies.py and reconstruct_butterflies.py first"))
        spots = config[n]["spots"]
        row = dict(n=n, species=config[n]["species"], spots=spots,
                   image=cs.render_srgb(recon, d65, xyz_cmf),
                   rec={s: spot_stats(recon, spots[s]) for s in spots},
                   hsi={s: (refs["hsi_mean"][f"b{n}_{s}"], refs["hsi_std"][f"b{n}_{s}"]) for s in spots},
                   spec={s: refs["spec"][f"b{n}_{s}"] for s in spots})
        row["rrmse"] = {s: spot_rrmse(row["rec"][s][0], refs["wl_spec"], row["spec"][s]) for s in spots}
        # one y range per row, fitted to that row's curves and error bars, with room for the RRMSE text
        top = max(max(np.max((row["rec"][s][0] + row["rec"][s][1])[in_plot_range(WL_RECON)]),
                      np.max((row["hsi"][s][0] + row["hsi"][s][1])[in_plot_range(refs["wl_hsi"])]),
                      np.max(row["spec"][s][in_plot_range(refs["wl_spec"])])) for s in PANEL_ORDER)
        row["ylim"] = (-0.03, 1.25 * top)
        rows.append(row)
        print(f"butterfly {n}: RRMSE vs spectrometer", {s: round(float(row['rrmse'][s]), 3) for s in PANEL_ORDER})

    n_rows = len(rows)
    fig_w = L_MARGIN + PW + IMG_GAP + 4 * PW + 3 * CURVE_GAP + R_MARGIN
    fig_h = T_MARGIN + n_rows * PH + (n_rows - 1) * ROW_GAP + B_MARGIN
    fig = plt.figure(figsize=(fig_w, fig_h))

    def add_axes(x_in, y_top_in):
        return fig.add_axes([x_in / fig_w, (fig_h - y_top_in - PH) / fig_h, PW / fig_w, PH / fig_h])

    col_x = [L_MARGIN] + [L_MARGIN + PW + IMG_GAP + k * (PW + CURVE_GAP) for k in range(4)]
    for r, row in enumerate(rows):
        bottom = r == n_rows - 1
        y_top = T_MARGIN + r * (PH + ROW_GAP)

        ax = add_axes(col_x[0], y_top)
        ax.imshow(row["image"], aspect="auto")
        draw_spot_markers(ax, row["spots"])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_ylabel(row["species"], fontstyle="italic", fontsize=SPECIES_FS)
        if r == 0:
            ax.set_title("Sample", fontsize=TITLE_FS)
        fig.text(0.05 / fig_w, (fig_h - y_top) / fig_h, ROW_LABELS[r], fontsize=PANEL_LABEL_FS,
                 fontweight="bold", va="top", ha="left")

        for c, s in enumerate(PANEL_ORDER):
            ax = add_axes(col_x[c + 1], y_top)
            handles = plot_spot(ax, row["rec"][s], row["hsi"][s], refs["wl_hsi"], refs["wl_spec"], row["spec"][s],
                                row["ylim"])
            if r == 0:
                ax.set_title(s.capitalize(), fontsize=TITLE_FS)
            if bottom:
                ax.set_xlabel("Wavelength (nm)", fontsize=LABEL_FS)
            else:
                ax.tick_params(labelbottom=False)
            if c == 0:
                ax.set_ylabel("Reflectance", fontsize=LABEL_FS)
            else:
                ax.tick_params(labelleft=False)
            label = rrmse_label(row["rrmse"][s])
            if r == 0 and c == 0:
                leg = ax.legend(handles=handles, labels=["Hyperspectral", "Spectrometer", "Reconstruction"],
                                loc="upper right", fontsize=LEGEND_FS)
                text_below_legend(leg, label, ha="right", pad_points=9, fontsize=LEGEND_FS)
            else:
                ax.text(0.03, 0.96, label, transform=ax.transAxes, ha="left", va="top", fontsize=LEGEND_FS)

    out = output_path("figureS4.png")
    save_figure(fig, out, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
