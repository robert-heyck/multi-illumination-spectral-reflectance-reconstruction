"""Figure 5: hyperspectral reconstruction of butterfly 3 (H. erato amazona).

Requires output/butterfly3_reconstruction_w1.0_370to720nm_400x600x71.npy
(run process_butterflies.py, then reconstruct_butterflies.py).

(a) reflectance of four wing spots: reconstruction (mean and standard deviation over a 10 x 10 pixel
    region), hyperspectral imaging (HSI; mean and standard deviation over a 10 x 10 pixel region) and
    spectrometer. RRMSE: reconstruction vs spectrometer, 380-720 nm.
(b) measured (HSI) and reconstructed images at 400-700 nm
(c) cone sensitivities of human, bird (blue tit) and butterfly (H. erato), and sRGB renderings of the
    reconstruction under D65 from the L, M, S cones of each species

External data: data/external/CIE_xyz_1931_2deg.csv and CIE_std_illum_D65.csv (see README.md).
Output: output/figure5.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import color_space as cs
from butterflies import (BUTTERFLY_DATA, PANEL_ORDER, SPOTS, draw_spot_markers, in_plot_range, load_butterflies,
                         recon_file, spot_references, spot_rrmse, spot_stats)
from cone_sensitivities import blue_tit_cones, heliconius_erato_cones
from common import OUTPUT, WL_RECON, output_path, require, rrmse_label, save_figure, text_below_legend

BUTTERFLY = 3
WL_HSI = np.arange(350, 755, 5)          # hyperspectral images: 350-750 nm, 5 nm
WL_CONES = np.arange(300, 705, 5)        # range plotted in (c)
# The spot regions in the reconstruction are in data/butterflies/spot_regions.csv; these are the same
# spots in the 1000 x 1000 hyperspectral image (their spectra are also in hyperspectral_spot_reflectance_*.csv).
HSI_SPOTS = {"orange": (462, 472, 414, 424), "black": (492, 502, 600, 610),
             "yellow": (415, 425, 470, 480), "gray": (570, 580, 556, 566)}
WL_BANDS = [400, 450, 500, 550, 600, 650, 700]
HSI_CROP = (215, 805, 305, 780)          # (b) crops showing the right wings
RECON_CROP = (5, 370, 310, None)

HSI_COLOR, SPEC_COLOR, RECON_COLOR = "black", "tab:blue", "tab:orange"
CONE_COLORS = ["red", "green", "blue", "violet", "purple"]
CONE_LABELS = ["L", "M", "S", "UV1", "UV2"]
LABEL_FS, TICK_FS, TITLE_FS, PANEL_LABEL_FS, LEGEND_FS = 16, 14, 16, 18, 16


def main():
    recon = np.load(require(OUTPUT / recon_file(BUTTERFLY),
                            "run process_butterflies.py and reconstruct_butterflies.py first"))
    hsi = np.load(require(BUTTERFLY_DATA / "b3_hyperspectral_350to750nm_81x1000x1000.npy"), mmap_mode="r")
    refs = spot_references()
    wl_spec = refs["wl_spec"]
    spec = {s: refs["spec"][f"b{BUTTERFLY}_{s}"] for s in SPOTS}
    recon_spots = load_butterflies()[BUTTERFLY]["spots"]

    # ---------------------------------------------------------------- data for (a)
    hsi_hwl = np.moveaxis(hsi, 0, -1)                       # view as (row, column, wavelength)
    rec_stats = {s: spot_stats(recon, recon_spots[s]) for s in SPOTS}
    hsi_stats = {s: spot_stats(hsi_hwl, HSI_SPOTS[s]) for s in SPOTS}
    rrmse = {s: spot_rrmse(rec_stats[s][0], wl_spec, spec[s]) for s in SPOTS}
    print("RRMSE reconstruction vs spectrometer:", {s: round(float(v), 3) for s, v in rrmse.items()})

    # ---------------------------------------------------------------- renderings for (c)
    d65 = cs.cie_d65(WL_RECON)
    xyz_cmf = cs.cie_xyz_cmf(WL_RECON)
    bird_lms = blue_tit_cones(WL_RECON)[:, [3, 2, 1]]       # (L, M, S)
    bfly_lms = heliconius_erato_cones(WL_RECON)[:, :3]      # (L, M, S)
    img_human = cs.render_srgb(recon, d65, xyz_cmf)
    img_bird = cs.render_srgb(recon, d65, cs.lms_to_xyz_like(bird_lms), normalize_white=True)
    img_bfly = cs.render_srgb(recon, d65, cs.lms_to_xyz_like(bfly_lms), normalize_white=True)

    # cone sensitivities for (c); human LMS defined from 370 nm (zero below)
    human = np.zeros((WL_CONES.size, 3))
    human[WL_CONES >= WL_RECON[0]] = cs.human_lms(WL_RECON)[:np.sum(WL_CONES >= WL_RECON[0])]
    cones = {"Human": human,
             "Bird": blue_tit_cones(WL_CONES)[:, [3, 2, 1, 0]],      # L, M, S, UV
             "Butterfly": heliconius_erato_cones(WL_CONES)}           # L, M, S, UV1, UV2

    # ---------------------------------------------------------------- figure
    fig = plt.figure(figsize=(16, 15.5), constrained_layout=True)
    outer = fig.add_gridspec(4, 1, height_ratios=[1.3, 1.8, 0.08, 2.80])

    # (a)
    gs_a = outer[0].subgridspec(1, 4, wspace=0.06)
    for k, name in enumerate(PANEL_ORDER):
        a = fig.add_subplot(gs_a[0, k])
        a.set_box_aspect(2 / 3)
        m, e = rec_stats[name]
        sel = in_plot_range(WL_RECON)
        a.fill_between(WL_RECON[sel], (m - e)[sel], (m + e)[sel], color=RECON_COLOR, alpha=0.5, linewidth=0, zorder=1)
        h_rec, = a.plot(WL_RECON[sel], m[sel], color=RECON_COLOR, linewidth=4.0, zorder=3)
        sel = in_plot_range(wl_spec)
        h_spec, = a.plot(wl_spec[sel], spec[name][sel], linestyle="--", color=SPEC_COLOR, linewidth=4.0, zorder=3)
        m, e = hsi_stats[name]
        x, m, e = WL_HSI[in_plot_range(WL_HSI)], m[in_plot_range(WL_HSI)], e[in_plot_range(WL_HSI)]
        a.plot(x, m, "o", markersize=3.4, color=HSI_COLOR, zorder=5)
        every = np.arange(0, x.size, 4)
        h_hsi = a.errorbar(x[every], m[every], yerr=e[every], fmt="o", color=HSI_COLOR, ecolor=HSI_COLOR,
                           alpha=0.75, elinewidth=2.0, markersize=3.4, capsize=3.3, zorder=4)
        a.set_title(name.capitalize(), fontsize=TITLE_FS)
        a.set_ylim(-0.03, 0.63); a.set_xlim(370, 730)
        a.set_xlabel("Wavelength (nm)", fontsize=LABEL_FS)
        a.set_xticks([400, 500, 600, 700])
        a.tick_params(axis="both", labelsize=TICK_FS)
        a.grid(True, alpha=0.2)
        if k == 0:
            a.set_ylabel("Reflectance", fontsize=LABEL_FS)
            a.set_yticks([0.0, 0.2, 0.4, 0.6])
            leg = a.legend(handles=[h_hsi, h_spec, h_rec], labels=["Hyperspectral", "Spectrometer", "Reconstruction"],
                           loc="upper right", fontsize=LEGEND_FS)
            text_below_legend(leg, rrmse_label(rrmse[name]), ha="right", pad_points=9, fontsize=LEGEND_FS)
        else:
            a.tick_params(labelleft=False)
            a.text(0.03, 0.96, rrmse_label(rrmse[name]), transform=a.transAxes, ha="left", va="top",
                   fontsize=LEGEND_FS)
    fig.text(0.01, 0.995, "(a)", fontsize=PANEL_LABEL_FS, fontweight="bold", va="top")

    # (b)
    gs_b = outer[1].subgridspec(2, 7, wspace=0.02, hspace=0.03)
    hy1, hy2, hx1, hx2 = HSI_CROP
    ry1, ry2, rx1, rx2 = RECON_CROP
    for j, lam in enumerate(WL_BANDS):
        for i in range(2):
            a = fig.add_subplot(gs_b[i, j])
            if i == 0:
                img = np.asarray(hsi[int(np.flatnonzero(WL_HSI == lam)[0]), hy1:hy2, hx1:hx2])
                a.set_title(f"{lam} nm", fontsize=16)
            else:
                img = recon[ry1:ry2, rx1:rx2, int(np.flatnonzero(WL_RECON == lam)[0])]
            a.imshow(img, cmap="gray", vmin=0, vmax=1, aspect="auto")
            if j == 0:
                a.set_ylabel("Measured" if i == 0 else "Reconstruction", fontsize=16)
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
    fig.text(0.01, 0.79, "(b)", fontsize=PANEL_LABEL_FS, fontweight="bold", va="top")

    # (c) cone sensitivities
    gs_c = outer[3].subgridspec(2, 1, height_ratios=[1.0, 0.9], hspace=0.02)
    gs_c1 = gs_c[0].subgridspec(1, 3, wspace=0.06)
    for k, (title, sens) in enumerate(cones.items()):
        a = fig.add_subplot(gs_c1[0, k])
        a.set_box_aspect(2 / 3)
        for i in range(sens.shape[1]):
            a.plot(WL_CONES, sens[:, i], color=CONE_COLORS[i], linewidth=3.5)
        a.set_title(title, fontsize=TITLE_FS)
        a.set_xlim(295, 705); a.set_ylim(-0.02, 1.05)
        a.set_xlabel("Wavelength (nm)", fontsize=LABEL_FS)
        a.set_xticks([300, 400, 500, 600, 700])
        a.tick_params(axis="both", labelsize=TICK_FS)
        a.grid(True, alpha=0.2)
        if k == 0:
            a.set_ylabel("Normalized sensitivity", fontsize=LABEL_FS)
            a.legend(handles=[Line2D([0], [0], color=c, lw=2.5) for c in CONE_COLORS], labels=CONE_LABELS,
                     loc="upper left", fontsize=15, frameon=True, handlelength=1.8, borderpad=0.35,
                     labelspacing=0.3, handletextpad=0.5)
        else:
            a.tick_params(labelleft=False)

    # (c) renderings
    gs_c2 = gs_c[1].subgridspec(1, 4, width_ratios=[0.10, 1, 1, 1], wspace=0.03)
    for k, img in enumerate([img_human, img_bird, img_bfly]):
        a = fig.add_subplot(gs_c2[0, k + 1])
        a.imshow(img)
        a.axis("off")
        if k == 0:
            draw_spot_markers(a, recon_spots)
    fig.text(0.01, 0.47, "(c)", fontsize=PANEL_LABEL_FS, fontweight="bold", va="top")

    out = output_path("figure5.png")
    save_figure(fig, out, dpi=300, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
