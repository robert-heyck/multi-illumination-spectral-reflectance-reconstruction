"""Figure 1: photo-box, LED irradiance spectra, camera, and camera spectral sensitivities.

(b) plots data/leds/led_spectra_350to750nm.csv directly (spectrometer-corrected LED irradiance at the
centre of the imaging area, on a common scale with the V peak at 1).
(d) plots the camera sensitivities, each channel normalized to its peak, with a marker at each measured
wavelength (every 5 nm). (The standard deviations in the data file are not drawn.)

Output: output/figure1.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (CHANNEL_NAMES, DATA, LED_NAMES, camera_sensitivity, led_spectra, output_path, save_figure,
                    style_legend)

LED_COLORS = ["red", "green", "blue", "cyan", "orange", "magenta", "purple", "pink"]
CHANNEL_COLORS = ["r", "g", "b"]

LABEL_FS, TICK_FS, LEGEND_FS, PANEL_LABEL_FS = 13, 11, 10.4, 14
LW = 2.2


def panel_image(ax, path):
    ax.imshow(plt.imread(path))
    ax.axis("off")


def panel_leds(ax):
    wl = np.arange(350, 751, 1)
    leds = led_spectra()
    for i in range(leds.shape[0]):
        ax.plot(wl, leds[i], color=LED_COLORS[i], label=LED_NAMES[i], linewidth=LW)

    ax.set_xlabel("Wavelength (nm)", fontsize=LABEL_FS)
    ax.set_ylabel("Relative power (normalized)", fontsize=LABEL_FS)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
    ax.set_xticks([350, 450, 550, 650, 750])
    ax.grid(True, which="major", alpha=0.25)
    ax.set_axisbelow(True)

    # 2 x 4 legend, filled column-major so each column pairs two LEDs
    handles, labels = ax.get_legend_handles_labels()
    n_rows, n_cols = 2, 4
    order = [r + c * n_rows for c in range(n_cols) for r in range(n_rows) if r + c * n_rows < len(handles)]
    style_legend(ax.legend([handles[i] for i in order], [labels[i] for i in order],
                           ncol=n_cols, fontsize=LEGEND_FS, frameon=True, loc="upper center",
                           bbox_to_anchor=(0.625, 0.98), columnspacing=1.5, handlelength=2.5))


def panel_cam_sens(ax):
    wl = np.arange(350, 755, 5)
    mean = camera_sensitivity()
    mean = mean / mean.max(axis=1, keepdims=True)
    for i in range(3):
        ax.plot(wl, mean[i], linewidth=LW, color=CHANNEL_COLORS[i], label=CHANNEL_NAMES[i],
                marker="o", markersize=3.5)
    ax.set_ylim(top=1.1)

    ax.set_xlabel("Wavelength (nm)", fontsize=LABEL_FS)
    ax.set_ylabel("Normalized sensitivity", fontsize=LABEL_FS)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
    ax.set_xticks([350, 450, 550, 650, 750])
    ax.grid(True, which="major", alpha=0.25)
    ax.set_axisbelow(True)
    style_legend(ax.legend(fontsize=LEGEND_FS, frameon=True, loc="best"))


def main():
    fig, axs = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    panel_image(axs[0, 0], DATA / "figure1" / "photobox_with_cartoon.png")
    panel_leds(axs[0, 1])
    panel_image(axs[1, 0], DATA / "figure1" / "camera_cartoon.png")
    panel_cam_sens(axs[1, 1])
    fig.align_ylabels(axs[:, 1])

    # Panel letters at the top-left of each panel; placed after a draw so the image axes
    # have their final (aspect-locked) size, and so each row's letters share one height.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for row, letters in zip(axs, [("(a)", "(b)"), ("(c)", "(d)")]):
        y_top = max(ax.get_position().y1 for ax in row)
        for ax, letter in zip(row, letters):
            x_left = inv.transform(ax.get_tightbbox(renderer).p0)[0]
            fig.text(x_left, y_top + 0.01, letter, fontsize=PANEL_LABEL_FS, fontweight="bold",
                     ha="left", va="bottom")

    out = output_path("figure1.png")
    save_figure(fig, out, dpi=300, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
