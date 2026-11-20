"""Figure 4: reconstruction of the CAVE multispectral images from synthetic CIE XYZ values.

For every pixel of every CAVE scene, synthetic stimulus values are computed with Equation (1) from the
scene's reflectance (400-700 nm, 10 nm), our six non-UV LED spectra and the CIE 1931 XYZ
colour-matching functions, normalized by a 100% reflector under the W LED. The reflectance is then
reconstructed pixel by pixel for six values of w, and compared with the original under D65 in CIELAB
(CIEDE2000).

External data (see README.md), in data/external/:
    CAVE/   the CAVE multispectral image database (Yasuma et al. 2008), unzipped as downloaded:
            CAVE/<scene>_ms/<scene>_ms/<scene>_ms_01.png ... _31.png  (one folder per scene)
    CIE_xyz_1931_2deg.csv, CIE_std_illum_D65.csv

The full run (32 scenes x 512 x 512 pixels x 6 values of w) is cached scene by scene in output/cave/,
so an interrupted run resumes where it stopped.

Usage:
    python figure4.py            all 32 scenes (panel (e) pools all scenes)
    python figure4.py --quick    only the two scenes shown in panels (a) and (b)
    python figure4.py --jobs N   processes for the bounded (L-BFGS-B) fallback
Output: output/figure4.png (or output/figure4_quick.png)
"""
import argparse
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from PIL import Image

import color_space as cs
import reflectance_solver as rs
from common import (EXTERNAL, LED_NAMES, box_style, led_spectra, limit_blas_threads, output_path, require,
                    save_figure)

CAVE_DIR = EXTERNAL / "CAVE"
WL = np.arange(400, 710, 10)
DL = 10.0
ILLUM_ORDER = ["W", "T", "V", "G", "R", "B"]
WS = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])
W_SHOWN = 1e-1                                       # reconstructions shown in (a) and (b)
EXAMPLE_SCENES = ["fake_and_real_lemon_slices_ms", "balloons_ms"]   # Image 1, Image 2
JND = 1.0


# ----------------------------------------------------------------------------- data
def cave_scenes():
    """Sorted scene folder names in data/external/CAVE."""
    require(CAVE_DIR, "download the CAVE database and unzip it into data/external/CAVE; see README.md")
    scenes = sorted(p.name for p in CAVE_DIR.iterdir() if p.is_dir() and p.name.endswith("_ms"))
    if not scenes:
        raise FileNotFoundError(f"no '<scene>_ms' folders in {CAVE_DIR}")
    return scenes


def load_scene(scene):
    """Reflectance cube (31, H, W) in [0, 1] from the 31 band images (400-700 nm) of one scene."""
    files = [p for p in (CAVE_DIR / scene).rglob("*.png") if re.search(r"_ms_\d+\.png$", p.name)]
    files = sorted(files, key=lambda p: int(re.search(r"_(\d+)\.png$", p.name).group(1)))
    if len(files) != WL.size:
        raise ValueError(f"{scene}: expected {WL.size} band images, found {len(files)}")
    bands = []
    for f in files:
        a = np.array(Image.open(f))
        if a.ndim == 3:                       # one scene is stored as RGBA; all channels are equal
            a = a[..., 0]
        bands.append(a.astype(np.float64) / np.iinfo(a.dtype).max)
    return np.stack(bands)


# ----------------------------------------------------------------------------- model
def forward_model():
    """Constraint matrix C (18, 31) of Equation (4), normalized by a 100% reflector under the W LED."""
    xyz = cs.cie_xyz_cmf(WL).T                                           # (3, 31)
    leds = led_spectra(WL)[[LED_NAMES.index(c) for c in ILLUM_ORDER]]    # (6, 31)
    white_W = (xyz * leds[0][None, :]).sum(axis=1) * DL
    return (leds[:, None, :] * xyz[None, :, :] * DL / white_W[None, :, None]).reshape(18, WL.size)


def process_scene(scene, C, xyz_d65, white, executor, cache_dir):
    """Reconstruct one scene for all w; cache CIEDE2000 maps and the XYZ images used for display."""
    cache = cache_dir / f"{scene}.npz"
    if cache.exists():
        with np.load(cache) as f:
            cached = {k: f[k] for k in f.files}
        # reuse only if computed for the same w values
        if ("ws" in cached and np.allclose(cached["ws"], WS)
                and np.isclose(cached.get("w_shown", np.nan), W_SHOWN)):
            return cached
        print(f"    cache for {scene} was computed for other w values; recomputing", flush=True)
    refl = load_scene(scene)                                             # (31, H, W)
    _, H, W = refl.shape
    R = refl.reshape(WL.size, -1).T                                      # (H*W, 31)
    pixels = R @ C.T                                                     # synthetic, noise-free
    xyz_true = R @ xyz_d65                                               # (H*W, 3)
    lab_true = cs.xyz_to_lab(xyz_true, white)

    de = np.zeros((WS.size, H * W), dtype=np.float32)
    xyz_shown = None
    for j, w in enumerate(WS):
        t = time.time()
        recon = rs.solve_batch(w, pixels, C, WL, executor=executor)
        xyz_rec = recon @ xyz_d65
        de[j] = cs.delta_e_2000(cs.xyz_to_lab(xyz_rec, white), lab_true)
        if np.isclose(w, W_SHOWN):
            xyz_shown = xyz_rec
        print(f"    w = {w:g}: {time.time() - t:.1f} s, median dE00 {np.median(de[j]):.3f}", flush=True)

    out = dict(de=de.reshape(WS.size, H, W),
               xyz_true=xyz_true.T.reshape(3, H, W).astype(np.float32),
               xyz_shown=xyz_shown.T.reshape(3, H, W).astype(np.float32),
               ws=WS, w_shown=np.float64(W_SHOWN))
    np.savez(cache, **out)
    return out


# ----------------------------------------------------------------------------- figure
def boxplot_panel(ax, data_per_w, label, legend=False):
    pos = np.arange(1, WS.size + 1)
    ax.boxplot(data_per_w, positions=pos, **box_style(widths=0.6))   # whiskers: 10th / 90th percentile
    ax.axhline(JND, color="k", linestyle="--", linewidth=1.6, alpha=0.9, label="JND")
    ax.set_xticks(pos)
    ax.set_xticklabels([rf"$10^{{{int(np.round(np.log10(w)))}}}$" for w in WS], fontsize=9)
    ax.set_xlabel(r"Reg. parameter $w$", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.text(0.04, 0.96, label, transform=ax.transAxes, ha="left", va="top", fontsize=12)
    if legend:
        ax.legend(loc="upper right", fontsize=12)
    ax.set_ylim(-0.1, 2.1)
    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.set_box_aspect(1)


def make_figure(results, white, pooled_label, out_name):
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    fig.subplots_adjust(wspace=0.08, hspace=0.10)
    titles = ("Original", rf"Reconstructed ($w={W_SHOWN:g}$)", r"CIEDE2000 $\Delta E_{00}$")
    j_shown = int(np.argmin(np.abs(WS - W_SHOWN)))

    for row, scene in enumerate(EXAMPLE_SCENES):
        r = results[scene]
        to_img = lambda xyz: cs.xyz_to_srgb(np.moveaxis(np.maximum(xyz, 0.0), 0, -1), white)
        axs[row, 0].imshow(to_img(r["xyz_true"]))
        axs[row, 1].imshow(to_img(r["xyz_shown"]))
        im = axs[row, 2].imshow(r["de"][j_shown], cmap="magma", vmin=0.0, vmax=5.0)
        for c in range(3):
            axs[row, c].set_xticks([]); axs[row, c].set_yticks([])
            if row == 0:
                axs[row, c].set_title(titles[c])
        axs[row, 0].set_ylabel(f"Image {row + 1}", fontsize=12)

        cax = inset_axes(axs[row, 2], width="4%", height="100%", loc="lower left",
                         bbox_to_anchor=(1.04, 0.0, 1, 1), bbox_transform=axs[row, 2].transAxes, borderpad=0)
        cbar = fig.colorbar(im, cax=cax)
        ticks = sorted(set(list(cbar.get_ticks()) + [JND]))
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(["JND" if t == JND else f"{t:.1f}" for t in ticks])
        cbar.ax.axhline(y=JND, color="white", linewidth=2)

    for c, scene in enumerate(EXAMPLE_SCENES):
        boxplot_panel(axs[2, c], [results[scene]["de"][j].ravel() for j in range(WS.size)],
                      f"Image {c + 1}", legend=(c == 0))
    pooled = [np.concatenate([r["de"][j].ravel() for r in results.values()]) for j in range(WS.size)]
    boxplot_panel(axs[2, 2], pooled, pooled_label)
    axs[2, 2].set_ylabel(r"$\Delta E_{00}$", fontsize=12)
    axs[2, 2].yaxis.set_label_position("right"); axs[2, 2].yaxis.tick_right()
    axs[2, 0].tick_params(axis="y", left=False, labelleft=False)
    axs[2, 1].tick_params(axis="y", left=False, labelleft=False)

    # bottom row: same size as the image panels
    ref = axs[0, 0].get_position()
    for ax in axs[2]:
        pos = ax.get_position()
        cx, cy = pos.x0 + pos.width / 2, pos.y0 + pos.height / 2
        ax.set_position([cx - ref.width / 2, cy - ref.height / 2, ref.width, ref.height])

    labels = [(axs[0, 0], "(a)", 0), (axs[1, 0], "(b)", 0), (axs[2, 0], "(c)", 0),
              (axs[2, 1], "(d)", 0.018), (axs[2, 2], "(e)", 0.018)]
    for ax, lab, dx in labels:
        ax.text(-0.01 + dx, 1.01, lab, transform=ax.transAxes, fontsize=14, fontweight="bold",
                ha="right", va="bottom", clip_on=False)

    out = output_path(out_name)
    save_figure(fig, out, dpi=300, bbox_inches="tight")
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="only the two example scenes")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()
    limit_blas_threads()

    scenes = cave_scenes()
    for s in EXAMPLE_SCENES:
        if s not in scenes:
            raise FileNotFoundError(f"example scene {s} not found in {CAVE_DIR}")
    if args.quick:
        scenes = EXAMPLE_SCENES
    print(f"{len(scenes)} scene(s)")

    C = forward_model()
    xyz_cmf = cs.cie_xyz_cmf(WL)
    d65 = cs.cie_d65(WL)
    xyz_d65 = xyz_cmf * d65[:, None] * DL                                # (31, 3)
    white = np.ones(WL.size) @ xyz_d65

    cache_dir = output_path("cave")
    cache_dir.mkdir(exist_ok=True)
    results = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for k, scene in enumerate(scenes):
            print(f"[{k + 1}/{len(scenes)}] {scene}", flush=True)
            results[scene] = process_scene(scene, C, xyz_d65, white, ex, cache_dir)

    if args.quick:
        make_figure(results, white, "Images 1 and 2", "figure4_quick.png")
    else:
        make_figure(results, white, "All Images", "figure4.png")
    pooled = np.concatenate([r["de"][int(np.argmin(np.abs(WS - W_SHOWN)))].ravel() for r in results.values()])
    print(f"median dE00 over {len(results)} scene(s) at w = {W_SHOWN:g}: {np.median(pooled):.3f}")


if __name__ == "__main__":
    main()
