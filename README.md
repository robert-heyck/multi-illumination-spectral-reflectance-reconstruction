# Multi-Illumination Spectral Reflectance Reconstruction — code and data

Code and data to reproduce the figures of

> R. Heyck, V. Gillaspy, S. J. Hagen, B. Xue. *Multi-Illumination Spectral Reflectance Reconstruction: A method for
> hyperspectral imaging using digital photography.*

The reconstruction algorithm (Section 2.1) is in `reflectance_solver.py`. The other scripts process our raw
images, run the reconstructions, and draw Figures 1–5 and Figures S2 and S4.

## Setup

```
pip install -r requirements.txt
```

The raw images and the two largest data files are archived on Zenodo: https://doi.org/10.5281/zenodo.24383916.
Download `multi-illumination-spectral-reflectance-reconstruction_data.zip` and unzip it into this folder. Its
files go to the `data/` paths marked "(Zenodo)" under [Data](#data).

Some third-party data are not redistributed here. Download them into `data/external/` as described in
[External data](#external-data) below.

## Running

Run each script from any directory. All outputs are written to `output/`.

| Step | Script | Needs | Produces | Time* |
|---|---|---|---|---|
| 1 | `python figure1.py` | — | `figure1.png` | seconds |
| 2 | `python figure2.py` | Munsell, CIE | `figure2.png` | seconds |
| 3 | `python process_color_checker.py` | — | `color_checker_patches.npz` | ~1 min |
| 4 | `python figure3.py` | step 3, CIE | `figure3.png`, `figureS2.png` | ~30 s |
| 5 | `python figure4.py` | CAVE, CIE | `figure4.png` | ~1–2 h |
|   | `python figure4.py --quick` | CAVE, CIE | `figure4_quick.png` (only the two scenes shown) | ~1 min |
| 6 | `python process_butterflies.py` | — | `butterfly<n>_rgb_8xHxWx3.npy`, n = 1–4 | ~4 min |
| 7 | `python reconstruct_butterflies.py` | step 6 | `butterfly<n>_reconstruction_w1.0_370to720nm_HxWx71.npy` | ~2 min |
| 8 | `python figure5.py` | step 7 (butterfly 3), CIE | `figure5.png` | ~15 s |
| 9 | `python figureS4.py` | step 7, CIE | `figureS4.png` | ~15 s |

\*On an 8-thread laptop. `figure4.py` caches results scene by scene in `output/cave/`. If it is interrupted,
running it again resumes from the last finished scene. Delete `output/cave/` to force a full recomputation,
for example after changing the LED data.

Steps 6 and 7 process all four butterflies by default. To run only some, give their numbers, for example
`python process_butterflies.py 3` and `python reconstruct_butterflies.py 3` for Figure 5 alone.

`python cone_sensitivities.py` writes the bird and butterfly cone sensitivities used in Figure 5 to `output/`.

## Code

| File | Contents |
|---|---|
| `reflectance_solver.py` | Reconstruction (Eqs. 2–6): the closed-form solution, the bounded L-BFGS-B fallback, and the adaptive choice of `w` (Section 4.1). `solve_batch()` gives the same result as `solve()` applied to each pixel, but vectorized. |
| `common.py` | Paths, LED/channel names, and loaders for the LED spectra, spatial profiles and camera sensitivities. `constraint_matrices()` builds the forward model **C** of Eq. 4 for our photo-box images. Shared plotting helpers. |
| `butterflies.py` | Reads the butterfly configuration (`data/butterflies/butterflies.csv`, `spot_regions.csv`) and the spot reference spectra; spot statistics and RRMSE. |
| `raw_processing.py` | Raw NEF → normalized RGB images (Section 2.2.1). |
| `color_space.py` | CIE data loaders; XYZ → CIELAB / sRGB; CIEDE2000; sRGB renderings for human, bird and butterfly cones. |
| `cone_sensitivities.py` | Govardovskii et al. (2000) template; blue-tit (Hart & Vorobyev 2005) and *H. erato* cone sensitivities. |
| `process_color_checker.py`, `process_butterflies.py` | Image processing for the color checker and the butterflies. |
| `reconstruct_butterflies.py` | Pixel-by-pixel reconstruction of the butterflies (`w` = 1.0), each over its crop region. |
| `figure1.py` … `figure5.py`, `figureS4.py` | The figures. `figure3.py` also makes Figure S2. |

The RRMSE values printed on the panels of Figures 3a, 5a, S2 and S4 are RMS(reconstruction − reference) /
mean(reference). The reference is the spectrometer for the butterfly spots (380–720 nm). For the color checker
it is the hyperspectral measurement (370–720 nm, as in Fig. 3c).

### Image processing (`raw_processing.py`)
1. Average the duplicate raw frames taken under each LED. (The color checker has one frame per LED. Its
   patches are uniform, so averaging over each patch region reduces the noise instead.)
2. Subtract the averaged dark frame and clip at zero.
3. Demosaic and downscale: each 4 × 4 Bayer block becomes one RGB pixel. Each channel is the median of that
   block's pixels of that colour: 4 red, 8 green (both green sites of the RGGB pattern) and 4 blue.
4. Rotate by 180°.
5. Divide each channel by the mean of a 10 × 10 pixel region on a Spectralon white standard imaged under the
   W LED in the same session.

No white balance or other camera processing is applied, so the pixel values are linear in light intensity.
The processed images are 1005 × 1509 pixels.

### Units and normalization
- **LED spectra** (`data/leds/led_spectra_350to750nm.csv`): relative spectral irradiance at the centre of the
  imaging area, corrected for the spectrometer's spectral response. All eight LEDs share one scale, with the
  V peak equal to 1. This is the curve plotted in Fig. 1b.
- **Spatial profiles** (`data/leds/led_spatial_profiles_8x1005x1509.npy`): for each LED, the irradiance at
  each pixel of the processed image grid relative to the box centre (value 1 at pixel (504, 761)). The
  spectral shape of each LED is the same everywhere (Section 2.2.2). So the irradiance of LED *k* at pixel
  (y, x) is `profile[k, y, x] * led_spectrum[k](λ)`.
- **Camera sensitivities** (`data/camera/camera_sensitivity_350to750nm.csv`): mean and standard deviation for
  R, G, B. They are scaled so that, with the two files above, Eq. 1 predicts the pixel values normalized by the
  Spectralon under the W LED (step 5 above). Fig. 1d shows each channel divided by its peak.

## Data

```
data/
  leds/
    led_spectra_350to750nm.csv                 LED spectra (350-750 nm, 1 nm)
    led_spatial_profiles_8x1005x1509.npy       LED spatial profiles (Zenodo)
  camera/          camera sensitivities (350-750 nm, 5 nm)
  figure1/         photo-box and camera images for Fig. 1a, 1c
  color_checker/
    raw/           (Zenodo)
    raw/dark/        DSC_5670, DSC_5672        dark frames (no LED on)
    raw/spectralon/  DSC_5684                  Spectralon under the W LED
    raw/chart/       DSC_5689-DSC_5696         ColorChecker under R, G, B, W, T, V, UV-1, UV-2 (one frame each)
    spectrometer_reflectance_350to750nm.csv            spectrometer probe, 24 patches
    hyperspectral_reflectance_{mean,std}_350to750nm.csv  hyperspectral imaging, mean/std over 10 x 10 pixels
  butterflies/     four Heliconius specimens (Fig. 5: butterfly 3; Fig. S4: all four), imaged in one session
    butterflies.csv  species and crop region (in the processed image) of each butterfly
    spot_regions.csv the four 10 x 10 pixel wing-spot regions of each butterfly (in the cropped image)
    raw/           (Zenodo)
    raw/dark/        DSC_5274-DSC_5278         dark frames, shared by all butterflies
    raw/spectralon/  DSC_5294-DSC_5298         Spectralon under the W LED, shared by all butterflies
    raw/b1/          DSC_5400-DSC_5439         butterfly 1: 5 frames under each LED, in the order above
    raw/b2/          DSC_5440-DSC_5479         butterfly 2
    raw/b3/          DSC_5480-DSC_5519         butterfly 3
    raw/b4/          DSC_5520-DSC_5559         butterfly 4
    spectrometer_spot_reflectance_350to750nm.csv           spectrometer probe, all 16 wing spots
    hyperspectral_spot_reflectance_{mean,std}_350to750nm.csv  hyperspectral imaging, mean/std over each spot
    b3_hyperspectral_350to750nm_81x1000x1000.npy           butterfly 3 hyperspectral image (float32; band, row, column) (Zenodo)
  external/        third-party data (not included; see below)
```

All `.csv` tables have `#` comment lines describing the contents, then a header line with the column names,
then one row per wavelength. `common.load_table()` reads them.

The raw images are Nikon D7100 NEF files: 14 bit, exposure 1/100 s, taken with the camera's UV/IR-blocking filter
removed and a Coastal Optics 60 mm UV-VIS-IR lens.

## External data

Place these in `data/external/` with the names shown.

| File / folder | Source | Used by |
|---|---|---|
| `CIE_xyz_1931_2deg.csv` | CIE 1931 2° colour-matching functions, 1 nm (CIE data tables, https://cie.co.at/data-tables) | Figs. 2–5, S2, S4 |
| `CIE_std_illum_D65.csv` | CIE standard illuminant D65, 1 nm (CIE data tables) | Figs. 2–5, S2, S4 |
| `SpectralReflectancesOf2007MunsellBookOfColorGlossy.txt` | P. Centore, https://www.munsellcolourscienceforpainters.com/MunsellResources/SpectralReflectancesOf2007MunsellBookOfColorGlossy.txt | Fig. 2 |
| `CAVE/` | CAVE multispectral image database (Yasuma et al. 2008, Columbia University), https://cave.cs.columbia.edu/repository/Multispectral. Unzip so that each scene is `CAVE/<scene>_ms/<scene>_ms/<scene>_ms_01.png` … `_31.png` | Fig. 4 |

The CIE files are comma-separated with no header: wavelength in nm, then the values. The Munsell file has one
line per colour: the name, then reflectances from 380 nm in 10 nm steps.

## Notes
- The reconstruction solves many small (71 × 71) linear systems. Multithreaded BLAS is about 100× slower than
  a single thread for this. The scripts call `common.limit_blas_threads()`, which uses `threadpoolctl` if it
  is installed and also sets `OPENBLAS_NUM_THREADS=1` (and the OMP/MKL equivalents) for worker processes.
- `figure4.py` and `reconstruct_butterflies.py` run the bounded fallback in parallel worker processes. Use
  `--jobs N` to set how many.
