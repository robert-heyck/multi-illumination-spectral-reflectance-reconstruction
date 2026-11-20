"""Colour-space functions: CIE data, XYZ -> CIELAB / sRGB, CIEDE2000, and sRGB renderings
for human, bird and butterfly cone sensitivities (Section 2.1.3).

The CIE tables are not distributed with this code. Download them from the CIE
(https://cie.co.at/data-tables) and place them in data/external/:
    CIE_xyz_1931_2deg.csv   CIE 1931 2-degree colour-matching functions, 1 nm (columns: nm, x, y, z)
    CIE_std_illum_D65.csv   CIE standard illuminant D65, 1 nm (columns: nm, relative power)
"""
import numpy as np
from skimage.color import deltaE_ciede2000

from common import EXTERNAL, require

XYZ_FILE = EXTERNAL / "CIE_xyz_1931_2deg.csv"
D65_FILE = EXTERNAL / "CIE_std_illum_D65.csv"

# XYZ -> linear sRGB (IEC 61966-2-1, as in Rowlands, Physics of Digital Photography, Eq. 4.33)
M_XYZ_TO_LINEAR_SRGB = np.array([[3.2406, -1.5372, -0.4986],
                                 [-0.9689, 1.8758, 0.0415],
                                 [0.0557, -0.2040, 1.0570]])

# XYZ -> LMS, Hunt-Pointer-Estevez transformation
M_XYZ_TO_LMS_HPE = np.array([[0.38971, 0.68898, -0.07868],
                             [-0.22981, 1.18340, 0.04641],
                             [0.00000, 0.00000, 1.00000]])
M_LMS_TO_XYZ_HPE = np.linalg.inv(M_XYZ_TO_LMS_HPE)


# ----------------------------------------------------------------------------- CIE data
def _load_cie_table(path, wl, hint):
    table = np.loadtxt(require(path, hint), delimiter=",")
    out = np.empty((len(wl), table.shape[1] - 1))
    for i, w in enumerate(np.asarray(wl, dtype=float)):
        hit = np.flatnonzero(np.isclose(table[:, 0], w))
        if hit.size != 1:
            raise ValueError(f"{path.name}: expected one row at {w} nm, found {hit.size}")
        out[i] = table[hit[0], 1:]
    return out


def cie_xyz_cmf(wl):
    """CIE 1931 2-degree colour-matching functions at wavelengths wl, shape (len(wl), 3)."""
    return _load_cie_table(XYZ_FILE, wl, "download the CIE 1931 2-degree CMFs (1 nm) from cie.co.at; see README.md")


def cie_d65(wl):
    """CIE standard illuminant D65 at wavelengths wl, shape (len(wl),)."""
    return _load_cie_table(D65_FILE, wl, "download CIE standard illuminant D65 (1 nm) from cie.co.at; see README.md")[:, 0]


# ----------------------------------------------------------------------------- CIELAB / CIEDE2000
def xyz_to_lab(xyz, white_xyz):
    """CIE XYZ (..., 3) -> CIELAB (..., 3) relative to the white point white_xyz (3,)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    Xn, Yn, Zn = map(float, white_xyz)
    eps, k = 216 / 24389, 24389 / 27

    def f(t):
        return np.where(t > eps, np.cbrt(t), (k * t + 16.0) / 116.0)

    fx, fy, fz = f(xyz[..., 0] / Xn), f(xyz[..., 1] / Yn), f(xyz[..., 2] / Zn)
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1)


def delta_e_2000(lab1, lab2):
    """CIEDE2000 colour difference between two CIELAB arrays (..., 3)."""
    return deltaE_ciede2000(lab1, lab2)


def reflectance_to_xyz(reflectance, illuminant, cmf, dl):
    """Tristimulus values of reflectance (..., n) under an illuminant (n,), with CMFs (n, 3)."""
    return np.asarray(reflectance) @ (cmf * illuminant[:, None]) * dl


def color_difference(refl1, refl2, illuminant, cmf, dl):
    """CIEDE2000 between two reflectance spectra under an illuminant (its own white point)."""
    white = illuminant @ cmf * dl
    lab1 = xyz_to_lab(reflectance_to_xyz(refl1, illuminant, cmf, dl), white)
    lab2 = xyz_to_lab(reflectance_to_xyz(refl2, illuminant, cmf, dl), white)
    return delta_e_2000(lab1, lab2)


# ----------------------------------------------------------------------------- sRGB
def linear_to_srgb(rgb_linear):
    """sRGB transfer function (gamma encoding) applied to linear RGB in [0, 1]."""
    lin = np.asarray(rgb_linear, dtype=np.float64)
    pos = np.maximum(lin, 0.0)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * pos ** (1 / 2.4) - 0.055)


def xyz_to_srgb(xyz, white_xyz=None):
    """
    CIE XYZ (..., 3) -> gamma-encoded sRGB in [0, 1].
    If white_xyz is given, XYZ is first divided by it per component, so that this white maps to
    (approximately) [1, 1, 1] (used for the CAVE renderings in Figure 4).
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if white_xyz is not None:
        xyz = xyz / np.asarray(white_xyz, dtype=np.float64)
    lin = np.clip(xyz @ M_XYZ_TO_LINEAR_SRGB.T, 0.0, None)
    return np.clip(linear_to_srgb(lin), 0.0, 1.0)


def to_uint8(srgb01):
    """[0, 1] sRGB -> 8-bit, by scaling and truncation."""
    return np.clip(np.asarray(srgb01) * 255.0, 0.0, 255.0).astype(np.uint8)


# ----------------------------------------------------------------------------- species renderings
def lms_to_xyz_like(lms):
    """
    Map cone sensitivities (n, 3), ordered (L, M, S), to XYZ-like functions (n, 3) with the inverse
    Hunt-Pointer-Estevez matrix. For human LMS (HPE applied to the CIE CMFs) this returns the CIE CMFs.
    """
    return np.asarray(lms) @ M_LMS_TO_XYZ_HPE.T


def render_srgb(reflectance, illuminant, xyz_like, normalize_white=False):
    """
    sRGB rendering (uint8) of a reflectance image under an illuminant, for an observer given by
    XYZ-like sensitivities.
    input:
        reflectance: array (..., n).
        illuminant: (n,) spectral power, e.g. D65.
        xyz_like: (n, 3) colour-matching functions: the CIE CMFs for humans, or lms_to_xyz_like()
                  of a species' L, M, S cone sensitivities.
        normalize_white: if True, scale linear sRGB per channel so that 100% reflectance maps to
                         [1, 1, 1] (i.e. [255, 255, 255]); used for bird and butterfly vision.
    return:
        uint8 array (..., 3).
    Tristimulus values are normalized by the Y-like value of the illuminant, so a perfect white has Y = 1.
    The renderings in the paper use D65, the sRGB white point, so no chromatic adaptation is applied.
    """
    weights = xyz_like * illuminant[:, None] / (illuminant @ xyz_like[:, 1])     # (n, 3)
    lin = np.asarray(reflectance) @ weights @ M_XYZ_TO_LINEAR_SRGB.T            # (..., 3)
    if normalize_white:
        white_lin = np.ones(len(illuminant)) @ weights @ M_XYZ_TO_LINEAR_SRGB.T
        lin = lin / white_lin
    return to_uint8(linear_to_srgb(lin))


def human_lms(wl):
    """Human L, M, S cone fundamentals from the CIE CMFs via HPE, (len(wl), 3), each peak-normalized."""
    lms = cie_xyz_cmf(wl) @ M_XYZ_TO_LMS_HPE.T
    return lms / lms.max(axis=0, keepdims=True)
