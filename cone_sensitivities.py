"""Cone spectral sensitivities of the blue tit and of Heliconius erato (Section 3.2.2, Figure 5c).

Visual-pigment absorbance: Govardovskii et al. (2000) A1 template (alpha band + beta band).
Bird cones (blue tit, Parus caeruleus): Hart & Vorobyev (2005) model
    S(lambda) = T_oil(lambda) * T_ocular(lambda) * [1 - 10^(-alpha * L * a(lambda))],
    with oil-droplet transmittance T_oil(lambda) = exp(-2.93 exp(-2.89 * 0.35 b (lambda - lambda_cut)))
    and a logistic ocular-media transmittance, using the blue-tit parameters of Hart et al. (2000).
Butterfly cones (H. erato, female): the same template with perfect transmittance (no oil droplets or
    ocular filtering) and the peak wavelengths of McCulloch, Osorio & Briscoe (2016).

Running this file writes the sensitivities to output/ and plots them.
"""
import numpy as np

# ----------------------------------------------------------------------------- templates

def govardovskii_a1_absorbance(lam_nm, lam_max_nm, include_beta=True):
    """Govardovskii et al. (2000) A1 visual-pigment template, alpha band normalized to 1 at lambda_max."""
    lam_nm = np.asarray(lam_nm, dtype=float)
    lam_max_nm = float(lam_max_nm)
    A, B, C = 69.7, 28.0, -14.9
    bpar, cpar, D = 0.922, 1.104, 0.674
    apar = 0.8795 + 0.0459 * np.exp(-((lam_max_nm - 300.0) ** 2) / 11940.0)

    def alpha(x):
        return 1.0 / (np.exp(A * (apar - x)) + np.exp(B * (bpar - x)) + np.exp(C * (cpar - x)) + D)

    alpha_band = alpha(lam_max_nm / lam_nm) / alpha(1.0)
    if not include_beta:
        return alpha_band

    A_beta = 0.26
    lam_mB = 189.0 + 0.315 * lam_max_nm
    b_beta = -40.5 + 0.195 * lam_max_nm
    return alpha_band + A_beta * np.exp(-((lam_nm - lam_mB) / b_beta) ** 2)


def oil_droplet_transmittance(lam_nm, lam_cut_nm, b_param):
    """Hart & Vorobyev (2005) oil-droplet transmittance, with B_mid = 0.35 b."""
    lam_nm = np.asarray(lam_nm, dtype=float)
    return np.exp(-2.93 * np.exp(-2.89 * 0.35 * b_param * (lam_nm - lam_cut_nm)))


def ocular_transmittance(lam_nm, lambda_half_nm=317.0, k_per_nm=0.25):
    """Logistic ocular-media transmittance, 0.5 at lambda_half_nm."""
    lam_nm = np.asarray(lam_nm, dtype=float)
    return 1.0 / (1.0 + np.exp(-k_per_nm * (lam_nm - lambda_half_nm)))


def cone_sensitivity(lam_nm, lam_max_nm, L_um=16.0, alpha_um_inv=0.014, include_beta=True,
                     lam_cut_nm=None, b_param=None, ocular=None):
    """
    Cone sensitivity S = T_oil * T_ocular * [1 - 10^(-alpha L a(lambda))].
    lam_cut_nm=None means no oil droplet; ocular=None means T_ocular = 1, otherwise a dict of
    keyword arguments for ocular_transmittance().
    """
    a = govardovskii_a1_absorbance(lam_nm, lam_max_nm, include_beta=include_beta)
    S = 1.0 - 10.0 ** (-(alpha_um_inv * L_um * a))
    if lam_cut_nm is not None:
        S = S * oil_droplet_transmittance(lam_nm, lam_cut_nm, b_param)
    if ocular is not None:
        S = S * ocular_transmittance(lam_nm, **ocular)
    return S


# ----------------------------------------------------------------------------- species

BIRD_CONE_NAMES = ["UV", "S", "M", "L"]
BIRD_LAMBDA_MAX = {"UV": 372.0, "S": 449.0, "M": 502.0, "L": 563.0}
BIRD_OIL = {"UV": None, "S": (413.0, 0.106), "M": (508.0, 0.068), "L": (573.0, 0.059)}
BIRD_OCULAR = dict(lambda_half_nm=317.0, k_per_nm=0.25)

# Female H. erato. Names as used in Figure 5c (UV1 = 389 nm, UV2 = 356 nm).
BUTTERFLY_CONE_NAMES = ["L", "M", "S", "UV1", "UV2"]
BUTTERFLY_LAMBDA_MAX = [600.0, 555.0, 470.0, 389.0, 356.0]


def blue_tit_cones(wl=None):
    """
    Blue-tit cone sensitivities, columns (UV, S, M, L), each peak-normalized on a 300-720 nm, 1 nm grid.
    Returns shape (len(wl), 4); wl must lie on that grid (default: the full 1 nm grid).
    """
    lam = np.linspace(300, 720, 421)
    cones = []
    for name in BIRD_CONE_NAMES:
        oil = BIRD_OIL[name]
        S = cone_sensitivity(lam, BIRD_LAMBDA_MAX[name],
                             lam_cut_nm=None if oil is None else oil[0],
                             b_param=None if oil is None else oil[1],
                             ocular=BIRD_OCULAR)
        cones.append(S / S.max())
    cones = np.stack(cones, axis=1)
    if wl is None:
        return cones
    idx = np.round(np.asarray(wl) - 300).astype(int)
    return cones[idx]


def heliconius_erato_cones(wl=None):
    """
    H. erato cone sensitivities, columns (L, M, S, UV1, UV2), each peak-normalized on a 300-720 nm,
    5 nm grid (no beta band, perfect transmittance). Returns shape (len(wl), 5); wl on that grid.
    """
    lam = np.arange(300, 725, 5)
    cones = np.stack([cone_sensitivity(lam, lm, include_beta=False) for lm in BUTTERFLY_LAMBDA_MAX], axis=1)
    cones = cones / cones.max(axis=0, keepdims=True)
    if wl is None:
        return cones
    idx = np.round((np.asarray(wl) - 300) / 5).astype(int)
    return cones[idx]


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common import output_path, save_figure

    lam_bird = np.linspace(300, 720, 421)
    lam_bfly = np.arange(300, 725, 5)
    bird, bfly = blue_tit_cones(), heliconius_erato_cones()
    np.savetxt(output_path("blue_tit_cones_300to720nm.csv"), np.column_stack([lam_bird, bird]),
               delimiter=",", header="wavelength_nm," + ",".join(BIRD_CONE_NAMES), fmt="%.8g")
    np.savetxt(output_path("heliconius_erato_cones_300to720nm.csv"), np.column_stack([lam_bfly, bfly]),
               delimiter=",", header="wavelength_nm," + ",".join(BUTTERFLY_CONE_NAMES), fmt="%.8g")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    for i, n in enumerate(BIRD_CONE_NAMES):
        ax[0].plot(lam_bird, bird[:, i], label=n, color=["violet", "blue", "green", "red"][i])
    for i, n in enumerate(BUTTERFLY_CONE_NAMES):
        ax[1].plot(lam_bfly, bfly[:, i], label=n, color=["red", "green", "blue", "violet", "purple"][i])
    for a, t in zip(ax, ["Blue tit", "H. erato"]):
        a.set_title(t); a.set_xlabel("Wavelength (nm)"); a.legend()
    ax[0].set_ylabel("Normalized sensitivity")
    save_figure(fig, output_path("cone_sensitivities.png"), dpi=150, bbox_inches="tight")
    print("saved cone sensitivities to", output_path(""))
