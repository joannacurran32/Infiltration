"""
Transient Pore Water Pressure in a Vegetated Unsaturated Infinite Slope
========================================================================
Implements Ng et al. (2015) Eq. (25):

    k*(z*, t) = k1*(z*) + Σ_n  C_n · φ_n(z*) · exp(-β_n · t)

where:
    φ_n(z*) = exp(-α·z*/2) · sin(ω_n·z*)          [eigenfunctions]
    β_n     = (ω_n² + α²/4) · ks · α / (θ_s − θ_r) [decay rates]
    k0*(z*) = steady-state relative permeability (pre-rain, flux q0) — Eq. (19)
    k1*(z*) = steady-state relative permeability (new target, flux q1) — Eq. (19)
    C_n     = ∫₀^H₀ [k0* − k1*] · exp(α·z*/2) · sin(ω_n·z*) dz*
              ──────────────────────────────────────────────────────
              ∫₀^H₀ sin²(ω_n·z*) dz*

The spatial frequencies ω_n satisfy the characteristic equation (from the
Robin BC at the slope surface z* = H₀):
    tan(ω_n · H₀) = −2·ω_n / α

The initial condition is the Eq. (19) steady state with surface flux q0.
Rainfall begins at t = 0, changing the surface flux to q1 (> q0 typically).
Transpiration T and root architecture are assumed constant throughout.

Because k0*(z*) − k1*(z*) = (q0 − q1)/ks · [exp(−α·z*) − 1] (the vegetation
terms cancel exactly), the coefficients C_n are independent of architecture and
can be computed analytically or with a single numerical integration per mode.

Reference
---------
    Ng, C.W.W., Ni, J.J., Leung, A.K., Wang, Z.J. (2015).
    Analytical solutions for calculating pore-water pressure in an infinite
    unsaturated slope with different root architectures.
    Can. Geotech. J. 52(12):1980–1993. doi:10.1139/cgj-2014-0537
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — avoids Qt/Tk dependency
import matplotlib.pyplot as plt
import warnings

from ng2015_eq19_pore_pressure import pore_water_pressure_eq19, root_uptake_term


# ═══════════════════════════════════════════════════════════════════════════
# STEADY-STATE k* (internal helper, avoids re-importing via public API)
# ═══════════════════════════════════════════════════════════════════════════

def _k_steady(z_star, H0, L1_vert, L2_vert, alpha, psi_0, q, ks, phi, T, architecture):
    """
    Steady-state relative permeability k*(z*) for surface flux q.

    k*(z*) = exp[α(ψ₀ − z*)] + (q/ks)[exp(−α·z*) − 1] + root_uptake_term(...)
    """
    term1 = np.exp(alpha * (psi_0 - z_star))
    term2 = (q / ks) * (np.exp(-alpha * z_star) - 1.0)
    term3 = (root_uptake_term(z_star, L1_vert, L2_vert, alpha, phi, ks, T, architecture)
             if T != 0.0 else 0.0)
    return term1 + term2 + term3


# ═══════════════════════════════════════════════════════════════════════════
# EIGENVALUES
# ═══════════════════════════════════════════════════════════════════════════

def compute_eigenvalues(alpha, H0, N=50):
    """
    Compute the first N spatial frequencies ω_n satisfying:

        tan(ω_n · H₀) = −2·ω_n / α        (n = 1, 2, 3, …)

    These arise from the Robin boundary condition at the slope surface
    (z* = H₀) of the homogeneous transient problem.  Roots are bracketed
    in intervals where tan < 0, i.e. ((n − ½)π/H₀,  n·π/H₀) for n ≥ 1.

    Parameters
    ----------
    alpha : float   Gardner desaturation coefficient  [1/m]
    H0    : float   Vertical depth to water table     [m]
    N     : int     Number of eigenvalues to return

    Returns
    -------
    list of float  Eigenvalues ω_n in ascending order
    """
    eigenvalues = []

    def f(omega):
        return np.tan(omega * H0) + 2.0 * omega / alpha

    n = 1
    while len(eigenvalues) < N:
        lo = (n - 0.5) * np.pi / H0 + 1e-12
        hi = n * np.pi / H0 - 1e-12
        try:
            flo, fhi = f(lo), f(hi)
        except Exception:
            n += 1
            continue
        if flo * fhi < 0:
            try:
                root = brentq(f, lo, hi, xtol=1e-14, rtol=1e-12)
                eigenvalues.append(root)
            except Exception:
                pass
        n += 1
        if n > N + 500:          # safety guard
            break

    return sorted(eigenvalues)


# ═══════════════════════════════════════════════════════════════════════════
# SERIES COEFFICIENTS  C_n
# ═══════════════════════════════════════════════════════════════════════════

def _series_coefficients(alpha, H0, q0, q1, ks, eigenvalues):
    """
    Compute the eigenfunction expansion coefficients C_n.

    Because k0*(z*) − k1*(z*) = (q0 − q1)/ks · [exp(−α·z*) − 1], the
    vegetation terms cancel and C_n is independent of root architecture:

        C_n = (q0−q1)/ks · ∫₀^H₀ [exp(−α·z*/2) − exp(α·z*/2)]·sin(ω_n·z*) dz*
              ──────────────────────────────────────────────────────────────────
                            ∫₀^H₀ sin²(ω_n·z*) dz*

    Both integrals are evaluated analytically.

    Parameters
    ----------
    alpha       : float
    H0          : float
    q0          : float   Pre-rainfall surface flux  [m/s]
    q1          : float   Rainfall surface flux      [m/s]
    ks          : float
    eigenvalues : list of float   ω_n values

    Returns
    -------
    list of float   Coefficients C_n (one per eigenvalue)
    """
    dq_ks = (q0 - q1) / ks
    coeffs = []

    for omega in eigenvalues:
        denom_sq = alpha**2 / 4.0 + omega**2   # shared denominator for integrals

        # ∫₀^H₀ exp(−α·z*/2) · sin(ω·z*) dz*
        def _I(a):
            """∫₀^H₀ exp(a·z*) · sin(ω·z*) dz*  (analytic)"""
            D = a**2 + omega**2
            at_H0 = np.exp(a * H0) * (a * np.sin(omega * H0) - omega * np.cos(omega * H0)) / D
            at_0  = -omega / D                     # exp(0)*(0 - omega*1)/D
            return at_H0 - at_0

        I_neg = _I(-alpha / 2.0)   # exp(−α·z*/2) term
        I_pos = _I( alpha / 2.0)   # exp(+α·z*/2) term

        numerator = dq_ks * (I_neg - I_pos)

        # ∫₀^H₀ sin²(ω·z*) dz* = H₀/2 − sin(2ω·H₀)/(4ω)
        denominator = H0 / 2.0 - np.sin(2.0 * omega * H0) / (4.0 * omega)

        coeffs.append(numerator / denominator if abs(denominator) > 1e-15 else 0.0)

    return coeffs


# ═══════════════════════════════════════════════════════════════════════════
# TRANSIENT PORE WATER PRESSURE  uw(z*, t)  —  Eq. (25)
# ═══════════════════════════════════════════════════════════════════════════

def pore_water_pressure_eq25(z_star, t, H0, L1_vert, L2_vert,
                              alpha, psi_0, q0, q1, ks, phi,
                              theta_s, theta_r, T, architecture,
                              gamma_w=10.0, eigenvalues=None, N_terms=50):
    """
    Transient pore water pressure  uw [kPa]  at depth z* and time t.

    Ng et al. (2015) Eq. (25):

        k*(z*, t) = k1*(z*) + Σ_n  C_n · exp(−α·z*/2) · sin(ω_n·z*) · exp(−β_n·t)

        β_n = (ω_n² + α²/4) · ks · α / (θ_s − θ_r)

        uw  = (γ_w / α) · ln[k*(z*, t)]

    Parameters
    ----------
    z_star       : float    Depth in z* coordinates [m]  (0 = water table)
    t            : float    Time since rainfall start [s]
    H0           : float    Vertical depth to water table [m]
    L1_vert      : float    Vertical depth of non-rooted zone [m]
    L2_vert      : float    Vertical thickness of root zone [m]
    alpha        : float    Gardner desaturation coefficient [1/m]
    psi_0        : float    Pressure head at water table [m]
    q0           : float    Pre-rainfall surface flux [m/s]
    q1           : float    Rainfall surface flux [m/s]
    ks           : float    Saturated hydraulic conductivity [m/s]
    phi          : float    Slope angle [radians]
    theta_s      : float    Saturated volumetric water content [-]
    theta_r      : float    Residual volumetric water content [-]
    T            : float    Transpiration rate [m/s]
    architecture : str      Root architecture type
    gamma_w      : float    Unit weight of water [kN/m³]
    eigenvalues  : list     Pre-computed ω_n (pass to avoid recomputation)
    N_terms      : int      Number of series terms to use

    Returns
    -------
    float  [kPa]
        Negative = suction.
        np.nan  = model breakdown (k* ≤ 0) or full saturation (k* > 1).
    """
    if eigenvalues is None:
        eigenvalues = compute_eigenvalues(alpha, H0, N=N_terms)
    eigs = eigenvalues[:N_terms]

    # Specific storage coefficient in k* space: μ = (θ_s − θ_r) / (ks · α)
    mu = (theta_s - theta_r) / (ks * alpha)

    # New steady-state k*(z*) with flux q1
    k1_star = _k_steady(z_star, H0, L1_vert, L2_vert, alpha, psi_0, q1, ks, phi, T, architecture)

    if t == 0.0:
        # At t=0 return the initial Eq. (19) steady state directly
        k0_star = _k_steady(z_star, H0, L1_vert, L2_vert, alpha, psi_0, q0, ks, phi, T, architecture)
        k_star = k0_star
    else:
        coeffs = _series_coefficients(alpha, H0, q0, q1, ks, eigs)
        k_star = k1_star
        for omega, C_n in zip(eigs, coeffs):
            beta_n = (omega**2 + alpha**2 / 4.0) / mu
            phi_n  = np.exp(-alpha * z_star / 2.0) * np.sin(omega * z_star)
            k_star += C_n * phi_n * np.exp(-beta_n * t)

    if k_star <= 0.0:
        warnings.warn(
            f"k* = {k_star:.3e} ≤ 0 at z* = {z_star:.3f} m, t = {t:.0f} s "
            "(model breakdown). Returning NaN.",
            RuntimeWarning, stacklevel=2
        )
        return np.nan

    if k_star > 1.0 + 1e-9:
        # Physically saturated: positive pore pressure — signal with NaN
        return np.nan

    return (gamma_w / alpha) * np.log(k_star)


# ═══════════════════════════════════════════════════════════════════════════
# PLOT
# ═══════════════════════════════════════════════════════════════════════════

def plot_transient_profiles(times_hours, H0, L1_vert, L2_vert,
                            alpha, psi_0, q0, q1, ks, phi,
                            theta_s, theta_r, T, architecture,
                            gamma_w=10.0, n_points=80, N_terms=50):
    """
    Plot transient uw vs vertical depth from surface for one root architecture.

    Parameters
    ----------
    times_hours  : list of float   Time snapshots [hours since rainfall]
    (all other parameters as in pore_water_pressure_eq25)

    Returns
    -------
    matplotlib.figure.Figure
    """
    eigs   = compute_eigenvalues(alpha, H0, N=N_terms)
    z_arr  = np.linspace(0.0, H0, n_points)
    depth  = H0 - z_arr          # vertical depth from surface

    fig, ax = plt.subplots(figsize=(7, 9))
    cmap    = plt.cm.plasma
    t_max   = max(times_hours) if max(times_hours) > 0 else 1.0

    for t_hr in times_hours:
        t_sec  = t_hr * 3600.0
        uw_arr = np.full(n_points, np.nan)
        for i, zs in enumerate(z_arr):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                uw_arr[i] = pore_water_pressure_eq25(
                    zs, t_sec, H0, L1_vert, L2_vert,
                    alpha, psi_0, q0, q1, ks, phi,
                    theta_s, theta_r, T, architecture,
                    gamma_w=gamma_w, eigenvalues=eigs, N_terms=N_terms
                )
        color = cmap(t_hr / t_max)
        ax.plot(uw_arr, depth, linewidth=2.0, color=color,
                label=f"t = {int(t_hr)} h")

    ax.axhspan(0.0, L2_vert, alpha=0.12, color="green",
               label=f"Root zone  (0 – {L2_vert*100:.0f} cm from surface)")
    ax.axhline(L2_vert, color="green", linestyle="--", linewidth=1.0)
    ax.axvline(0, color="gray",  linestyle=":",  linewidth=1.0)

    phi_deg = np.degrees(phi)
    ax.invert_yaxis()
    ax.set_xlabel("Pore water pressure  $u_w$  [kPa]", fontsize=12)
    ax.set_ylabel("Vertical depth from surface  [m]",  fontsize=12)
    ax.set_title(
        "Transient Pore Water Pressure  —  Ng et al. (2015) Eq. (25)\n"
        f"{architecture.capitalize()} architecture  |  "
        f"$\\phi$ = {phi_deg:.0f}°,  $\\alpha$ = {alpha} m⁻¹,  "
        f"$k_s$ = {ks:.2e} m/s\n"
        f"$q_1$ = {q1*86400*1e3:.1f} mm/day,  "
        f"$T$ = {T*86400*1e3:.2f} mm/day",
        fontsize=10
    )
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE  (reproduces parametric verification against Ng et al. 2015 Fig. 9)
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Soil
    ks      = 2.2e-6
    alpha   = 1.1

    # Geometry
    phi_deg = 40.0
    H0      = 5.0
    L2_vert = 0.30
    L1_vert = H0 - L2_vert

    # Boundary conditions
    psi_0   = 0.0
    q0      = 0.0                            # pre-rain: no flux
    q1      = 181e-3 / 86400.0              # rainfall: 181 mm/day

    # Vegetation
    T       = 6.6e-3 / 86400.0             # 6.6 mm/day

    # Water content
    theta_s = 0.45
    theta_r = 0.05

    # Output
    gamma_w = 10.0
    arch    = "triangular"

    phi     = np.radians(phi_deg)

    print("=" * 62)
    print("  Ng et al. (2015) Eq. (25) — Transient Pore Water Pressure")
    print("=" * 62)
    print(f"  ks={ks:.2e} m/s | alpha={alpha} | phi={phi_deg}° | H0={H0} m")
    print(f"  q0={q0} m/s | q1={q1*86400*1e3:.1f} mm/day | T={T*86400*1e3} mm/day")
    print(f"  theta_s={theta_s} | theta_r={theta_r}")
    print()

    print("  Computing eigenvalues ...", end="  ", flush=True)
    eigs = compute_eigenvalues(alpha, H0, N=50)
    print(f"done ({len(eigs)} found).")

    times_hours = [0, 10, 25, 50, 100]
    depths      = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                   0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

    col = 10
    hdr = f"  {'Depth (m)':>10}" + "".join(
        f"  {('t=' + str(t) + 'h'):>{col}}" for t in times_hours)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for d in depths:
        zs  = max(0.0, H0 - d)
        row = f"  {d:>10.2f}"
        for t_hr in times_hours:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                uw = pore_water_pressure_eq25(
                    zs, t_hr * 3600.0, H0, L1_vert, L2_vert,
                    alpha, psi_0, q0, q1, ks, phi,
                    theta_s, theta_r, T, arch,
                    gamma_w=gamma_w, eigenvalues=eigs
                )
            row += f"  {uw:>{col}.2f}" if not np.isnan(uw) else f"  {'sat.':>{col}}"
        print(row)

    print()
    print("  kPa | Negative = suction | 'sat.' = fully saturated")
