"""
Root Architecture Sink Term  S(z')  —  Ng et al. (2015) Eq. (4)
================================================================
Implements the water uptake sink term for the triangular and parabolic
root architectures defined in Eq. (4) of:

    Ng, C.W.W., Ni, J.J., Leung, A.K., Wang, Z.J. (2015).
    Analytical solutions for calculating pore-water pressure in an infinite
    unsaturated slope with different root architectures.
    Can. Geotech. J. 52(12):1980-1993.

Equation (4):
-------------
    Triangular:  S(z') = 2T * (z' - L1') / L2'^2

    Parabolic:   S(z') = 6T * (z' - L1') * (L2' - (z' - L1')) / L2'^3

Both are only active inside the root zone:  L1' <= z' <= L1' + L2'
Outside the root zone, S = 0  (enforced by the Heaviside function H, Eq. 3).

Coordinate convention (Fig. 3 of Ng et al. 2015):
    z' = perpendicular depth FROM the water table TO the slope surface.
    z' = 0         : water table (bottom)
    z' = L1'       : base of root zone (start of roots)
    z' = L1' + L2' : slope surface (top of root zone)

Parameters used here:
    L1' = 2.0  m    perpendicular depth of the non-rooted zone (outside root zone)
    L2' = 0.30 m    perpendicular depth of the root zone
    T   = 6.6 mm/day  transpiration rate (converted to m/s in calculations)

Key property — both architectures integrate to the same total transpiration T:
    INT_{L1'}^{L1'+L2'} S(z') dz' = T   (verified numerically below)
"""

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for pkg in ["numpy", "scipy", "matplotlib"]:
    try:
        __import__(pkg)
    except ImportError:
        print(f"Installing {pkg}...")
        install(pkg)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import numpy as np
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

L1_perp = 2.00        # perpendicular depth outside root zone  [m]
L2_perp = 0.30        # perpendicular root depth               [m]
T_mmday = 6.6         # transpiration rate                     [mm/day]
T       = T_mmday * 1e-3 / 86400.0   # convert to [m/s]

# Root zone spans:  z' in [L1', L1' + L2']
z_root_base = L1_perp              # bottom of root zone  [m]
z_root_top  = L1_perp + L2_perp   # top of root zone (= slope surface)  [m]


# ---------------------------------------------------------------------------
# Eq. (4) — Sink term functions
# ---------------------------------------------------------------------------

def S_triangular(z_perp, L1_perp, L2_perp, T):
    """
    Triangular root architecture sink term  [1/s]  (Eq. 4, Ng et al. 2015).

    S(z') = 2T * (z' - L1') / L2'^2   inside root zone
    S(z') = 0                          outside root zone

    Root density increases linearly from zero at the root zone base (z' = L1')
    to a maximum at the slope surface (z' = L1' + L2').

    Parameters
    ----------
    z_perp  : float or array, perpendicular depth z' [m]
    L1_perp : float, perpendicular depth of non-rooted zone [m]
    L2_perp : float, perpendicular root depth [m]
    T       : float, transpiration rate [m/s]

    Returns
    -------
    S : float or array  [1/s]
    """
    z_perp = np.asarray(z_perp, dtype=float)
    inside = (z_perp >= L1_perp) & (z_perp <= L1_perp + L2_perp)
    S = np.zeros_like(z_perp)
    u = z_perp[inside] - L1_perp          # offset from root zone base [m]
    S[inside] = 2.0 * T * u / L2_perp**2
    return S


def S_parabolic(z_perp, L1_perp, L2_perp, T):
    """
    Parabolic root architecture sink term  [1/s]  (Eq. 4, Ng et al. 2015).

    S(z') = 6T * (z' - L1') * (L2' - (z' - L1')) / L2'^3   inside root zone
    S(z') = 0                                                 outside root zone

    Root density follows a parabolic profile — zero at both the root zone base
    (z' = L1') and the slope surface (z' = L1' + L2'), with maximum at mid-depth.
    This matches the parabolic distribution observed by Leung et al. (2015)
    for vegetation in Hong Kong.

    Parameters
    ----------
    z_perp  : float or array, perpendicular depth z' [m]
    L1_perp : float, perpendicular depth of non-rooted zone [m]
    L2_perp : float, perpendicular root depth [m]
    T       : float, transpiration rate [m/s]

    Returns
    -------
    S : float or array  [1/s]
    """
    z_perp = np.asarray(z_perp, dtype=float)
    inside = (z_perp >= L1_perp) & (z_perp <= L1_perp + L2_perp)
    S = np.zeros_like(z_perp)
    u = z_perp[inside] - L1_perp          # offset from root zone base [m]
    S[inside] = 6.0 * T * u * (L2_perp - u) / L2_perp**3
    return S


# ---------------------------------------------------------------------------
# Verification: integrate each S over root zone — should equal T
# ---------------------------------------------------------------------------

def verify_integration(L1_perp, L2_perp, T):
    """Numerically integrate each sink term and check it equals T."""
    T_tri, _ = quad(S_triangular,  L1_perp, L1_perp + L2_perp,
                    args=(L1_perp, L2_perp, T))
    T_par, _ = quad(S_parabolic,   L1_perp, L1_perp + L2_perp,
                    args=(L1_perp, L2_perp, T))
    return T_tri, T_par


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_sink_profiles(L1_perp, L2_perp, T, n_points=400):
    """
    Plot S(z') vs perpendicular depth z' for both architectures.
    Depth axis is displayed with slope surface at top (mirroring Fig. 6
    of Ng et al. 2015 — depth increasing downward).
    """
    z_total = L1_perp + L2_perp          # total perpendicular depth to surface
    z_arr   = np.linspace(0, z_total, n_points)

    S_tri = S_triangular(z_arr, L1_perp, L2_perp, T)
    S_par = S_parabolic( z_arr, L1_perp, L2_perp, T)

    # Convert to 1/day for readable axis labels
    S_tri_day = S_tri * 86400
    S_par_day = S_par * 86400

    # Depth from surface (for conventional geotechnical plot)
    depth = z_total - z_arr

    fig, ax = plt.subplots(figsize=(6, 8))

    ax.plot(S_tri_day, depth, color="darkorange",  linewidth=2.5,
            label="Triangular")
    ax.plot(S_par_day, depth, color="steelblue",   linewidth=2.5,
            label="Parabolic")

    # Shade root zone
    ax.axhspan(0, L2_perp, alpha=0.10, color="green",
               label=f"Root zone  (0 – {L2_perp*100:.0f} cm from surface)")
    ax.axhline(L2_perp, color="green", linestyle="--", linewidth=1.0)

    # Annotations
    ax.set_xlabel("Sink term  $S(z')$  [day$^{-1}$]", fontsize=12)
    ax.set_ylabel("Perpendicular depth from surface  [m]", fontsize=12)
    ax.set_title(
        f"Root Water Uptake Sink Term  —  Ng et al. (2015) Eq. (4)\n"
        f"$L_1'$ = {L1_perp:.2f} m,  $L_2'$ = {L2_perp*100:.0f} cm,  "
        f"$T$ = {T*86400*1e3:.1f} mm/day",
        fontsize=11
    )
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 55)
    print("Ng et al. (2015) Eq. (4) — Root Sink Term S(z')")
    print("=" * 55)
    print(f"  L1' (outside root zone depth) : {L1_perp:.2f} m")
    print(f"  L2' (root zone depth)         : {L2_perp:.2f} m  ({L2_perp*100:.0f} cm)")
    print(f"  T   (transpiration rate)      : {T_mmday} mm/day  = {T:.3e} m/s")
    print(f"  Root zone                     : z' in [{L1_perp:.2f}, {z_root_top:.2f}] m")
    print()

    # --- Verify integrals equal T ---
    T_tri, T_par = verify_integration(L1_perp, L2_perp, T)
    print("Integration check (should equal T = {:.3e} m/s):".format(T))
    print(f"  Triangular  ∫S dz' = {T_tri:.6e} m/s   "
          f"{'✓' if abs(T_tri - T)/T < 1e-6 else '✗'}")
    print(f"  Parabolic   ∫S dz' = {T_par:.6e} m/s   "
          f"{'✓' if abs(T_par - T)/T < 1e-6 else '✗'}")
    print()

    # --- Tabulate S at key depths within root zone ---
    print(f"  S(z') values inside root zone  [units: day⁻¹]:")
    print(f"  {'z_perp (m)':>12}  {'Depth from surf (cm)':>22}  "
          f"{'Triangular':>12}  {'Parabolic':>12}")
    print("  " + "-" * 62)

    # Sample 7 evenly-spaced points across the root zone
    z_sample = np.linspace(L1_perp, z_root_top, 7)
    for z in z_sample:
        d_surf_cm = (z_root_top - z) * 100   # depth from surface in cm
        s_tri = S_triangular(z, L1_perp, L2_perp, T) * 86400
        s_par = S_parabolic( z, L1_perp, L2_perp, T) * 86400
        print(f"  {z:>10.4f}  {d_surf_cm:>22.1f}  {s_tri:>12.6f}  {s_par:>12.6f}")
    print()

    # Check S=0 outside root zone
    z_outside = [0.5, 1.0, 1.5, L1_perp - 0.001]
    print("  S = 0 outside root zone check:")
    for z in z_outside:
        s_tri = S_triangular(z, L1_perp, L2_perp, T)
        s_par = S_parabolic( z, L1_perp, L2_perp, T)
        print(f"    z' = {z:.3f} m  ->  S_tri = {s_tri:.2e}, S_par = {s_par:.2e}")
    print()

    # --- Plot ---
    fig = plot_sink_profiles(L1_perp, L2_perp, T)
    fig.savefig("sink_term_eq4.png", dpi=150, bbox_inches="tight")
    print("  Plot saved to sink_term_eq4.png")
