"""
run_pore_pressure.py
====================
Interactive terminal interface for the Ng et al. (2015) pore water pressure
models — both steady-state (Eq. 19) and transient (Eq. 25).

HOW TO USE
----------
1. Put this file in the SAME FOLDER as:
       ng2015_eq19_pore_pressure.py
       ng2015_eq25_transient.py
2. Open a terminal in that folder and run:
       python run_pore_pressure.py
3. Answer each prompt — press Enter to accept the default shown in [ ]
4. A results table is printed and a plot is saved to the same folder

REQUIREMENTS
------------
    pip install numpy scipy matplotlib
"""
import sys
import os
import warnings
import numpy as np

# ─── Check for interactive terminal ──────────────────────────────────────────
if not sys.stdin.isatty():
    print()
    print("  This script requires an interactive terminal.")
    print()
    print("  In VSCode:  open the integrated terminal  (Ctrl+` or View > Terminal)")
    print("  then run:   python run_pore_pressure.py")
    print()
    print("  The Run button does not work because the script uses input() prompts.")
    print()
    sys.exit(0)

# ─── Import steady-state model ───────────────────────────────────────────────
try:
    from ng2015_eq19_pore_pressure import (
        pore_water_pressure_eq19,
        plot_profiles as plot_steady,
    )
except ImportError:
    print()
    print("  ERROR: Could not find 'ng2015_eq19_pore_pressure.py'")
    print("  Make sure it is in the same folder as this script.")
    print()
    sys.exit(1)

# ─── Import transient model ───────────────────────────────────────────────────
try:
    from ng2015_eq25_transient import (
        pore_water_pressure_eq25,
        compute_eigenvalues,
        plot_transient_profiles,
    )
except ImportError:
    print()
    print("  ERROR: Could not find 'ng2015_eq25_transient.py'")
    print("  Make sure it is in the same folder as this script.")
    print()
    sys.exit(1)

# =============================================================================
# INPUT HELPERS
# =============================================================================

def ask_float(prompt, default, min_val=None, max_val=None):
    """Prompt for a number with a default; repeats until valid."""
    while True:
        raw = input(f"  {prompt} [{default}]: ").strip()
        if raw == "":
            return float(default)
        try:
            value = float(raw)
        except ValueError:
            print(f"    x  '{raw}' is not a number. Please try again.")
            continue
        if min_val is not None and value < min_val:
            print(f"    x  Value must be >= {min_val}. Please try again.")
            continue
        if max_val is not None and value > max_val:
            print(f"    x  Value must be <= {max_val}. Please try again.")
            continue
        return value


def ask_choice(prompt, options, default):
    """Prompt for a single choice from a numbered list."""
    print(f"  {prompt}")
    for i, opt in enumerate(options, start=1):
        marker = "  <-- default" if opt == default else ""
        print(f"    {i}. {opt}{marker}")
    while True:
        raw = input(f"  Enter number [default: {default}]: ").strip()
        if raw == "":
            return default
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
            print(f"    x  Please enter a number between 1 and {len(options)}.")
        except ValueError:
            print("    x  Please enter a number.")


def ask_multi_choice(prompt, options):
    """Prompt for one or more choices from a numbered list."""
    print(f"  {prompt}")
    for i, opt in enumerate(options, start=1):
        print(f"    {i}. {opt}")
    print("    (Enter numbers separated by commas, e.g. 1,2  -- or press Enter for all)")
    while True:
        raw = input("  Your selection: ").strip()
        if raw == "":
            return list(options)
        try:
            indices = [int(x.strip()) - 1 for x in raw.split(",")]
            if all(0 <= idx < len(options) for idx in indices):
                return [options[idx] for idx in indices]
            print(f"    x  Numbers must be between 1 and {len(options)}.")
        except ValueError:
            print("    x  Please enter numbers separated by commas.")


def ask_time_list(default_str):
    """Prompt for a comma-separated list of times in hours."""
    print("  Enter hours separated by commas, e.g.  0, 10, 50, 100")
    while True:
        raw = input(f"  Times (hours) [{default_str}]: ").strip()
        if raw == "":
            raw = default_str
        try:
            times = sorted(set(float(x.strip()) for x in raw.split(",")))
            if all(t >= 0 for t in times):
                return times
            print("    x  All times must be >= 0.")
        except ValueError:
            print("    x  Please enter numbers separated by commas.")


def divider(title, n, total):
    print()
    print("-" * 62)
    print(f"  SECTION {n} of {total} -- {title}")
    print("-" * 62)
    print()


# =============================================================================
# SHARED INPUT SECTIONS
# =============================================================================

def collect_soil(total_sections):
    divider("Soil Properties", 1, total_sections)
    print("  Saturated hydraulic conductivity ks  [m/s]")
    print("  Typical values:")
    print("    CDG silty sand : 2.2e-6")
    print("    Fine sand      : 2.7e-4")
    print("    Silt           : 9.0e-7")
    ks = ask_float("ks (m/s)", default=2.2e-6, min_val=1e-12)
    print()
    print("  Gardner desaturation coefficient alpha  [1/m]")
    print("  Controls how fast permeability drops with suction.")
    print("  Typical CDG value: 1.1")
    alpha = ask_float("alpha (1/m)", default=1.1, min_val=0.001)
    return ks, alpha


def collect_geometry(total_sections, section_n):
    divider("Slope Geometry", section_n, total_sections)
    phi_deg = ask_float("Slope angle (degrees)", default=40.0,
                        min_val=1.0, max_val=89.0)
    print()
    H0 = ask_float("Vertical depth to water table H0 (m)",
                   default=5.0, min_val=0.1)
    print()
    print("  Root zone vertical depth L2  [m]")
    print("  Measured from the slope surface downward.")
    print(f"  Must be less than H0 = {H0} m")
    L2_vert = ask_float("Root zone depth L2 (m)", default=0.30,
                        min_val=0.01, max_val=H0 - 0.01)
    L1_vert = H0 - L2_vert
    print(f"    -> Non-rooted zone L1 = {H0} - {L2_vert} = {L1_vert:.3f} m")
    return phi_deg, H0, L1_vert, L2_vert


def collect_vegetation(ks, alpha, H0, total_sections, section_n):
    divider("Boundary Conditions and Vegetation", section_n, total_sections)
    print("  Pressure head at water table psi_0  [m]")
    print("  Use 0 for a free (atmospheric) water table.")
    psi_0 = ask_float("psi_0 (m)", default=0.0)
    print()
    print("  Pre-rainfall surface flux q0  [m/s]")
    print("  Sets the initial steady-state condition before any rainfall.")
    print("  0 = no flux (dry season)  |  positive = infiltration  |  negative = evaporation")
    q0 = ask_float("q0 (m/s)", default=0.0)
    print()
    print("  Transpiration rate T  [mm/day]")
    print("  Typical value: 6.6 mm/day  (set to 0 for bare slope)")
    T_mmday = ask_float("T (mm/day)", default=6.6, min_val=0.0)
    T = T_mmday * 1e-3 / 86400.0
    print()
    T_over_ks = T / ks
    if T_over_ks > 0.01:
        print(f"  WARNING: T/ks = {T_over_ks:.4f}  (> 0.01)")
        print(f"  With alpha x H0 = {alpha*H0:.1f}, the model may give invalid results")
        print(f"  (k0* <= 0) near the surface. Check for 'NaN' in output.")
    else:
        print(f"  OK: T/ks = {T_over_ks:.5f}  (within valid range)")
    return psi_0, q0, T_mmday, T


def collect_arch_and_output(default_plot, total_sections, section_n):
    divider("Root Architecture and Output", section_n, total_sections)
    arch_options = ["triangular", "parabolic", "uniform", "exponential"]
    archs = ask_multi_choice(
        "Which root architectures to compute?",
        arch_options
    )
    print()
    gamma_w = ask_float("Unit weight of water gamma_w (kN/m3)",
                        default=10.0, min_val=9.0, max_val=11.0)
    print()
    raw = input(f"  Save plot as [{default_plot}]: ").strip()
    plot_filename = raw if raw else default_plot
    return archs, gamma_w, plot_filename


# =============================================================================
# STEADY-STATE RUNNER  (Eq. 19)
# =============================================================================

def run_steady():
    print()
    print("=" * 62)
    print("  Ng et al. (2015) -- STEADY-STATE Pore Water Pressure")
    print("  Infinite Vegetated Unsaturated Slope  --  Eq. (19)")
    print("=" * 62)
    print()
    print("  Press Enter at any prompt to accept the default shown in [ ].")

    N = 4
    ks, alpha              = collect_soil(N)
    phi_deg, H0, L1, L2   = collect_geometry(N, 2)
    psi_0, q0, T_mmday, T = collect_vegetation(ks, alpha, H0, N, 3)
    archs, gamma_w, fname  = collect_arch_and_output("steady_state_results.png", N, 4)

    phi     = np.radians(phi_deg)
    cos_phi = np.cos(phi)

    print()
    print("-" * 62)
    print("  COMPUTING  (Eq. 19 -- steady state) ...")
    print("-" * 62)
    print()
    print(f"  ks={ks:.2e} m/s | alpha={alpha} m^-1 | phi={phi_deg:.1f} deg | H0={H0} m")
    print(f"  L2={L2} m | L1={L1:.3f} m | psi_0={psi_0} m | q0={q0} m/s")
    print(f"  T={T_mmday} mm/day | gamma_w={gamma_w} kN/m3")
    print(f"  Architectures: {', '.join(archs)}")
    print()

    # Table
    depths_root  = list(np.linspace(0.0, L2, 7))
    depths_below = [d for d in [0.5, 1.0, 2.0, 3.0, 4.0, H0] if d > L2]
    depths       = sorted(set(round(d, 4) for d in depths_root + depths_below))

    col = 13
    hdr = f"  {'Depth (m)':>10}  {'z* (m)':>7}"
    for a in archs:
        hdr += f"  {a.capitalize():>{col}}"
    hdr += f"  {'No veg (T=0)':>{col}}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for d in depths:
        zs  = max(0.0, H0 - d)
        row = f"  {d:>10.3f}  {zs:>7.3f}"
        for a in archs:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                uw = pore_water_pressure_eq19(
                    zs, H0, L1, L2, alpha, psi_0, q0,
                    ks, phi, T, a, gamma_w
                )
            row += f"  {uw:>{col}.2f}" if not np.isnan(uw) else f"  {'NaN':>{col}}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            uw0 = pore_water_pressure_eq19(
                zs, H0, L1, L2, alpha, psi_0, q0,
                ks, phi, 0.0, "triangular", gamma_w
            )
        row += f"  {uw0:>{col}.2f}"
        print(row)

    print()
    print(f"  kPa | Negative = suction | NaN = model breakdown (reduce T)")
    print(f"  Root zone: 0 - {L2*100:.0f} cm from surface")
    print()

    # Plot
    print("  Generating plot ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = plot_steady(
            H0, L1, L2, alpha, psi_0, q0, ks, phi, T,
            architectures=archs, gamma_w=gamma_w, n_points=120
        )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path  = os.path.join(script_dir, fname)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved -> {save_path}")


# =============================================================================
# TRANSIENT RUNNER  (Eq. 25)
# =============================================================================

def run_transient():
    print()
    print("=" * 62)
    print("  Ng et al. (2015) -- TRANSIENT Pore Water Pressure")
    print("  Infinite Vegetated Unsaturated Slope  --  Eq. (25)")
    print("=" * 62)
    print()
    print("  The initial condition is the Eq. (19) steady state.")
    print("  Rainfall begins at t = 0 and continues at constant intensity.")
    print()
    print("  Press Enter at any prompt to accept the default shown in [ ].")

    N = 5
    ks, alpha              = collect_soil(N)
    phi_deg, H0, L1, L2   = collect_geometry(N, 2)
    psi_0, q0, T_mmday, T = collect_vegetation(ks, alpha, H0, N, 3)

    # Section 4: Transient-specific parameters
    divider("Rainfall and Water Content Parameters", 4, N)
    print("  Saturated volumetric water content theta_s  [-]")
    print("  Typical CDG: 0.45")
    theta_s = ask_float("theta_s", default=0.45, min_val=0.01, max_val=1.0)
    print()
    print("  Residual volumetric water content theta_r  [-]")
    print("  Typical CDG: 0.05")
    theta_r = ask_float("theta_r", default=0.05, min_val=0.0,
                        max_val=theta_s - 0.01)
    print()
    print("  Rainfall intensity q1  [mm/day]")
    print("  This is the surface flux once rainfall begins (t > 0).")
    print("  Paper verification value: 181 mm/day")
    print("  If q1 > ks (in same units), the slope will fully saturate near")
    print("  the surface -- 'sat.' will appear in the output table.")
    q1_mmday = ask_float("q1 (mm/day)", default=181.0, min_val=0.0)
    q1 = q1_mmday * 1e-3 / 86400.0
    print()
    q1_over_ks = q1 / ks
    if q1_over_ks > 1.0:
        print(f"  NOTE: q1/ks = {q1_over_ks:.3f}  (> 1.0)")
        print(f"  Rainfall exceeds permeability -- 'sat.' entries are expected.")
    else:
        print(f"  OK: q1/ks = {q1_over_ks:.4f}")
    print()
    print("  Time snapshots  [hours since rainfall started]")
    print("  t = 0 is always included (= the steady-state initial condition).")
    times_hours = ask_time_list(default_str="0, 10, 25, 50, 100")
    if 0.0 not in times_hours:
        times_hours = sorted([0.0] + times_hours)
    print(f"    -> Times: {', '.join(str(int(t)) + 'h' for t in times_hours)}")
    print()
    print("  Number of eigenvalue series terms N")
    print("  50 is sufficient for most practical rainfall durations.")
    N_terms = int(ask_float("N_terms", default=50, min_val=5, max_val=200))

    # Section 5: Architecture and output
    archs, gamma_w, fname = collect_arch_and_output("transient_results.png", N, 5)

    phi = np.radians(phi_deg)

    print()
    print("-" * 62)
    print("  COMPUTING  (Eq. 25 -- transient) ...")
    print("-" * 62)
    print()
    print(f"  ks={ks:.2e} m/s | alpha={alpha} m^-1 | phi={phi_deg:.1f} deg | H0={H0} m")
    print(f"  theta_s={theta_s} | theta_r={theta_r}")
    print(f"  q0={q0} m/s | q1={q1_mmday} mm/day | T={T_mmday} mm/day")
    print(f"  Times: {[str(int(t))+'h' for t in times_hours]}")
    print(f"  Architectures: {', '.join(archs)}")
    print()

    # Pre-compute eigenvalues once (reused for all times and architectures)
    print("  Computing eigenvalues ...", end="  ", flush=True)
    eigs = compute_eigenvalues(alpha, H0, N=N_terms)
    print(f"done ({len(eigs)} found).")
    print()

    # Table
    depths_root  = list(np.linspace(0.0, L2, 5))
    depths_below = [d for d in [0.5, 1.0, 2.0, 3.0, 4.0, H0] if d > L2]
    depths       = sorted(set(round(d, 4) for d in depths_root + depths_below))

    col = 10
    for arch in archs:
        print(f"  -- {arch.capitalize()} root architecture --")
        hdr = f"  {'Depth (m)':>10}"
        for t_hr in times_hours:
            hdr += f"  {('t=' + str(int(t_hr)) + 'h'):>{col}}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for d in depths:
            zs  = max(0.0, H0 - d)
            row = f"  {d:>10.3f}"
            for t_hr in times_hours:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    uw = pore_water_pressure_eq25(
                        zs, t_hr * 3600.0, H0, L1, L2,
                        alpha, psi_0, q0, q1, ks, phi,
                        theta_s, theta_r, T, arch,
                        gamma_w=gamma_w,
                        eigenvalues=eigs,
                        N_terms=N_terms
                    )
                if np.isnan(uw):
                    row += f"  {'sat.':>{col}}"
                else:
                    row += f"  {uw:>{col}.2f}"
            print(row)
        print()

    print(f"  kPa | Negative = suction | 'sat.' = fully saturated at that depth/time")
    print(f"  Root zone: 0 - {L2*100:.0f} cm from surface")
    print()

    # Plots
    print("  Generating plot(s) ...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base, ext  = os.path.splitext(fname)
    if not ext:
        ext = ".png"
    saved = []
    for arch in archs:
        fig = plot_transient_profiles(
            times_hours, H0, L1, L2,
            alpha, psi_0, q0, q1, ks, phi,
            theta_s, theta_r, T, arch,
            gamma_w=gamma_w, n_points=120, N_terms=N_terms
        )
        suffix    = f"_{arch}" if len(archs) > 1 else ""
        save_path = os.path.join(script_dir, f"{base}{suffix}{ext}")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        saved.append(save_path)
    for p in saved:
        print(f"  Plot saved -> {p}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run():
    print()
    print("=" * 62)
    print("  Ng et al. (2015) -- Pore Water Pressure Model")
    print("  Vegetated Infinite Unsaturated Slope")
    print("=" * 62)
    print()
    mode = ask_choice(
        "Which analysis would you like to run?",
        options=[
            "Steady state  (Eq. 19) -- dry-season or long-term equilibrium",
            "Transient     (Eq. 25) -- rainfall event from steady-state initial condition",
        ],
        default="Steady state  (Eq. 19) -- dry-season or long-term equilibrium"
    )
    print()
    if mode.startswith("Steady"):
        run_steady()
    else:
        run_transient()

    print()
    again = input("  Run again with different parameters? (y/n) [n]: ").strip().lower()
    if again == "y":
        run()
    else:
        print()
        print("  Done. Goodbye!")
        print()


if __name__ == "__main__":
    run()
