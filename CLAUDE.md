# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Script

```bash
# Run the pore water pressure calculation and generate the plot
python ng2015_pore_water_pressure.py
```

This produces a table of `uw` values printed to stdout and saves a plot to `ng2015_eq19_pore_pressure.png` in the current directory.

## Architecture

The repository contains a single self-contained Python script implementing **Ng et al. (2015) Eq. (19)** — an analytical solution for steady-state pore water pressure in a vegetated unsaturated infinite slope.

**Key equations and their implementations:**

| Equation | Function | Description |
|---|---|---|
| Eq. (4) | `S_triangular`, `S_parabolic`, `S_uniform`, `S_exponential` | Root water uptake sink term S(z') for each root architecture |
| Eq. (18) | `green_function(z_star, x_star, alpha)` | Green function G(z*, x*) for the flow ODE |
| Eq. (17) | `root_uptake_term(...)` | Numerical integration of Term 3 (vegetation effect on k0*) |
| Eq. (19) | `pore_water_pressure_eq19(...)` | Full uw calculation combining all three terms |

**Coordinate system:**
- `z*` = transformed depth (vertical depth × cos(φ)), 0 at water table, increases toward surface
- `z'` = perpendicular depth from water table; `z' = z* / cos(φ)`
- `depth_from_surface` = `H0 - z*` (used for plotting)

**Call chain:**
`plot_profiles` → `compute_uw_profile` → `pore_water_pressure_eq19` → `root_uptake_term` → `green_function` + `S_*()`

**k0* validity:** The linearized Gardner model breaks down (k0* ≤ 0) when `T/ks` is not small relative to `exp(alpha * H0)`. The code returns `np.nan` at affected depths with a `RuntimeWarning`. To avoid this, keep `T/ks < ~0.005` for CDG soils with `alpha=1.1, H0=5 m`.

## Dependencies

- `numpy`, `scipy` (for `quad` numerical integration), `matplotlib`
