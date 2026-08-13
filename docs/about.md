# About

Scihub Boltzmann is a Python research workspace for two-dimensional lattice Boltzmann method simulations. The current implementation focuses on low-Mach incompressible and Boussinesq thermal flow, using D2Q9 for momentum and D2Q5 for temperature.

The project is designed for reproducible research rather than maximum raw speed. It keeps numerical kernels, boundary conditions, configuration, desktop UI, post-processing, tests, and run outputs separated so that simulation cases can be extended without turning the repository into a pile of one-off scripts.

## Goals

- Provide a complete, runnable D2Q9 solver with selectable BGK, TRT, and MRT collision operators.
- Support common 2D benchmark cases such as lid-driven cavity, channel flow, Couette flow, periodic channel flow, and obstacle flow.
- Offer a desktop UI for configuring, running, monitoring, and saving simulations.
- Track numerical stability indicators such as `tau`, `omega`, Mach number, residual, convergence factors, and mass drift.
- Keep public code, local data, logs, databases, and generated results clearly separated.

## Current Scope

Implemented in the first public version:

- D2Q9 lattice constants, weights, opposite directions, equilibrium distribution, collision, streaming, and macroscopic variables.
- BGK single-relaxation-time collision model.
- TRT symmetric/antisymmetric collision model with configurable magic parameter `Lambda`.
- MRT moment-space collision model with configurable non-conserved relaxation rates.
- Reynolds-derived, direct-`tau`, physical-unit conversion, and Rayleigh/Prandtl parameter modes.
- D2Q9+D2Q5 double-distribution thermal flow with isothermal, adiabatic, periodic, and outlet thermal boundaries.
- Natural-convection cavity, Rayleigh-Benard convection, and heated periodic-channel presets.
- Two-dimensional Guo forcing with configurable startup profiles.
- Runtime safety limits for residual, velocity, and mass drift.
- Core boundary conditions: periodic, no-slip bounce-back, moving-wall bounce-back, non-equilibrium extrapolation, fully developed outlet, specular reflection, and mixed bounce/specular reflection.
- Solid masks for cylinder and rectangular obstacles.
- Command-line runner.
- PySide6 desktop UI.
- Optional Streamlit UI retained as a legacy entry point.
- Result export to `npz`, `csv`, `json`, and figures.
- Smoke tests for solver, database setup, and core validation.

Planned or reserved for future versions:

- Zou-He velocity and pressure boundaries.
- Symmetry and mass-corrected outlet boundaries.
- Animation export.
- Additional collision families such as regularized, central-moment, or entropic LBM.

## Non-Goals

- This project does not claim to support arbitrary complex geometry. The first version supports rectangular domains and simple solid masks.
- This project is not yet optimized for large-scale production CFD.
- Generated simulation results are not stored in Git by default.

## License

The project is released under the MIT License. See `LICENSE`.

## Numerical References

- P. Lallemand and L.-S. Luo, "Theory of the Lattice Boltzmann Method: Dispersion, Dissipation, Isotropy, Galilean Invariance, and Stability," *Physical Review E* 61 (2000), DOI: [10.1103/PhysRevE.61.6546](https://doi.org/10.1103/PhysRevE.61.6546).
- I. Ginzburg, F. Verhaeghe, and D. d'Humieres, "Two-Relaxation-Time Lattice Boltzmann Scheme: About Parametrization, Velocity, Pressure and Mixed Boundary Conditions," *Communications in Computational Physics* 3 (2008), [paper PDF](https://doc.global-sci.org/uploads/admin/article_pdf/20200723/3f11f5c9e96890cca5fcc3c60f5e84c0.pdf).
- Z. Guo, C. Zheng, and B. Shi, "Discrete Lattice Effects on the Forcing Term in the Lattice Boltzmann Method," *Physical Review E* 65 (2002), DOI: [10.1103/PhysRevE.65.046308](https://doi.org/10.1103/PhysRevE.65.046308).
- A. A. Mohamad and R. Bennacer, "Simulation of High Rayleigh Number Natural Convection in a Square Cavity Using the Lattice Boltzmann Method," *International Journal of Heat and Mass Transfer* 49 (2006), DOI: [10.1016/j.ijheatmasstransfer.2005.07.046](https://doi.org/10.1016/j.ijheatmasstransfer.2005.07.046).
- Y. Chen, H. Ohashi, and M. Akiyama, "Simulation of Laminar Flow over a Backward-Facing Step Using the Lattice BGK Method," *JSME International Journal Series B* 40 (1997), DOI: [10.1299/jsmeb.40.25](https://doi.org/10.1299/jsmeb.40.25).
