# About

Scihub Boltzmann is a Python research workspace for two-dimensional lattice Boltzmann method simulations. The current implementation focuses on low-Mach incompressible flow with the D2Q9 lattice and the single-relaxation-time LBGK collision model.

The project is designed for reproducible research rather than maximum raw speed. It keeps numerical kernels, boundary conditions, configuration, desktop UI, post-processing, tests, and run outputs separated so that simulation cases can be extended without turning the repository into a pile of one-off scripts.

## Goals

- Provide a complete, runnable D2Q9-BGK solver in Python.
- Support common 2D benchmark cases such as lid-driven cavity, channel flow, Couette flow, periodic channel flow, and obstacle flow.
- Offer a desktop UI for configuring, running, monitoring, and saving simulations.
- Track numerical stability indicators such as `tau`, `omega`, Mach number, residual, convergence factors, and mass drift.
- Keep public code, local data, logs, databases, and generated results clearly separated.

## Current Scope

Implemented in the first public version:

- D2Q9 lattice constants, weights, opposite directions, equilibrium distribution, collision, streaming, and macroscopic variables.
- BGK collision model.
- Core boundary conditions: periodic, no-slip bounce-back, moving-wall bounce-back, non-equilibrium extrapolation, and fully developed outlet.
- Solid masks for cylinder and rectangular obstacles.
- Command-line runner.
- PySide6 desktop UI.
- Optional Streamlit UI retained as a legacy entry point.
- Result export to `npz`, `csv`, `json`, and figures.
- Smoke tests for solver, database setup, and core validation.

Planned or reserved for future versions:

- Zou-He velocity and pressure boundaries.
- Symmetry, specular reflection, mixed bounce/specular reflection, and mass-corrected outlet.
- Physical-unit conversion workflows.
- Animation export.
- More robust high-Re models such as TRT or MRT.

## Non-Goals

- This project does not claim to support arbitrary complex geometry. The first version supports rectangular domains and simple solid masks.
- This project is not yet optimized for large-scale production CFD.
- Generated simulation results are not stored in Git by default.

## License

The project is released under the MIT License. See `LICENSE`.

