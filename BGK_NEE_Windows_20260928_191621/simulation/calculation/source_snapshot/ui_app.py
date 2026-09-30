"""Streamlit front end for the 2D LBM simulator."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
import io
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from config import (
    BOUNDARY_TYPES,
    BoundaryConfig,
    ConvergenceConfig,
    FlowConfig,
    GridConfig,
    OutputConfig,
    SimulationConfig,
    default_boundaries_for_case,
    recommend_parameters,
    validate_config,
)
from lbm_solver import LBMSolver
from postprocess import speed, vorticity


st.set_page_config(page_title="2D LBM Simulator", layout="wide")


def _session_defaults() -> None:
    st.session_state.setdefault("running", False)
    st.session_state.setdefault("stop_requested", False)
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("recommended_u", 0.05)


def _boundary_editor(side: str, default: BoundaryConfig) -> BoundaryConfig:
    with st.container(border=True):
        st.caption(side)
        bc_type = st.selectbox(
            "type",
            BOUNDARY_TYPES,
            index=BOUNDARY_TYPES.index(default.type) if default.type in BOUNDARY_TYPES else 0,
            key=f"{side}_type",
        )
        cols = st.columns(4)
        ux = cols[0].number_input("ux", value=float(default.ux), format="%.6f", key=f"{side}_ux")
        uy = cols[1].number_input("uy", value=float(default.uy), format="%.6f", key=f"{side}_uy")
        rho = cols[2].number_input("rho", min_value=1.0e-9, value=float(default.rho), format="%.6f", key=f"{side}_rho")
        rb = cols[3].slider("rb", min_value=0.0, max_value=1.0, value=float(default.rb), key=f"{side}_rb")
    return BoundaryConfig(type=bc_type, ux=ux, uy=uy, rho=rho, rb=rb)


def _uploaded_mask(uploaded, nx: int, ny: int) -> np.ndarray | None:
    if uploaded is None:
        return None
    try:
        import PIL.Image
    except Exception as exc:
        st.warning(f"Mask image upload needs Pillow: {exc}")
        return None
    image = PIL.Image.open(uploaded).convert("L").resize((nx, ny))
    arr = np.array(image)
    return arr < 128


def _plot_field(field: np.ndarray, title: str, solid_mask: np.ndarray, cmap: str = "viridis") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    im = ax.imshow(field, origin="lower", cmap=cmap, interpolation="nearest")
    if np.any(solid_mask):
        masked = np.ma.masked_where(~solid_mask, solid_mask)
        ax.imshow(masked, origin="lower", cmap="gray_r", vmin=0, vmax=1, alpha=0.85, interpolation="nearest")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return fig


def _history_chart(history: list[dict[str, float]]) -> None:
    if not history:
        return
    data = {
        "iteration": [row["iteration"] for row in history],
        "residual": [row["residual"] for row in history],
        "mass_drift": [row["mass_drift"] for row in history],
    }
    st.line_chart(data, x="iteration", y=["residual", "mass_drift"])


def _zip_output_dir(output_dir: str) -> bytes | None:
    root = Path(output_dir)
    if not root.exists():
        return None
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in root.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(root.parent))
    return buffer.getvalue()


def main() -> None:
    _session_defaults()
    st.title("2D LBM Simulator")

    with st.sidebar:
        st.subheader("模型")
        case_type = st.selectbox(
            "case_type",
            [
                "lid_driven_cavity",
                "poiseuille_channel",
                "couette_flow",
                "periodic_channel",
                "force_poiseuille",
                "cylinder_flow",
                "custom",
            ],
        )
        collision_model = st.selectbox("collision_model", ["BGK"])
        boundary_scheme_default = st.selectbox(
            "boundary_scheme_default",
            ["halfway_bounce_back", "non_equilibrium_extrapolation", "no_slip_bounce_back"],
        )

        st.subheader("网格")
        nx = st.number_input("NX", min_value=8, max_value=1024, value=128, step=8)
        ny = st.number_input("NY", min_value=8, max_value=1024, value=128, step=8)
        l_ref = st.number_input("L_ref (0 = automatic)", min_value=0.0, value=0.0, step=1.0)
        obstacle_type = st.selectbox("obstacle type", ["none", "circle", "rectangle", "uploaded_mask"], index=0)
        obstacle_x = st.slider("obstacle x", 0.0, 1.0, 0.33)
        obstacle_y = st.slider("obstacle y", 0.0, 1.0, 0.5)
        obstacle_radius = st.slider("obstacle radius", 0.01, 0.40, 0.08)
        obstacle_width = st.slider("obstacle width", 0.01, 0.80, 0.12)
        obstacle_height = st.slider("obstacle height", 0.01, 0.80, 0.20)
        mask_file = st.file_uploader("mask image", type=["png", "jpg", "jpeg"]) if obstacle_type == "uploaded_mask" else None

        st.subheader("流动参数")
        rho0 = st.number_input("rho0", min_value=1.0e-9, value=1.0, format="%.6f")
        re = st.number_input("Re", min_value=1.0e-9, value=1000.0, format="%.6f")
        u_ref = st.number_input("U_ref", min_value=0.0, value=float(st.session_state.recommended_u), format="%.6f")
        nu_explicit = st.number_input("nu_lattice (0 = derive from Re)", min_value=0.0, value=0.0, format="%.8f")
        force_x = st.number_input("ax lattice acceleration", value=0.0, format="%.10f")
        force_y = st.number_input("ay lattice acceleration", value=0.0, format="%.10f")
        initial_velocity = st.selectbox("initial velocity", ["rest", "uniform", "couette"])
        fixed_steps = st.checkbox("Fixed steps (transient)", value=False)
        steady_windows = st.number_input("consecutive steady windows", min_value=1, value=3)
        physical_mode = st.checkbox("physical input mode", value=False)
        l_phys = st.number_input("L_phys", min_value=1.0e-12, value=1.0, format="%.6g")
        u_phys = st.number_input("U_phys", min_value=1.0e-12, value=1.0, format="%.6g")
        nu_phys = st.number_input("nu_phys", min_value=1.0e-15, value=1.0e-6, format="%.6g")

        with st.expander("自动推荐参数", expanded=False):
            desired_ma_max = st.number_input("desired_Ma_max", min_value=0.001, value=0.1, format="%.4f")
            desired_tau_min = st.number_input("desired_tau_min", min_value=0.501, value=0.55, format="%.4f")
            desired_tau_target = st.number_input("desired_tau_target", min_value=0.501, value=0.60, format="%.4f")
            if st.button("计算推荐 U_ref"):
                rec = recommend_parameters(
                    case_type,
                    int(nx),
                    int(ny),
                    float(re),
                    desired_ma_max,
                    desired_tau_min,
                    desired_tau_target,
                    l_ref=float(l_ref) if l_ref > 0 else None,
                )
                st.session_state.recommended_u = float(rec["U_ref"])
                st.write(rec)

        st.subheader("收敛")
        tol = st.number_input("tol", min_value=1.0e-14, value=1.0e-6, format="%.1e")
        max_iter = st.number_input("max_iter", min_value=1, value=5000, step=100)
        min_iter = st.number_input("min_iter", min_value=0, value=500, step=100)
        report_interval = st.number_input("report_interval", min_value=1, value=100, step=10)
        ramp_steps = st.number_input("ramp_steps", min_value=0, value=1000, step=100)

        st.subheader("输出")
        output_dir = st.text_input("output_dir", value="outputs")
        save_npz = st.checkbox("save_npz", value=True)
        save_csv = st.checkbox("save_csv", value=True)
        save_png = st.checkbox("save_png", value=True)
        save_animation = st.checkbox("save_animation", value=False)
        plot_interval = st.number_input("plot_interval", min_value=1, value=500, step=100)

    defaults = default_boundaries_for_case(case_type, float(u_ref), boundary_scheme_default)

    boundary_tab, run_tab, results_tab = st.tabs(["边界条件", "运行", "结果"])
    with boundary_tab:
        cols = st.columns(4)
        boundaries = {}
        for col, side in zip(cols, ("left", "right", "bottom", "top")):
            with col:
                boundaries[side] = _boundary_editor(side, defaults[side])

    grid = GridConfig(
        NX=int(nx),
        NY=int(ny),
        L_ref=None if l_ref <= 0 else float(l_ref),
        obstacle_type="none" if obstacle_type == "uploaded_mask" else obstacle_type,
        obstacle_x=float(obstacle_x),
        obstacle_y=float(obstacle_y),
        obstacle_radius=float(obstacle_radius),
        obstacle_width=float(obstacle_width),
        obstacle_height=float(obstacle_height),
    )
    flow = FlowConfig(
        rho0=float(rho0),
        U_ref=float(u_ref),
        Re=float(re),
        physical_mode=bool(physical_mode),
        L_phys=float(l_phys),
        U_phys=float(u_phys),
        nu_phys=float(nu_phys),
        nu_lattice=float(nu_explicit) if nu_explicit > 0 else None,
        body_force_x=float(force_x), body_force_y=float(force_y),
        initial_velocity=initial_velocity,
    )
    convergence = ConvergenceConfig(
        tol=float(tol),
        max_iter=int(max_iter),
        min_iter=int(min_iter),
        report_interval=int(report_interval),
        ramp_steps=int(ramp_steps),
        consecutive_reports=int(steady_windows), steady=not fixed_steps,
    )
    output = OutputConfig(
        output_dir=output_dir,
        save_npz=save_npz,
        save_csv=save_csv,
        save_png=save_png,
        save_animation=save_animation,
        plot_interval=int(plot_interval),
    )
    config = SimulationConfig(
        case_type=case_type,
        collision_model=collision_model,
        boundary_scheme_default=boundary_scheme_default,
        grid=grid,
        flow=flow,
        convergence=convergence,
        output=output,
        boundaries=boundaries,
    )
    errors, warnings, transport = validate_config(config)

    with run_tab:
        metric_cols = st.columns(6)
        metric_cols[0].metric("Re", f"{transport['Re']:.4g}")
        metric_cols[1].metric("Ma", f"{transport['Ma']:.4g}")
        metric_cols[2].metric("nu_lattice", f"{transport['nu_lattice']:.4g}")
        metric_cols[3].metric("tau", f"{transport['tau']:.4g}")
        metric_cols[4].metric("omega", f"{transport['omega']:.4g}")
        metric_cols[5].metric("L_ref", f"{transport['L_ref']:.4g}")

        if errors:
            for error in errors:
                st.error(error)
        for warning in warnings:
            st.warning(warning)

        control_cols = st.columns(4)
        start = control_cols[0].button("Start simulation", type="primary", disabled=bool(errors))
        if control_cols[1].button("Pause"):
            st.session_state.running = False
        if control_cols[2].button("Stop"):
            st.session_state.stop_requested = True
        if control_cols[3].button("Reset"):
            st.session_state.last_result = None
            st.session_state.running = False
            st.session_state.stop_requested = False

        status_box = st.empty()
        progress = st.progress(0)
        live_metrics = st.empty()
        chart_box = st.empty()
        field_cols = st.columns(2)

        if start:
            st.session_state.running = True
            st.session_state.stop_requested = False
            solid_mask = _uploaded_mask(mask_file, int(nx), int(ny)) if obstacle_type == "uploaded_mask" else None
            solver = LBMSolver(config, solid_mask=solid_mask)
            last_report = None
            for report in solver.run():
                last_report = report
                progress.progress(min(1.0, report.iteration / max(1, int(max_iter))))
                live_metrics.metric(
                    "status",
                    f"iter {report.iteration} | residual {report.residual:.3e} | "
                    f"q {report.q:.3g} | q_avg {report.q_avg:.3g} | "
                    f"mass {report.mass_drift:.3e} | max_u {report.max_velocity:.4f}",
                )
                status_box.info(report.message)
                if report.iteration % max(1, int(plot_interval)) == 0 or report.converged or report.diverged:
                    snap = solver.field_snapshot()
                    with field_cols[0]:
                        st.pyplot(_plot_field(snap["speed"], "velocity magnitude", snap["solid_mask"]))
                    with field_cols[1]:
                        st.pyplot(_plot_field(snap["vorticity"], "vorticity", snap["solid_mask"], cmap="coolwarm"))
                    chart_box.empty()
                    with chart_box.container():
                        _history_chart(solver.history)
                if st.session_state.stop_requested:
                    solver.cancel()
                    status_box.warning("Stop requested; finalizing current state.")
                    break
            result = solver.finalize(save_outputs=True)
            st.session_state.last_result = result
            st.session_state.running = False
            if last_report and last_report.diverged:
                st.error("Simulation diverged. Try lower Re, larger grid, smaller U_ref, or gentler boundaries.")
            elif last_report and last_report.converged:
                st.success("Simulation converged.")
            else:
                st.info("Simulation finished or stopped.")

    with results_tab:
        result = st.session_state.last_result
        if result is None:
            st.info("No completed simulation yet.")
        else:
            st.write(f"Message: {result.message}")
            st.write(result.output_files)
            cols = st.columns(2)
            cols[0].pyplot(_plot_field(speed(result.ux, result.uy), "velocity magnitude", result.solid_mask))
            cols[1].pyplot(_plot_field(vorticity(result.ux, result.uy, result.solid_mask), "vorticity", result.solid_mask, cmap="coolwarm"))
            _history_chart(result.history)
            zipped = _zip_output_dir(result.config.output.output_dir)
            if zipped:
                st.download_button("Download results", zipped, file_name="lbm_results.zip")


if __name__ == "__main__":
    main()
