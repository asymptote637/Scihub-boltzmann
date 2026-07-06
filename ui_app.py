"""Streamlit front end for the customized 2D LBM simulator."""

from __future__ import annotations

from pathlib import Path
import tempfile
import zipfile

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    CORE_BOUNDARY_TYPES,
    BoundaryConfig,
    ObstacleConfig,
    OutputConfig,
    SolverConfig,
    case_preset,
    derived_parameters,
    recommend_u_ref,
    validate_config,
)
from lbm_solver import LBMSolver
from postprocess import save_results, speed, vorticity


st.set_page_config(page_title="2D LBM Simulator", layout="wide")


def boundary_editor(label: str, default: BoundaryConfig) -> BoundaryConfig:
    with st.expander(f"{label} boundary", expanded=False):
        bc_type = st.selectbox(
            "type",
            sorted(CORE_BOUNDARY_TYPES),
            index=sorted(CORE_BOUNDARY_TYPES).index(default.type)
            if default.type in CORE_BOUNDARY_TYPES
            else 0,
            key=f"{label}_type",
        )
        c1, c2, c3, c4 = st.columns(4)
        ux = c1.number_input("ux", value=float(default.ux), format="%.6f", key=f"{label}_ux")
        uy = c2.number_input("uy", value=float(default.uy), format="%.6f", key=f"{label}_uy")
        rho = c3.number_input("rho", value=float(default.rho), min_value=0.0001, format="%.6f", key=f"{label}_rho")
        rb = c4.slider("rb", 0.0, 1.0, float(default.rb), 0.05, key=f"{label}_rb")
    return BoundaryConfig(type=bc_type, ux=ux, uy=uy, rho=rho, rb=rb)


def make_zip(path: Path) -> bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        zip_path = Path(tmp.name)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in path.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(path.parent))
    data = zip_path.read_bytes()
    zip_path.unlink(missing_ok=True)
    return data


st.title("2D D2Q9-BGK LBM Simulator")

with st.sidebar:
    st.header("Model")
    case_type = st.selectbox(
        "case_type",
        ["lid_driven_cavity", "poiseuille_channel", "couette_flow", "periodic_channel", "cylinder_flow", "custom"],
    )
    collision_model = st.selectbox("collision_model", ["BGK"])
    boundary_scheme_default = st.selectbox(
        "boundary_scheme_default",
        ["non_equilibrium_extrapolation", "no_slip_bounce_back", "moving_wall_bounce_back"],
    )

    st.header("Grid")
    nx = st.number_input("NX", min_value=16, max_value=1024, value=128, step=16)
    ny = st.number_input("NY", min_value=16, max_value=1024, value=128, step=16)

    st.header("Flow")
    rho0 = st.number_input("rho0", min_value=0.01, value=1.0, step=0.01)
    reynolds = st.number_input("Re", min_value=1.0, value=1000.0, step=50.0)
    u_ref = st.number_input("U_ref", min_value=0.0, value=0.05, step=0.005, format="%.5f")
    l_ref = st.number_input("L_ref, 0 = auto", min_value=0.0, value=0.0, step=1.0)

    with st.expander("Auto recommend parameters"):
        desired_ma = st.number_input("desired_Ma_max", min_value=0.01, max_value=0.3, value=0.1)
        tau_target = st.number_input("desired_tau_target", min_value=0.55, max_value=1.2, value=0.6)
        rec = recommend_u_ref(
            reynolds=reynolds,
            nx=int(nx),
            ny=int(ny),
            case_type=case_type,
            desired_ma_max=desired_ma,
            desired_tau_target=tau_target,
        )
        st.write(rec)

    st.header("Convergence")
    tol = st.number_input("tol", min_value=1e-12, value=1e-6, format="%.1e")
    max_iter = st.number_input("max_iter", min_value=1, value=5000, step=1000)
    min_iter = st.number_input("min_iter", min_value=0, value=500, step=100)
    report_interval = st.number_input("report_interval", min_value=1, value=100, step=50)
    ramp_steps = st.number_input("ramp_steps", min_value=1, value=1000, step=100)

preset = case_preset(case_type, nx=int(nx), ny=int(ny))
preset.u_ref = float(u_ref)
preset.reynolds = float(reynolds)
if case_type == "lid_driven_cavity":
    preset.top.ux = float(u_ref)

tab_setup, tab_run, tab_results = st.tabs(["Setup", "Run", "Results"])

with tab_setup:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Boundary Conditions")
        left = boundary_editor("left", preset.left)
        right = boundary_editor("right", preset.right)
        bottom = boundary_editor("bottom", preset.bottom)
        top = boundary_editor("top", preset.top)
    with c2:
        st.subheader("Obstacle")
        obstacle_type = st.selectbox("obstacle type", ["none", "cylinder", "rectangle"])
        oc1, oc2 = st.columns(2)
        obs = ObstacleConfig(
            type=obstacle_type,
            cx=oc1.slider("obstacle cx", 0.0, 1.0, preset.obstacle.cx, 0.01),
            cy=oc2.slider("obstacle cy", 0.0, 1.0, preset.obstacle.cy, 0.01),
            radius=oc1.slider("cylinder radius", 0.01, 0.30, preset.obstacle.radius, 0.01),
            width=oc1.slider("rectangle width", 0.02, 0.60, preset.obstacle.width, 0.01),
            height=oc2.slider("rectangle height", 0.02, 0.60, preset.obstacle.height, 0.01),
        )
        st.subheader("Output")
        save_npz = st.checkbox("save_npz", True)
        save_csv = st.checkbox("save_csv", True)
        save_png = st.checkbox("save_png", True)
        save_animation = st.checkbox("save_animation", False, disabled=True)
        output_dir = st.text_input("output_dir", "results/ui_runs")

cfg = SolverConfig(
    case_type=case_type,
    collision_model=collision_model,
    boundary_scheme_default=boundary_scheme_default,
    nx=int(nx),
    ny=int(ny),
    rho0=float(rho0),
    u_ref=float(u_ref),
    reynolds=float(reynolds),
    l_ref=None if l_ref <= 0 else float(l_ref),
    tol=float(tol),
    max_iter=int(max_iter),
    min_iter=int(min_iter),
    report_interval=int(report_interval),
    ramp_steps=int(ramp_steps),
    left=left,
    right=right,
    bottom=bottom,
    top=top,
    obstacle=obs,
    output=OutputConfig(
        output_dir=output_dir,
        save_npz=save_npz,
        save_csv=save_csv,
        save_png=save_png,
        save_animation=save_animation,
        plot_interval=int(report_interval),
    ),
)

errors, warnings = validate_config(cfg)
derived = derived_parameters(cfg)

with tab_setup:
    st.subheader("Stability Diagnostics")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Re", f"{derived['Re']:.3g}")
    d2.metric("Ma", f"{derived['Ma']:.4f}")
    d3.metric("tau", f"{derived['tau']:.5f}")
    d4.metric("omega", f"{derived['omega']:.5f}")
    d5.metric("level", str(derived["stability_level"]))
    for warning in warnings:
        st.warning(warning)
    for error in errors:
        st.error(error)

with tab_run:
    st.subheader("Run Control")
    start = st.button("Start simulation", type="primary", disabled=bool(errors))
    st.button("Pause", disabled=True)
    st.button("Stop", disabled=True)
    if st.button("Reset"):
        st.session_state.pop("last_run_dir", None)
        st.session_state.pop("last_solver", None)

    metrics_box = st.empty()
    plot_col1, plot_col2 = st.columns(2)
    history_plot = st.empty()
    status_box = st.empty()

    if start:
        solver = LBMSolver(cfg)
        progress = st.progress(0.0)
        last_report = None
        for report in solver.run():
            last_report = report
            progress.progress(min(1.0, report.iteration / max(1, cfg.max_iter)))
            metrics_box.json(report.as_dict())
            fields = solver.fields()
            spd = speed(fields["ux"], fields["uy"])
            vort = vorticity(fields["ux"], fields["uy"])
            plot_col1.image(spd / max(float(np.max(spd)), 1e-30), caption="velocity magnitude")
            plot_col2.image(vort, caption="vorticity", clamp=True)
            if len(solver.residual_history) > 1:
                hist = pd.DataFrame([r.as_dict() for r in solver.residual_history])
                history_plot.line_chart(hist.set_index("iteration")[["residual", "mass_drift"]])
            status_box.info(f"status: {report.status}")
        run_dir = save_results(solver, cfg)
        st.session_state["last_run_dir"] = str(run_dir)
        st.session_state["last_solver"] = solver.summary()
        st.success(f"Saved results to {run_dir}")
        if last_report is not None:
            st.write(last_report.as_dict())

with tab_results:
    run_dir_text = st.session_state.get("last_run_dir")
    if run_dir_text:
        run_dir = Path(run_dir_text)
        st.write(f"Latest output: `{run_dir}`")
        if run_dir.exists():
            st.download_button("Download results", data=make_zip(run_dir), file_name=f"{run_dir.name}.zip")
            figs = run_dir / "figures"
            if figs.exists():
                cols = st.columns(3)
                for col, name in zip(cols, ["velocity_magnitude.png", "streamlines.png", "vorticity.png"]):
                    path = figs / name
                    if path.exists():
                        col.image(str(path), caption=name)
    else:
        st.info("No completed UI run yet.")

