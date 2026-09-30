"""Validated workbench experiment definitions, independent of any GPU runtime."""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import math

LATTICES = {"D2Q9":"D2Q9 · 二维 9 速度", "D2Q25":"D2Q25 · 二维 25 速度（周期域）",
            "D3Q19":"D3Q19 · 三维 19 速度", "D3Q27":"D3Q27 · 三维 27 速度"}
SCENARIOS = {"cavity":"顶盖驱动腔", "couette":"平行壁剪切流", "poiseuille":"体力驱动通道",
             "obstacle":"周期障碍通道", "shear_wave":"周期剪切波衰减", "taylor_green":"Taylor–Green 涡",
             "thermal_wave":"温度对流扩散", "natural_convection":"差温腔自然对流 / 导热",
             "open_channel":"速度入口 / 压力出口通道", "cylinder_wake":"开放通道障碍物尾流",
             "oscillatory_channel":"周期体力驱动通道", "oscillating_couette":"振荡壁面剪切流",
             "rayleigh_benard":"底热顶冷 Rayleigh–Bénard 对流", "mixed_convection":"顶盖剪切与差温混合对流"}
EXTRA_DEFAULTS = dict(scenario="cavity", lattice="D2Q9", grid_y=0, grid_z=16,
    run_mode="steady", viscosity=0.0, force_x=0.0, force_y=0.0, force_z=0.0, trt_magic=3/16,
    obstacle="circle", obstacle_x=.35, obstacle_y=.5, obstacle_z=.5, obstacle_size=.15,
    diffusivity=.02, rayleigh=1000.0, prandtl=.71, rho0=1.0,
    snapshot_interval=0, checkpoint_interval=0, resume_from="",
    drive_period=1000.0, drive_phase=0.0, force_amplitude=1e-6, thermal_perturbation=.001,
    inlet_profile="parabolic", outlet_rho=1.0,
    obstacle_count_x=1, obstacle_count_y=1, obstacle_count_z=1,
    obstacle_spacing_x=.2, obstacle_spacing_y=.3, obstacle_spacing_z=.3, probes=[])
PERIODIC = {"shear_wave", "taylor_green", "thermal_wave"}
TRANSIENT = PERIODIC | {"oscillatory_channel", "oscillating_couette"}
OPEN = {"open_channel", "cylinder_wake"}
OBSTACLES = {"obstacle", "cylinder_wake"}
BUOYANT = {"natural_convection", "rayleigh_benard", "mixed_convection"}
THERMAL = BUOYANT | {"thermal_wave"}
MOVING = {"cavity", "couette", "oscillating_couette", "mixed_convection"}
CHANNEL = {"couette", "poiseuille", "obstacle", "oscillatory_channel", "oscillating_couette", "rayleigh_benard"}
RECIPES = {
    "二维顶盖腔 · MRT/HBB": dict(scenario="cavity",lattice="D2Q9",boundary="halfway",collision="MRT",grid=64,re=100),
    "二维剪切通道 · TRT": dict(scenario="couette",lattice="D2Q9",collision="TRT",boundary="halfway",grid=64,grid_y=32,re=10),
    "二维 Poiseuille · 解析对照": dict(scenario="poiseuille",lattice="D2Q9",collision="MRT",boundary="halfway",grid=32,grid_y=32,re=10),
    "二维周期圆柱阵列": dict(scenario="obstacle",lattice="D2Q9",collision="TRT",boundary="halfway",grid=128,grid_y=48,re=50,max_iter=30000,run_mode="transient"),
    "D2Q25 · 剪切波": dict(scenario="shear_wave",lattice="D2Q25",collision="BGK",boundary="periodic",grid=32,re=10,run_mode="transient",ramp_steps=0,max_iter=2000),
    "三维顶盖腔 · D3Q19": dict(scenario="cavity",lattice="D3Q19",collision="TRT",boundary="halfway",grid=32,grid_y=32,grid_z=32,re=100,max_iter=30000),
    "三维通道 · D3Q27": dict(scenario="poiseuille",lattice="D3Q27",collision="TRT",boundary="halfway",grid=32,grid_y=24,grid_z=16,re=10,max_iter=15000),
    "三维球形障碍物": dict(scenario="obstacle",lattice="D3Q19",collision="TRT",boundary="halfway",grid=64,grid_y=32,grid_z=32,re=30,max_iter=20000,run_mode="transient"),
    "三维 Taylor–Green 涡": dict(scenario="taylor_green",lattice="D3Q27",collision="TRT",boundary="periodic",grid=32,grid_y=32,grid_z=32,re=100,run_mode="transient",ramp_steps=0,max_iter=2000),
    "二维温度波 · D2Q5": dict(scenario="thermal_wave",lattice="D2Q9",collision="BGK",boundary="periodic",grid=64,grid_y=16,re=10,run_mode="transient",ramp_steps=0,max_iter=2000,lid_speed=.02),
    "二维自然对流 · Ra1000": dict(scenario="natural_convection",lattice="D2Q9",collision="MRT",boundary="halfway",grid=32,grid_y=32,viscosity=.02,rayleigh=1000,prandtl=.71,max_iter=80000),
    "三维自然对流 · D3Q19+D3Q7": dict(scenario="natural_convection",lattice="D3Q19",collision="TRT",boundary="halfway",grid=24,grid_y=24,grid_z=24,viscosity=.02,rayleigh=1000,prandtl=.71,max_iter=80000),
}
RECIPES.update({
    "入口出口通道 · D2Q9/TRT": dict(scenario="open_channel",lattice="D2Q9",collision="TRT",boundary="inlet_outlet",grid=96,grid_y=32,viscosity=.08,lid_speed=.025,ramp_steps=500,max_iter=20000,probes=[[.25,.5],[.75,.5]]),
    "圆柱尾流 · 开放通道": dict(scenario="cylinder_wake",lattice="D2Q9",collision="TRT",boundary="inlet_outlet",grid=240,grid_y=64,viscosity=.0048,lid_speed=.03,obstacle_x=.2,obstacle_y=.47,obstacle_size=.1,run_mode="transient",ramp_steps=1000,max_iter=60000,report_interval=20,snapshot_interval=500,probes=[[.35,.5],[.6,.5]]),
    "周期驱动力 · Womersley 通道": dict(scenario="oscillatory_channel",lattice="D2Q9",collision="TRT",boundary="halfway",grid=32,grid_y=32,viscosity=.1,force_amplitude=2e-5,drive_period=2000,run_mode="transient",ramp_steps=0,max_iter=12000,report_interval=20,snapshot_interval=200,probes=[[.5,.5],[.5,.25]]),
    "三维振荡剪切 · D3Q19": dict(scenario="oscillating_couette",lattice="D3Q19",collision="TRT",boundary="halfway",grid=24,grid_y=24,grid_z=16,viscosity=.1,lid_speed=.02,drive_period=2000,run_mode="transient",ramp_steps=0,max_iter=10000,report_interval=20,probes=[[.5,.5,.5]]),
    "底热顶冷对流 · Ra10000": dict(scenario="rayleigh_benard",lattice="D2Q9",collision="MRT",boundary="halfway",grid=64,grid_y=32,viscosity=.02,rayleigh=10000,prandtl=.71,thermal_perturbation=.005,max_iter=80000,snapshot_interval=1000),
    "三维底热顶冷 · D3Q27": dict(scenario="rayleigh_benard",lattice="D3Q27",collision="TRT",boundary="halfway",grid=32,grid_y=16,grid_z=16,viscosity=.01,rayleigh=3000,prandtl=.71,thermal_perturbation=.005,max_iter=60000),
    "热剪切混合对流": dict(scenario="mixed_convection",lattice="D2Q9",collision="MRT",boundary="halfway",grid=32,grid_y=32,viscosity=.02,lid_speed=.02,rayleigh=1000,prandtl=.71,max_iter=60000),
    "六圆柱周期阵列": dict(scenario="obstacle",lattice="D2Q9",collision="TRT",boundary="halfway",grid=128,grid_y=64,re=30,obstacle_x=.5,obstacle_y=.5,obstacle_size=.06,obstacle_count_x=3,obstacle_count_y=2,obstacle_spacing_x=.25,obstacle_spacing_y=.4,run_mode="transient",max_iter=30000,probes=[[.9,.5]])
})


def is_extended(params):
    return any(k in params for k in EXTRA_DEFAULTS) or params.get("collision") == "TRT"


def generic_engine(p):
    return (p.get("lattice", "D2Q9") != "D2Q9" or p.get("scenario", "cavity") != "cavity"
            or p.get("collision") == "TRT")


def transport(p):
    from lattices import lattice
    lat = lattice(p.get("lattice", "D2Q9"))
    nx, ny = p["grid"], p.get("grid_y", 0) or p["grid"]
    nz = p.get("grid_z", 16) if lat.dim == 3 else 1
    scenario = p.get("scenario", "cavity")
    length = min(nx, ny, nz) if lat.dim == 3 and scenario == "cavity" else (min(nx,ny) if scenario == "cavity" else ny)
    if scenario == "cavity" and p.get("boundary") == "nee":
        length -= 1
    nu = p.get("viscosity", 0) or p["lid_speed"] * length / p["re"]
    thermal = scenario in THERMAL
    alpha = nu/p.get("prandtl", .71) if scenario in BUOYANT else p.get("diffusivity", .02)
    scalar_cs2 = 1/3 if lat.dim == 2 else 1/4
    force = p.get("force_x", 0)
    if scenario in {"poiseuille", "obstacle"}:
        force += 8*nu*p["lid_speed"]/ny**2
    nodes = nx*ny*nz
    return dict(length=length, nu=nu, tau=.5+nu/lat.cs2, mach=p["lid_speed"]/math.sqrt(lat.cs2),
                effective_re=p["lid_speed"]*length/nu,
                nx=nx,ny=ny,nz=nz,dim=lat.dim,q=lat.q,cs2=lat.cs2,alpha=alpha,
                thermal_tau=.5+alpha/scalar_cs2 if thermal else None, force_x=force,
                buoyancy=p.get("rayleigh",0)*nu*alpha/ny**3 if scenario in BUOYANT else 0,
                estimated_host_bytes=nodes*(lat.q*64+512), estimated_gpu_bytes=nodes*(lat.q*80+1024))


def validate(raw):
    from dashboard.store import validate_legacy_parameters
    base = dict(raw)
    base["collision"] = "BGK" if raw.get("collision") == "TRT" else raw.get("collision", "BGK")
    base["boundary"] = "halfway" if raw.get("boundary") in {"periodic","inlet_outlet"} else raw.get("boundary", "nee")
    p = validate_legacy_parameters(base)
    p.update({k:raw.get(k,v) for k,v in EXTRA_DEFAULTS.items()})
    p["collision"], p["boundary"] = raw.get("collision", "BGK"), raw.get("boundary", "nee")
    if p["lattice"] not in LATTICES or p["scenario"] not in SCENARIOS:
        raise ValueError("离散速度模型或物理场景无效")
    if p["collision"] not in {"BGK","TRT","MRT"} or p["run_mode"] not in {"steady","transient"}:
        raise ValueError("碰撞模型或运行方式无效")
    for key, minimum in (("grid_y",0),("grid_z",8),("snapshot_interval",0),("checkpoint_interval",0),
                         ("obstacle_count_x",1),("obstacle_count_y",1),("obstacle_count_z",1)):
        try:
            value = Decimal(str(p[key]))
        except InvalidOperation:
            raise ValueError(f"{key} 必须是整数") from None
        if not value.is_finite() or value != value.to_integral_value() or not minimum <= value <= 2147483647:
            raise ValueError(f"{key} 整数超出范围")
        p[key] = int(value)
    if 0 < p["grid_y"] < 8:
        raise ValueError("Ny 为 0（跟随 Nx）或至少 8")
    for key in ("viscosity","force_x","force_y","force_z","trt_magic","obstacle_x","obstacle_y","obstacle_z",
                "obstacle_size","diffusivity","rayleigh","prandtl","rho0", "drive_period","drive_phase",
                "force_amplitude","thermal_perturbation","outlet_rho","obstacle_spacing_x","obstacle_spacing_y","obstacle_spacing_z"):
        try:
            p[key] = float(p[key])
        except (TypeError,ValueError,OverflowError):
            raise ValueError(f"{key} 必须是有限数") from None
        if not math.isfinite(p[key]):
            raise ValueError(f"{key} 必须是有限数")
    if min(p[k] for k in ("trt_magic","diffusivity","prandtl","rho0")) <= 0 or p["viscosity"] < 0 or p["rayleigh"] < 0:
        raise ValueError("黏度/Ra 不得为负；TRT 参数、热扩散率、Pr、密度必须为正")
    if p["obstacle"] not in {"circle","rectangle"}:
        raise ValueError("障碍物类型无效")
    if not .02 <= p["obstacle_size"] <= .4 or any(not .05 <= p[k] <= .95 for k in ("obstacle_x","obstacle_y","obstacle_z")):
        raise ValueError("障碍物尺度范围 0.02–0.4，中心坐标范围 0.05–0.95")
    s, lat = p["scenario"], p["lattice"]
    if p['drive_period'] < 4 or not 0 <= p['thermal_perturbation'] <= .05 or p['outlet_rho'] <= 0:
        raise ValueError('驱动周期至少 4 步；温度扰动范围 0–0.05；出口密度必须为正')
    if p['inlet_profile'] not in {'parabolic','uniform'}:
        raise ValueError('入口剖面无效')
    if any(not 1 <= p['obstacle_count_'+a] <= 8 or not 0 < p['obstacle_spacing_'+a] <= 1 for a in 'xyz'):
        raise ValueError('阵列各方向数量为 1–8，间距范围 (0,1]')
    if math.prod(p['obstacle_count_'+a] for a in 'xyz') > 64:
        raise ValueError('阵列最多 64 个障碍物')
    if lat.startswith('D2') and p['obstacle_count_z'] != 1:
        raise ValueError('二维障碍物阵列的 z 向数量必须为 1')
    from advanced_physics import parse_probes, obstacle_centers
    p['probes'] = parse_probes(p['probes'], 3 if lat.startswith('D3') else 2)
    if p['probes'] and not generic_engine(p):
        raise ValueError('点探针用于扩展内核；二维方腔选择 TRT 可启用')
    if s in OPEN and (lat != 'D2Q9' or any(p[k] for k in ('force_x','force_y','force_z'))):
        raise ValueError('入口出口当前支持无附加体力的 D2Q9；三维请选择闭合或周期场景')
    if lat != "D2Q9" and p["collision"] == "MRT":
        raise ValueError("当前 MRT 已验证矩基仅支持 D2Q9；该速度模型请选 BGK 或 TRT")
    if lat == "D2Q25" and s not in PERIODIC:
        raise ValueError("D2Q25 的多格距迁移目前仅支持无障碍物的全周期场景")
    required = "inlet_outlet" if s in OPEN else ("periodic" if s in PERIODIC else "halfway")
    if s == "cavity" and lat == "D2Q9" and p["collision"] != "TRT":
        if p["boundary"] not in {"nee","halfway"}: raise ValueError("方腔请选择 NEE 或 HBB")
    elif p["boundary"] != required:
        raise ValueError("此场景请选择" + {"periodic":"全周期边界","halfway":" HBB（半格点反弹）壁面","inlet_outlet":"入口 / 出口 + HBB 壁面"}[required])
    if s in TRANSIENT and p["run_mode"] != "transient":
        raise ValueError("衰减/对流算例须选择瞬态运行，不能按速度残差提前判稳态")
    if not generic_engine(p) and any(p[k] for k in ("force_x","force_y","force_z")):
        raise ValueError("原 NEE/HBB 方腔内核无外力；施加外力请选择通道等扩展场景")
    if not generic_engine(p) and (p["snapshot_interval"] or p["checkpoint_interval"] or p["resume_from"]):
        raise ValueError("检查点/场快照用于扩展内核；二维方腔使用 TRT 可启用，原 BGK/MRT 方腔保留原计算路径")
    if lat.startswith("D2") and p["force_z"]:
        raise ValueError("二维模型不能施加 z 向体力")
    d = transport(p)
    if d["mach"] > .1:
        raise ValueError("参考速度对应 Ma 超过 0.1")
    if s in OBSTACLES:
        lengths=[d['n'+a]-(1 if a=='x' and s in OPEN else 0) for a in 'xyz'[:d['dim']]]
        radius = p["obstacle_size"]*min(lengths)
        for center in obstacle_centers(p,d['dim']):
            for a,axis in enumerate('xyz'[:d['dim']]):
                n = lengths[a]
                if min(center[a],1-center[a])*n <= radius+1:
                    raise ValueError('每个障碍物必须与边界留出至少一个完整流体格点')
        for axis in 'xyz'[:d['dim']]:
            if p['obstacle_count_'+axis]>1 and p['obstacle_spacing_'+axis]*d['n'+axis]<=2*radius+1:
                raise ValueError('相邻障碍物之间必须留出流体格点')
    if d["nx"]*d["ny"]*d["nz"] > 2147483647:
        raise ValueError("总格点数超出索引范围")
    if s == "taylor_green" and (d["nx"] != d["ny"] or (d["dim"]==3 and d["nz"] != d["nx"])):
        raise ValueError("Taylor–Green 预设需要等长的周期方向")
    if not math.isfinite(d["tau"]) or d["tau"] <= .5 or (d["thermal_tau"] is not None and d["thermal_tau"] <= .5):
        raise ValueError("流体/温度松弛时间必须大于 0.5")
    p["resume_from"] = str(p["resume_from"] or "")
    return p


def make_config(params):
    from reference import make_config as cavity_config
    from cavity_models import CavityConfig
    from config import GridConfig, BoundaryConfig
    from dashboard.store import DEFAULTS
    p = validate(params)
    base = cavity_config(**{k:p[k] for k in DEFAULTS if k != "backend" and k not in {"collision","boundary"}},
                         collision="BGK",boundary="halfway")
    @dataclass
    class ResearchGrid(GridConfig):
        NZ: int = 1
    @dataclass
    class ResearchConfig(CavityConfig):
        research: dict = None
    cfg = ResearchConfig(**base.__dict__, research=p)
    cfg.grid = ResearchGrid(**base.grid.__dict__, NZ=p["grid_z"] if p["lattice"].startswith("D3") else 1)
    cfg.grid.NY = p["grid_y"] or p["grid"]
    cfg.flow.rho0 = p["rho0"]
    cfg.flow.nu_lattice = transport(p)["nu"]
    cfg.flow.Re = transport(p)["effective_re"]
    cfg.flow.body_force_x = transport(p)["force_x"] if generic_engine(p) else 0
    cfg.flow.body_force_y = p["force_y"]
    if p["scenario"] in OBSTACLES:
        cfg.grid.obstacle_type = p["obstacle"]
    for key in ("mrt_s_e", "mrt_s_eps", "mrt_s_q"):
        setattr(cfg,key,p[key])
    cfg.collision_model = p["collision"]
    cfg.convergence.steady = p["run_mode"] == "steady"
    cfg.boundary_scheme_default = {"nee":"non_equilibrium_extrapolation","halfway":"halfway_bounce_back","periodic":"periodic","inlet_outlet":"halfway_bounce_back"}[p["boundary"]]
    for side in cfg.boundaries:
        cfg.boundaries[side] = BoundaryConfig(cfg.boundary_scheme_default, ux=p["lid_speed"] if side=="top" and p["scenario"]=="cavity" else 0)
    if p["boundary"] == "halfway" and p["scenario"] in MOVING:
        cfg.boundaries["top"].type = "halfway_moving_wall"
        cfg.boundaries["top"].ux = p["lid_speed"]
    if p["scenario"] in CHANNEL:
        cfg.boundaries["left"] = BoundaryConfig("periodic")
        cfg.boundaries["right"] = BoundaryConfig("periodic")
    if p["lattice"].startswith("D3"):
        kind = "periodic" if p["scenario"] in PERIODIC | CHANNEL else "halfway_bounce_back"
        cfg.boundaries.update(front=BoundaryConfig(kind),back=BoundaryConfig(kind))
    cfg.case_type = "lid_driven_cavity" if p["scenario"] in {"cavity","natural_convection"} else "custom"
    if p['scenario'] in OPEN:
        cfg.boundaries['left'] = BoundaryConfig('velocity_zou_he',ux=p['lid_speed'])
        cfg.boundaries['right'] = BoundaryConfig('pressure_zou_he',rho=p['outlet_rho'])
    return cfg
