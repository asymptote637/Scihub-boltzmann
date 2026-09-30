"""Run a parameterized BGK cavity with Chinese prompts and durable records."""
from __future__ import annotations

import argparse
import copy
from decimal import Decimal, InvalidOperation
import math
from pathlib import Path
import sys
import json
import time
import datetime
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source_snapshot'))
from workbench.project import config_from_dict, atomic_json
from workbench.validation import evaluate, reference_kind
from lbm_solver import LBMSolver
from config import SimulationConfig, config_to_dict, validate_config, default_boundaries_for_case

STOP_NAMES = {
    'running': '计算中', 'converged': '已收敛',
    'max_iter': '达到步数上限，尚未收敛',
    'completed_steps': '已完成指定步数', 'cancelled': '已取消',
    'diverged': '已触发数值保护并停止',
}
VALIDATION_NAMES = {
    'passed': '通过（仅限当前工况）', 'failed': '未通过',
    'not_converged': '尚未收敛，不能认定通过',
    'not_applicable': '没有适用于该配置的现有参考',
}
BOUNDARY_NAMES = {
    'halfway_bounce_back': '半格静壁反弹',
    'halfway_moving_wall': '半格移动壁反弹',
    'non_equilibrium_extrapolation': '非平衡外推',
}


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError('请输入有效数字，例如 100 或 1e-7。') from None
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError('必须是大于 0 的有限数字，不能是 NaN 或 Inf。')
    return parsed


def integer_at_least(value: str, minimum: int) -> int:
    try:
        parsed = int(value)
    except ValueError:
        # Decimal preserves large integer inputs exactly; int(float(...)) would
        # silently round counts such as 9007199254740993e0.
        try:
            decimal = Decimal(value)
        except InvalidOperation:
            raise argparse.ArgumentTypeError('请输入整数或结果为整数的科学计数法，例如 4096 或 1e8。') from None
        if not decimal.is_finite() or decimal != decimal.to_integral_value():
            raise argparse.ArgumentTypeError('必须是有限整数；科学计数法的结果也必须为整数。')
        if decimal.copy_abs() > Decimal(sys.float_info.max):
            raise argparse.ArgumentTypeError('数量级超出当前双精度求解器可表示的范围。')
        parsed = int(decimal)
    if parsed < minimum:
        raise argparse.ArgumentTypeError(f'必须是大于或等于 {minimum} 的整数。')
    return parsed


def lid_speed(value: str) -> float:
    parsed = positive_float(value)
    # The simple cavity entry keeps the existing low-Mach protection limit.
    if parsed * math.sqrt(3) > 0.1:
        raise argparse.ArgumentTypeError('顶盖速度过大；Ma 上限为 0.1，请输入不超过 0.057735 的格子速度。')
    return parsed


# dest, CLI flag, prompt, parser, config section, config field
PARAMETERS = (
    ('re', '--re', '雷诺数 Re（正数，无固定上限，例如 1e6）', positive_float, 'flow', 'Re'),
    ('grid', '--grid', '每边网格数 N（整数≥8，例如 4096 或 1e4）',
     lambda v: integer_at_least(v, 8), 'grid', 'NX'),
    ('lid_speed', '--lid-speed', '顶盖速度 U（格子单位，建议 0.04）', lid_speed, 'flow', 'U_ref'),
    ('max_iter', '--max-iter', '最大迭代步数（正整数，例如 1e8 或 1e12）', lambda v: integer_at_least(v, 1), 'convergence', 'max_iter'),
    ('tol', '--tol', '收敛阈值（正数，例如 1e-12 或 1e-16）', positive_float, 'convergence', 'tol'),
    ('ramp_steps', '--ramp-steps', '顶盖从静止加速到 U 的步数（0 表示立即启动）',
     lambda v: integer_at_least(v, 0), 'convergence', 'ramp_steps'),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='D2Q9-BGK 方腔：无参数运行时逐项输入；提供命令行参数时，其余项使用默认值。',
        epilog='默认值：Re=100，N=64，U=0.04，最多 50000 步，阈值 1e-6，启动 500 步。')
    for dest, flag, label, convert, _, _ in PARAMETERS:
        parser.add_argument(flag, dest=dest, type=convert, help=label)
    parser.add_argument('--min-iter', type=lambda v: integer_at_least(v, 0), help='最早允许判断收敛的步数，默认 2000')
    parser.add_argument('--report-interval', type=lambda v: integer_at_least(v, 1), help='进度输出间隔，默认 200 步；收敛每步检查')
    parser.add_argument('--defaults', action='store_true', help='不提问，未指定参数全部使用默认值')
    parser.add_argument('--dry-run', action='store_true', help='只检查并显示参数，不开始计算、不写结果')
    parser.add_argument('--loop', action='store_true', help='每次计算后返回继续运行菜单；无参数启动时默认启用')
    parser.add_argument('--boundary', choices=['nee', 'halfway'], help='壁面格式；默认 nee，顶角随盖运动、下角静止（书附录D）')
    parser.add_argument('--mass-policy', choices=['strict', 'diagnostic'], help='strict：质量参与收敛；diagnostic：只记录质量漂移，不修正')
    return parser


def needs_input(args: argparse.Namespace) -> bool:
    return not (args.defaults or args.dry_run or args.boundary is not None or args.mass_policy is not None or args.min_iter is not None
                or args.report_interval is not None
                or any(getattr(args, p[0]) is not None for p in PARAMETERS))


def configure(args: argparse.Namespace, previous: SimulationConfig | None = None) -> SimulationConfig:
    config = (copy.deepcopy(previous) if previous is not None else
              config_from_dict(json.loads((ROOT / 'baseline_request.json').read_text(encoding='utf-8'))['config']))
    if previous is None:
        config.convergence.tol = 1e-6
        config.convergence.consecutive_reports = 1
    interactive = needs_input(args)
    if interactive:
        print('\n请逐项输入；直接按回车使用方括号中的默认值，Ctrl+C 可退出。', flush=True)
        if previous is not None:
            print('本轮默认值沿用上一组参数。', flush=True)
        print('方括号内是默认值，不是上限。Re、网格、步数和阈值均支持科学计数法。', flush=True)
    for dest, _, label, convert, section, field in PARAMETERS:
        group = getattr(config, section)
        default = getattr(group, field)
        value = getattr(args, dest)
        if interactive:
            while True:
                raw = input(f'{label} [{default:g}]：').strip()
                try:
                    value = convert(raw if raw else str(default))
                    break
                except argparse.ArgumentTypeError as exc:
                    print(f'  输入无效：{exc}', flush=True)
        setattr(group, field, default if value is None else value)
    config.grid.NY = config.grid.NX
    if args.boundary is not None:
        scheme = 'non_equilibrium_extrapolation' if args.boundary == 'nee' else 'halfway_bounce_back'
        config.boundary_scheme_default = scheme
        config.boundaries = default_boundaries_for_case('lid_driven_cavity', config.flow.U_ref, scheme)
    # Re and U control transport and the actual lid velocity together.
    config.grid.L_ref = None
    config.flow.nu_lattice = None
    config.flow.physical_mode = False
    config.boundaries['top'].ux = config.flow.U_ref
    config.boundaries['top'].uy = 0.0
    config.convergence.steady = True
    if args.mass_policy is not None:
        config.convergence.mass_criterion = args.mass_policy == 'strict'
    if args.min_iter is not None:
        config.convergence.min_iter = args.min_iter
    if args.report_interval is not None:
        config.convergence.report_interval = args.report_interval
    return config


def preflight(config: SimulationConfig) -> dict:
    try:
        errors, warnings, transport = validate_config(config)
    except OverflowError:
        raise ValueError('参数数量级超过当前双精度计算可表示的范围。') from None
    if errors:
        raise ValueError('参数组合不合法：\n  ' + '\n  '.join(errors))
    cells = config.grid.NX * config.grid.NY
    if cells > np.iinfo(np.intp).max // (9 * np.dtype(np.float64).itemsize):
        raise ValueError('网格的 D2Q9 分布数组超过 NumPy 可寻址的大小；请减小网格。')
    print('\n参数检查通过', flush=True)
    print(f"  Re = {transport['Re']:g} | 网格 {config.grid.NX} × {config.grid.NY} | 顶盖 U = {config.flow.U_ref:g}", flush=True)
    print(f"  自动换算：L = {transport['L_ref']:g}，ν = {transport['nu_lattice']:.8g}，τ = {transport['tau']:.16g}，设计 Ma = {transport['Ma']:.6f}", flush=True)
    print(f"  τ 与 0.5 的差值：{transport['tau'] - 0.5:.6e}", flush=True)
    print(f'  最大步数 {config.convergence.max_iter:,}；顶盖启动 {config.convergence.ramp_steps} 步。', flush=True)
    print(f'  收敛阈值 {config.convergence.tol:g}；最短步数 {config.convergence.min_iter}；输出间隔 {config.convergence.report_interval} 步（每步检查收敛）。', flush=True)
    # f (9), rho/ux/uy (3), previous ux/uy (2): float64; solid mask: bool.
    resident_gib = cells * (14 * 8 + 1) / 1024**3
    print(f'  主要常驻数组约 {resident_gib:.4g} GiB；计算临时数组、历史记录及输出另计。', flush=True)
    if config.convergence.tol < np.finfo(np.float64).eps:
        print('  提示：阈值小于 float64 在 1 附近的分辨尺度，可能长期无法满足，不能据此宣称同等物理精度。', flush=True)
    reports = config.convergence.max_iter // config.convergence.report_interval + 2
    if reports > 100_000:
        print(f'  提示：若跑满上限，将记录约 {reports:,} 次诊断；可用 --report-interval 增大间隔以减少内存和日志。', flush=True)
    translations = {
        'min_iter > max_iter: this run cannot be labelled converged.': '最大步数小于最短收敛步数，本次只能用于短跑检查。',
        'tau is close to 0.5; positive viscosity is not a stability guarantee.': 'τ 接近 0.5，数值稳定性仍需实际检查。',
        'Large tau: check resolution and viscosity sensitivity.': 'τ 较大，需要检查网格分辨率及黏度敏感性。',
    }
    for warning in warnings:
        print('  提示：' + translations.get(warning, warning), flush=True)
    if config.convergence.max_iter <= config.convergence.ramp_steps:
        print('  提示：本次步数不足以观察顶盖启动后的稳态流动。', flush=True)
    kind, note = reference_kind(config)
    print('  基准参考：' + (note if kind else '暂无适用于该工况的内置参考；仍记录收敛和数值诊断。'), flush=True)
    return transport


def number(value: object) -> str:
    """Format small diagnostics without converting absent values into zeros."""
    return '未提供' if value is None else f'{float(value):.3e}'


def result_text(data: dict, validation: dict, summary: dict,
                config: SimulationConfig, timing_only: bool = False) -> str:
    """Describe actual status; a fixed-step timing run is never called validated."""
    diag = summary.get('final_diagnostics', {})
    lines = [
        '\n' + '─' * 54,
        '短时计时结果' if timing_only else '方腔计算结果',
        f"  本次参数：Re = {config.flow.Re:g}，网格 {config.grid.NX} × {config.grid.NY}，顶盖 U = {config.flow.U_ref:g}",
        f"  停止状态：{'速度残差达标（质量另列）' if data['stop_reason'] == 'converged' and not config.convergence.mass_criterion else STOP_NAMES.get(data['stop_reason'], data['stop_reason'])}",
        f"  实际步数：{data['iterations']:,} 步",
        f"  计算耗时：{data['compute_seconds']:.2f} 秒（含报告与日志，不含初始化）",
        f"  无量纲时间 T = 步数 × U / L：{data['dimensionless_time']:.5g}",
    ]
    if timing_only:
        lines.extend([
            f"  平均每步：{data['seconds_per_step']:.6f} 秒",
            '  用途：仅测计算速度，不作为收敛或精度验证。',
            f"  当前速度残差：{number(diag.get('residual'))}",
        ])
        if data['iterations'] < config.convergence.ramp_steps:
            lines.append('  当前仍在顶盖启动阶段，不能与稳态参考直接比较。')
    else:
        lines.append(f"  基准对照：{VALIDATION_NAMES.get(validation['status'], validation['status'])}")
        checks = validation.get('checks', {})
        if checks:
            lines.append('')
        for key, label in [
            ('converged', '满足稳态收敛条件'),
            ('finite_positive_fields', '密度为正，宏观场无 NaN/Inf'),
            ('actual_max_mach', '实际最大 Ma 符合基准限制'),
            ('mass_drift', '闭域质量漂移符合基准限制'),
            ('u_normalized_RMSE', '水平速度中线与参考相符'),
            ('v_normalized_RMSE', '竖直速度中线与参考相符'),
        ]:
            if key in checks:
                lines.append(f"  [{'通过' if checks[key] else '未通过'}] {label}")
        lines.extend([
            '',
            f"  最终残差：{number(diag.get('residual'))}；收敛阈值 < {config.convergence.tol:.1e}",
            '  残差为相邻一个时间步速度变化的相对 L2 范数，以当前速度场归一化。',
        ])
        metrics = validation.get('metrics', {})
        for axis, name in [('u', '水平速度'), ('v', '竖直速度')]:
            key = f'{axis}_normalized_RMSE'
            if key in metrics:
                limit = validation.get('thresholds', {}).get(key)
                lines.append(f"  {name}中线 RMSE：{metrics[key]:.6f}；阈值 < {number(limit)}")
        if 'u_normalized_RMSE' in metrics:
            lines.append('  RMSE 使用顶盖速度归一化；它不是各点相对误差的百分比。')
        if validation.get('reference_note'):
            lines.append(f"  参考说明：{validation['reference_note']}")
    lines.extend([
        f"  总质量相对漂移：{number(diag.get('mass_drift'))}",
        f"  实际最大 Ma：{number(diag.get('max_mach'))}；运行保护上限 {config.convergence.max_mach:g}",
        f"  密度范围：{number(diag.get('min_density'))} ～ {number(diag.get('max_density'))}",
        f"  最终分布：{'全部有限' if data['population_check']['finite'] else '含非有限值'}；负值数量 {data['population_check']['negative_count']}",
    ])
    if summary.get('error'):
        lines.append(f"  停止原因详情：{summary['error']}")
    lines.extend([
        '', '  结果已保存到：', f"  {data['path']}",
        '  先看「结果说明.txt」；完整数值见 residual_history.csv 和 validation.json。',
        '─' * 54,
    ])
    return '\n'.join(lines)


def run(config: SimulationConfig, label: str, timing_only: bool = False) -> dict:
    errors, warnings, transport = validate_config(config)
    if errors:
        raise ValueError('\n'.join(errors))
    dest = ROOT / (label + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    dest.mkdir()
    config.output.output_dir = str(dest)
    config.output.save_png = False
    atomic_json(dest / 'run_request.json', dict(
        created_at=datetime.datetime.now().astimezone().isoformat(),
        residual_definition="relative_L2_single_step_current_velocity", residual_steps=1,
        config=config_to_dict(config, transport), warnings=warnings))
    solver = LBMSolver(config)
    t = solver.transport
    title = '方腔短时计时（不是稳态验证）' if timing_only else '开始方腔计算'
    print(f'\n{title}', flush=True)
    print(f"  网格 {config.grid.NX} × {config.grid.NY} | {config.collision_model} | Re = {t['Re']:g}", flush=True)
    print(f"  顶盖速度 {config.boundaries['top'].ux:g} | 有效参考长度 {t['L_ref']:g} | 黏度 {t['nu_lattice']:.6g} | 松弛时间 {t['tau']:.16g}", flush=True)
    walls = '；'.join(f"{name}：{BOUNDARY_NAMES.get(config.boundaries[side].type, config.boundaries[side].type)}"
                     for side, name in [('left', '左'), ('right', '右'), ('bottom', '下'), ('top', '上')])
    print('  ' + walls, flush=True)
    print('  残差求和区域：' + ('内部流体节点（按书附录D，排除节点壁面）' if solver.residual_domain == 'interior_without_on_node_walls' else '所有流体节点'), flush=True)
    initial = {'rest': '静止', 'uniform': '均匀速度', 'couette': '线性剪切速度'}.get(config.flow.initial_velocity, config.flow.initial_velocity)
    print(f'  初场：{initial}；顶盖启动 {config.convergence.ramp_steps} 步。', flush=True)
    if timing_only:
        print(f'  本阶段只运行 {config.convergence.max_iter} 步，用于估算耗时。', flush=True)
    else:
        print(f'  最多 {config.convergence.max_iter:,} 步；每步检查收敛，每 {config.convergence.report_interval} 步显示一次进度。', flush=True)
        mass_note = f'、质量漂移 ≤ {config.convergence.mass_tolerance:.1e}' if config.convergence.mass_criterion else '；质量漂移仅记录，不修正'
        print(f'  达到最短步数且启动结束后：残差 < {config.convergence.tol:.1e}{mass_note}；连续 {config.convergence.consecutive_reports} 个时间步达标。', flush=True)
    print('  进度中的步数上限是计算预算，不代表到该步才会收敛。', flush=True)
    start = time.perf_counter()
    last = 0.0
    with (dest / 'progress.jsonl').open('w', encoding='utf-8') as log:
        for report in solver.run():
            elapsed = time.perf_counter() - start
            row = dict(iteration=report.iteration, elapsed_seconds=elapsed,
                       residual=report.residual, mass_drift=report.mass_drift,
                       max_mach=report.max_mach, stop_reason=report.stop_reason)
            log.write(json.dumps(row) + '\n')
            log.flush()
            if report.iteration == 1 or elapsed - last >= 5 or report.stop_reason != 'running':
                state = STOP_NAMES.get(report.stop_reason, report.stop_reason)
                print(f'  {report.iteration:>6,}/{config.convergence.max_iter:,} 步 | '
                      f'{elapsed:>5.1f} 秒 | 残差 {report.residual:.2e} | '
                      f'质量漂移 {report.mass_drift:.2e} | Ma {report.max_mach:.4f} | {state}', flush=True)
                last = elapsed
    compute = time.perf_counter() - start
    pop = dict(finite=bool(np.isfinite(solver.f).all()), min=float(solver.f.min()),
               max=float(solver.f.max()), negative_count=int((solver.f < 0).sum()),
               dtype=str(solver.f.dtype))
    result = solver.finalize()
    summary = json.loads((dest / 'run_summary.json').read_text())
    validation = evaluate(result.config, summary, solver.field_snapshot())
    atomic_json(dest / 'validation.json', validation)
    data = dict(path=str(dest), stop_reason=solver.stop_reason, iterations=solver.iteration,
                compute_seconds=compute, total_seconds=time.perf_counter() - start,
                seconds_per_step=compute / solver.iteration,
                dimensionless_time=solver.iteration * config.flow.U_ref / t['L_ref'],
                population_check=pop, validation_status=validation['status'], metrics=validation['metrics'])
    atomic_json(dest / 'timing_and_population.json', data)
    readable = result_text(data, validation, summary, config, timing_only)
    (dest / '结果说明.txt').write_text(readable + '\n', encoding='utf-8')
    print(readable, flush=True)
    return data


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    loop = args.loop or needs_input(args)
    print('BGK 方腔参数输入：' + ('计算后可继续下一组。' if loop else '本次运行一组工况。'), flush=True)
    try:
        config = configure(args)
        while True:
            preflight(config)
            if args.dry_run:
                print('仅检查参数，尚未运行计算。', flush=True)
                return 0
            result = run(config, f'cavity_N{config.grid.NX}_Re{config.flow.Re:g}')
            # Keep the historical index for the N64/Re100-specific plotting script.
            atomic_json(ROOT / 'latest_run.json', {'simulation': result})
            print('\n本组运行结束。数值数据已保存；本脚本不会自动生成或更新图件。', flush=True)
            print('最新结果索引：' + str(ROOT / 'latest_run.json'), flush=True)
            exit_code = 1 if result['stop_reason'] == 'diverged' else 0
            if not loop:
                return exit_code
            while True:
                print('\n下一步：1 修改参数再运行（回车默认） | 2 原参数重跑 | 0 退出', flush=True)
                try:
                    choice = input('请选择 [1]：').strip().lower()
                except EOFError:
                    print('\n输入已结束，已保存本组结果，退出。', flush=True)
                    return exit_code
                if choice in ('0', 'q', 'quit', 'exit'):
                    print('已退出，所有已完成的结果均已保留。', flush=True)
                    return exit_code
                if choice in ('', '1'):
                    config = configure(build_parser().parse_args([]), previous=config)
                    break
                if choice == '2':
                    print('按原参数从初始条件重新计算，结果将保存到新目录。', flush=True)
                    break
                print('请输入 1、2 或 0。', flush=True)
    except EOFError:
        print('\n输入提前结束，未开始计算下一组；已完成的结果不受影响。自动运行请使用 --defaults 或指定 --re、--grid 等参数。', flush=True)
        return 2
    except (ValueError, OSError, MemoryError) as exc:
        if isinstance(exc, MemoryError):
            exc = '内存分配失败；请减小网格，或在内存更大的计算机上运行。'
        print(f'\n无法完成运行：{exc}', file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        print('\n用户已中止；已完成的结果已保留，未完成的计算不能作为最终结果。', flush=True)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
