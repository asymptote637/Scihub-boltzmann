"""Read-only adapter for the BGK search output protocol. No solver imports."""
from __future__ import annotations
import json
import math
import subprocess
import time
from pathlib import Path
from control import snapshot, inspect_process
import psutil


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def read_records(path: Path) -> tuple:
    """Never consume an incomplete final append; expose corrupt complete rows."""
    if not path.exists():
        return [], ['尚无 progress.jsonl']
    data = path.read_bytes()
    lines = data.split(b'\n')
    warnings = []
    if lines[-1]:
        warnings.append('末行尚未完整写入，等待下一次刷新')
    records = []
    for i, line in enumerate(lines[:-1], 1):
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get('iteration'), (int, float)):
                raise ValueError('invalid iteration')
            records.append(row)
        except (ValueError, TypeError):
            warnings.append(f'第 {i} 行损坏，图中留空；请检查原始记录')
            records.append({'iteration': math.nan})
    return records, warnings


def process_info(pid: object, expected_script: Path) -> dict:
    try:
        p=inspect_process(pid)
        verified=bool(p and str(expected_script) in p['command'][1:] and p['uid']==psutil.Process().username())
        return {'verified':verified,'paused':bool(p and p['status']==psutil.STATUS_STOPPED),'detail':p or '进程不存在'}
    except psutil.Error as exc:
        return {'verified':False,'detail':str(exc)}


def status_text(summary: dict, latest: dict, matched: bool, age: float | None) -> str:
    reason = summary.get('stop_reason') or latest.get('stop_reason')
    names = {'converged': '速度残差达标 · 质量与验证另列', 'max_iter': '步数预算耗尽',
             'diverged': '数值保护停止', 'cancelled': '已取消', 'completed_steps': '指定步数完成'}
    if reason in names:
        return names[reason]
    if matched:
        return '进程已核验 · 记录延迟' if age is None or age > 60 else '计算中 · 进程已核验'
    return '未核验到对应进程 · 不能认定仍在计算'


def collect(root: Path, selected: str = '') -> dict:
    now = time.time()
    warnings = []
    state = read_json(root / 'search_state.json')
    current = state.get('current') or {}
    controls = snapshot(root, state)
    event_path = root / 'dashboard_control/state.json'
    event = read_json(event_path) if event_path.exists() else {}
    folders = sorted((root / 'calculation').glob('cavity_*'))
    target = selected or current.get('path')
    if not target and state.get('results'):
        target = state['results'][-1].get('path')
    folder = Path(target) if target else None
    output = {'root': str(root), 'read_at': now, 'state': state,
              'folders': [str(p) for p in folders], 'warnings': warnings, 'folder': str(folder) if folder else '',
              'records': [], 'latest': {}, 'config': {}, 'summary': {}, 'validation': {}, 'fields': None,
              'controls': controls, 'control_event': event, 'age': None, 'process': {'verified': False, 'detail': '无选中运行'}, 'status': '等待控制器提供运行路径'}
    if folder is None:
        return output
    # Do not silently select a different run if a source disappears.
    request = read_json(folder / 'run_request.json')
    records, record_warnings = read_records(folder / 'progress.jsonl')
    warnings.extend(record_warnings)
    latest = next((r for r in reversed(records) if math.isfinite(r['iteration'])), {})
    summary = read_json(folder / 'run_summary.json') if (folder / 'run_summary.json').exists() else {}
    validation = read_json(folder / 'validation.json') if (folder / 'validation.json').exists() else {}
    progress = folder / 'progress.jsonl'
    age = max(0, now - progress.stat().st_mtime) if progress.exists() else None
    is_current = str(folder) == current.get('path')
    process = process_info(current.get('pid'), root / 'calculation/run_baseline.py') if is_current else {'verified': False, 'detail': '历史运行；不使用当前 PID'}
    fields = None
    # Final summary + NPZ: never display an older run's field as the current field.
    if summary and (folder / 'results.npz').exists():
        import numpy as np
        try:
            with np.load(folder / 'results.npz', allow_pickle=False) as data:
                fields = {k: data[k].copy() for k in ('x', 'y', 'speed', 'solid_mask')}
        except (OSError, ValueError, KeyError, EOFError) as exc:
            warnings.append(f'流场尚不可读：{exc}')
    output.update(records=records, latest=latest, request=request, config=request['config'], summary=summary,
                  validation=validation, fields=fields, age=age, process=process,
                  status=status_text(summary, latest, process['verified'], age))
    if process.get('verified') and process.get('paused'):
        output['status'] = '计算进程已暂停 · 内存保留，可继续'
    if controls.get('available') and controls.get('supervisor_paused') and not controls.get('child_paused'):
        output['status'] += ' · 调度器已暂停'
    if not process['verified'] and event.get('action') == 'terminate' and event.get('outcome') == 'confirmed' and event.get('folder') == str(folder):
        output['status'] = '已由看板终止 · 当前计算与队列均已停止'
    return output
