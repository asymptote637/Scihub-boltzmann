"""Saved experiment configurations, Cartesian scans and sampled signals."""
import itertools
import json
from pathlib import Path
import re
import numpy as np
from .store import atomic_json, validate_parameters
from research_options import BUOYANT


def scan_requests(params, axes, title='', group=''):
    if len(axes)>2 or len({a for a,_ in axes}) != len(axes):
        raise ValueError('最多使用两个不同的扫描参数')
    allowed = {'re','grid','grid_y','grid_z','viscosity','rayleigh','prandtl','diffusivity',
               'force_x','lid_speed','drive_period','force_amplitude','obstacle_size'}
    items=[]
    for key,text in axes:
        if key not in allowed: raise ValueError('扫描参数无效')
        values = str(text).replace('，',',').split(',')
        if not 1<=len(values)<=100 or any(not v.strip() for v in values):
            raise ValueError('每个扫描轴需要 1–100 个逗号分隔的数值')
        if key=='re' and params['viscosity']:
            raise ValueError('扫描 Re 时将 ν 设为 0，使黏度随 Re 改变')
        if key in {'rayleigh','prandtl'} and params['scenario'] not in BUOYANT:
            raise ValueError('Ra/Pr 扫描用于浮力热对流场景')
        if key=='diffusivity' and params['scenario']!='thermal_wave':
            raise ValueError('α 扫描用于温度波；浮力热对流使用 α=ν/Pr')
        if key=='grid_z' and not params['lattice'].startswith('D3'):
            raise ValueError('Nz 扫描只适用于三维模型')
        if key=='drive_period' and params['scenario'] not in {'oscillatory_channel','oscillating_couette'}:
            raise ValueError('周期扫描用于振荡驱动场景')
        if key=='force_amplitude' and params['scenario']!='oscillatory_channel':
            raise ValueError('加速度幅值扫描用于周期体力通道')
        if key=='obstacle_size' and params['scenario'] not in {'obstacle','cylinder_wake'}:
            raise ValueError('障碍尺寸扫描用于含障碍物的场景')
        if key=='lid_speed' and params['viscosity'] and params['scenario'] in {'natural_convection','rayleigh_benard'}:
            raise ValueError('指定 ν 的纯浮力场景不由参考 U 驱动；请扫描 Ra、Pr 或 ν')
        items.append(values)
    count=int(np.prod([len(v) for v in items]))
    if count>100: raise ValueError('双参数笛卡尔积最多 100 个任务')
    requests=[]
    for index,values in enumerate(itertools.product(*items),1):
        updates=dict(zip([k for k,_ in axes],values))
        p=validate_parameters(dict(params,**updates))
        label=title
        if count>1:
            label=(title or '参数矩阵')+' · '+', '.join(f'{k}={p[k]:g}' for k in updates)+f' ({index}/{count})'
        requests.append(dict(params=p,name=label,group=group))
    return requests


def save_configuration(path, params, name='', group=''):
    p=validate_parameters(params)
    atomic_json(Path(path),dict(format='lbm-workbench-experiment',version=1,params=p,name=name,group=group))


def read_configuration(path):
    source=Path(path)
    if source.stat().st_size>1024*1024: raise ValueError('配置文件超过 1 MB')
    data=json.loads(source.read_text(encoding='utf-8-sig'))
    if not isinstance(data,dict) or data.get('format')!='lbm-workbench-experiment' or data.get('version')!=1:
        raise ValueError('需要工作台导出的版本 1 实验配置 JSON')
    data['params']=validate_parameters(data['params'])
    return data


def snapshots(folder):
    root=Path(folder)/'snapshots'
    result=[]
    for path in root.glob('fields_*.npz'):
        match=re.fullmatch(r'fields_(\d+)\.npz',path.name)
        if match: result.append((int(match[1]),str(path)))
    return sorted(result)


def load_snapshot(path, folder):
    from .analysis import _load_field_archive
    source=Path(path).resolve()
    parent=Path(folder).resolve()
    if source.parent != parent/'snapshots' or not re.fullmatch(r'fields_\d+\.npz',source.name):
        raise ValueError('快照必须来自当前算例的 snapshots 目录')
    fields=_load_field_archive(source,parent,False)
    step=int(source.stem.split('_')[1])
    if fields['preview_iteration']!=step: raise ValueError('快照步数与文件名不符')
    fields['snapshot']=True
    return fields


def signal_series(records, key, probe=None):
    times,values=[],[]
    for row in records:
        data=row
        if probe is not None:
            probes=row.get('probes',[])
            data=probes[probe] if isinstance(probes,list) and probe<len(probes) else {}
        try: t=float(row['iteration']);v=float(data[key])
        except (KeyError,TypeError,ValueError): t=float(row.get('iteration') or np.nan);v=np.nan
        times.append(t);values.append(v)
    return np.asarray(times),np.asarray(values)


def spectrum(times, values, start_fraction=0):
    """Hann-window amplitude; trim only irregular edges, never resample gaps."""
    times=np.asarray(times,dtype=float);values=np.asarray(values,dtype=float)
    if times.shape!=values.shape or times.ndim!=1 or not 0<=start_fraction<1:
        raise ValueError('频谱输入形状或起始比例无效')
    first=int(len(times)*start_fraction)
    t,y=times[first:],values[first:]
    if len(t)<16 or not np.isfinite(t).all() or not np.isfinite(y).all():
        raise ValueError('频谱至少需要 16 个连续、有效的报告点；固体内探针没有流体值')
    dt_values=np.diff(t)
    dt=float(np.median(dt_values))
    if dt<=0: raise ValueError('报告时间必须严格递增')
    trimmed=0
    if not np.isclose(t[1]-t[0],dt,rtol=0,atol=1e-9): t,y=t[1:],y[1:];trimmed+=1
    if not np.isclose(t[-1]-t[-2],dt,rtol=0,atol=1e-9): t,y=t[:-1],y[:-1];trimmed+=1
    if len(t)<16 or not np.allclose(np.diff(t),dt,rtol=0,atol=1e-9):
        raise ValueError('报告间隔不均匀或存在内部缺口；不能直接计算 FFT')
    win=np.hanning(len(y));mean=float(y.mean())
    amplitude=2*np.abs(np.fft.rfft((y-mean)*win))/win.sum()
    amplitude[0]*=.5
    if len(y)%2==0: amplitude[-1]*=.5
    frequency=np.fft.rfftfreq(len(y),dt)
    peak=int(np.argmax(amplitude[1:]))+1
    resolved=bool(amplitude[peak]>1e-12*max(1,np.max(abs(y))))
    return dict(frequency=frequency,amplitude=amplitude,peak_frequency=float(frequency[peak]),
                peak_amplitude=float(amplitude[peak]),resolution=1/(len(y)*dt),nyquist=.5/dt,
                samples=len(y),dt=dt,trimmed=trimmed,start_dropped=first,mean=mean,resolved=resolved)
