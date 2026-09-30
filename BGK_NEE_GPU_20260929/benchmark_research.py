"""Matched warm timings of the research GPU solver, including step guards/reports.

Use --source-root to execute an archived implementation with the same benchmark.
The current user workspace is never enqueued or modified.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--grid', type=int, default=128)
    ap.add_argument('--lattice', default='D3Q19')
    ap.add_argument('--scenario', default='cavity')
    ap.add_argument('--backend', default='fused')
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--warmup', type=int, default=30)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--report-interval', type=int, default=20)
    ap.add_argument('--compare', type=Path)
    ap.add_argument('--checkpoint', type=Path)
    ap.add_argument('--save-fields', action='store_true')
    ap.add_argument('--profile', action='store_true')
    args = ap.parse_args()
    if min(args.steps,args.warmup,args.repeats,args.report_interval)<1:
        raise ValueError('Benchmark counts must be positive')
    sys.path.insert(0, str(args.source_root.resolve()))
    import gpu_environment
    import cupy as cp
    import numpy as np
    from dashboard.store import DEFAULTS, validate_parameters, atomic_json
    from research_options import EXTRA_DEFAULTS
    from research_runtime import build_config, solver_class
    from research_validation import fingerprints
    periodic = args.scenario in {'thermal_wave', 'shear_wave', 'taylor_green'}
    p = validate_parameters(DEFAULTS | EXTRA_DEFAULTS | dict(backend=args.backend,
        grid=args.grid, grid_y=args.grid, grid_z=args.grid, lattice=args.lattice, scenario=args.scenario,
        boundary='periodic' if periodic else 'halfway', collision='TRT', re=5000, lid_speed=.04,
        max_iter=100000, min_iter=2000, ramp_steps=500, report_interval=args.report_interval,
        tol=1e-6, run_mode='steady' if args.scenario=='cavity' else 'transient'))
    if args.checkpoint:
        with np.load(args.checkpoint,allow_pickle=False) as saved:
            metadata=json.loads(str(saved['metadata']))
            start_iteration=int(saved['iteration'])
        p=validate_parameters(metadata['params'] | dict(backend=args.backend,
            resume_from=str(args.checkpoint.resolve()),report_interval=args.report_interval,
            max_iter=max(metadata['params']['max_iter'],start_iteration+args.warmup+args.steps+100),
            snapshot_interval=0,checkpoint_interval=0))
    result = dict(passed=False, parameters=p, source_sha256=fingerprints(), trials=[],
        warmup=args.warmup, steps=args.steps, repetitions=args.repeats,
        excludes='initialization, compilation, final file compression; includes reports and every-step protection',
        device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(), dtype='float64')
    result['benchmark_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    stream = cp.cuda.get_current_stream()
    def telemetry():
        fields='name,driver_version,temperature.gpu,utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem'
        info=subprocess.run(['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader'],capture_output=True,text=True,timeout=10)
        return dict(fields=fields,value=info.stdout.strip(),returncode=info.returncode)
    result['gpu_before']=telemetry()
    def advance(s, count):
        for _ in range(count):
            s.step()
            if s.iteration==1 or s.iteration % args.report_interval==0: s.report()
            if s.stop_reason!='running': raise AssertionError((s.iteration,s.stop_reason,s._failure))
        stream.synchronize()
    for repeat in range(args.repeats):
        s=solver_class(p)(build_config(p))
        assert all(getattr(s,k).dtype==cp.float64 for k in ('f','rho','ux','uy','uz','g','T') if getattr(s,k,None) is not None)
        advance(s,args.warmup)
        start=time.perf_counter()
        advance(s,args.steps)
        seconds=time.perf_counter()-start
        row=dict(repeat=repeat+1, seconds=seconds, ms_per_step=1000*seconds/args.steps,
                 steps_per_second=args.steps/seconds, iteration=s.iteration,
                 pool_used_bytes=cp.get_default_memory_pool().used_bytes(),
                 pool_reserved_bytes=cp.get_default_memory_pool().total_bytes())
        if args.compare:
            with np.load(args.compare,allow_pickle=False) as ref:
                row['max_absolute_field_difference']=max(float(np.max(abs(cp.asnumpy(getattr(s,k))-ref[k]))) for k in ref.files)
            assert row['max_absolute_field_difference']<2e-11,row
        if args.save_fields and repeat==0:
            np.savez_compressed(args.output.with_suffix('.npz'), **{k:cp.asnumpy(getattr(s,k))
                for k in ('f','rho','ux','uy','uz','g','T') if getattr(s,k,None) is not None})
        result['trials'].append(row)
        atomic_json(args.output,result)
        print(json.dumps(row),flush=True)
        if args.profile and repeat==args.repeats-1:
            # Synchronized phase timings are diagnostic, separate from uninstrumented trials.
            phases={}
            for name in ('advance_populations','macros','report'):
                original=getattr(s,name)
                def measured(*a,_name=name,_fn=original,**kw):
                    stream.synchronize();begin=time.perf_counter()
                    value=_fn(*a,**kw);stream.synchronize()
                    phases[_name]=phases.get(_name,0)+time.perf_counter()-begin
                    return value
                setattr(s,name,measured)
            begin=time.perf_counter();advance(s,40);total=time.perf_counter()-begin
            result['profile_seconds_for_40_steps']=dict(phases,total=total,other=total-sum(phases.values()))
        del s
        gc.collect();cp.get_default_memory_pool().free_all_blocks()
    times=[t['ms_per_step'] for t in result['trials']]
    result.update(passed=True,median_ms_per_step=statistics.median(times),min_ms_per_step=min(times),max_ms_per_step=max(times))
    result['gpu_after']=telemetry()
    atomic_json(args.output,result)
    print('PASS',result['median_ms_per_step'],flush=True)


if __name__=='__main__': main()
