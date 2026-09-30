"""Sequential, locked, evidence-preserving NEE comparison search."""
from pathlib import Path
import datetime,hashlib,json,math,os,subprocess,sys,time
from portable_os import exclusive_lock,assert_no_runner

ROOT=Path(__file__).resolve().parent
CALC=ROOT/'calculation'
PLAN=json.loads((ROOT/'plan.json').read_text())
STATE=ROOT/'search_state.json'
PASS='velocity_criterion_met'
LABELS={PASS:'速度残差达标（质量另列）','budget_exhausted':'100万步内速度未达标',
        'mach_protection_stop':'Ma保护停止','numerical_failure':'非有限或非正密度停止'}


def atomic(path,data):
    def clean(v):
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        if isinstance(v,list):return [clean(x) for x in v]
        if isinstance(v,float) and not math.isfinite(v):return None
        return v
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(data),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def classify(summary):
    reason=summary['stop_reason'];error=summary.get('error','')
    if reason=='converged':return PASS
    if reason=='max_iter':return 'budget_exhausted'
    if reason=='diverged':
        if error=='Actual maximum Mach exceeded configured limit':return 'mach_protection_stop'
        if error in ('Non-finite population or macroscopic field','Non-positive density'):return 'numerical_failure'
    raise RuntimeError('Unclassified execution state: '+reason+' '+error)


def bracket(results):
    passed=[r['Re'] for r in results if r['classification']==PASS]
    failed=[r['Re'] for r in results if r['classification']!=PASS]
    low=max(passed) if passed else None;high=min(failed) if failed else None
    if low is not None and high is not None and low>=high:
        raise RuntimeError('Tested outcomes are nonmonotonic; do not bisect or call a critical Re.')
    return low,high


def next_case(results):
    tested={r['Re'] for r in results}
    for re in PLAN['seed_res']:
        if re not in tested:return re,'running'
    low,high=bracket(results)
    if low is None:
        if high<=PLAN['min_search_re']:return None,'no_pass_at_minimum'
        return max(PLAN['min_search_re'],high//2),'running'
    if high is None:return max(10000,low*2),'running'
    if high-low<=PLAN['resolution_re']:return None,'complete'
    return (low+high)//2,'running'


def save(state):
    state['updated_at']=datetime.datetime.now().astimezone().isoformat()
    try:
        state['low'],state['high']=bracket(state['results'])
    except RuntimeError as exc:
        state.update(low=None,high=None,status='nonmonotonic_results',error=str(exc))
    atomic(STATE,state)
    lines=['# 非平衡外推边界比较进度','',f"更新：{state['updated_at']}；状态：{state['status']}",'',
        '256×256存储节点，L=255，U=.04；书附录D角点及内部节点单步残差<1e-6；质量独立报告、不修正；每组上限100万步，Ma保护.1。','',
        '| Re | 结果 | 步数 | 末步残差 | 质量相对漂移 | 末步空间最大Ma |','|---:|---|---:|---:|---:|---:|']
    for r in state['results']:
        def fmt(x):return f'{x:.12g}' if isinstance(x,(float,int)) else '无有限值'
        lines.append(f"| {r['Re']} | {LABELS[r['classification']]} | {r['iteration']} | {fmt(r['residual'])} | {fmt(r['mass_drift'])} | {fmt(r['max_mach'])} |")
    lines += ['',f"NEE已测通过端：{state['low']}；未通过端：{state['high']}。",'',
              '半格反弹历史：Re=5312达到判据，Re=5468触发Ma保护。不能把历史结果作为NEE已通过点。']
    if state.get('current'):lines += ['', '当前计算：'+json.dumps(state['current'],ensure_ascii=False)]
    if state.get('error'):lines += ['', '执行异常（不计数值失败）：'+state['error']]
    lines += ['', '速度达标不保证质量守恒；测试区间不等于物理失稳临界值，不证明所有Re的单调性。']
    (ROOT/'搜索进度.md').write_text('\n'.join(lines)+'\n')


def verify_source():
    for rel,digest in PLAN['source_sha256'].items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Source changed after experiment freeze: '+rel)


def read_progress(folder):
    p=folder/'progress.jsonl'
    if not p.exists():return None
    with p.open('rb') as f:
        f.seek(max(0,p.stat().st_size-8192));lines=f.read().decode(errors='replace').splitlines()
    for line in reversed(lines):
        try:return json.loads(line)
        except ValueError:pass
    return None


def ingest(state,folder,re):
    s=json.loads((folder/'run_summary.json').read_text())
    timing=json.loads((folder/'timing_and_population.json').read_text())
    request=json.loads((folder/'run_request.json').read_text());c=request['config']
    for key,value in [('NX',256),('NY',256)]:assert c['grid'][key]==value
    assert c['flow']['Re']==re and c['flow']['U_ref']==.04 and c['flow']['initial_velocity']=='rest'
    assert c['flow']['body_force_x']==c['flow']['body_force_y']==0
    assert c['derived']['L_ref']==255
    for key in ['tol','max_iter','min_iter','ramp_steps','report_interval','mass_tolerance','max_mach']:
        assert c['convergence'][key]==PLAN[key],key
    assert c['convergence']['mass_criterion'] is False and c['convergence']['consecutive_reports']==1
    assert all(b['type']=='non_equilibrium_extrapolation' for b in c['boundaries'].values())
    assert c['boundaries']['top']['ux']==.04
    assert all(c['boundaries'][side]['ux']==c['boundaries'][side]['uy']==0 for side in ('left','right','bottom'))
    assert s['convergence_policy']=='velocity_only_mass_reported'
    assert s['residual_domain']=='interior_without_on_node_walls'
    for name,digest in s['source_sha256'].items():
        assert PLAN['source_sha256']['calculation/source_snapshot/'+name]==digest
    assert timing['iterations']==s['iteration'] and timing['stop_reason']==s['stop_reason']
    category=classify(s);diag=s['final_diagnostics']
    if category==PASS:assert diag['residual']<PLAN['tol'] and s['iteration']>=PLAN['min_iter']
    if any(r['Re']==re for r in state['results']):raise RuntimeError('Duplicate result ingestion')
    state['results'].append(dict(Re=re,classification=category,iteration=s['iteration'],residual=diag['residual'],
        mass_drift=diag['mass_drift'],max_mach=diag['max_mach'],error=s.get('error',''),path=str(folder)))
    state['current']=None;save(state)


def assert_no_child():
    assert_no_runner(CALC/'run_baseline.py')


def main():
    try:lock=exclusive_lock(ROOT/'search.lock')
    except BlockingIOError:raise SystemExit('A controller is already running.')
    state=json.loads(STATE.read_text()) if STATE.exists() else dict(status='running',results=[],current=None)
    if state['status']=='complete':return
    try:
        verify_source();assert_no_child()
        prior=state.get('current')
        if prior:
            folders=set(CALC.glob('cavity_N256_*'))-{Path(p) for p in prior['before_folders']}
            if len(folders)!=1:raise RuntimeError('Unfinished prior launch requires inspection; no rerun')
            folder=folders.pop()
            if not (folder/'timing_and_population.json').exists():raise RuntimeError('Interrupted calculation; preserve '+str(folder))
            ingest(state,folder,prior['Re'])
        accounted={Path(r['path']) for r in state['results']}
        unaccounted=set(CALC.glob('cavity_N256_*'))-accounted
        if unaccounted:raise RuntimeError('Unaccounted outputs; inspect before launching: '+str(unaccounted))
        state.update(status='running',supervisor_pid=os.getpid());state.pop('error',None)
        atomic(ROOT/'launch.json',dict(pid=os.getpid(),command=[sys.executable,'-u',str(ROOT/'search.py')],executable=sys.executable,launched_at=datetime.datetime.now().astimezone().isoformat()))
        save(state)
        (ROOT/'logs').mkdir(exist_ok=True)
        while True:
            re,status=next_case(state['results'])
            if re is None:state['status']=status;save(state);return
            if (ROOT/'STOP').exists():state['status']='stopped_by_request';save(state);return
            if re>PLAN['safety_re_cap'] or len(state['results'])>=PLAN['max_new_runs']:
                state['status']='search_limit_reached';save(state);return
            verify_source();assert_no_child()
            cmd=[sys.executable,'-u',str(CALC/'run_baseline.py'),'--boundary','nee','--mass-policy','diagnostic','--re',str(re),'--grid','256','--lid-speed','0.04']
            for key in ['tol','max_iter','min_iter','ramp_steps','report_interval']:
                cmd += ['--'+key.replace('_','-'),str(PLAN[key])]
            before=set(CALC.glob('cavity_N256_*'));log=ROOT/'logs'/f'Re{re}_{len(state["results"]):02d}.log'
            state['current']=dict(Re=re,pid=None,command=cmd,log=str(log),before_folders=[str(p) for p in before]);save(state)
            with log.open('a') as f:
                child=subprocess.Popen(cmd,cwd=CALC,stdout=f,stderr=subprocess.STDOUT)
                state['current']['pid']=child.pid;save(state)
                while child.poll() is None:
                    folders=set(CALC.glob('cavity_N256_*'))-before
                    if len(folders)==1:
                        folder=next(iter(folders));state['current'].update(path=str(folder),progress=read_progress(folder))
                    save(state);time.sleep(15)
            folders=set(CALC.glob('cavity_N256_*'))-before
            if len(folders)!=1 or child.returncode not in (0,1):raise RuntimeError('Execution error; inspect '+str(log))
            ingest(state,folders.pop(),re)
    except Exception as exc:
        state.update(status='error',error=repr(exc));save(state);raise


if __name__=='__main__':
    if '--self-test' in sys.argv:
        assert next_case([])[0]==5468
        assert next_case([{'Re':5468,'classification':'mach_protection_stop'}])[0]==5312
        examples=[{'Re':5468,'classification':PASS},{'Re':5312,'classification':PASS}]
        assert next_case(examples)[0]==10936
        assert next_case([{'Re':5312,'classification':PASS},{'Re':5468,'classification':'mach_protection_stop'}])==(None,'complete')
        assert next_case([{'Re':5312,'classification':'mach_protection_stop'},{'Re':5468,'classification':'mach_protection_stop'}])[0]==2656
        assert classify({'stop_reason':'max_iter'})=='budget_exhausted'
        assert classify({'stop_reason':'diverged','error':'Actual maximum Mach exceeded configured limit'})=='mach_protection_stop'
        try:classify({'stop_reason':'diverged','error':'unrecognized error'})
        except RuntimeError:pass
        else:raise AssertionError('Unknown errors cannot be numerical failures')
        print('PASS: seed queue, expansion, downward search, bisection and stop classification')
    else:main()
