"""Check the current source against correctness, performance and native UI evidence."""
import hashlib
import json
from pathlib import Path
import re

from dashboard.store import Store,atomic_json,ACTIVE
from reference import verify_reference
from research_options import generic_engine
from research_solver import physical_parameters
from research_validation import ROOT,fingerprints,verified_research
from run_gpu import verified_backend
from model_validation import verified_models


def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source=fingerprints()
    for backend in ('cpu','array','fused'):assert verified_research(backend)==source
    verified_models();verified_backend('array');verified_backend('fused')
    scientific=read('validation/correctness_research.json')
    assert len(scientific['checks'])==18 and all(c['passed'] for c in scientific['checks'].values())
    pairs=[v for name in ('gpu_agreement','rectangular_obstacles','advanced_gpu')
           for v in scientific['checks'][name]['result'].values()]
    assert len(pairs)==112 and max(pairs)<5e-12
    timings={}
    for label in ('128','resume','thermal64'):
        before=read('validation/performance_before_'+label+'.json')
        after=read('validation/performance_after_'+label+'.json')
        assert before['passed'] and after['passed'] and after['source_sha256']==source
        assert physical_parameters(before['parameters'])==physical_parameters(after['parameters'])
        for key in ('warmup','steps','repetitions'):assert before[key]==after[key]
        assert before['parameters']['report_interval']==after['parameters']['report_interval']==20
        assert len(before['trials'])==len(after['trials'])==3
        for a,b in zip(before['trials'],after['trials']):
            assert a['iteration']==b['iteration'] and b['max_absolute_field_difference']<2e-11
        archived=ROOT/'validation/performance_before_20260930'
        assert all(sha(archived/name)==value for name,value in before['source_sha256'].items())
        speedup=before['median_ms_per_step']/after['median_ms_per_step']
        assert speedup>2,(label,speedup)
        timings[label]=dict(before_ms=before['median_ms_per_step'],after_ms=after['median_ms_per_step'],
            speedup=speedup,max_field_error=max(t['max_absolute_field_difference'] for t in after['trials']),
            before_range_ms=[before['min_ms_per_step'],before['max_ms_per_step']],
            after_range_ms=[after['min_ms_per_step'],after['max_ms_per_step']])
    log=(ROOT/'validation/performance_pytest.log').read_text(encoding='utf-8-sig')
    count=int(re.findall(r'(\d+) passed',log)[-1]);assert count>=149
    ui=read('validation/advanced_ui.json')
    assert ui['passed'] and not ui['errors'] and len(ui['jobs'])==28
    assert all(sha(ROOT/name)==value for name,value in ui['source_sha256'].items())
    store=Store(Path(ui['workspace']))
    for identifier in ui['jobs']:
        case=store.get(identifier)
        assert case['status'] in ('completed_steps','max_iter','converged')
        if generic_engine(case['params']):
            request=json.loads((Path(case['data_dir'])/'run_request.json').read_text(encoding='utf-8'))
            assert request['research_source_sha256']==source
    migration=read('workspace/runs/74a3deda04a343269c6f760ed20d0dc8/checkpoint_optimized.migration.json')
    assert migration['arrays_unchanged'] and migration['source_sha256']==source
    assert sha(Path(migration['input']))==migration['original_sha256']
    baseline=Store().get('74a3deda04a343269c6f760ed20d0dc8')
    assert baseline['status']=='cancelled' and baseline['metrics']['iteration']==3721
    active=[c['id'] for c in Store().cases() if c['status'] in (*ACTIVE,'queued')]
    files=[p for p in ROOT.glob('*.py')]+[p for p in (ROOT/'tests').glob('*.py')]+[p for p in (ROOT/'dashboard').glob('*.py')]
    files += [ROOT/n for n in ('PERFORMANCE_OPTIMIZATION.md','README_WORKBENCH.md','open_dashboard_windows.bat')]
    receipts=[ROOT/('validation/'+n) for n in ('correctness_research.json','advanced_ui.json','performance_pytest.log')]
    receipts += list((ROOT/'validation').glob('performance_before_*.json'))+list((ROOT/'validation').glob('performance_after_*.json'))
    report=dict(passed=True,source_sha256=source,frozen_reference_files=verify_reference(),
        scientific_check_groups=18,cpu_gpu_trajectory_pairs=112,max_trajectory_difference=max(pairs),
        tests_passed=count,native_ui_jobs=28,benchmark_results=timings,migration=migration,
        default_workspace_active_at_audit=active,
        delivery_file_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
        receipt_sha256={str(p.relative_to(ROOT)):sha(p) for p in receipts})
    atomic_json(ROOT/'validation/performance_delivery.json',report)
    print(json.dumps({k:v for k,v in report.items() if 'sha256' not in k and k!='migration'},indent=2),flush=True)


if __name__=='__main__':main()
