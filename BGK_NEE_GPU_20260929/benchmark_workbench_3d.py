"""Exercise the real worker at 128 cubed, including SQLite, previews and exports."""
import datetime as dt
import json
from pathlib import Path
import time
import numpy as np

from dashboard.store import ROOT,Store,atomic_json
from dashboard.worker import run_case
from research_validation import fingerprints


def main():
    base=json.loads((ROOT/'validation/performance_after_128.json').read_text(encoding='utf-8'))
    p=base['parameters']|dict(max_iter=1000)
    folder=ROOT/'validation'/('performance_worker_'+dt.datetime.now().strftime('%Y%m%d_%H%M%S'))
    store=Store(folder)
    identifier=store.enqueue([dict(name='128 cubed optimized real worker',params=p)])[0]
    assert store.claim_next()['id']==identifier
    start=time.perf_counter();run_case(store,identifier);total=time.perf_counter()-start
    case=store.get(identifier)
    assert case['status']=='max_iter' and case['metrics']['iteration']==1000,case
    data=Path(case['data_dir'])
    request=json.loads((data/'run_request.json').read_text(encoding='utf-8'))
    timing=json.loads((data/'timing_and_population.json').read_text(encoding='utf-8'))
    assert request['research_source_sha256']==fingerprints() and timing['population_check']['finite']
    with np.load(data/'results.npz',allow_pickle=False) as fields:
        assert fields['rho'].shape==(128,128,128) and all(np.isfinite(fields[k]).all() for k in ('rho','ux','uy','uz'))
    with np.load(data/'preview.npz',allow_pickle=False) as preview:
        assert int(preview['iteration'])==1000
    rows=[json.loads(line) for line in (data/'progress.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [row['iteration'] for row in rows]==[1]+list(range(20,1001,20))
    assert case['metrics']['preview_skipped_updates']==0
    receipt=dict(passed=True,workspace=str(folder),case_id=identifier,source_sha256=fingerprints(),
        steps=1000,active_seconds=timing['compute_seconds'],active_steps_per_second=1000/timing['compute_seconds'],
        total_seconds_including_initialization_and_exports=total,progress_reports=len(rows),
        complete_fields=True,preview_iteration=1000,checkpoint_written=(data/'checkpoint.npz').is_file())
    atomic_json(ROOT/'validation/performance_worker.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
