"""Copy a checkpoint from the verified pre-optimization engine to the current one.

This is deliberately limited to the saved 2026-09-30 implementation. Unknown
source fingerprints remain rejected by the solver; the original is not changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from research_validation import ROOT, fingerprints, verified_research
from research_solver import atomic_npz


def migrate(input_path,output_path):
    input_path,output_path=Path(input_path).resolve(),Path(output_path).resolve()
    if input_path==output_path:raise ValueError('必须另存为新检查点，不能覆盖原文件')
    if output_path.exists():raise FileExistsError(output_path)
    current=verified_research('fused')
    before=ROOT/'validation/performance_before_20260930'
    receipt=json.loads((before/'correctness_research.json').read_text(encoding='utf-8'))
    if not receipt.get('passed'):raise ValueError('原版本未通过数值验收')
    old=receipt['source_sha256']
    if any(hashlib.sha256((before/name).read_bytes()).hexdigest()!=value for name,value in old.items()):
        raise ValueError('原版本存档源码与验收记录不一致')
    with np.load(input_path,allow_pickle=False) as saved:
        meta=json.loads(str(saved['metadata']))
        if meta.get('version')!=1 or meta.get('source_sha256')!=old:
            raise ValueError('仅接受本次优化前已验证版本的检查点')
        arrays={key:saved[key] for key in saved.files if key!='metadata'}
    for key in ('f','g'):
        if key in arrays and (arrays[key].dtype!=np.float64 or not np.isfinite(arrays[key]).all()):
            raise ValueError('分布函数精度或数值无效')
    meta['source_sha256']=current
    meta['migration']=dict(reason='2026-09-30 numerically verified GPU optimization; canonical arrays unchanged',
        from_source_sha256=old,original_checkpoint_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest())
    arrays['metadata']=np.array(json.dumps(meta))
    atomic_npz(output_path,**arrays)
    with np.load(input_path,allow_pickle=False) as old_data,np.load(output_path,allow_pickle=False) as new_data:
        for key in old_data.files:
            if key!='metadata':np.testing.assert_array_equal(old_data[key],new_data[key])
    return dict(input=str(input_path),output=str(output_path),iteration=int(arrays['iteration']),
                arrays_unchanged=True,source_sha256=current,original_sha256=meta['migration']['original_checkpoint_sha256'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args()
    report=migrate(args.input,args.output)
    args.output.with_suffix('.migration.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
