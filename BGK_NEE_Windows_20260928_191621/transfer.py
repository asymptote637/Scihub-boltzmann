"""Portable entry point. Always use UTF-8; every numerical run starts fresh."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
PRESET=['--boundary','nee','--mass-policy','diagnostic','--grid','256','--re','5468',
        '--lid-speed','0.04','--tol','1e-6','--max-iter','1000000','--min-iter','2000',
        '--ramp-steps','500','--report-interval','200']

def verify():
    manifest=json.loads((ROOT/'MANIFEST_SHA256.json').read_text(encoding='utf-8'))
    errors=[name for name,digest in manifest.items() if not (ROOT/name).is_file() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
    if errors:raise SystemExit('Checksum mismatch: '+', '.join(errors))
    print('PASS: package SHA256,',len(manifest),'files',flush=True)

def run(path,args=()):
    return subprocess.call([sys.executable,'-u',str(ROOT/path),*args],cwd=ROOT,env=os.environ.copy())

def main():
    os.environ['PYTHONUTF8']='1'
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
    if not sys.flags.utf8_mode:
        return subprocess.call([sys.executable,'-X','utf8',str(Path(__file__).resolve()),*sys.argv[1:]],env=os.environ.copy())
    parser=argparse.ArgumentParser(description='BGK NEE portable package (fresh initial state only)')
    parser.add_argument('action',choices=['verify','check','smoke','run-current','search','dashboard','boundary-check'])
    args=parser.parse_args()
    if args.action=='verify':verify();return 0
    if args.action=='check':
        verify()
        import numpy,matplotlib,psutil
        print('Python:',sys.version,'NumPy:',numpy.__version__,'Matplotlib:',matplotlib.__version__,'psutil:',psutil.__version__,flush=True)
        code=run('simulation/search.py',['--self-test'])
        return code or run('simulation/calculation/run_baseline.py',PRESET+['--dry-run'])
    if args.action=='smoke':
        return run('simulation/calculation/run_baseline.py',['--boundary','nee','--mass-policy','diagnostic','--grid','16','--re','100','--lid-speed','0.04','--tol','1e-6','--max-iter','300','--min-iter','2000','--ramp-steps','500','--report-interval','100'])
    if args.action=='run-current':return run('simulation/calculation/run_baseline.py',PRESET)
    if args.action=='search':return run('simulation/search.py')
    if args.action=='dashboard':return run('dashboard/dashboard.py',['--root',str(ROOT/'simulation')])
    if args.action=='boundary-check':return run('simulation/verify_boundary.py')

if __name__=='__main__':raise SystemExit(main())
