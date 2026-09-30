"""Real signals on disposable processes; never touch the research computation."""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dashboard'))
from control import ControlError, execute, inspect_process, snapshot

CHILD = '''from pathlib import Path
import json,time
folder=Path(__file__).parent/'cavity_test'
i=0
try:
    while True:
        i+=1
        temp=folder/'counter.tmp';temp.write_text(str(i));temp.replace(folder/'counter')
        with (folder/'progress.jsonl').open('a') as f:f.write(json.dumps(dict(iteration=i,residual=1/i,stop_reason='running'))+'\\n')
        time.sleep(.03)
except KeyboardInterrupt:
    (folder/'interrupted').write_text(str(i))
'''
SUPERVISOR = '''from pathlib import Path
import subprocess,sys,os,json,time
root=Path(__file__).parent
command=[sys.executable,'-u',str(root/'calculation/run_baseline.py')]
p=subprocess.Popen(command)
(root/'launch.json').write_text(json.dumps(dict(pid=os.getpid(),command=[sys.executable,'-u',str(root/'search.py')])))
(root/'search_state.json').write_text(json.dumps(dict(status='running',supervisor_pid=os.getpid(),current=dict(pid=p.pid,command=command,path=str(root/'calculation/cavity_test')),results=[])))
while p.poll() is None:time.sleep(.03)
if not (root/'STOP').exists():(root/'next_launched').write_text('unexpected next case')
'''


class ProcessControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='LBM_控制测试_')
        self.root = Path(self.tmp.name).resolve()
        self.folder = self.root / 'calculation/cavity_test'; self.folder.mkdir(parents=True)
        (self.folder / 'run_request.json').write_text('{"config":{}}')
        (self.root / 'calculation/run_baseline.py').write_text(CHILD)
        (self.root / 'search.py').write_text(SUPERVISOR)
        self.proc = subprocess.Popen([sys.executable, '-u', str(self.root / 'search.py')], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.token = None
        until = time.monotonic() + 5
        while time.monotonic() < until:
            token = snapshot(self.root)
            if token.get('available') and (self.folder / 'counter').exists():
                self.token = token; break
            time.sleep(.03)
        self.assertIsNotNone(self.token, token)

    def tearDown(self):
        if self.token:
            for role in ('supervisor', 'child'):
                saved = self.token[role]; p = inspect_process(saved['pid'])
                if p and p['started'] == saved['started'] and p['command'] == saved['command']:
                    import psutil
                    try:
                        proc=psutil.Process(p['pid']);proc.terminate()
                        if os.name!='nt': proc.resume()
                    except psutil.NoSuchProcess:
                        pass
        self.proc.wait(timeout=5)
        self.tmp.cleanup()

    def test_pause_resume_keeps_memory_and_pid(self):
        execute(self.root, 'pause', self.token)
        s = snapshot(self.root)
        self.assertTrue(s['child_paused'] and s['supervisor_paused'])
        count = int((self.folder / 'counter').read_text())
        time.sleep(.2)
        self.assertEqual(int((self.folder / 'counter').read_text()), count)
        execute(self.root, 'resume', s)
        time.sleep(.2)
        s = snapshot(self.root)
        self.assertEqual(s['child']['pid'], self.token['child']['pid'])
        self.assertFalse(s['child_paused'] or s['supervisor_paused'])
        self.assertGreater(int((self.folder / 'counter').read_text()), count)

    def assert_terminated(self):
        result = execute(self.root, 'terminate', snapshot(self.root))
        self.assertEqual(result['outcome'], 'confirmed')
        self.proc.wait(timeout=5)
        self.assertIsNone(inspect_process(self.token['child']['pid']))
        self.assertTrue((self.root / 'STOP').exists())
        self.assertFalse((self.root / 'next_launched').exists())
        self.assertTrue((self.folder / 'progress.jsonl').exists())
        from reader import collect
        self.assertIn('已由看板终止', collect(self.root)['status'])

    def test_terminate_running_stops_queue_and_preserves_logs(self):
        self.assert_terminated()

    def test_terminate_paused_processes(self):
        execute(self.root, 'pause', self.token)
        self.assert_terminated()

    def test_stale_identity_rejected_without_signal(self):
        token = json.loads(json.dumps(self.token)); token['child']['started'] = 'stale'
        with self.assertRaises(ControlError): execute(self.root, 'terminate', token)
        self.assertFalse((self.root / 'STOP').exists())
        self.assertFalse(snapshot(self.root)['child_paused'])

    def test_stop_marker_blocks_resume(self):
        execute(self.root, 'pause', self.token)
        (self.root / 'STOP').write_text('existing user stop')
        with self.assertRaises(ControlError): execute(self.root, 'resume', self.token)
        self.assertTrue(snapshot(self.root)['child_paused'])
        self.assertEqual((self.root / 'STOP').read_text(), 'existing user stop')

    def test_history_or_other_root_cannot_control_current(self):
        token = dict(self.token); token['folder'] = str(self.root / 'calculation/old')
        with self.assertRaises(ControlError): execute(self.root, 'pause', token)
        self.assertFalse(snapshot(self.root)['supervisor_paused'])


if __name__ == '__main__': unittest.main()
