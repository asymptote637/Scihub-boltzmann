"""Verified Windows/POSIX control via psutil. No solver changes."""
from __future__ import annotations
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
import psutil
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'simulation'))
from portable_os import exclusive_lock

class ControlError(RuntimeError): pass

def worker_process(pid: int) -> psutil.Process:
    p = psutil.Process(pid)
    # Windows venv launchers wait for a second Python process running the script.
    def python_in_directory(executable: str, reference: str) -> bool:
        path = Path(executable)
        return (path.name.lower() in ('python.exe', 'pythonw.exe')
                and os.path.normcase(str(path.parent)) == os.path.normcase(str(Path(reference).parent)))
    if os.name == 'nt' and python_in_directory(p.exe(), sys.executable):
        command = p.cmdline()
        base = getattr(sys, '_base_executable', sys.executable)
        candidates = [child for child in p.children()
                      if python_in_directory(child.exe(), base)
                      and child.cmdline()[1:] == command[1:]
                      and child.username() == p.username()]
        if len(candidates) > 1:
            raise ControlError('Python launcher has multiple matching workers')
        if candidates:
            return candidates[0]
    return p

def inspect_process(pid: object) -> dict | None:
    if not isinstance(pid,int) or isinstance(pid,bool) or pid<=1:return None
    try:
        p=worker_process(pid)
        if p.status()==psutil.STATUS_ZOMBIE:return None
        return dict(pid=p.pid,ppid=p.ppid(),uid=p.username(),started=p.create_time(),command=p.cmdline(),status=p.status())
    except psutil.NoSuchProcess:return None

def identity(p:dict)->dict:
    return {k:p[k] for k in ('pid','ppid','uid','started','command')}

def matches(p:dict,expected:list,script:Path)->bool:
    return (str(script) in expected and len(expected)>1 and p['command'][1:]==expected[1:]
            and p['uid']==psutil.Process().username())

def snapshot(root:Path,state:dict|None=None)->dict:
    root=root.resolve()
    try:
        state=state if state is not None else json.loads((root/'search_state.json').read_text(encoding='utf-8'))
        current=state.get('current') or {}
        if not current.get('path'):raise ControlError('当前没有可控制的运行')
        parent=inspect_process(state.get('supervisor_pid'));child=inspect_process(current.get('pid'))
        if not parent or not child:raise ControlError('调度器或计算进程已退出')
        launch=json.loads((root/'launch.json').read_text(encoding='utf-8'))
        if launch.get('pid')!=parent['pid'] or not matches(parent,launch.get('command',[]),root/'search.py'):
            raise ControlError('调度器启动身份不符')
        launched_child=psutil.Process(current['pid'])
        if not matches(child,current.get('command',[]),root/'calculation/run_baseline.py') or launched_child.ppid()!=parent['pid']:
            raise ControlError('计算命令或父子关系不符')
        folder=Path(current['path']).resolve()
        if folder.parent!=root/'calculation':raise ControlError('运行目录不在当前任务中')
        return dict(available=True,root=str(root),folder=str(folder),supervisor=identity(parent),child=identity(child),
                    supervisor_paused=parent['status']==psutil.STATUS_STOPPED,child_paused=child['status']==psutil.STATUS_STOPPED)
    except (OSError,ValueError,KeyError,psutil.Error,ControlError) as exc:return dict(available=False,reason=str(exc))

def verify_token(root:Path,token:dict)->dict:
    live=snapshot(root)
    if not live.get('available'):raise ControlError(live.get('reason','无法核验'))
    if any(live.get(k)!=token.get(k) for k in ('root','folder','supervisor','child')):
        raise ControlError('工况或进程身份改变，请刷新重试')
    return live

def checked_action(target:dict,action:str)->None:
    p=inspect_process(target['pid'])
    if not p or identity(p)!=target:raise ControlError('操作前进程身份已变化')
    proc=psutil.Process(target['pid'])
    if proc.create_time()!=target['started']:raise ControlError('PID 已复用')
    getattr(proc,action)()

def wait_for(target:dict,stopped:bool|None,timeout:float=4)->bool:
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        p=inspect_process(target['pid'])
        if not p or identity(p)!=target:return stopped is None
        if stopped is not None and (p['status']==psutil.STATUS_STOPPED)==stopped:return True
        time.sleep(.05)
    return False

def event(directory:Path,record:dict)->None:
    value={**record,'at':dt.datetime.now().astimezone().isoformat()}
    with (directory/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')
    temp=directory/'state.json.tmp';temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(directory/'state.json')

def execute(root:Path,action:str,token:dict)->dict:
    if action not in ('pause','resume','terminate'):raise ControlError('无效操作')
    root=root.resolve();directory=root/'dashboard_control';directory.mkdir(exist_ok=True)
    with exclusive_lock(directory/'control.lock'):
        live=verify_token(root,token);parent=live['supervisor'];child=live['child']
        record=dict(action=action,folder=live['folder'],target=token)
        event(directory,{**record,'outcome':'requested'})
        froze=False;changed_child=False
        try:
            if action=='resume':
                if (root/'STOP').exists():raise ControlError('已有 STOP 标记，拒绝继续')
                checked_action(child,'resume')
                if not wait_for(child,False):raise ControlError('计算恢复未确认')
                checked_action(parent,'resume')
                if not wait_for(parent,False):raise ControlError('调度器恢复未确认')
                message='已继续原进程与原内存状态'
            else:
                if not live['supervisor_paused']:checked_action(parent,'suspend');froze=True
                if not wait_for(parent,True):raise ControlError('调度器暂停未确认')
                verify_token(root,token)
                if action=='pause':
                    checked_action(child,'suspend');changed_child=True
                    if not wait_for(child,True):raise ControlError('计算暂停未确认')
                    message='已暂停计算与调度器；保留内存'
                else:
                    try:(root/'STOP').open('x',encoding='utf-8').close()
                    except FileExistsError:pass
                    checked_action(parent,'terminate')
                    p=inspect_process(parent['pid'])
                    if p and p['status']==psutil.STATUS_STOPPED:checked_action(parent,'resume')
                    if not wait_for(parent,None):raise ControlError('调度器退出未确认，未终止计算')
                    p=inspect_process(child['pid'])
                    if p:
                        if any(p[k]!=child[k] for k in ('pid','uid','started','command')):raise ControlError('计算身份已变化')
                        child=identity(p);checked_action(child,'terminate');changed_child=True
                        p=inspect_process(child['pid'])
                        if p and p['status']==psutil.STATUS_STOPPED:checked_action(child,'resume')
                        if not wait_for(child,None,8):raise ControlError('计算退出未确认')
                    message='已终止计算与队列；已有文件保留，未落盘状态不可恢复'
            result={**record,'outcome':'confirmed','message':message};event(directory,result);return result
        except Exception as exc:
            if action=='pause' and froze and not changed_child:
                try:checked_action(parent,'resume')
                except (psutil.Error,ControlError):pass
            event(directory,{**record,'outcome':'failed','message':str(exc)});raise
