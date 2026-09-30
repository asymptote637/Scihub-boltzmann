"""OS-only adaptations; numerical kernels are unchanged."""
from pathlib import Path
import os
import psutil

def exclusive_lock(path):
    handle = Path(path).open('a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            if handle.seek(0, 2) == 0:
                handle.write(b'0'); handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handle
    except OSError:
        handle.close()
        raise BlockingIOError('Another controller holds the lock') from None

def assert_no_runner(script):
    expected = os.path.normcase(str(Path(script).resolve()))
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            args = proc.info['cmdline'] or []
            if any(os.path.normcase(str(Path(arg).resolve())) == expected for arg in args[1:] if arg and not arg.startswith('-')):
                raise RuntimeError('Existing calculation process: PID ' + str(proc.pid))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
