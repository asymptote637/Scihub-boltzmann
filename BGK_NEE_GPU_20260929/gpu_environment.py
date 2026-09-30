"""Discover environment-local CUDA wheels without changing system settings."""
import os
from pathlib import Path
import sys
import sysconfig

_DLL_HANDLES = []


def configure_cuda():
    site = Path(sysconfig.get_paths()["purelib"])
    runtime = site / "nvidia/cuda_runtime"
    compiler = site / "nvidia/cuda_nvrtc"
    if runtime.is_dir():
        os.environ.setdefault("CUDA_PATH", str(runtime))
    if os.name == "nt":
        for folder in (runtime / "bin", compiler / "bin"):
            if folder.is_dir():
                _DLL_HANDLES.append(os.add_dll_directory(str(folder)))
                os.environ["PATH"] = str(folder) + os.pathsep + os.environ.get("PATH", "")
        cache = Path(sys.prefix) / "cupy_cache"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("CUPY_CACHE_DIR", str(cache))


configure_cuda()
