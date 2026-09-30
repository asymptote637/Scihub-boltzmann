import contextlib
import io
import json
from pathlib import Path
import platform
import sys

import gpu_environment
import cupy as cp
import numpy as np
from reference import verify_reference

capture = io.StringIO()
with contextlib.redirect_stdout(capture):
    cp.show_config()
values = cp.ones((256, 256, 9), dtype=cp.float64)
total = float(values.sum())
assert total == 589824.0
kernel = cp.RawKernel('extern "C" __global__ void twice(double* a) { a[threadIdx.x] *= 2.0; }',
                      "twice", options=("--fmad=false",))
kernel((1,), (32,), (values,))
cp.cuda.get_current_stream().synchronize()
assert float(values.sum()) == 589856.0
device = cp.cuda.runtime.getDeviceProperties(0)
name = device["name"]
if isinstance(name, bytes):
    name = name.decode()
report = dict(passed=True, python=sys.version, executable=sys.executable, numpy=np.__version__,
              cupy=cp.__version__, platform=platform.platform(), gpu=name,
              compute_capability=f'{device["major"]}.{device["minor"]}',
              vram_bytes=device["totalGlobalMem"], frozen_sources=verify_reference(),
              cuda_runtime=cp.cuda.runtime.runtimeGetVersion(), cuda_configuration=capture.getvalue(),
              double_precision_kernel_verified=True)
Path("validation/gpu_environment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
