"""Source-bound numerical acceptance for the expanded solver."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = ("lattices.py","research_options.py","research_solver.py","research_gpu.py","research_kernels.cu","research_cuda.py",
           "research_runtime.py","research_validation.py","cavity_models.py","reference.py","gpu_environment.py","advanced_physics.py")


def fingerprints():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}


def verified_research(backend="cpu"):
    path = ROOT/"validation/correctness_research.json"
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError,ValueError):
        receipt = {}
    if (not receipt.get("passed") or receipt.get("source_sha256")!=fingerprints()
            or not receipt.get("backends",{}).get(backend)):
        raise RuntimeError("扩展求解器未通过当前源码的数值验收，请运行 validate_research.py")
    return receipt["source_sha256"]
