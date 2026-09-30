"""Source-bound verification receipt for the extended cavity model family."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = ("cavity_models.py", "reference.py", "gpu_solver.py", "gpu_fused.py", "kernels.cu", "model_validation.py")


def fingerprints():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def verified_models():
    path = ROOT / "validation/correctness_models.json"
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise RuntimeError("Run validate_models.py first to validate BGK/MRT and NEE/halfway") from None
    if not receipt.get("passed") or receipt.get("source_sha256") != fingerprints():
        raise RuntimeError("Model sources changed or verification failed; run validate_models.py")
    return receipt["source_sha256"]
