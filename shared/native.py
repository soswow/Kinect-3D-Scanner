"""Optional native kernels; no compilation or dependency installation at runtime.

KINECT_NATIVE=off retains the NumPy reference. auto uses a compatible installed
extension when available. on requires it, so benchmarks cannot time a fallback
while claiming native performance.
"""

import importlib
import os
from functools import lru_cache


@lru_cache(maxsize=1)
def _extension():
    try:
        module = importlib.import_module("_kinect_native")
    except (ImportError, OSError) as exc:
        return None, str(exc)
    if getattr(module, "API_VERSION", None) != 2:
        return (
            None,
            "Installed native extension has an incompatible API; rebuild ./native",
        )
    return module, None


def native_mode():
    mode = os.environ.get("KINECT_NATIVE", "auto").lower()
    if mode not in ("auto", "on", "off"):
        raise ValueError("KINECT_NATIVE must be auto, on, or off")
    return mode


def kernels():
    mode = native_mode()
    if mode == "off":
        return None
    module, reason = _extension()
    if module is None and mode == "on":
        raise RuntimeError(
            f"Native kernels requested but unavailable: {reason}. "
            "Install with the scanner's Python: python -m pip install ./native"
        )
    return module


def native_status():
    module = kernels()
    mode = native_mode()
    return {
        "requested": mode,
        "active": module is not None,
        "api_version": getattr(module, "API_VERSION", None),
        "fallback_reason": _extension()[1]
        if mode != "off" and module is None
        else None,
    }
