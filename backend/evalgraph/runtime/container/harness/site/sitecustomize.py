"""Auto-imported by Python when this directory is on PYTHONPATH: turns on EvalGraph probes in the app process."""

import os
import sys

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _here not in sys.path:
    sys.path.insert(0, _here)
try:
    import probe

    probe.install_from_env()
except Exception as _exc:  # instrumentation must never break the app
    sys.stderr.write(f"[evalgraph probe disabled: {_exc}]\n")
