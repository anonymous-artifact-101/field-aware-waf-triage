
from __future__ import annotations

import importlib.util
from pathlib import Path

_SMOKE = Path(__file__).resolve().parents[1] / "scripts" / "98_smoke_test.py"

def test_smoke_pipeline_runs():
    spec = importlib.util.spec_from_file_location("_pecti_smoke", _SMOKE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main() == 0
