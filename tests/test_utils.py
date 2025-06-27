import time
import pytest
import sys
import types
import importlib
from pathlib import Path

# Ensure repository root is on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Provide lightweight stubs for optional dependencies
if 'torch' not in sys.modules:
    torch_stub = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    )
    sys.modules['torch'] = torch_stub

if 'numpy' not in sys.modules:
    sys.modules['numpy'] = types.ModuleType('numpy')

if 'psutil' not in sys.modules:
    psutil_stub = types.ModuleType('psutil')
    psutil_stub.virtual_memory = lambda: types.SimpleNamespace(total=0, percent=0, available=0)
    psutil_stub.cpu_count = lambda: 0
    psutil_stub.Process = lambda: types.SimpleNamespace(memory_info=lambda: types.SimpleNamespace(rss=0))
    sys.modules['psutil'] = psutil_stub

if 'rich' not in sys.modules:
    rich_stub = types.ModuleType('rich')
    class _DummyRichHandler:
        def __init__(self, *a, **k):
            pass
        def emit(self, *a, **k):
            pass
    rich_stub.console = types.ModuleType('rich.console')
    rich_stub.console.Console = object
    rich_stub.highlighter = types.ModuleType('rich.highlighter')
    rich_stub.highlighter.ReprHighlighter = object
    rich_stub.logging = types.ModuleType('rich.logging')
    rich_stub.logging.RichHandler = _DummyRichHandler
    rich_stub.theme = types.ModuleType('rich.theme')
    rich_stub.theme.Theme = object
    rich_stub.traceback = types.ModuleType('rich.traceback')
    rich_stub.traceback.install = lambda *a, **k: None
    sys.modules['rich'] = rich_stub
    sys.modules['rich.console'] = rich_stub.console
    sys.modules['rich.highlighter'] = rich_stub.highlighter
    sys.modules['rich.logging'] = rich_stub.logging
    sys.modules['rich.theme'] = rich_stub.theme
    sys.modules['rich.traceback'] = rich_stub.traceback

if 'python.utils.data.quality' not in sys.modules:
    quality_stub = types.ModuleType('quality')
    class SyntheticDataQualityTracker:
        def log_generation_quality(self, *a, **k):
            pass
    quality_stub.SyntheticDataQualityTracker = SyntheticDataQualityTracker
    sys.modules['python.utils.data.quality'] = quality_stub

# Helper functions to import modules with graceful skip

def import_module(name: str):
    try:
        return importlib.import_module(name)
    except Exception as exc:
        pytest.skip(f"Could not import {name}: {exc}")


def test_get_logger_returns_logger():
    logger_mod = import_module('python.utils.logging.logger')
    logger = logger_mod.get_logger('test_logger')
    assert logger.name == 'test_logger'


def test_profile_records_metrics():
    prof_mod = import_module('python.utils.performance.profiling')
    profiler = prof_mod.get_profiler()
    profiler.clear_metrics()
    with prof_mod.profile('sample_op'):
        time.sleep(0.01)
    metrics = profiler.get_all_metrics()
    assert any(m.operation_name == 'sample_op' for m in metrics)


def test_memory_monitor_snapshot():
    prof_mod = import_module('python.utils.performance.profiling')
    monitor = prof_mod.MemoryMonitor()
    snapshot = monitor.get_memory_snapshot('test')
    assert snapshot.process_memory_gb >= 0


