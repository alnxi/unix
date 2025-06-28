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

try:
    import numpy  # noqa: F401
except Exception:
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

if 'python.utils.performance.experiment_logger' not in sys.modules:
    exp_stub = types.ModuleType('experiment_logger')
    class ExperimentLogger:
        def __init__(self, *a, **k):
            pass
        def log_metric(self, *a, **k):
            pass
        def get_metric_summary(self, *a, **k):
            return {}
    exp_stub.ExperimentLogger = ExperimentLogger
    sys.modules['python.utils.performance.experiment_logger'] = exp_stub

if 'python.utils.performance.performance_tracker' not in sys.modules:
    tracker_stub = types.ModuleType('performance_tracker')
    class PerformanceTracker:
        def __init__(self):
            self.operations = {}
        def record_operation(self, *a, **k):
            pass
        def get_operation_stats(self, *a, **k):
            return {}
    tracker_stub.PerformanceTracker = PerformanceTracker
    sys.modules['python.utils.performance.performance_tracker'] = tracker_stub

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


def test_text_dataset_loader(tmp_path):
    dl_mod = import_module('python.utils.data.dataset_loader')

    data_file = tmp_path / 'sample.txt'
    data_file.write_text('hello\nworld\n')

    cfg = dl_mod.DatasetConfig(name='test', path=str(data_file))
    loader = dl_mod.UnifiedDatasetLoader()
    dataset = loader.load_dataset(cfg)

    assert len(dataset) == 2
    assert dataset[0]['text'] == 'hello'

    with pytest.raises(ImportError):
        dl_mod.create_simple_dataloader(dataset)


def test_preprocessing_pipeline():
    pre_mod = import_module('python.utils.data.preprocessing')
    pipeline = pre_mod.create_text_pipeline()
    processed = pipeline.process('<b>Hello</b> World!')
    assert 'hello' in processed


def test_data_validation():
    val_mod = import_module('python.utils.data.validation')
    data = [{'text': 'a'}, {'text': 'a'}, {'text': ''}, {'text': 'b'}]
    validator = val_mod.DataValidator(val_mod.ValidationConfig(min_samples=1))
    result = validator.validate_dataset(data)
    assert result.metrics['total_samples'] == 4
    assert 'duplicate_count' in result.metrics


def test_quality_tracker_analysis():
    if 'python.utils.data.quality' in sys.modules:
        del sys.modules['python.utils.data.quality']
    quality_mod = import_module('python.utils.data.quality')
    tracker = quality_mod.SyntheticDataQualityTracker()
    tracker.log_generation_quality({'batch_size': 2},
                                   {'coherence': 0.8, 'correctness': 0.9})
    analysis = tracker.analyze_data_quality_trends()
    assert analysis['total_batches_analyzed'] == 1


def test_encryption_hash_and_compare():
    enc_mod = import_module('python.utils.security.encryption_utils')
    digest, salt = enc_mod.hash_data('secret')
    assert isinstance(digest, str)
    assert enc_mod.secure_compare(digest, digest)


def test_config_loader_merge_and_env(tmp_path, monkeypatch):
    cfg_mod = import_module('python.utils.setup.config_utils')
    cfg1 = {'a': 1, 'b': {'c': 2}}
    cfg2 = {'b': {'d': 3}}
    loader = cfg_mod.ConfigLoader(config_dir=str(tmp_path))
    merged = loader.merge_configs(cfg1, cfg2)
    assert merged['b']['c'] == 2 and merged['b']['d'] == 3

    env_file = tmp_path / 'env.yaml'
    env_file.write_text('value: ${TEST_ENV:42}')
    monkeypatch.setenv('TEST_ENV', '99')
    loaded = loader.load_yaml(env_file)
    assert loaded['value'] == '99'


def test_experiment_logger_and_tracker(tmp_path):
    if 'python.utils.performance.experiment_logger' in sys.modules:
        del sys.modules['python.utils.performance.experiment_logger']
    if 'python.utils.performance.performance_tracker' in sys.modules:
        del sys.modules['python.utils.performance.performance_tracker']

    exp_mod = import_module('python.utils.performance.experiment_logger')
    tracker_mod = import_module('python.utils.performance.performance_tracker')

    logger = exp_mod.ExperimentLogger('exp', log_dir=str(tmp_path))
    logger.log_metric('m', 1.0, step=1)
    summary = logger.get_metric_summary('m')
    assert summary['count'] == 1

    tracker = tracker_mod.PerformanceTracker()
    tracker.record_operation('op', 10.0, memory_delta_mb=0)
    stats = tracker.get_operation_stats('op')
    assert stats['count'] == 1


def test_progress_context():
    term_mod = import_module('python.utils.logging.terminal_display')
    with term_mod.progress_context('task', total=2) as progress:
        progress.advance()
        progress.advance()
    assert hasattr(progress, 'advance')


