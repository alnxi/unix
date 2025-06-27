# UnifiedTransformer Utilities

This repository provides a collection of utility modules used by the
**UnifiedTransformer** project.  The utilities focus on advanced logging,
performance profiling and Apple Silicon optimisation but are designed to be
platform agnostic.  Key features include:

- Modular logging with rich terminal output
- Dataset loading and preprocessing helpers
- Performance and memory profiling tools
- MLX integration for Apple Silicon devices
- Model analysis and optimisation helpers
- Security utilities for encryption and key management
- Configuration and hardware detection helpers

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

## Basic Usage

Import the utilities that you need from the `utils` package:

```python
from python.utils import get_logger, get_memory_monitor, load_config

logger = get_logger(__name__)
monitor = get_memory_monitor()
config = load_config("configs/model/base.yaml")
```

See `python/utils/README.md` for detailed information on each module.

## Development

1. Create a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run your preferred test runner (for example `pytest`) to ensure changes
   do not break existing functionality.

