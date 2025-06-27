"""
Hardware detection utilities for Apple Silicon systems.
Used to validate system requirements for MLX training.
"""

import subprocess
import sys
import platform
import re
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def get_apple_silicon_generation() -> Optional[str]:
    """
    Detect the Apple Silicon chip generation.
    
    Returns:
        Apple Silicon generation (M1, M2, M3, M4, etc.) or None if not Apple Silicon
    """
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None
    
    try:
        # Get system profiler info for hardware
        result = subprocess.run([
            "system_profiler", "SPHardwareDataType"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            # Look for chip name in output
            for line in result.stdout.split('\n'):
                if 'Chip:' in line:
                    # Extract chip name (e.g., "Apple M4 Pro")
                    chip_match = re.search(r'Apple (M\d+)(?:\s+\w+)?', line)
                    if chip_match:
                        return chip_match.group(1)
        
        # Fallback: try sysctl
        result = subprocess.run([
            "sysctl", "-n", "machdep.cpu.brand_string"
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            chip_match = re.search(r'Apple (M\d+)', result.stdout)
            if chip_match:
                return chip_match.group(1)
                
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        logger.warning("Could not detect Apple Silicon generation")
    
    return None


def get_system_memory_gb() -> float:
    """
    Get total system memory in GB.
    
    Returns:
        Total system memory in GB
    """
    if platform.system() == "Darwin":
        try:
            # Use sysctl to get memory info on macOS
            result = subprocess.run([
                "sysctl", "-n", "hw.memsize"
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                memory_bytes = int(result.stdout.strip())
                return memory_bytes / (1024 ** 3)  # Convert to GB
                
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError):
            logger.warning("Could not detect system memory via sysctl")
    
    # Fallback for other systems or if sysctl fails
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        # If psutil not available, try /proc/meminfo on Linux
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        memory_kb = int(line.split()[1])
                        return memory_kb / (1024 ** 2)  # Convert KB to GB
        except (FileNotFoundError, ValueError, IndexError):
            logger.debug("Could not parse /proc/meminfo, continuing with other methods.")
    
    logger.warning("Could not detect system memory")
    return 0.0


def check_mlx_requirements() -> Dict[str, Any]:
    """
    Check system requirements for MLX training.
    
    Returns:
        Dictionary with requirement check results
    """
    silicon_gen = get_apple_silicon_generation()
    memory_gb = get_system_memory_gb()
    
    # Check if Apple Silicon generation is M3 or newer
    is_silicon_compatible = False
    silicon_warning = None
    
    if silicon_gen:
        # Extract generation number (M1 -> 1, M2 -> 2, etc.)
        try:
            gen_num = int(silicon_gen[1:])  # Skip 'M' prefix
            is_silicon_compatible = gen_num >= 3
            if not is_silicon_compatible:
                silicon_warning = f"Apple {silicon_gen} detected. MLX training optimized for M3+ chips. Performance may be limited."
        except ValueError:
            silicon_warning = f"Could not parse Apple Silicon generation: {silicon_gen}"
    else:
        silicon_warning = "Not running on Apple Silicon. MLX backend may not be available."
    
    # Check memory requirements (18GB minimum recommended)
    is_memory_sufficient = memory_gb >= 18.0
    memory_warning = None
    if not is_memory_sufficient and memory_gb > 0:
        memory_warning = f"Only {memory_gb:.1f}GB unified memory detected. MLX training recommends ≥18GB for optimal performance."
    elif memory_gb == 0:
        memory_warning = "Could not detect system memory. Unable to verify MLX requirements."
    
    return {
        'silicon_generation': silicon_gen,
        'memory_gb': memory_gb,
        'is_silicon_compatible': is_silicon_compatible,
        'is_memory_sufficient': is_memory_sufficient,
        'silicon_warning': silicon_warning,
        'memory_warning': memory_warning,
        'mlx_recommended': is_silicon_compatible and is_memory_sufficient
    }


def validate_mlx_requirements(strict: bool = False) -> bool:
    """
    Validate system requirements for MLX and log warnings/errors.
    
    Args:
        strict: If True, raise SystemExit on requirement failures
        
    Returns:
        True if requirements are met or warnings only, False if strict validation fails
    """
    requirements = check_mlx_requirements()
    
    # Log silicon generation info
    if requirements['silicon_generation']:
        logger.info(f"🍎 Detected Apple {requirements['silicon_generation']} chip")
    
    # Log memory info
    if requirements['memory_gb'] > 0:
        logger.info(f"💾 System memory: {requirements['memory_gb']:.1f}GB")
    
    # Handle warnings and errors
    validation_passed = True
    
    if requirements['silicon_warning']:
        if strict and not requirements['is_silicon_compatible']:
            logger.error(requirements['silicon_warning'])
            validation_passed = False
        else:
            logger.warning(requirements['silicon_warning'])
    
    if requirements['memory_warning']:
        if strict and not requirements['is_memory_sufficient']:
            logger.error(requirements['memory_warning'])
            logger.error("MLX training requires ≥18GB unified memory. Aborting.")
            validation_passed = False
        else:
            logger.warning(requirements['memory_warning'])
    
    if requirements['mlx_recommended']:
        logger.info("✅ System meets MLX training recommendations")
    elif not strict:
        logger.warning("⚠️  System may not be optimal for MLX training")
    
    if strict and not validation_passed:
        logger.error("System requirements validation failed. Use UT_USE_TORCH=1 to fall back to PyTorch.")
        sys.exit(1)
    
    return validation_passed


if __name__ == "__main__":
    # Command-line usage for testing
    import json
    requirements = check_mlx_requirements()
    print(json.dumps(requirements, indent=2))