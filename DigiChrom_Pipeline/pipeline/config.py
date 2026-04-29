import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def set_config(config_module):
    """Set the active pipeline configuration module."""
    global _config
    _config = config_module


def get_config():
    """Return the currently active pipeline configuration module."""
    if _config is None:
        raise RuntimeError(
            "Pipeline config has not been initialized. "
            "Call pipeline.config.set_config(your_config_module) before using pipeline functions."
        )
    return _config


def reset_config():
    """Reset the pipeline config to the built-in default module."""
    global _config
