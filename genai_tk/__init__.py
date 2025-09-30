"""GenAI Toolkit - Core AI components for building applications."""

__version__ = "0.1.0"

# Import main components to make them easily accessible
try:
    from . import core
    from . import extra  
    from . import utils
except ImportError:
    # Handle case where dependencies aren't installed
    pass