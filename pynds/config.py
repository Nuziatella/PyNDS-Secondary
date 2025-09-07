"""Configuration interface for PyNDS emulator settings.

This module provides access to the underlying C++ configuration system
from the NooDS emulator. Configuration options control various aspects
of emulation behavior including rendering quality, performance settings,
and compatibility options.
"""

from cnds import config

__all__ = [config]
