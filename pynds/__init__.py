"""PyNDS: A Python wrapper for the Nintendo DS emulator.

Your digital time machine to the Nintendo DS universe! This package brings the
power of NooDS emulation to Python, perfect for bot development and automated gaming magic.

Main Components:
    PyNDS: Your core emulator class for loading ROMs and controlling emulation
    config: Configuration management for your emulator settings
    button: Input handling for buttons and touch screen interactions
    memory: Memory access and manipulation toolkit
    window: Display and rendering interface for your games

Examples
--------
    >>> import pynds
    >>> # Basic emulator usage
    >>> nds = pynds.PyNDS("game.nds")
    >>> nds.tick()
    >>> top, bottom = nds.get_frame()
    
    >>> # Advanced features (Roadmap 3)
    >>> state = nds.save_state()
    >>> nds.load_state(state)
    >>> nds.export_frame("screenshot.png")
"""

from .config import config
from .pynds import PyNDS

__all__ = [PyNDS, config]
