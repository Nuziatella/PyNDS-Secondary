"""PyNDS GUI Package.

This package contains the PyNDS graphical user interface components.

Modules:
    simple_gui: Main GUI application for PyNDS emulator
    run_gui: GUI launcher with dependency checking

Usage:
    python gui/run_gui.py
    # or
    python -m gui.run_gui
"""

from .simple_gui import main as run_gui

__all__ = ["run_gui"]
